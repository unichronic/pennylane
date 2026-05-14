import json
import time
from typing import Any, Type

from agno.agent import Agent
from agno.models.base import Model
from agno.models.message import Message
from agno.models.mistral import MistralChat
from agno.models.response import ModelResponse
from pydantic import BaseModel

from agents.local_llm import LocalStructuredLLM, LocalTradingLLM
from config import get_config, llm_max_tokens_for, select_mistral_api_key
from core.llm_observability import LLMProviderError, raise_if_agno_error_content, traced_llm_call


class AgnoResponse:
    def __init__(self, content):
        self.content = content


class LocalAgnoModel(Model):
    def __init__(self, schema: Type[BaseModel] | None = None):
        super().__init__(id="local-trading-adapter", name="Local Trading Adapter", provider="local")
        self.local = LocalTradingLLM()
        self.schema = schema

    def invoke(self, *args, **kwargs) -> ModelResponse:
        messages = kwargs.get("messages") or []
        response_format = kwargs.get("response_format")
        prompt = self._prompt_from_messages(messages)
        schema = self.schema or response_format
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            obj = LocalStructuredLLM(self.local, schema).invoke(prompt)
            return ModelResponse(role="assistant", content=obj.model_dump_json(), parsed=obj)
        return ModelResponse(role="assistant", content=self.local.invoke(prompt).content)

    async def ainvoke(self, *args, **kwargs) -> ModelResponse:
        return self.invoke(*args, **kwargs)

    def invoke_stream(self, *args, **kwargs):
        yield self.invoke(*args, **kwargs)

    async def ainvoke_stream(self, *args, **kwargs):
        yield self.invoke(*args, **kwargs)

    def _parse_provider_response(self, response: Any, **kwargs) -> ModelResponse:
        return response

    def _parse_provider_response_delta(self, response_delta: Any, **kwargs) -> ModelResponse:
        return response_delta

    def _prompt_from_messages(self, messages):
        parts = []
        for msg in messages:
            if isinstance(msg, Message):
                if msg.content:
                    parts.append(str(msg.content))
            elif isinstance(msg, dict):
                parts.append(str(msg.get("content", "")))
            else:
                parts.append(str(msg))
        return "\n".join(parts)


class AgnoLLMAdapter:
    def __init__(self, name="Trading Agent", role=None, schema: Type[BaseModel] | None = None, kind="quick"):
        self.name = name
        self.schema = schema
        self.kind = kind
        self.mistral_key_slot = None
        model = self._model()
        self.model = model
        self.model_id = getattr(model, "id", None) or getattr(model, "name", None)
        self.provider = getattr(model, "provider", None) or get_config()["llm_provider"]
        self._local_model = isinstance(model, LocalAgnoModel)
        self.agent = Agent(
            name=name,
            role=role,
            model=model,
            output_schema=schema,
            parse_response=schema is not None,
            markdown=True,
            telemetry=False,
        )

    def invoke(self, prompt):
        prompt_text = self._prompt(prompt)
        cfg = get_config()
        max_attempts = 1 + max(0, cfg.get("llm_rate_limit_retries", 0))
        for attempt in range(1, max_attempts + 1):
            try:
                with traced_llm_call(
                    agent_name=self.name,
                    provider=self.provider,
                    model=self.model_id,
                    kind=self.kind,
                    key_slot=self.mistral_key_slot,
                    prompt=prompt_text,
                    call_site=f"AgnoLLMAdapter.invoke attempt={attempt}",
                ) as trace:
                    run = self.agent.run(prompt_text)
                    content = getattr(run, "content", "")
                    trace.set_response(content)
                    raise_if_agno_error_content(content)
                break
            except LLMProviderError as exc:
                if not getattr(exc, "retryable", False) or attempt >= max_attempts:
                    raise
                time.sleep(float(cfg.get("llm_rate_limit_backoff_seconds", 30)) * attempt)
        content = run.content
        if self.schema is not None:
            if isinstance(content, self.schema):
                return content
            if isinstance(content, BaseModel):
                return self.schema.model_validate(content.model_dump())
            if isinstance(content, dict):
                return self.schema.model_validate(content)
            if isinstance(content, str):
                return self.schema.model_validate_json(content)
        return AgnoResponse(content)

    def with_structured_output(self, schema):
        return AgnoLLMAdapter(name=self.name, schema=schema, kind=self.kind)

    def _prompt(self, prompt):
        if isinstance(prompt, list):
            return "\n\n".join(str(x.get("content", x)) if isinstance(x, dict) else str(x) for x in prompt)
        return str(prompt)

    def _model(self):
        cfg = get_config()
        provider = cfg["llm_provider"].lower()
        if provider == "local":
            return LocalAgnoModel(self.schema)
        if provider == "mistral":
            import os

            key, slot = select_mistral_api_key(self.kind, self.name)
            self.mistral_key_slot = slot
            if not key:
                raise RuntimeError(
                    f"Missing Mistral API key for {self.name}. Set MISTRAL_API_KEY1/MISTRAL_API_KEY2 or use LLM_PROVIDER=local explicitly."
                )
            model_id = cfg["quick_think_llm"] if self.kind == "quick" else cfg["deep_think_llm"]
            client_params = {"timeout_ms": cfg.get("llm_call_timeout_seconds", 120) * 1000}
            if cfg["backend_url"] and cfg["backend_url"] != "https://api.mistral.ai/v1":
                client_params["server_url"] = cfg["backend_url"]
            kwargs = {
                "id": model_id,
                "api_key": key,
                "retries": cfg.get("llm_max_retries", 0),
                "max_tokens": llm_max_tokens_for(self.kind, self.name),
                "client_params": client_params,
            }
            model = MistralChat(**kwargs)
            object.__setattr__(model, "tradeage_mistral_key_slot", slot)
            return model
        raise RuntimeError(f"Unsupported LLM_PROVIDER={cfg['llm_provider']}")


def get_agno_llm(kind="quick", name="Trading Agent", role=None):
    return AgnoLLMAdapter(name=name, role=role, kind=kind)


def create_agno_model(kind="quick", name="Trading Agent", role=None, schema: Type[BaseModel] | None = None):
    return AgnoLLMAdapter(name=name, role=role, schema=schema, kind=kind)._model()
