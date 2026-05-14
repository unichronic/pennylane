import pytest

from agents.agno_runtime import AgnoLLMAdapter
from agents.structured import bind_structured, invoke_structured_or_freetext
from agents.schemas import ResearchPlan


def test_mistral_without_keys_fails_loudly(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mistral")
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    for idx in range(1, 17):
        monkeypatch.delenv(f"MISTRAL_API_KEY{idx}", raising=False)
    monkeypatch.delenv("MISTRAL_API_KEYS", raising=False)

    with pytest.raises(RuntimeError, match="Missing Mistral API key"):
        AgnoLLMAdapter(name="no fallback")


def test_structured_output_failure_is_not_retried_as_free_text(monkeypatch):
    class BrokenStructured:
        def invoke(self, prompt):
            raise ValueError("structured failed")

    class PlainShouldNotRun:
        called = False

        def invoke(self, prompt):
            self.called = True
            raise AssertionError("plain fallback should not run")

    plain = PlainShouldNotRun()

    with pytest.raises(ValueError, match="structured failed"):
        invoke_structured_or_freetext(
            BrokenStructured(),
            plain,
            "not a trading prompt that local structured parser can convert",
            lambda x: "should not render",
            "Research Manager",
        )
    assert plain.called is False
