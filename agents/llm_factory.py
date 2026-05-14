from agents.agno_runtime import get_agno_llm


def get_llm(kind="quick", role=None):
    name = role or f"{kind.title()} Trading Agent"
    return get_agno_llm(kind=kind, name=name, role=role)
