import json
import re
from pathlib import Path
from typing import Any

from config import get_config


_SAFE_ID = re.compile(r"[^A-Za-z0-9_.-]+")


def _safe_id(value: str) -> str:
    return _SAFE_ID.sub("_", str(value))[:160]


def checkpoint_path(session_id: str) -> Path:
    cfg = get_config()
    directory = Path(cfg["data_cache_dir"]) / "checkpoints"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{_safe_id(session_id)}.json"


def _jsonable(value: Any):
    try:
        json.dumps(value)
        return value
    except TypeError:
        if isinstance(value, dict):
            return {str(k): _jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_jsonable(v) for v in value]
        return str(value)


def load_checkpoint(session_id: str) -> dict:
    path = checkpoint_path(session_id)
    if not path.exists():
        return {"completed": [], "outputs": {}, "session_state": {}}
    return json.loads(path.read_text())


def save_checkpoint(session_id: str, step: str, session_state: dict, output: Any) -> None:
    checkpoint = load_checkpoint(session_id)
    if step not in checkpoint["completed"]:
        checkpoint["completed"].append(step)
    checkpoint["outputs"][step] = _jsonable(output)
    checkpoint["session_state"] = _jsonable(session_state)
    path = checkpoint_path(session_id)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(checkpoint, indent=2))
    tmp.replace(path)


def clear_checkpoint(session_id: str) -> None:
    path = checkpoint_path(session_id)
    if path.exists():
        path.unlink()
