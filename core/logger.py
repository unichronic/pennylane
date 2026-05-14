import json
from pathlib import Path


def log_stage(path, stage, payload):
    if not path:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as f:
        f.write(json.dumps({"stage": stage, "payload": payload}, default=str) + "\n")
