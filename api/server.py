from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from api.orchestrator import (
    default_config,
    get_run_snapshot,
    initial_snapshot,
    start_run_job,
    stop_run_job,
)


class TradeageApiHandler(BaseHTTPRequestHandler):
    server_version = "penny-lane-capital-api/0.1"

    def do_OPTIONS(self):
        self._send_json({"ok": True})

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/health":
            self._send_json({"ok": True, "service": "penny-lane-capital-api"})
            return
        if path == "/api/config/default":
            config = default_config()
            self._send_json({"config": config, "snapshot": initial_snapshot(config)})
            return
        if path.startswith("/api/runs/") and path.endswith("/snapshot"):
            run_id = path.split("/")[3]
            snapshot = get_run_snapshot(run_id)
            if not snapshot:
                self._send_json({"error": "run not found"}, status=404)
                return
            self._send_json({"snapshot": snapshot})
            return
        self._send_json({"error": "not found"}, status=404)

    def do_POST(self):
        path = urlparse(self.path).path
        if path == "/api/runs":
            payload = self._read_json()
            snapshot = start_run_job(payload.get("config", payload))
            self._send_json({"snapshot": snapshot}, status=202)
            return
        if path.startswith("/api/runs/") and path.endswith("/stop"):
            run_id = path.split("/")[3]
            snapshot = stop_run_job(run_id)
            if not snapshot:
                self._send_json({"error": "run not found"}, status=404)
                return
            self._send_json({
                "ok": True,
                "snapshot": snapshot,
                "message": "Stop acknowledged. The UI-visible backend job was marked stopped.",
            })
            return
        self._send_json({"error": "not found"}, status=404)

    def log_message(self, fmt, *args):
        print(f"{self.address_string()} - {fmt % args}")

    def _read_json(self):
        length = int(self.headers.get("Content-Length", "0") or "0")
        if not length:
            return {}
        body = self.rfile.read(length).decode("utf-8")
        try:
            data = json.loads(body)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON body: {exc}") from exc
        return data if isinstance(data, dict) else {}

    def _send_json(self, payload, *, status=200):
        data = json.dumps(payload, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(data)


def run(host="127.0.0.1", port=8000):
    server = ThreadingHTTPServer((host, port), TradeageApiHandler)
    print(f"Penny Lane Capital API listening on http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    run()
