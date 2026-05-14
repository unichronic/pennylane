# Penny Lane Capital UI

React/Vite frontend for visualizing and launching the multi-agent trading workflow.

The frontend connects to the Python API in `api/server.py` through
`src/api/orchestratorClient.ts`. During local development Vite proxies `/api`
to `http://127.0.0.1:8000`.

## Run

```bash
python -m api.server
```

In a second shell:

```bash
npm install
npm run dev
```

Or from repo root:

```bash
./start.sh
```

## Backend Connection Points

The client implements:

```ts
interface OrchestratorClient {
  loadDefault(): Promise<{ config: RunConfig; snapshot: RunSnapshot }>;
  startRun(config: RunConfig): AsyncIterable<RunSnapshot>;
  stopRun(runId: string): Promise<void>;
}
```

Backend endpoints:

```text
GET  /api/config/default
GET  /api/health
POST /api/runs
GET  /api/runs/:id/snapshot
POST /api/runs/:id/stop
```

The UI is already structured around:

- run controls
- orchestration stream
- agent status lanes
- tool calls and data-quality labels
- reports
- portfolio/final decision/audit panels
