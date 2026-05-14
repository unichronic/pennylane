import { AgentNode, OrchestratorClient, RunConfig, RunMode, RunSnapshot, ToolCall } from "../types";

const API_BASE = import.meta.env.VITE_TRADEAGE_API_BASE ?? "/api";
const sleep = (ms: number) => new Promise((resolve) => window.setTimeout(resolve, ms));

export const defaultConfig: RunConfig = {
  symbol: "AAPL",
  symbols: "AAPL,GOOGL,AMZN",
  mode: "live",
  startDate: "2024-03-01",
  endDate: "2024-03-08",
  cash: 10000,
  dataProvider: "yfinance",
  baseline: "buy_and_hold",
  llmProvider: "local",
  debateRounds: 1,
  checkpointing: true,
  logPath: "runs/latest.jsonl"
};

const baseAgents: AgentNode[] = [
  { id: "load_market", label: "Market Loader", group: "data", status: "idle", summary: "Fetches the visible OHLCV window", progress: 0 },
  { id: "market", label: "Market Analyst", group: "analysis", status: "idle", summary: "Price, indicators, volatility, and volume", progress: 0 },
  { id: "news", label: "News Analyst", group: "analysis", status: "idle", summary: "Company, macro, and insider context", progress: 0 },
  { id: "sentiment", label: "Sentiment Analyst", group: "analysis", status: "idle", summary: "Public mood and discussion proxies", progress: 0 },
  { id: "fundamentals", label: "Fundamentals Analyst", group: "analysis", status: "idle", summary: "Statements and company quality", progress: 0 },
  { id: "bull", label: "Bull Researcher", group: "research", status: "idle", summary: "Builds the pro-trade thesis", progress: 0 },
  { id: "bear", label: "Bear Researcher", group: "research", status: "idle", summary: "Challenges the trade and downside case", progress: 0 },
  { id: "research_manager", label: "Research Manager", group: "research", status: "idle", summary: "Synthesizes the investment plan", progress: 0 },
  { id: "trader", label: "Trader", group: "trading", status: "idle", summary: "Turns the plan into a transaction", progress: 0 },
  { id: "risk_debate", label: "Risk Debate", group: "risk", status: "idle", summary: "Aggressive, conservative, and neutral review", progress: 0 },
  { id: "portfolio_manager", label: "Portfolio Manager", group: "risk", status: "idle", summary: "Makes the final portfolio call", progress: 0 },
  { id: "execution", label: "Paper Execution", group: "execution", status: "idle", summary: "Applies the signal and tracks equity", progress: 0 }
];

const baselineAgents: AgentNode[] = [
  { id: "load_market", label: "Market Loader", group: "data", status: "idle", summary: "Fetches the historical window for the baseline run", progress: 0 },
  { id: "baseline", label: "Baseline Strategy", group: "trading", status: "idle", summary: "Runs the selected rule-based benchmark", progress: 0 },
  { id: "execution", label: "Paper Execution", group: "execution", status: "idle", summary: "Tracks fills, equity, and drawdown", progress: 0 }
];

const experimentAgents: AgentNode[] = [
  { id: "experiment_inputs", label: "Experiment Loader", group: "data", status: "idle", summary: "Loads one historical window per symbol", progress: 0 },
  { id: "batch_runner", label: "Benchmark Runner", group: "execution", status: "idle", summary: "Runs Penny Lane and baselines across the batch", progress: 0 },
  { id: "aggregate", label: "Aggregate Results", group: "execution", status: "idle", summary: "Builds the combined strategy comparison", progress: 0 }
];

function sanitizeConfig(raw: Partial<RunConfig> & Record<string, unknown>): RunConfig {
  return {
    symbol: String(raw.symbol ?? defaultConfig.symbol),
    symbols: String(raw.symbols ?? defaultConfig.symbols),
    mode: (raw.mode ?? defaultConfig.mode) as RunConfig["mode"],
    startDate: String(raw.startDate ?? defaultConfig.startDate),
    endDate: String(raw.endDate ?? defaultConfig.endDate),
    cash: Number(raw.cash ?? defaultConfig.cash),
    dataProvider: (raw.dataProvider ?? defaultConfig.dataProvider) as RunConfig["dataProvider"],
    baseline: (raw.baseline ?? defaultConfig.baseline) as RunConfig["baseline"],
    llmProvider: (raw.llmProvider ?? defaultConfig.llmProvider) as RunConfig["llmProvider"],
    debateRounds: Number(raw.debateRounds ?? defaultConfig.debateRounds),
    checkpointing: Boolean(raw.checkpointing ?? defaultConfig.checkpointing),
    logPath: String(raw.logPath ?? defaultConfig.logPath)
  };
}

function sanitizeSnapshot(raw: RunSnapshot): RunSnapshot {
  return {
    ...raw,
    config: sanitizeConfig(raw.config as Partial<RunConfig> & Record<string, unknown>)
  };
}

function agentsForMode(mode: RunMode): AgentNode[] {
  if (mode === "baseline") return baselineAgents;
  if (mode === "paper_experiment") return experimentAgents;
  return baseAgents;
}

function placeholderReports(config: RunConfig): RunSnapshot["reports"] {
  if (config.mode === "baseline") {
    return {
      market: "Baseline report will show the selected rule set, period, and metrics.",
      news: "News analysis is not used in baseline mode.",
      sentiment: "Sentiment analysis is not used in baseline mode.",
      fundamentals: "Fundamentals analysis is not used in baseline mode."
    };
  }
  if (config.mode === "paper_experiment") {
    return {
      market: "Experiment output will show the aggregate comparison once the batch completes.",
      news: "The experiment view does not produce one combined news report.",
      sentiment: "The experiment view does not produce one combined sentiment report.",
      fundamentals: "The experiment view does not produce one combined fundamentals report."
    };
  }
  return {
    market: "No market report yet.",
    news: "News context is not loaded yet.",
    sentiment: "Sentiment report is not loaded yet.",
    fundamentals: "Fundamentals report is not loaded yet."
  };
}

function toolPlan(config: RunConfig, status: ToolCall["status"]): ToolCall[] {
  if (config.mode === "baseline") {
    const calls: ToolCall[] = [
      {
        id: "tool-1",
        agentId: "load_market",
        tool: "get_stock_data",
        args: { symbol: config.symbol, start_date: config.startDate, end_date: config.endDate },
        source: config.dataProvider,
        dataQuality: "strict_point_in_time",
        status,
        observation: "Waiting for the historical baseline window."
      },
      {
        id: "tool-2",
        agentId: "baseline",
        tool: "run_baseline_strategy",
        args: { strategy: config.baseline, symbol: config.symbol },
        source: "backend baseline engine",
        dataQuality: "simulated_or_derived",
        status,
        observation: "Waiting for the rule-based benchmark run."
      }
    ];
    if (config.baseline !== "buy_and_hold") {
      calls.splice(1, 0, {
        id: "tool-1b",
        agentId: "baseline",
        tool: "get_indicators",
        args: { strategy: config.baseline, inputs: "price-derived indicators" },
        source: "backend indicators",
        dataQuality: "simulated_or_derived",
        status,
        observation: "Waiting for the derived indicator inputs."
      });
    }
    return calls;
  }

  if (config.mode === "paper_experiment") {
    return [
      {
        id: "tool-1",
        agentId: "experiment_inputs",
        tool: "load_symbol_windows",
        args: { symbols: config.symbols, start_date: config.startDate, end_date: config.endDate },
        source: config.dataProvider,
        dataQuality: "strict_point_in_time",
        status,
        observation: "Waiting for the experiment market windows."
      },
      {
        id: "tool-2",
        agentId: "batch_runner",
        tool: "run_strategy_batch",
        args: { symbols: config.symbols, strategies: "PennyLaneCapital,buy_and_hold,macd,kdj_rsi,zmr,sma" },
        source: "backend paper engine",
        dataQuality: "simulated_or_derived",
        status,
        observation: "Waiting for the multi-strategy experiment batch."
      },
      {
        id: "tool-3",
        agentId: "aggregate",
        tool: "aggregate_equity_curves",
        args: { symbols: config.symbols },
        source: "backend aggregator",
        dataQuality: "simulated_or_derived",
        status,
        observation: "Waiting for the aggregate comparison table."
      }
    ];
  }

  return [
    {
      id: "tool-1",
      agentId: "market",
      tool: "get_stock_data",
      args: { symbol: config.symbol, start_date: config.startDate, end_date: config.endDate },
      source: config.dataProvider,
      dataQuality: "strict_point_in_time",
      status,
      observation: "Waiting for the historical market window."
    },
    {
      id: "tool-2",
      agentId: "market",
      tool: "get_indicators",
      args: { indicators: "rsi,macd,boll,atr,vwma" },
      source: "local",
      dataQuality: "simulated_or_derived",
      status,
      observation: "Waiting for the indicator set."
    },
    {
      id: "tool-3",
      agentId: "news",
      tool: "get_news",
      args: {},
      source: "backend configured",
      dataQuality: "publication_time",
      status,
      observation: "Waiting for the news checks."
    },
    {
      id: "tool-4",
      agentId: "fundamentals",
      tool: "get_fundamentals",
      args: { label: "current_snapshot" },
      source: "configured backend vendor",
      dataQuality: "current_snapshot",
      status,
      observation: "Waiting for the company snapshot."
    }
  ];
}

export function cliEquivalent(config: RunConfig) {
  const args = ["python", "main.py"];
  if (config.mode !== "paper_experiment") {
    args.push(config.symbol);
  }
  args.push("--start", config.startDate);
  args.push("--end", config.endDate);
  args.push("--cash", String(config.cash));
  args.push("--log", config.logPath);
  args.push("--data-provider", config.dataProvider);
  if (config.mode === "backtest") {
    args.push("--backtest");
  }
  if (config.mode === "paper_experiment") {
    args.push("--paper-experiment", "--symbols", config.symbols);
  }
  if (config.mode === "baseline") {
    args.push("--baseline", config.baseline);
  }
  return args.join(" ");
}

export function initialSnapshot(config = defaultConfig, status: RunSnapshot["status"] = "idle"): RunSnapshot {
  const agents = agentsForMode(config.mode);
  return {
    runId: `client-${Date.now()}`,
    status,
    config,
    agents: agents.map((agent) => ({ ...agent, status: status === "running" ? "queued" : "idle", progress: 0 })),
    events: [],
    toolCalls: toolPlan(config, "queued"),
    reports: placeholderReports(config),
    portfolio: {
      cash: config.cash,
      shares: 0,
      equity: config.cash,
      position: "flat",
      pnl: 0,
      stopLoss: null
    },
    signal: {
      rating: config.mode === "paper_experiment" ? "Experiment pending" : "Pending",
      action: config.mode === "paper_experiment" ? "summary" : "hold",
      confidence: 0,
      riskNotes: "Run has not started yet."
    },
    audit: {
      fullStateLog: "",
      jsonlLog: config.logPath,
      checkpointDb: "data_cache/workflows.db",
      cliEquivalent: cliEquivalent(config),
      rowsVisible: 0,
      latestMarketDate: "-"
    }
  };
}

function submittedSnapshot(config: RunConfig): RunSnapshot {
  const snapshot = initialSnapshot(config, "running");
  const runningId = snapshot.agents[0]?.id;
  return {
    ...snapshot,
    agents: snapshot.agents.map((agent) => agent.id === runningId ? { ...agent, status: "running", progress: 35 } : { ...agent, status: "queued" }),
    events: [
      {
        id: `event-submit-${Date.now()}`,
        ts: new Date().toLocaleTimeString(),
        level: "info",
        title: "Run started",
        detail: "The workspace accepted the run and handed it off to the backend."
      }
    ]
  };
}

export class HttpOrchestratorClient implements OrchestratorClient {
  private stoppedRuns = new Set<string>();

  async loadDefault(): Promise<{ config: RunConfig; snapshot: RunSnapshot }> {
    const response = await fetch(`${API_BASE}/config/default`);
    const payload = await response.json().catch(() => ({}));
    if (!response.ok || !payload.config || !payload.snapshot) {
      throw new Error(payload.error || `Unable to load workspace defaults with HTTP ${response.status}`);
    }
    return {
      config: sanitizeConfig(payload.config),
      snapshot: sanitizeSnapshot(payload.snapshot as RunSnapshot)
    };
  }

  async *startRun(config: RunConfig): AsyncIterable<RunSnapshot> {
    const pending = submittedSnapshot(config);
    yield pending;

    const startResponse = await fetch(`${API_BASE}/runs`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ config })
    });
    const startPayload = await startResponse.json().catch(() => ({}));
    if (!startPayload.snapshot) {
      throw new Error(startPayload.error || `Run failed to start with HTTP ${startResponse.status}`);
    }

    let snapshot = sanitizeSnapshot(startPayload.snapshot as RunSnapshot);
    this.stoppedRuns.delete(snapshot.runId);
    yield snapshot;

    while (snapshot.status === "running" || snapshot.status === "configuring") {
      if (this.stoppedRuns.has(snapshot.runId)) {
        return;
      }
      await sleep(800);
      const pollResponse = await fetch(`${API_BASE}/runs/${encodeURIComponent(snapshot.runId)}/snapshot`);
      const pollPayload = await pollResponse.json().catch(() => ({}));
      if (!pollPayload.snapshot) {
        throw new Error(pollPayload.error || `Run update failed with HTTP ${pollResponse.status}`);
      }
      snapshot = sanitizeSnapshot(pollPayload.snapshot as RunSnapshot);
      yield snapshot;
    }
  }

  async stopRun(runId: string): Promise<void> {
    this.stoppedRuns.add(runId);
    await fetch(`${API_BASE}/runs/${encodeURIComponent(runId)}/stop`, { method: "POST" });
  }
}

export const orchestratorClient = new HttpOrchestratorClient();
