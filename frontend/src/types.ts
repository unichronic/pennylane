export type RunMode = "live" | "backtest" | "paper_experiment" | "baseline";
export type DataProvider = "auto" | "yfinance" | "twelvedata" | "alpha_vantage";
export type BaselineStrategy = "buy_and_hold" | "macd" | "kdj_rsi" | "zmr" | "sma";
export type AgentStatus = "idle" | "queued" | "running" | "waiting" | "complete" | "blocked";
export type EventLevel = "info" | "tool" | "decision" | "risk" | "error";

export interface RunConfig {
  symbol: string;
  symbols: string;
  mode: RunMode;
  startDate: string;
  endDate: string;
  cash: number;
  dataProvider: DataProvider;
  baseline: BaselineStrategy;
  llmProvider: "local" | "mistral";
  debateRounds: number;
  checkpointing: boolean;
  logPath: string;
}

export interface AgentNode {
  id: string;
  label: string;
  group: "data" | "analysis" | "research" | "trading" | "risk" | "execution";
  status: AgentStatus;
  summary: string;
  progress: number;
  startedAt?: string;
  completedAt?: string;
}

export interface ToolCall {
  id: string;
  agentId: string;
  tool: string;
  args: Record<string, string | number | boolean | null>;
  source: string;
  dataQuality: "strict_point_in_time" | "publication_time" | "current_snapshot" | "simulated_or_derived";
  status: "queued" | "running" | "complete" | "failed";
  observation: string;
}

export interface OrchestrationEvent {
  id: string;
  ts: string;
  level: EventLevel;
  agentId?: string;
  title: string;
  detail: string;
}

export interface ReportBundle {
  market: string;
  news: string;
  sentiment: string;
  fundamentals: string;
}

export interface PortfolioState {
  cash: number;
  shares: number;
  equity: number;
  position: "flat" | "long" | "short";
  pnl: number;
  stopLoss?: number | null;
}

export interface RunSnapshot {
  runId: string;
  status: "idle" | "configuring" | "running" | "complete" | "failed";
  config: RunConfig;
  agents: AgentNode[];
  events: OrchestrationEvent[];
  toolCalls: ToolCall[];
  reports: ReportBundle;
  portfolio: PortfolioState;
  signal: {
    rating: string;
    action: string;
    confidence: number;
    riskNotes: string;
  };
  audit: {
    fullStateLog: string;
    jsonlLog: string;
    checkpointDb: string;
    cliEquivalent: string;
    rowsVisible: number;
    latestMarketDate: string;
  };
}

export interface OrchestratorClient {
  loadDefault(): Promise<{ config: RunConfig; snapshot: RunSnapshot }>;
  startRun(config: RunConfig): AsyncIterable<RunSnapshot>;
  stopRun(runId: string): Promise<void>;
}
