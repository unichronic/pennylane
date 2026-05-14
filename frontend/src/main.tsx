import React from "react";
import ReactDOM from "react-dom/client";
import {
  Activity,
  AlertTriangle,
  BarChart3,
  Brain,
  CheckCircle2,
  Circle,
  Clock3,
  Database,
  FileJson,
  Gauge,
  Layers3,
  ListChecks,
  RefreshCcw,
  ShieldCheck,
  Square,
  Wrench,
  Zap
} from "lucide-react";
import { defaultConfig, initialSnapshot, orchestratorClient } from "./api/orchestratorClient";
import { AgentNode, EventLevel, RunConfig, RunMode, RunSnapshot } from "./types";
import "./styles.css";

type ReportKey = keyof RunSnapshot["reports"];
type ReportBlock =
  | { kind: "metric"; label: string; value: string }
  | { kind: "list"; items: string[] }
  | { kind: "heading"; text: string }
  | { kind: "paragraph"; text: string }
  | { kind: "callout"; text: string }
  | { kind: "code"; text: string };

const groupLabels: Record<AgentNode["group"], string> = {
  data: "Data",
  analysis: "Analysis",
  research: "Research",
  trading: "Trading",
  risk: "Risk",
  execution: "Execution"
};

const levelIcon: Record<EventLevel, React.ReactNode> = {
  info: <Activity size={14} />,
  tool: <Wrench size={14} />,
  decision: <Brain size={14} />,
  risk: <ShieldCheck size={14} />,
  error: <AlertTriangle size={14} />
};

const modeMeta: Record<RunMode, { label: string; description: string; actionLabel: string }> = {
  live: {
    label: "Live committee",
    description: "Run the full agent committee over the selected visible market window.",
    actionLabel: "Start live run"
  },
  backtest: {
    label: "Paper backtest",
    description: "Replay the committee daily across the date window and track paper performance.",
    actionLabel: "Start backtest"
  },
  paper_experiment: {
    label: "Paper experiment",
    description: "Run Penny Lane and the baseline set across multiple symbols and compare aggregate curves.",
    actionLabel: "Run experiment"
  },
  baseline: {
    label: "Baseline benchmark",
    description: "Run a rule-based benchmark over the same historical window without the agent committee.",
    actionLabel: "Run baseline"
  }
};

const providerLabels: Record<RunConfig["dataProvider"], string> = {
  yfinance: "Yahoo Finance",
  auto: "Auto source",
  twelvedata: "Twelve Data",
  alpha_vantage: "Alpha Vantage"
};

const baselineLabels: Record<RunConfig["baseline"], string> = {
  buy_and_hold: "Buy and hold",
  macd: "MACD",
  kdj_rsi: "KDJ + RSI",
  zmr: "ZMR",
  sma: "SMA crossover"
};

const toolLabels: Record<string, string> = {
  get_stock_data: "Market window",
  get_indicators: "Indicator set",
  get_news: "Company news",
  get_global_news: "Macro news",
  get_fundamentals: "Fundamentals",
  get_balance_sheet: "Balance sheet",
  get_cashflow: "Cash flow",
  get_income_statement: "Income statement",
  get_insider_transactions: "Insider activity",
  run_baseline_strategy: "Baseline engine",
  load_symbol_windows: "Experiment window batch",
  run_strategy_batch: "Strategy batch",
  aggregate_equity_curves: "Aggregate comparison"
};

const qualityLabels: Record<RunSnapshot["toolCalls"][number]["dataQuality"], string> = {
  strict_point_in_time: "Point-in-time",
  publication_time: "Publication-timed",
  current_snapshot: "Current snapshot",
  simulated_or_derived: "Derived"
};

const reportMeta: Record<ReportKey, { label: string; kicker: string; blurb: string; icon: React.ReactNode }> = {
  market: {
    label: "Market",
    kicker: "Price structure",
    blurb: "Trend, momentum, volatility, and the visible market setup for the selected window.",
    icon: <BarChart3 size={16} />
  },
  news: {
    label: "News",
    kicker: "Event context",
    blurb: "Company-specific headlines, macro context, and time-sensitive narrative risk.",
    icon: <Activity size={16} />
  },
  sentiment: {
    label: "Sentiment",
    kicker: "Crowd read",
    blurb: "Public mood, chatter, and whether the market tone is helping or fighting the trade.",
    icon: <Brain size={16} />
  },
  fundamentals: {
    label: "Fundamentals",
    kicker: "Company quality",
    blurb: "Business quality, statement context, leverage, and whether the balance sheet supports the case.",
    icon: <Database size={16} />
  }
};

function usesCommittee(mode: RunMode) {
  return mode === "live" || mode === "backtest";
}

function formatMoney(value: number) {
  const rounded = Number.isFinite(value) ? value : 0;
  return `$${rounded.toLocaleString(undefined, { maximumFractionDigits: 2 })}`;
}

function humanizeStatus(status: RunSnapshot["status"]) {
  return status.charAt(0).toUpperCase() + status.slice(1);
}

function toneForAction(action: string) {
  if (action === "buy") return "positive";
  if (action === "sell") return "negative";
  return "neutral";
}

function toneForLevel(level: EventLevel) {
  if (level === "error") return "negative";
  if (level === "risk") return "warning";
  if (level === "decision") return "positive";
  return "neutral";
}

function Marker({ tone }: { tone: "positive" | "negative" | "warning" | "neutral" }) {
  return <span className={`markerDot ${tone}`} aria-hidden="true" />;
}

function normalizeSymbolInput(value: string) {
  return value.toUpperCase().replace(/\s+/g, "");
}

function splitSymbols(value: string) {
  return value.split(",").map((item) => item.trim()).filter(Boolean);
}

function describeTool(tool: RunSnapshot["toolCalls"][number]) {
  const args = tool.args ?? {};
  const symbol = typeof args.symbol === "string" ? args.symbol : "";
  const symbols = typeof args.symbols === "string" ? args.symbols : "";
  const strategy = typeof args.strategy === "string" ? args.strategy : "";
  const strategies = typeof args.strategies === "string" ? args.strategies : "";
  const start = typeof args.start_date === "string" ? args.start_date : "";
  const end = typeof args.end_date === "string" ? args.end_date : "";
  const indicators = typeof args.indicators === "string" ? args.indicators : "";
  if (symbol && start && end) return `${symbol} · ${start} to ${end}`;
  if (symbols && start && end) return `${symbols} · ${start} to ${end}`;
  if (strategy && symbol) return `${strategy} · ${symbol}`;
  if (strategies) return strategies.split(",").join(" · ");
  if (symbol) return symbol;
  if (symbols) return symbols;
  if (indicators) return indicators.split(",").join(" · ");
  return "Run evidence";
}

function issuesForConfig(config: RunConfig) {
  const issues: string[] = [];
  if (config.mode === "paper_experiment") {
    if (splitSymbols(config.symbols).length === 0) {
      issues.push("Enter at least one symbol for the experiment batch.");
    }
  } else if (!config.symbol.trim()) {
    issues.push("Enter a symbol to run.");
  }
  if (!config.startDate || !config.endDate) {
    issues.push("Set both the start and end date.");
  } else if (config.startDate > config.endDate) {
    issues.push("The end date must be on or after the start date.");
  }
  if (!Number.isFinite(config.cash) || config.cash <= 0) {
    issues.push("Cash must be a positive number.");
  }
  if (usesCommittee(config.mode) && (!Number.isFinite(config.debateRounds) || config.debateRounds < 1 || config.debateRounds > 5)) {
    issues.push("Debate rounds must stay between 1 and 5.");
  }
  return issues;
}

function scopeLabel(config: RunConfig) {
  if (config.mode === "paper_experiment") {
    const count = splitSymbols(config.symbols).length;
    return `${count} symbol${count === 1 ? "" : "s"}`;
  }
  return config.symbol || "No symbol";
}

function truncate(text: string, max = 120) {
  if (text.length <= max) return text;
  return `${text.slice(0, max - 1).trimEnd()}…`;
}

function cleanInlineMarkdown(value: string) {
  return value
    .replace(/\*\*(.*?)\*\*/g, "$1")
    .replace(/`([^`]+)`/g, "$1")
    .trim();
}

function looksLikeJson(text: string) {
  const trimmed = text.trim();
  return trimmed.startsWith("{") || trimmed.startsWith("[");
}

function parseReportBlocks(text: string): ReportBlock[] {
  const trimmed = text.trim();
  if (!trimmed) {
    return [{ kind: "paragraph", text: "No report is available yet." }];
  }
  if (looksLikeJson(trimmed)) {
    try {
      return [{ kind: "code", text: JSON.stringify(JSON.parse(trimmed), null, 2) }];
    } catch {
      return [{ kind: "code", text: trimmed }];
    }
  }

  const lines = trimmed.split("\n").map((line) => line.trimEnd());
  const blocks: ReportBlock[] = [];
  let idx = 0;

  while (idx < lines.length) {
    const raw = lines[idx].trim();
    if (!raw) {
      idx += 1;
      continue;
    }

    if (raw.startsWith("- ")) {
      const items: string[] = [];
      while (idx < lines.length && lines[idx].trim().startsWith("- ")) {
        items.push(cleanInlineMarkdown(lines[idx].trim().slice(2)));
        idx += 1;
      }
      blocks.push({ kind: "list", items });
      continue;
    }

    const metricMatch = raw.match(/^\*\*(.+?)\*\*:\s*(.+)$/);
    if (metricMatch) {
      blocks.push({
        kind: "metric",
        label: cleanInlineMarkdown(metricMatch[1]),
        value: cleanInlineMarkdown(metricMatch[2])
      });
      idx += 1;
      continue;
    }

    const headingMatch = raw.match(/^([A-Z][A-Za-z0-9 /&()+-]{1,48}):$/);
    if (headingMatch) {
      blocks.push({ kind: "heading", text: cleanInlineMarkdown(headingMatch[1]) });
      idx += 1;
      continue;
    }

    if (/^\*\*(.+?)\*\*$/.test(raw)) {
      blocks.push({ kind: "heading", text: cleanInlineMarkdown(raw) });
      idx += 1;
      continue;
    }

    if (raw.startsWith("FINAL TRANSACTION PROPOSAL")) {
      blocks.push({ kind: "callout", text: cleanInlineMarkdown(raw) });
      idx += 1;
      continue;
    }

    const parts = [cleanInlineMarkdown(raw)];
    idx += 1;
    while (idx < lines.length) {
      const next = lines[idx].trim();
      if (!next) break;
      if (
        next.startsWith("- ") ||
        /^\*\*(.+?)\*\*:\s*(.+)$/.test(next) ||
        /^([A-Z][A-Za-z0-9 /&()+-]{1,48}):$/.test(next) ||
        /^\*\*(.+?)\*\*$/.test(next) ||
        next.startsWith("FINAL TRANSACTION PROPOSAL")
      ) {
        break;
      }
      parts.push(cleanInlineMarkdown(next));
      idx += 1;
    }
    blocks.push({ kind: "paragraph", text: parts.join(" ") });
  }

  return blocks;
}

function reportPreview(text: string) {
  const blocks = parseReportBlocks(text);
  const first = blocks[0];
  if (!first) return "No report yet.";
  if (first.kind === "metric") return truncate(`${first.label}: ${first.value}`, 96);
  if (first.kind === "list") return truncate(first.items.join(" · "), 96);
  return truncate(first.text, 96);
}

function reportState(text: string) {
  const normalized = text.trim().toLowerCase();
  if (!normalized) return "empty";
  if (
    normalized.startsWith("no ") ||
    normalized.includes("not used in") ||
    normalized.includes("does not produce one combined")
  ) {
    return "reference";
  }
  return "ready";
}

function reportWordCount(text: string) {
  const normalized = cleanInlineMarkdown(text);
  if (!normalized) return 0;
  return normalized.split(/\s+/).filter(Boolean).length;
}

function Field({
  label,
  hint,
  children
}: {
  label: string;
  hint?: string;
  children: React.ReactNode;
}) {
  return (
    <label className="field">
      <span>{label}</span>
      {children}
      {hint ? <small>{hint}</small> : null}
    </label>
  );
}

function ControlSection({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="controlSection">
      <div className="controlSectionTitle">{title}</div>
      {children}
    </section>
  );
}

function ModePicker({
  mode,
  onChange
}: {
  mode: RunMode;
  onChange: (mode: RunMode) => void;
}) {
  const modes = Object.keys(modeMeta) as RunMode[];
  return (
    <div className="modePicker" role="tablist" aria-label="Run mode">
      {modes.map((item) => (
        <button
          key={item}
          type="button"
          className={item === mode ? "active" : ""}
          onClick={() => onChange(item)}
        >
          {modeMeta[item].label}
        </button>
      ))}
    </div>
  );
}

function RunControls({
  config,
  setConfig,
  onStart,
  onStop,
  running
}: {
  config: RunConfig;
  setConfig: React.Dispatch<React.SetStateAction<RunConfig>>;
  onStart: () => void;
  onStop: () => void;
  running: boolean;
}) {
  const issues = issuesForConfig(config);
  const canStart = !running && issues.length === 0;

  const patch = <K extends keyof RunConfig>(key: K, value: RunConfig[K]) => {
    setConfig((prev) => ({ ...prev, [key]: value }));
  };

  return (
    <aside className="controlPane">
      <div className="paneTitle">
        <ListChecks size={18} />
        <span>Run setup</span>
      </div>
      <p className="paneIntro">Pick the run shape, scope, and data window. The model backend stays configured on the server.</p>

      <ControlSection title="Run type">
        <ModePicker mode={config.mode} onChange={(mode) => patch("mode", mode)} />
        <p className="sectionNote">{modeMeta[config.mode].description}</p>
      </ControlSection>

      <ControlSection title="Scope">
        {config.mode === "paper_experiment" ? (
          <Field label="Symbols" hint="Comma-separated tickers for the experiment batch.">
            <input
              value={config.symbols}
              onChange={(event) => patch("symbols", normalizeSymbolInput(event.target.value))}
              placeholder="AAPL,GOOGL,AMZN"
            />
          </Field>
        ) : (
          <Field label="Symbol">
            <input
              value={config.symbol}
              onChange={(event) => patch("symbol", normalizeSymbolInput(event.target.value))}
              placeholder="AAPL"
            />
          </Field>
        )}

        {config.mode === "baseline" ? (
          <Field label="Baseline strategy">
            <select value={config.baseline} onChange={(event) => patch("baseline", event.target.value as RunConfig["baseline"])}>
              <option value="buy_and_hold">{baselineLabels.buy_and_hold}</option>
              <option value="macd">{baselineLabels.macd}</option>
              <option value="kdj_rsi">{baselineLabels.kdj_rsi}</option>
              <option value="zmr">{baselineLabels.zmr}</option>
              <option value="sma">{baselineLabels.sma}</option>
            </select>
          </Field>
        ) : null}

        <div className="twoCol">
          <Field label="Start date">
            <input type="date" value={config.startDate} onChange={(event) => patch("startDate", event.target.value)} />
          </Field>
          <Field label="End date">
            <input type="date" value={config.endDate} onChange={(event) => patch("endDate", event.target.value)} />
          </Field>
        </div>

        <Field label="Starting cash">
          <input
            type="number"
            value={config.cash}
            min={0}
            step={1000}
            onChange={(event) => patch("cash", Number(event.target.value))}
          />
        </Field>
      </ControlSection>

      <ControlSection title="Data">
        <Field label="Market data source">
          <select value={config.dataProvider} onChange={(event) => patch("dataProvider", event.target.value as RunConfig["dataProvider"])}>
            <option value="yfinance">{providerLabels.yfinance}</option>
            <option value="auto">{providerLabels.auto}</option>
            <option value="twelvedata">{providerLabels.twelvedata}</option>
            <option value="alpha_vantage">{providerLabels.alpha_vantage}</option>
          </select>
        </Field>
      </ControlSection>

      {usesCommittee(config.mode) ? (
        <details className="advancedCard">
          <summary>Advanced</summary>
          <div className="advancedBody">
            <Field label="Debate rounds" hint="Use more rounds only when you want a longer committee exchange.">
              <input
                type="number"
                value={config.debateRounds}
                min={1}
                max={5}
                onChange={(event) => patch("debateRounds", Number(event.target.value))}
              />
            </Field>
            <p className="advancedNote">The LLM backend is configured on the server side and is intentionally not exposed as a public run control.</p>
          </div>
        </details>
      ) : null}

      <section className="runSummaryCard">
        <div className="controlSectionTitle">This run</div>
        <dl className="summaryList">
          <dt>Mode</dt><dd>{modeMeta[config.mode].label}</dd>
          <dt>Scope</dt><dd>{scopeLabel(config)}</dd>
          <dt>Window</dt><dd>{config.startDate} to {config.endDate}</dd>
          <dt>Source</dt><dd>{providerLabels[config.dataProvider]}</dd>
        </dl>
      </section>

      {issues.length > 0 ? (
        <div className="validationBox">
          <div className="validationTitle">
            <AlertTriangle size={14} />
            <span>Check the run setup</span>
          </div>
          <ul>
            {issues.map((issue) => (
              <li key={issue}>{issue}</li>
            ))}
          </ul>
        </div>
      ) : (
        <div className="controlFootnote">
          <Marker tone="neutral" />
          <span>Operational logs, credentials, and provider details stay behind the backend boundary.</span>
        </div>
      )}

      <div className="controlActions">
        <button className="primaryBtn" onClick={onStart} disabled={!canStart}>
          <Zap size={16} />
          {running ? "Running" : modeMeta[config.mode].actionLabel}
        </button>
        <button onClick={onStop} disabled={!running}>
          <Square size={16} />
          Stop
        </button>
      </div>
    </aside>
  );
}

function AgentMap({ agents }: { agents: AgentNode[] }) {
  const groups = Object.entries(groupLabels) as Array<[AgentNode["group"], string]>;
  const visibleGroups = groups.filter(([group]) => agents.some((agent) => agent.group === group));
  return (
    <section className="agentMap">
      {visibleGroups.map(([group, label]) => (
        <div className={`lane lane-${group}`} key={group}>
          <div className="laneLabel">
            <Marker
              tone={
                group === "analysis" || group === "research"
                  ? "warning"
                  : group === "risk"
                    ? "negative"
                    : "positive"
              }
            />
            <span>{label}</span>
          </div>
          <div className="laneNodes">
            {agents.filter((agent) => agent.group === group).map((agent) => (
              <article className={`agentNode ${agent.status}`} key={agent.id}>
                <div className="agentHeader">
                  {agent.status === "complete" ? (
                    <CheckCircle2 size={16} />
                  ) : agent.status === "running" ? (
                    <RefreshCcw size={16} className="spin" />
                  ) : (
                    <Circle size={16} />
                  )}
                  <span>{agent.label}</span>
                </div>
                <p>{agent.summary}</p>
                <div className="progressTrack">
                  <div style={{ width: `${agent.progress}%` }} />
                </div>
              </article>
            ))}
          </div>
        </div>
      ))}
    </section>
  );
}

function EventStream({ snapshot }: { snapshot: RunSnapshot }) {
  return (
    <section className="panel eventPanel">
      <div className="panelHeader">
        <Activity size={18} />
        <span>Activity</span>
      </div>
      <div className="eventList">
        {snapshot.events.length === 0 ? (
          <div className="emptyState">Start a run to see the backend timeline.</div>
        ) : (
          snapshot.events.map((item) => (
            <div className={`eventItem ${item.level}`} key={item.id}>
              <div className="eventIcon">{levelIcon[item.level]}</div>
              <div>
                <div className="eventTitle">
                  <span className="eventHeading">
                    <Marker tone={toneForLevel(item.level)} />
                    {item.title}
                  </span>
                  <time>{item.ts}</time>
                </div>
                <p>{item.detail}</p>
              </div>
            </div>
          ))
        )}
      </div>
    </section>
  );
}

function ToolCalls({ snapshot }: { snapshot: RunSnapshot }) {
  return (
    <section className="panel toolPanel">
      <div className="panelHeader">
        <Wrench size={18} />
        <span>Evidence and data timing</span>
      </div>
      <div className="toolTable">
        {snapshot.toolCalls.length === 0 ? (
          <div className="emptyState">Data access and derived inputs will show up here.</div>
        ) : (
          snapshot.toolCalls.map((tool) => (
            <div className="toolRow" key={tool.id}>
              <div className="toolTitleRow">
                <div className="toolIdentity">
                  <Marker tone={tool.status === "failed" ? "negative" : tool.status === "running" ? "warning" : "positive"} />
                  <strong>{toolLabels[tool.tool] ?? tool.tool}</strong>
                </div>
                <div className="pillRow">
                  <span className={`quality ${tool.dataQuality}`}>{qualityLabels[tool.dataQuality]}</span>
                  <span className={`statusPill ${tool.status}`}>{tool.status}</span>
                </div>
              </div>
              <span className="toolContext">{describeTool(tool)}</span>
              <p>{tool.observation}</p>
            </div>
          ))
        )}
      </div>
    </section>
  );
}

function Reports({ snapshot }: { snapshot: RunSnapshot }) {
  const [tab, setTab] = React.useState<ReportKey>("market");
  const tabs = Object.keys(snapshot.reports) as ReportKey[];
  const activeText = snapshot.reports[tab];
  const activeMeta = reportMeta[tab];
  const blocks = React.useMemo(() => parseReportBlocks(activeText), [activeText]);
  const status = reportState(activeText);
  const words = reportWordCount(activeText);
  const structuralCount = blocks.filter((block) => block.kind !== "paragraph").length || 1;

  React.useEffect(() => {
    if (!tabs.includes(tab)) {
      setTab("market");
    }
  }, [tab, tabs]);

  return (
    <section className="panel reportPanel">
      <div className="panelHeader">
        <FileJson size={18} />
        <span>Research notebook</span>
      </div>
      <div className="reportWorkspace">
        <aside className="reportNav">
          {tabs.map((item) => {
            const text = snapshot.reports[item];
            const state = reportState(text);
            return (
              <button
                key={item}
                type="button"
                className={`reportNavItem ${tab === item ? "active" : ""}`}
                onClick={() => setTab(item)}
              >
                <div className="reportNavTop">
                  <span className="reportNavIcon">{reportMeta[item].icon}</span>
                  <div>
                    <strong>{reportMeta[item].label}</strong>
                    <span>{reportMeta[item].kicker}</span>
                  </div>
                </div>
                <p>{reportPreview(text)}</p>
                <div className={`reportStatePill ${state}`}>
                  <Marker tone={state === "ready" ? "positive" : "neutral"} />
                  <span>{state === "ready" ? "Readable notes" : "Reference only"}</span>
                </div>
              </button>
            );
          })}
        </aside>

        <div className="reportReader">
          <div className={`reportHero ${tab}`}>
            <div className="reportHeroTop">
              <span className="reportKicker">{activeMeta.kicker}</span>
              <div className={`reportStatePill ${status}`}>
                <Marker tone={status === "ready" ? "positive" : "neutral"} />
                <span>{status === "ready" ? "Ready to read" : "Reference only"}</span>
              </div>
            </div>
            <h3>{activeMeta.label} report</h3>
            <p>{activeMeta.blurb}</p>
            <div className="reportMetaRow">
              <span>{words} words</span>
              <span>{structuralCount} sections</span>
              <span>{modeMeta[snapshot.config.mode].label}</span>
            </div>
          </div>

          <div className="reportBody">
            {blocks.map((block, idx) => {
              if (block.kind === "metric") {
                return (
                  <div className="reportMetricRow" key={`${block.kind}-${block.label}-${idx}`}>
                    <span>{block.label}</span>
                    <strong>{block.value}</strong>
                  </div>
                );
              }
              if (block.kind === "list") {
                return (
                  <ul className="reportList" key={`${block.kind}-${idx}`}>
                    {block.items.map((item, itemIdx) => (
                      <li key={`${item}-${itemIdx}`}>{item}</li>
                    ))}
                  </ul>
                );
              }
              if (block.kind === "heading") {
                return <h4 className="reportHeading" key={`${block.kind}-${block.text}-${idx}`}>{block.text}</h4>;
              }
              if (block.kind === "callout") {
                return <div className="reportCallout" key={`${block.kind}-${idx}`}>{block.text}</div>;
              }
              if (block.kind === "code") {
                return <pre className="reportCode" key={`${block.kind}-${idx}`}>{block.text}</pre>;
              }
              return <p className="reportParagraph" key={`${block.kind}-${idx}`}>{block.text}</p>;
            })}
          </div>
        </div>
      </div>
    </section>
  );
}

function StatePanels({ snapshot }: { snapshot: RunSnapshot }) {
  const actionTone = toneForAction(snapshot.signal.action);
  const mode = snapshot.config.mode;
  const showConfidence = snapshot.signal.confidence > 0 && usesCommittee(mode);

  const portfolioMetrics =
    mode === "paper_experiment"
      ? [
          { label: "Ending equity", value: formatMoney(snapshot.portfolio.equity) },
          { label: "Starting capital", value: formatMoney(snapshot.portfolio.cash) },
          { label: "Symbols", value: String(Math.round(snapshot.portfolio.shares)) },
          { label: "Net change", value: formatMoney(snapshot.portfolio.pnl) }
        ]
      : [
          { label: "Equity", value: formatMoney(snapshot.portfolio.equity) },
          { label: "Cash", value: formatMoney(snapshot.portfolio.cash) },
          { label: "Shares", value: String(snapshot.portfolio.shares) },
          { label: "P&L", value: formatMoney(snapshot.portfolio.pnl) },
          { label: "Position", value: snapshot.portfolio.position },
          { label: "Stop level", value: snapshot.portfolio.stopLoss ?? "Not set" }
        ];

  return (
    <div className="stateGrid">
      <section className="panel">
        <div className="panelHeader">
          <BarChart3 size={18} />
          <span>{mode === "paper_experiment" ? "Experiment snapshot" : "Portfolio snapshot"}</span>
        </div>
        <div className="metricGrid">
          {portfolioMetrics.map((item) => (
            <div key={item.label}>
              <span>{item.label}</span>
              <strong>{item.value}</strong>
            </div>
          ))}
        </div>
      </section>
      <section className="panel">
        <div className="panelHeader">
          <ShieldCheck size={18} />
          <span>{mode === "baseline" ? "Baseline outcome" : mode === "paper_experiment" ? "Run outcome" : "Final call"}</span>
        </div>
        <div className={`decisionBox ${actionTone}`}>
          <div className="decisionHeader">
            <Marker tone={actionTone} />
            <strong>{snapshot.signal.rating}</strong>
          </div>
          <span>{snapshot.signal.action}</span>
          {showConfidence ? (
            <div className="confidenceRow">
              <label>Confidence</label>
              <div className="confidenceTrack">
                <div style={{ width: `${Math.round(snapshot.signal.confidence * 100)}%` }} />
              </div>
              <b>{Math.round(snapshot.signal.confidence * 100)}%</b>
            </div>
          ) : null}
          <p>{snapshot.signal.riskNotes}</p>
        </div>
      </section>
      <section className="panel">
        <div className="panelHeader">
          <Gauge size={18} />
          <span>Run context</span>
        </div>
        <dl className="auditList">
          <dt>Mode</dt><dd>{modeMeta[snapshot.config.mode].label}</dd>
          <dt>Scope</dt><dd>{scopeLabel(snapshot.config)}</dd>
          <dt>Source</dt><dd>{providerLabels[snapshot.config.dataProvider]}</dd>
          <dt>Window</dt><dd>{snapshot.config.startDate} to {snapshot.config.endDate}</dd>
          <dt>Latest market date</dt><dd>{snapshot.audit.latestMarketDate}</dd>
          <dt>Rows visible</dt><dd>{snapshot.audit.rowsVisible}</dd>
        </dl>
        <div className="coverageNote">
          <Marker tone="neutral" />
          <span>Backend runtime details stay off the main screen unless they change the outcome.</span>
        </div>
      </section>
    </div>
  );
}

function App() {
  const [config, setConfig] = React.useState<RunConfig>(defaultConfig);
  const [snapshot, setSnapshot] = React.useState<RunSnapshot>(() => initialSnapshot(defaultConfig));
  const [running, setRunning] = React.useState(false);
  const cancelRef = React.useRef(false);

  React.useEffect(() => {
    let mounted = true;
    orchestratorClient.loadDefault()
      .then(({ config: backendConfig, snapshot: backendSnapshot }) => {
        if (!mounted) return;
        setConfig(backendConfig);
        setSnapshot(backendSnapshot);
      })
      .catch(() => {
        if (!mounted) return;
        setSnapshot((prev) => ({
          ...prev,
          status: "failed",
          events: [
            {
              id: `event-defaults-${Date.now()}`,
              ts: new Date().toLocaleTimeString(),
              level: "error",
              title: "Workspace unavailable",
              detail: "The app could not load its backend defaults."
            },
            ...prev.events
          ]
        }));
      });
    return () => {
      mounted = false;
    };
  }, []);

  async function startRun() {
    if (issuesForConfig(config).length > 0) {
      return;
    }
    cancelRef.current = false;
    setRunning(true);
    try {
      for await (const next of orchestratorClient.startRun(config)) {
        if (cancelRef.current) break;
        setSnapshot(next);
      }
    } catch (error) {
      setSnapshot((prev) => ({
        ...prev,
        status: "failed",
        events: [
          {
            id: `event-error-${Date.now()}`,
            ts: new Date().toLocaleTimeString(),
            level: "error",
            title: "Run failed",
            detail: error instanceof Error ? error.message.split("backend").join("service") : "The run could not be completed."
          },
          ...prev.events
        ]
      }));
    }
    setRunning(false);
  }

  async function stopRun() {
    cancelRef.current = true;
    await orchestratorClient.stopRun(snapshot.runId);
    setRunning(false);
    setSnapshot((prev) => ({
      ...prev,
      status: "idle",
      events: [
        {
          id: `event-stop-${Date.now()}`,
          ts: new Date().toLocaleTimeString(),
          level: "info",
          title: "Run stopped",
          detail: "The active run was stopped."
        },
        ...prev.events
      ]
    }));
  }

  return (
    <main className="appShell">
      <header className="topBar">
        <div>
          <h1>Penny Lane Capital</h1>
          <p>Inspect the decision flow, evidence trail, and paper results without dropping into the CLI.</p>
        </div>
        <div className={`runBadge ${snapshot.status}`}>
          {running ? <RefreshCcw size={16} className="spin" /> : snapshot.status === "complete" ? <CheckCircle2 size={16} /> : snapshot.status === "failed" ? <AlertTriangle size={16} /> : <Clock3 size={16} />}
          <span>{humanizeStatus(snapshot.status)}</span>
        </div>
      </header>

      <div className="layout">
        <RunControls
          config={config}
          setConfig={setConfig}
          onStart={startRun}
          onStop={stopRun}
          running={running}
        />
        <section className="workspace">
          <div className="workspaceHeader">
            <div>
              <h2>{modeMeta[config.mode].label}</h2>
              <p>{modeMeta[config.mode].description}</p>
            </div>
            <div className="connectionStrip">
              <span><Zap size={14} /> {scopeLabel(config)}</span>
              <span><Database size={14} /> {providerLabels[config.dataProvider]}</span>
              <span><Clock3 size={14} /> {config.startDate} to {config.endDate}</span>
              {snapshot.audit.latestMarketDate !== "-" ? <span><Gauge size={14} /> Latest {snapshot.audit.latestMarketDate}</span> : null}
              {config.mode === "paper_experiment" ? <span><Layers3 size={14} /> {splitSymbols(config.symbols).length} symbols</span> : null}
            </div>
          </div>

          <StatePanels snapshot={snapshot} />
          <AgentMap agents={snapshot.agents} />

          <div className="detailGrid">
            <Reports snapshot={snapshot} />
            <div className="detailRail">
              <EventStream snapshot={snapshot} />
              <ToolCalls snapshot={snapshot} />
            </div>
          </div>
        </section>
      </div>
    </main>
  );
}

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
