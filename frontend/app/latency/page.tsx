"use client";

import { useMemo, useState } from "react";

type RunMetrics = {
  label: string;
  task_id: string;
  start_ack_seconds: number;
  sse_open_seconds: number | null;
  first_event_seconds: number | null;
  generation_started_seconds: number | null;
  first_model_token_seconds: number | null;
  first_painted_frame_seconds: number | null;
  sse_receive_to_painted_frame_seconds: number | null;
  task_total_seconds: number | null;
  model_token_event_count: number;
  model_token_interarrival_median_seconds: number | null;
  live_event_delivery_median_seconds: number | null;
  runtime_metrics: Record<string, unknown>;
  stream_batching: Record<string, unknown>;
};

type ValidationReport = {
  benchmark_design: string;
  status: string;
  prewarm: Record<string, unknown>;
  warmup: RunMetrics | null;
  runs: RunMetrics[];
  summary: Record<string, number | null>;
  checks: Record<string, boolean>;
};

const api = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";
const prompt = "Answer with exactly four short bullets, one sentence each: why does an outer-race bearing fault create periodic vibration impulses?";

function cookie(name: string) {
  return document.cookie.split("; ").find((item) => item.startsWith(`${name}=`))?.split("=").slice(1).join("=") ?? "";
}

function apiFetch(path: string, init: RequestInit = {}) {
  const method = (init.method ?? "GET").toUpperCase();
  const headers = new Headers(init.headers);
  if (!["GET", "HEAD", "OPTIONS"].includes(method)) {
    const csrf = cookie("sovereign_csrf");
    if (csrf) headers.set("X-CSRF-Token", decodeURIComponent(csrf));
  }
  return fetch(`${api}${path}`, { ...init, headers, credentials: "include" });
}

function median(values: number[]): number | null {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  const value = sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
  return Number(value.toFixed(6));
}

function numeric(run: RunMetrics, key: keyof RunMetrics): number | null {
  const value = run[key];
  return typeof value === "number" ? value : null;
}

async function prewarmModel() {
  const response = await fetch("/api/phase12/prewarm", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    cache: "no-store",
  });
  if (!response.ok) throw new Error(`Phase 12 prewarm returned ${response.status}: ${await response.text()}`);
  const payload = await response.json() as Record<string, unknown>;
  if (payload.resident !== true) throw new Error("Exact-runner prewarm completed but qwen3:4b-instruct is not resident.");
  return payload;
}

function runTask(label: string, setVisibleText: (value: string) => void): Promise<RunMetrics> {
  return new Promise(async (resolve, reject) => {
    const startedAt = performance.now();
    let taskId = "";
    let firstEventAt: number | null = null;
    let generationStartedAt: number | null = null;
    let firstTokenAt: number | null = null;
    let firstPaintedFrameAt: number | null = null;
    let sseOpenAt: number | null = null;
    let tokenEvents = 0;
    let tokenText = "";
    const tokenReceiveTimes: number[] = [];
    const liveDeliverySeconds: number[] = [];
    let finished = false;

    try {
      setVisibleText("");
      const response = await apiFetch("/api/tasks/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          request: `${prompt}\n${label}`,
          model_override: null,
          execution_mode: "FAST",
          chat_mode: "GENERAL",
          attachments: [],
          use_case: "internal_assistant",
          workcell_id: null,
        }),
      });
      const ackAt = performance.now();
      if (!response.ok) throw new Error(`Task start returned ${response.status}: ${await response.text()}`);
      const startPayload = await response.json();
      taskId = String(startPayload.task_id ?? "");
      if (!taskId) throw new Error("Task start returned no task_id");

      const stream = new EventSource(`${api}/api/tasks/${taskId}/events`, { withCredentials: true });
      stream.onopen = () => { sseOpenAt = performance.now(); };

      const eventTypes = [
        "task_accepted", "governance_completed", "task_classified", "model_selected",
        "chat_mode_selected", "execution_mode_selected", "plan_created", "step_started",
        "generation_started", "model_token", "generation_completed", "step_completed",
        "warning", "task_completed", "task_cancelled", "task_failed",
      ];

      const finish = (record: RunMetrics) => {
        if (finished) return;
        finished = true;
        stream.close();
        requestAnimationFrame(() => resolve(record));
      };

      const receive = (raw: MessageEvent) => {
        const receivedAt = performance.now();
        const item = JSON.parse(raw.data) as { type?: string; timestamp?: string; payload?: Record<string, unknown> };
        const type = String(item.type ?? "");
        if (firstEventAt === null) firstEventAt = receivedAt;
        const createdAt = item.timestamp ? Date.parse(item.timestamp) : Number.NaN;
        if (Number.isFinite(createdAt)) {
          liveDeliverySeconds.push(Math.max(0, (Date.now() - createdAt) / 1000));
        }
        if (type === "generation_started" && generationStartedAt === null) generationStartedAt = receivedAt;
        if (type === "model_token") {
          tokenEvents += 1;
          tokenReceiveTimes.push(receivedAt);
          const text = String(item.payload?.text ?? "");
          tokenText += text;
          if (firstTokenAt === null) {
            firstTokenAt = receivedAt;
            setVisibleText(tokenText);
            requestAnimationFrame(() => {
              if (firstPaintedFrameAt === null) firstPaintedFrameAt = performance.now();
            });
          } else {
            setVisibleText(tokenText);
          }
          return;
        }
        if (type === "task_failed" || type === "task_cancelled") {
          const error = String(item.payload?.error ?? type);
          stream.close();
          reject(new Error(`${label}: ${error}`));
          return;
        }
        if (type === "task_completed") {
          const terminalAt = performance.now();
          const result = (item.payload?.result ?? {}) as Record<string, unknown>;
          const runtime = (result.runtime_metrics ?? {}) as Record<string, unknown>;
          const batching = (runtime.stream_batching ?? {}) as Record<string, unknown>;
          const intervals = tokenReceiveTimes.slice(1).map((value, index) => (value - tokenReceiveTimes[index]) / 1000);
          const record: RunMetrics = {
            label,
            task_id: taskId,
            start_ack_seconds: Number(((ackAt - startedAt) / 1000).toFixed(6)),
            sse_open_seconds: sseOpenAt === null ? null : Number(((sseOpenAt - startedAt) / 1000).toFixed(6)),
            first_event_seconds: firstEventAt === null ? null : Number(((firstEventAt - startedAt) / 1000).toFixed(6)),
            generation_started_seconds: generationStartedAt === null ? null : Number(((generationStartedAt - startedAt) / 1000).toFixed(6)),
            first_model_token_seconds: firstTokenAt === null ? null : Number(((firstTokenAt - startedAt) / 1000).toFixed(6)),
            first_painted_frame_seconds: firstPaintedFrameAt === null ? null : Number(((firstPaintedFrameAt - startedAt) / 1000).toFixed(6)),
            sse_receive_to_painted_frame_seconds: firstTokenAt === null || firstPaintedFrameAt === null
              ? null
              : Number(((firstPaintedFrameAt - firstTokenAt) / 1000).toFixed(6)),
            task_total_seconds: Number(((terminalAt - startedAt) / 1000).toFixed(6)),
            model_token_event_count: tokenEvents,
            model_token_interarrival_median_seconds: median(intervals),
            live_event_delivery_median_seconds: median(liveDeliverySeconds),
            runtime_metrics: runtime,
            stream_batching: batching,
          };
          finish(record);
        }
      };

      eventTypes.forEach((type) => stream.addEventListener(type, receive as EventListener));
      stream.onerror = () => {
        if (!finished && stream.readyState === EventSource.CLOSED) {
          reject(new Error(`${label}: SSE connection closed before terminal event`));
        }
      };
    } catch (error) {
      reject(error);
    }
  });
}

export default function Phase12LatencyPage() {
  const [running, setRunning] = useState(false);
  const [visibleText, setVisibleText] = useState("");
  const [report, setReport] = useState<ValidationReport | null>(null);
  const [error, setError] = useState("");

  const pretty = useMemo(() => report ? JSON.stringify(report, null, 2) : "", [report]);

  async function runSuite() {
    if (running) return;
    setRunning(true);
    setError("");
    setReport(null);
    try {
      const prewarm = await prewarmModel();
      const warmup = await runTask("Phase 12 full-path warm-up.", setVisibleText);
      const runs: RunMetrics[] = [];
      for (let index = 0; index < 3; index += 1) {
        runs.push(await runTask(`Phase 12 browser benchmark run ${index + 1}.`, setVisibleText));
      }

      const summary: Record<string, number | null> = {};
      const keys: Array<keyof RunMetrics> = [
        "start_ack_seconds", "sse_open_seconds", "first_event_seconds", "generation_started_seconds",
        "first_model_token_seconds", "first_painted_frame_seconds", "sse_receive_to_painted_frame_seconds",
        "task_total_seconds", "model_token_event_count", "model_token_interarrival_median_seconds",
        "live_event_delivery_median_seconds",
      ];
      for (const key of keys) {
        const values = runs.map((run) => numeric(run, key)).filter((value): value is number => value !== null);
        summary[`median_${String(key)}`] = median(values);
      }
      const callbackValues = runs.map((run) => {
        const breakdown = (run.runtime_metrics.latency_breakdown ?? {}) as Record<string, unknown>;
        return typeof breakdown.event_callback_seconds === "number" ? breakdown.event_callback_seconds : null;
      }).filter((value): value is number => value !== null);
      const decodeValues = runs.map((run) => {
        const breakdown = (run.runtime_metrics.latency_breakdown ?? {}) as Record<string, unknown>;
        return typeof breakdown.decode_tokens_per_second === "number" ? breakdown.decode_tokens_per_second : null;
      }).filter((value): value is number => value !== null);
      summary.median_agent_event_callback_seconds = median(callbackValues);
      summary.median_decode_tokens_per_second = median(decodeValues);

      const batchingLoaded = runs.every((run) => Number(run.stream_batching.max_delay_seconds) === 0.2);
      const allCompletedWithTokens = runs.every((run) => run.model_token_event_count > 0 && run.task_total_seconds !== null);
      const browserPaintMeasured = runs.every((run) => run.first_painted_frame_seconds !== null);
      const reportValue: ValidationReport = {
        benchmark_design: "phase12-one-click-browser-e2e-v1",
        status: batchingLoaded && allCompletedWithTokens && browserPaintMeasured ? "complete" : "invalid",
        prewarm,
        warmup,
        runs,
        summary,
        checks: {
          stream_batching_200ms_loaded: batchingLoaded,
          all_measured_runs_completed_with_tokens: allCompletedWithTokens,
          browser_painted_frame_measured: browserPaintMeasured,
        },
      };
      setReport(reportValue);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Phase 12 validation failed");
    } finally {
      setRunning(false);
    }
  }

  async function copyReport() {
    if (pretty) await navigator.clipboard.writeText(pretty);
  }

  return (
    <main style={{ maxWidth: 1100, margin: "0 auto", padding: 32, fontFamily: "system-ui, sans-serif" }}>
      <h1>Phase 12 — one-click user-visible latency validation</h1>
      <p>
        One click performs one exact-runner prewarm, one unmeasured full-path warm-up, then three measured FAST/GENERAL
        tasks through the real task API, SSE stream, React state update, and next browser paint frame.
      </p>
      <div style={{ display: "flex", gap: 12, marginBottom: 20 }}>
        <button onClick={runSuite} disabled={running} style={{ padding: "10px 16px" }}>
          {running ? "Running Phase 12…" : "Run Phase 12 validation"}
        </button>
        <button onClick={copyReport} disabled={!report} style={{ padding: "10px 16px" }}>Copy JSON</button>
      </div>
      {error && <pre style={{ whiteSpace: "pre-wrap", background: "#fee", padding: 16 }}>{error}</pre>}
      <section style={{ marginBottom: 24 }}>
        <h2>Live paint target</h2>
        <div id="phase12-live-output" style={{ minHeight: 100, border: "1px solid #aaa", padding: 16, whiteSpace: "pre-wrap" }}>
          {visibleText || "First streamed response text will render here."}
        </div>
      </section>
      <section>
        <h2>Validation report</h2>
        <pre id="phase12-results" style={{ overflowX: "auto", whiteSpace: "pre-wrap", background: "#111", color: "#eee", padding: 16 }}>
          {pretty || "No report yet."}
        </pre>
      </section>
    </main>
  );
}
