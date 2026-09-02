"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

type MetricPayload = { metrics: {
  routing: { accuracy: number };
  rag: { retrieval_precision_at_1: number | null; retrieval_recall_at_1: number | null; batch2?: { comparison: Record<string, { recall_at_5: number; mrr: number; mean_total_latency_ms: number; reranker_available?: boolean | null; warning?: string | null }>; context_compiler: { raw_candidate_count: number; final_evidence_count: number; compiled_context_tokens: number; required_fact_preserved: boolean } } };
  governance: { pii: { precision: number; recall: number }; prompt_injection: { precision: number; recall: number } };
  agent: { total_runs: number; task_completion_rate: number | null; failed_tool_calls: number };
  system: { cpu_percent: number; process_ram_mb: number; system_ram_percent: number; note: string };
}};

const pct = (value: number | null | undefined) => value == null ? "N/A" : `${Math.round(value * 100)}%`;

export default function MetricsPage() {
  const [data, setData] = useState<MetricPayload | null>(null);
  const [running, setRunning] = useState(false);
  const api = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
  useEffect(() => { fetch(`${api}/api/evaluation/metrics`, { credentials: "include" }).then((response) => response.json()).then(setData); }, [api]);
  async function run() { const csrf = document.cookie.split("; ").find((item) => item.startsWith("sovereign_csrf="))?.split("=")[1] ?? ""; setRunning(true); const response = await fetch(`${api}/api/evaluation/run`, { method: "POST", credentials: "include", headers: { "X-CSRF-Token": decodeURIComponent(csrf) } }); setData(await response.json()); setRunning(false); }
  const metrics = data?.metrics;
  return <main className="monitorPage">
    <div className="monitorHeader"><div><small>OFFLINE EVALUATION</small><h1>Quality & operations</h1></div><div><button className="evalButton" onClick={run} disabled={running}>{running ? "Running…" : "Run benchmarks"}</button> <Link href="/">← Workbench</Link></div></div>
    <div className="statGrid metricsGrid"><article><label>ROUTING ACCURACY</label><strong>{pct(metrics?.routing.accuracy)}</strong></article><article><label>RAG PRECISION @1</label><strong>{pct(metrics?.rag.retrieval_precision_at_1)}</strong></article><article><label>TASK COMPLETION</label><strong>{pct(metrics?.agent.task_completion_rate)}</strong></article></div>
    <div className="metricPanels"><section><label>GOVERNANCE</label><div className="metric"><span>PII precision / recall</span><b>{pct(metrics?.governance.pii.precision)} / {pct(metrics?.governance.pii.recall)}</b></div><div className="metric"><span>Injection precision / recall</span><b>{pct(metrics?.governance.prompt_injection.precision)} / {pct(metrics?.governance.prompt_injection.recall)}</b></div></section><section><label>AGENT</label><div className="metric"><span>Total runs</span><b>{metrics?.agent.total_runs ?? "—"}</b></div><div className="metric"><span>Failed tool calls</span><b>{metrics?.agent.failed_tool_calls ?? "—"}</b></div></section><section><label>SYSTEM</label><div className="metric"><span>CPU</span><b>{metrics?.system.cpu_percent ?? "—"}%</b></div><div className="metric"><span>Process RAM</span><b>{metrics?.system.process_ram_mb ?? "—"} MB</b></div><div className="metric"><span>System RAM</span><b>{metrics?.system.system_ram_percent ?? "—"}%</b></div></section></div>
    {metrics?.rag.batch2 && <div className="metricPanels"><section><label>BATCH 2 RETRIEVAL</label>{Object.entries(metrics.rag.batch2.comparison).map(([name, value]) => <div className="metric" key={name}><span>{name} Recall@5 / MRR</span><b>{pct(value.recall_at_5)} / {value.mrr.toFixed(2)}</b></div>)}</section><section><label>CONTEXT COMPILER</label><div className="metric"><span>Evidence selected</span><b>{metrics.rag.batch2.context_compiler.final_evidence_count}/{metrics.rag.batch2.context_compiler.raw_candidate_count}</b></div><div className="metric"><span>Compiled tokens</span><b>{metrics.rag.batch2.context_compiler.compiled_context_tokens}</b></div><div className="metric"><span>Required fact retained</span><b>{metrics.rag.batch2.context_compiler.required_fact_preserved ? "YES" : "NO"}</b></div></section></div>}
    <p className="monitorNote">{metrics?.system.note}</p>
  </main>;
}
