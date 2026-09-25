"use client";

import Link from "next/link";
import { FormEvent, useEffect, useState } from "react";

const api = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

type Health = { status: string; services: Record<string, { status: string; service?: string; detail?: string }> };
type Analysis = {
  run_id: string;
  released: boolean;
  status: string;
  final_response: string;
  service_status: Record<string, string>;
  graph_evidence?: {
    status?: string;
    chunks?: Array<{ id: string; content: string; similarity?: number; metadata?: Record<string, unknown>; authorization?: Record<string, unknown> }>;
    claims?: Array<{ id: string; claim_text: string }>;
    verification?: { summary?: Record<string, number>; results?: Array<{ label?: string }> };
  };
  diagnostic?: { decision?: string; detection?: { abnormal?: boolean }; hypotheses?: Array<{ label?: string; confidence?: number }>; recommended_actions?: string[] };
  controlplane?: { decision?: { action?: string; risk_score?: number; reason?: string }; findings?: Array<{ category: string; severity: string; message: string }> };
  model_runtime?: { provider?: string; model?: string; ttft_ms?: number; tokens_per_second?: number; prompt_tokens?: number; fallback?: boolean };
  timings_ms?: Record<string, number>;
  artifacts?: Array<{ id: string; name: string; sha256?: string; url: string }>;
  capsule?: { id?: string; state: string; capsule_root_hash?: string; signature_status?: string; verify_url?: string; download_url?: string; error?: string };
};

const pumpSignal = [
  0.11, 0.12, 0.1, 0.14, 0.13, 0.16, 0.18, 0.21,
  0.19, 0.24, 0.3, 0.36, 0.42, 0.49, 0.61, 0.73,
  0.68, 0.79, 0.86, 0.91, 0.82, 0.74, 0.66, 0.58,
  0.5, 0.43, 0.38, 0.31, 0.26, 0.22, 0.18, 0.15,
];

function cookie(name: string) {
  if (typeof document === "undefined") return "";
  return document.cookie.split("; ").find((item) => item.startsWith(`${name}=`))?.split("=").slice(1).join("=") ?? "";
}

function apiFetch(path: string, init: RequestInit = {}) {
  const headers = new Headers(init.headers);
  const csrf = cookie("sovereign_csrf");
  if (csrf && (init.method ?? "GET").toUpperCase() !== "GET") headers.set("X-CSRF-Token", decodeURIComponent(csrf));
  return fetch(`${api}${path}`, { ...init, headers, credentials: "include" });
}

export default function IntegrationsPage() {
  const [health, setHealth] = useState<Health | null>(null);
  const [query, setQuery] = useState("Assess Pump-102 using sensor history and authorized manuals/SOPs. Recommend the next maintenance action with citations.");
  const [useDiagnostics, setUseDiagnostics] = useState(true);
  const [domain, setDomain] = useState("bearing");
  const [inputs, setInputs] = useState(JSON.stringify({ signal: pumpSignal, sampling_rate_hz: 8000 }, null, 2));
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    apiFetch("/api/integrations/health").then(async (response) => {
      if (!response.ok) throw new Error("Sign in from the workbench to inspect integration services.");
      return response.json();
    }).then(setHealth).catch((caught) => setError(caught instanceof Error ? caught.message : "Health check failed"));
  }, []);

  async function submit(event: FormEvent) {
    event.preventDefault(); setBusy(true); setError(""); setAnalysis(null);
    try {
      const diagnosticInputs = useDiagnostics ? JSON.parse(inputs) : undefined;
      const response = await apiFetch("/api/integrations/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query,
          include_graph_evidence: true,
          diagnostic: useDiagnostics ? { domain, inputs: diagnosticInputs } : null,
          policy_profile: "internal_assistant",
          consequential: true,
        }),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail?.code ?? payload.detail ?? `Analysis failed (${response.status})`);
      setAnalysis(payload);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Integrated analysis failed");
    } finally { setBusy(false); }
  }

  return <main className="integrationPage">
    <header className="monitorHeader"><div><small>SOVEREIGNAI FLAGSHIP · CANONICAL DEMO</small><h1>Pump-102 evidence-to-maintenance workflow</h1><p>Sensor history → diagnostic agent → authorized GraphRAG → local model → ControlPlane → artifact → Evidence Capsule</p></div><Link href="/">Back to workbench</Link></header>
    <section className="integrationStatus">
      <div><label>ORCHESTRATOR</label><strong>{health?.status ?? "checking"}</strong></div>
      {Object.entries(health?.services ?? {}).map(([name, service]) => <div key={name}><label>{name}</label><strong className={service.status === "ok" ? "serviceReady" : "serviceUnavailable"}>{service.status}</strong><small>{service.detail}</small></div>)}
    </section>
    <div className="integrationLayout">
      <form className="integrationForm" onSubmit={submit}>
        <label>ANALYSIS REQUEST<textarea value={query} onChange={(event) => setQuery(event.target.value)} required minLength={3}/></label>
        <label className="integrationToggle"><input type="checkbox" checked={useDiagnostics} onChange={(event) => setUseDiagnostics(event.target.checked)}/> Include time-series diagnosis</label>
        {useDiagnostics && <div className="diagnosticInputs"><label>DOMAIN<select value={domain} onChange={(event) => setDomain(event.target.value)}><option>bearing</option><option>process</option><option>wind_scada</option><option>battery</option><option>turbofan</option><option>transformer</option></select></label><label>DIAGNOSTIC INPUTS (JSON)<textarea value={inputs} onChange={(event) => setInputs(event.target.value)} spellCheck={false}/></label></div>}
        <button disabled={busy || !query.trim()}>{busy ? "Running controlled workflow..." : "Run integrated analysis"}</button>
        {error && <p className="error">{error}</p>}
      </form>
      <section className="integrationResult">
        {!analysis && <div className="integrationEmpty"><h2>Evidence before release</h2><p>Graph evidence and sensor diagnostics are gathered independently. ControlPlane.ai decides whether the assembled response can be released.</p></div>}
        {analysis && <>
          <div className="releaseHeader"><div><label>{analysis.run_id}</label><h2>{analysis.released ? "Released by ControlPlane" : "Not released"}</h2></div><span className={analysis.released ? "released" : "held"}>{analysis.controlplane?.decision?.action ?? analysis.status}</span></div>
          <div className="flagshipGrid">
            <section><label>1 · AUTHORIZED EVIDENCE</label><b>{analysis.graph_evidence?.status ?? "not returned"}</b>{analysis.graph_evidence?.chunks?.slice(0, 5).map((chunk) => { const meta = chunk.metadata ?? {}; const auth = chunk.authorization ?? {}; return <details key={chunk.id} open><summary>{String(meta.file ?? meta.sourceDocId ?? chunk.id)}</summary><small>Rev {String(meta.revision ?? "—")} · Page {String(meta.page ?? meta.pageStart ?? "—")} · ACL {String(auth.decision ?? "not reported")}</small><p>{chunk.content}</p></details>; })}</section>
            <section><label>2 · SENSOR DIAGNOSIS</label><b>{analysis.diagnostic?.detection?.abnormal ? "ANOMALY DETECTED" : analysis.diagnostic?.decision ?? "not requested"}</b>{analysis.diagnostic?.hypotheses?.slice(0, 4).map((item, index) => <div className="metric" key={`${item.label}-${index}`}><span>{item.label}</span><b>{item.confidence != null ? `${Math.round(item.confidence * 100)}%` : "evidence-linked"}</b></div>)}{analysis.diagnostic?.recommended_actions?.map((action) => <p key={action}>{action}</p>)}</section>
            <section><label>3 · LOCAL MODEL RUNTIME</label><b>{analysis.model_runtime?.provider ?? analysis.service_status.model ?? "not called"}</b><p>{analysis.model_runtime?.model}</p><div className="metric"><span>TTFT</span><b>{analysis.model_runtime?.ttft_ms != null ? `${analysis.model_runtime.ttft_ms} ms` : "—"}</b></div><div className="metric"><span>Generation</span><b>{analysis.model_runtime?.tokens_per_second != null ? `${analysis.model_runtime.tokens_per_second} tok/s` : "—"}</b></div><div className="metric"><span>Prompt</span><b>{analysis.model_runtime?.prompt_tokens != null ? `${analysis.model_runtime.prompt_tokens} tokens` : "—"}</b></div></section>
            <section><label>4 · GOVERNANCE</label><b>{analysis.controlplane?.decision?.action ?? analysis.status}</b><div className="metric"><span>Supported</span><b>{analysis.graph_evidence?.verification?.summary?.supported ?? analysis.graph_evidence?.claims?.length ?? 0}</b></div><div className="metric"><span>Conflicting</span><b>{analysis.graph_evidence?.verification?.summary?.contradicted ?? 0}</b></div><div className="metric"><span>Release</span><b>{analysis.released ? "YES" : "NO"}</b></div><p>{analysis.controlplane?.decision?.reason}</p></section>
          </div>
          <article className="finalResponse"><label>5 · FINAL ANSWER</label><p>{analysis.final_response}</p><small>{analysis.graph_evidence?.claims?.length ?? 0} verified document claims · {(analysis.graph_evidence?.chunks ?? []).length} authorized chunks · {analysis.timings_ms?.total != null ? `${analysis.timings_ms.total} ms total` : "runtime recorded"}</small></article>
          <div className="outputRail"><section><label>6 · GENERATED ARTIFACT</label>{analysis.artifacts?.length ? analysis.artifacts.map((artifact) => <a href={`${api}${artifact.url}`} key={artifact.id}>Download {artifact.name}</a>) : <p>Created only after release.</p>}</section><section><label>7 · EVIDENCE CAPSULE</label>{analysis.capsule ? <><b>{analysis.capsule.state}</b><code>{analysis.capsule.capsule_root_hash ?? analysis.capsule.error}</code>{analysis.capsule.download_url && <a href={`${api}${analysis.capsule.download_url}`}>Download verifiable capsule</a>}</> : <p>No capsule is created for held or incomplete runs.</p>}</section></div>
          {analysis.controlplane?.findings?.length ? <section className="policyFindings"><label>POLICY FINDINGS</label>{analysis.controlplane.findings.map((finding, index) => <p key={`${finding.category}-${index}`}><b>{finding.severity} {finding.category}</b> {finding.message}</p>)}</section> : null}
        </>}
      </section>
    </div>
  </main>;
}
