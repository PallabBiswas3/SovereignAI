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
  graph_evidence?: { status?: string; chunks?: Array<{ id: string; content: string; metadata?: Record<string, unknown> }>; claims?: Array<{ id: string; claim_text: string }> };
  diagnostic?: { decision?: string; detection?: { abnormal?: boolean }; hypotheses?: Array<{ label?: string; confidence?: number }>; recommended_actions?: string[] };
  controlplane?: { decision?: { action?: string; risk_score?: number; reason?: string }; findings?: Array<{ category: string; severity: string; message: string }> };
};

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
  const [query, setQuery] = useState("Assess the current condition and explain the evidence.");
  const [useDiagnostics, setUseDiagnostics] = useState(false);
  const [domain, setDomain] = useState("bearing");
  const [inputs, setInputs] = useState("{}");
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
    <header className="monitorHeader"><div><small>CONTROLLED CROSS-SERVICE WORKFLOW</small><h1>Integrated industrial analysis</h1></div><Link href="/">Back to workbench</Link></header>
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
          <div className="releaseHeader"><div><label>{analysis.run_id}</label><h2>{analysis.released ? "Released by policy" : "Held by policy"}</h2></div><span className={analysis.released ? "released" : "held"}>{analysis.controlplane?.decision?.action ?? analysis.status}</span></div>
          <article className="finalResponse"><label>FINAL RESPONSE</label><p>{analysis.final_response}</p></article>
          <div className="integrationColumns">
            <section><label>DOCUMENT EVIDENCE</label><b>{analysis.graph_evidence?.status ?? "not returned"}</b>{analysis.graph_evidence?.claims?.map((claim) => <p key={claim.id}>{claim.claim_text}</p>)}{analysis.graph_evidence?.chunks?.slice(0, 4).map((chunk) => <details key={chunk.id}><summary>{chunk.id}</summary><p>{chunk.content}</p></details>)}</section>
            <section><label>SENSOR DIAGNOSIS</label><b>{analysis.diagnostic?.decision ?? "not requested"}</b>{analysis.diagnostic?.hypotheses?.slice(0, 4).map((item, index) => <p key={`${item.label}-${index}`}>{item.label} {item.confidence != null ? `(${Math.round(item.confidence * 100)}%)` : ""}</p>)}{analysis.diagnostic?.recommended_actions?.map((action) => <p key={action}>{action}</p>)}</section>
          </div>
          {analysis.controlplane?.findings?.length ? <section className="policyFindings"><label>POLICY FINDINGS</label>{analysis.controlplane.findings.map((finding, index) => <p key={`${finding.category}-${index}`}><b>{finding.severity} {finding.category}</b> {finding.message}</p>)}</section> : null}
        </>}
      </section>
    </div>
  </main>;
}
