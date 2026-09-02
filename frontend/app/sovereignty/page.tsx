"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

type Monitor = {
  sovereignty_status: string;
  external_ai_apis: number;
  external_requests: number;
  services: Array<{ name: string; endpoint: string; status: string }>;
  note: string;
};

type MonitorState =
  | { status: "loading" }
  | { status: "ready"; monitor: Monitor }
  | { status: "forbidden" }
  | { status: "unauthorized" }
  | { status: "error"; message: string };

export default function SovereigntyPage() {
  const [state, setState] = useState<MonitorState>({ status: "loading" });

  useEffect(() => {
    const controller = new AbortController();

    async function loadMonitor() {
      try {
        const response = await fetch(
          `${process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"}/api/monitor/network`,
          { credentials: "include", signal: controller.signal },
        );

        if (response.status === 403) {
          setState({ status: "forbidden" });
          return;
        }
        if (response.status === 401) {
          setState({ status: "unauthorized" });
          return;
        }
        if (!response.ok) {
          setState({ status: "error", message: `Monitor service returned HTTP ${response.status}.` });
          return;
        }

        const monitor = await response.json() as Monitor;
        setState({ status: "ready", monitor });
      } catch (caught) {
        if (caught instanceof DOMException && caught.name === "AbortError") return;
        setState({ status: "error", message: "The local monitoring service could not be reached." });
      }
    }

    void loadMonitor();
    return () => controller.abort();
  }, []);

  return <main className="monitorPage">
    <div className="monitorHeader">
      <div><small>SOVEREIGNTY MONITOR</small><h1>Air-gap control plane</h1></div>
      <Link href="/">← Workbench</Link>
    </div>

    {state.status === "loading" && <section className="monitorState" aria-live="polite">
      <span className="pulse"/>
      <div><label>STATUS</label><h2>CHECKING</h2><p>Requesting authorized monitoring data from the local control plane.</p></div>
    </section>}

    {state.status === "forbidden" && <section className="monitorAccessDenied" role="alert">
      <div className="monitorAccessIcon" aria-hidden="true">!</div>
      <div>
        <label>ACCESS RESTRICTED</label>
        <h2>Auditor access required</h2>
        <p>Your account is authenticated, but it does not have the <code>audit.read</code> permission required to view system-wide sovereignty telemetry.</p>
        <p>Asset and task access remain available according to your assigned role.</p>
        <Link href="/">Return to Workbench</Link>
      </div>
    </section>}

    {state.status === "unauthorized" && <section className="monitorAccessDenied" role="alert">
      <div className="monitorAccessIcon" aria-hidden="true">!</div>
      <div><label>SESSION REQUIRED</label><h2>Sign in to continue</h2><p>Your session is missing or has expired.</p><Link href="/">Return to sign in</Link></div>
    </section>}

    {state.status === "error" && <section className="monitorAccessDenied monitorServiceError" role="alert">
      <div className="monitorAccessIcon" aria-hidden="true">!</div>
      <div><label>MONITOR UNAVAILABLE</label><h2>Monitoring data could not be loaded</h2><p>{state.message}</p><p>Confirm that the backend is running locally, then refresh this page.</p></div>
    </section>}

    {state.status === "ready" && <>
      <div className="sovereignHero"><span className="pulse"/><div><label>STATUS</label><h2>{state.monitor.sovereignty_status.toUpperCase()}</h2><p>Runtime inference endpoints are restricted to local or private infrastructure.</p></div></div>
      <div className="statGrid"><article><label>EXTERNAL AI APIS</label><strong>{state.monitor.external_ai_apis}</strong></article><article><label>EXTERNAL REQUEST ATTEMPTS</label><strong>{state.monitor.external_requests}</strong></article><article><label>TELEMETRY</label><strong>OFF</strong></article></div>
      <section className="serviceTable"><label>LOCAL SERVICES</label>{state.monitor.services.map((service) => <div key={service.name}><b>{service.name}</b><span>{service.endpoint}</span><em className={service.status}>{service.status}</em></div>)}</section>
      <p className="monitorNote">{state.monitor.note}</p>
    </>}
  </main>;
}
