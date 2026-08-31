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

export default function SovereigntyPage() {
  const [monitor, setMonitor] = useState<Monitor | null>(null);
  const [error, setError] = useState("");
  useEffect(() => {
    fetch(`${process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000"}/api/monitor/network`)
      .then((response) => { if (!response.ok) throw new Error(`Monitor returned ${response.status}`); return response.json(); })
      .then(setMonitor).catch((caught) => setError(String(caught)));
  }, []);
  return <main className="monitorPage">
    <div className="monitorHeader"><div><small>SOVEREIGNTY MONITOR</small><h1>Air-gap control plane</h1></div><Link href="/">← Workbench</Link></div>
    {error && <p className="error">{error}</p>}
    <div className="sovereignHero"><span className="pulse"/><div><label>STATUS</label><h2>{monitor?.sovereignty_status?.toUpperCase() ?? "CHECKING"}</h2><p>Runtime inference endpoints are restricted to local or private infrastructure.</p></div></div>
    <div className="statGrid"><article><label>EXTERNAL AI APIS</label><strong>{monitor?.external_ai_apis ?? "—"}</strong></article><article><label>EXTERNAL REQUEST ATTEMPTS</label><strong>{monitor?.external_requests ?? "—"}</strong></article><article><label>TELEMETRY</label><strong>OFF</strong></article></div>
    <section className="serviceTable"><label>LOCAL SERVICES</label>{monitor?.services.map((service) => <div key={service.name}><b>{service.name}</b><span>{service.endpoint}</span><em className={service.status}>{service.status}</em></div>)}</section>
    <p className="monitorNote">{monitor?.note}</p>
  </main>;
}

