"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

type OrganizationPayload = {
  organization: { id: string; name: string; short_name: string } | null;
  departments: Array<{ id: string; name: string }>;
  workspaces: Array<{ id: string; name: string }>;
  metadata: {
    fictional?: boolean;
    plant?: { id: string; name: string; areas: string[] };
    assets?: Array<{ id: string; area: string; service: string }>;
    scenarios?: Array<{ id: string; user: string; prompt: string; expected: string }>;
  };
};

const api = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

export default function OrganizationPage() {
  const [payload, setPayload] = useState<OrganizationPayload | null>(null);
  const [error, setError] = useState("");
  useEffect(() => {
    fetch(`${api}/api/organization`, { credentials: "include" }).then(async (response) => {
      if (!response.ok) throw new Error("Sign in from the workbench to inspect organization context.");
      setPayload(await response.json());
    }).catch((caught) => setError(caught instanceof Error ? caught.message : "Organization unavailable"));
  }, []);
  return <main className="monitorPage organizationPage">
    <header className="monitorHeader"><div><small>SYNTHETIC ENTERPRISE DEMO</small><h1>{payload?.organization?.name ?? "Organization context"}</h1></div><Link href="/">Back to workbench</Link></header>
    {error && <p className="error">{error}</p>}
    {payload && <>
      <div className="sovereignHero"><div className="pulse"/><div><small>{payload.organization?.short_name} · COMPLETELY FICTIONAL</small><h2>{payload.metadata.plant?.name ?? payload.workspaces[0]?.name}</h2><p>{payload.metadata.plant?.areas?.join(" · ")}</p></div></div>
      <div className="orgColumns"><section><label>DEPARTMENTS</label>{payload.departments.map((item) => <div className="orgRow" key={item.id}><b>{item.name}</b><small>{item.id}</small></div>)}</section><section><label>ASSETS</label>{payload.metadata.assets?.map((item) => <div className="orgRow" key={item.id}><b>{item.id}</b><small>{item.area} · {item.service}</small></div>)}</section></div>
      <section className="scenarioList"><label>AVAILABLE DEMONSTRATION SCENARIOS</label>{payload.metadata.scenarios?.map((item) => <article key={item.id}><b>{item.id}</b><p>“{item.prompt}”</p><small>{item.user} · {item.expected}</small></article>)}</section>
    </>}
  </main>;
}
