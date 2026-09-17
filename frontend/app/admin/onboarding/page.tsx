"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

type Principal = {
  organization_id: string;
  display_name: string;
  roles: string[];
};

type Pack = {
  name: string;
  organization_id: string;
  fingerprint: string;
  valid: boolean;
  counts: Record<string, number>;
  issues: Array<{ code: string; message: string; path?: string | null }>;
  required_secret_environment_variables: string[];
};

type Report = {
  status: string;
  organization_id: string;
  pack_fingerprint: string;
  report_path?: string | null;
  warnings: string[];
};

const api = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

function cookie(name: string) {
  if (typeof document === "undefined") return "";
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

export default function OrganizationOnboardingPage() {
  const [principal, setPrincipal] = useState<Principal | null>(null);
  const [packs, setPacks] = useState<Pack[]>([]);
  const [confirmation, setConfirmation] = useState("");
  const [result, setResult] = useState<Pack | Report | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState("");

  async function load() {
    setError("");
    try {
      const me = await apiFetch("/api/auth/me");
      if (!me.ok) throw new Error("Sign in before opening organization onboarding.");
      const identity = (await me.json()).principal as Principal;
      setPrincipal(identity);
      if (!identity.roles.includes("ADMIN")) throw new Error("Organization onboarding is restricted to administrators.");
      const response = await apiFetch("/api/admin/organization-packs");
      if (!response.ok) throw new Error("Organization Pack catalog could not be loaded.");
      setPacks(await response.json());
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Onboarding unavailable");
    }
  }

  useEffect(() => { void load(); }, []);

  async function act(pack: Pack, action: "validate" | "import") {
    setBusy(`${pack.name}:${action}`); setError(""); setResult(null);
    try {
      const response = await apiFetch(`/api/admin/organization-packs/${pack.name}/${action}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: action === "import" ? JSON.stringify({
          confirm_organization_id: confirmation,
          dry_run: false,
          rotate_local_passwords: false,
        }) : undefined,
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload?.detail?.message ?? payload?.detail?.code ?? "Request rejected");
      setResult(payload);
      await load();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Request failed");
    } finally {
      setBusy("");
    }
  }

  return <main className="monitorPage onboardingPage">
    <header className="monitorHeader">
      <div><small>ADMINISTRATION · LOCAL ONLY</small><h1>Organization Pack onboarding</h1></div>
      <Link href="/">← Workbench</Link>
    </header>
    <p className="monitorNote">Validate first. Imports are additive, audited, organization-scoped, and never delete omitted records.</p>
    {principal && <div className="sovereignHero"><div className="pulse"/><div><small>AUTHENTICATED ORGANIZATION</small><h2>{principal.organization_id}</h2><p>{principal.display_name} · Administrator</p></div></div>}
    {error && <div className="monitorAccessDenied monitorServiceError"><div className="monitorAccessIcon">!</div><div><label>REQUEST REJECTED</label><h2>{error}</h2></div></div>}
    <div className="onboardingGrid">
      {packs.map((pack) => <article className="onboardingCard" key={pack.name}>
        <label>{pack.valid ? "PACK VALID" : "REVIEW REQUIRED"}</label>
        <h2>{pack.name}</h2>
        <p>Organization: <b>{pack.organization_id}</b></p>
        <code className="shortHash">{pack.fingerprint}</code>
        <div className="packCounts">{Object.entries(pack.counts).map(([name, count]) => <span key={name}><b>{count}</b>{name}</span>)}</div>
        <details><summary>Required secret names</summary>{pack.required_secret_environment_variables.length
          ? pack.required_secret_environment_variables.map((name) => <code className="shortHash" key={name}>{name}</code>)
          : <p>None</p>}</details>
        {pack.issues.map((issue) => <p className="error" key={`${issue.code}-${issue.path}`}>{issue.code}: {issue.message}</p>)}
        <button className="evalButton" disabled={Boolean(busy)} onClick={() => void act(pack, "validate")}>{busy === `${pack.name}:validate` ? "Validating…" : "Validate"}</button>
        <label className="confirmationField">TYPE ORGANIZATION ID TO IMPORT
          <input value={confirmation} onChange={(event) => setConfirmation(event.target.value)} placeholder={pack.organization_id}/>
        </label>
        <button className="importButton" disabled={Boolean(busy) || confirmation !== pack.organization_id || !pack.valid} onClick={() => void act(pack, "import")}>{busy === `${pack.name}:import` ? "Importing…" : "Import validated pack"}</button>
      </article>)}
    </div>
    {!error && principal && !packs.length && <div className="monitorAccessDenied"><div className="monitorAccessIcon">i</div><div><label>NO MATCHING PACK</label><h2>No pack is available for {principal.organization_id}</h2><p>Bootstrap a new organization with the local CLI. This page only updates the authenticated administrator&apos;s own organization.</p></div></div>}
    {result && <section className="serviceTable"><label>LAST RESULT</label><pre>{JSON.stringify(result, null, 2)}</pre></section>}
  </main>;
}
