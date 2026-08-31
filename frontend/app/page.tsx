"use client";

import Link from "next/link";
import { FormEvent, KeyboardEvent, useEffect, useRef, useState } from "react";

type TaskResult = {
  id: string;
  status: string;
  final_response: string | null;
  warnings: string[];
  governance: { decision?: string; privacy_risk?: string; grounding_score?: number | null };
  artifacts: Array<{ id: string; name: string; url: string }>;
  sources: Array<{ file?: string; page?: number | null; section?: string | null; text?: string }>;
  plan: { steps: Array<{ id: number; title: string; status: string; observation: string | null }> };
  routing: { model_id: string; confidence: number; reason: string };
};

type ModelStatus = {
  id: string;
  role: string;
  display_name: string;
  model_tag: string;
  availability: "READY" | "MODEL_NOT_INSTALLED" | "OLLAMA_UNAVAILABLE" | "CONFIGURATION_ERROR";
  detail: string;
};

type LiveEvent = { type: string; timestamp: string; payload: Record<string, unknown> };

type ChatEntry = {
  id: string;
  role: "user" | "assistant";
  content: string;
  waiting?: boolean;
  warnings?: string[];
  attachments?: string[];
};

const eventLabels: Record<string, string> = {
  task_accepted: "Task accepted locally",
  governance_completed: "Safety checks completed",
  task_classified: "Task classified",
  model_selected: "Local model selected",
  plan_created: "Execution plan created",
  step_started: "Working on the next step",
  tool_proposed: "Preparing a controlled tool",
  tool_started: "Running a local tool",
  tool_completed: "Local tool completed",
  source_retrieved: "Source evidence retrieved",
  artifact_created: "Artifact created",
  warning: "Reviewing a warning",
  step_completed: "Step completed",
};

const api = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

export default function Home() {
  const [message, setMessage] = useState("");
  const [result, setResult] = useState<TaskResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [modelOverride, setModelOverride] = useState("");
  const [attachments, setAttachments] = useState<Array<{ name: string; path: string }>>([]);
  const [models, setModels] = useState<ModelStatus[]>([]);
  const [liveEvents, setLiveEvents] = useState<LiveEvent[]>([]);
  const [chatEntries, setChatEntries] = useState<ChatEntry[]>([]);
  const conversationEnd = useRef<HTMLDivElement>(null);

  useEffect(() => {
    fetch(`${api}/api/models/status`).then((response) => response.json())
      .then((payload) => setModels(payload.models ?? [])).catch(() => setModels([]));
  }, []);

  useEffect(() => {
    conversationEnd.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [chatEntries, liveEvents]);

  async function submit(event: FormEvent) {
    event.preventDefault();
    if (loading || !message.trim()) return;
    const submittedMessage = message.trim();
    const submittedAttachments = attachments.map((item) => item.name);
    const assistantEntryId = `assistant-${Date.now()}`;
    setChatEntries((current) => [...current,
      { id: `user-${Date.now()}`, role: "user", content: submittedMessage, attachments: submittedAttachments },
      { id: assistantEntryId, role: "assistant", content: "", waiting: true },
    ]);
    setMessage("");
    setAttachments([]);
    setLoading(true); setError("");
    try {
      setResult(null); setLiveEvents([]);
      const response = await fetch(`${api}/api/tasks/start`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          request: submittedMessage,
          model_override: modelOverride || null,
          attachments: attachments.map((item) => item.path),
          use_case: submittedMessage.toLowerCase().includes("inspection") ? "engineering" : "internal_assistant",
        }),
      });
      if (!response.ok) throw new Error(`Backend returned ${response.status}: ${await response.text()}`);
      const started = await response.json();
      const stream = new EventSource(`${api}/api/tasks/${started.task_id}/events`);
      let finished = false;
      const eventTypes = ["task_accepted", "governance_completed", "task_classified", "model_selected", "plan_created", "step_started", "tool_proposed", "tool_started", "tool_completed", "source_retrieved", "artifact_created", "warning", "step_completed", "task_completed", "task_failed"];
      const receive = (event: MessageEvent) => {
        const item = JSON.parse(event.data) as LiveEvent;
        setLiveEvents((current) => [...current, item]);
        if (item.type === "task_completed") {
          const completed = item.payload.result as TaskResult;
          finished = true;
          setResult(completed);
          setChatEntries((current) => current.map((entry) => entry.id === assistantEntryId
            ? { ...entry, content: completed.final_response ?? "Task completed without a text response.", waiting: false, warnings: completed.warnings }
            : entry));
          setLoading(false); stream.close();
        } else if (item.type === "task_failed") {
          const failure = String(item.payload.error ?? "Task failed");
          finished = true;
          setError(failure);
          setChatEntries((current) => current.map((entry) => entry.id === assistantEntryId
            ? { ...entry, content: failure, waiting: false, warnings: ["The task stopped safely."] }
            : entry));
          setLoading(false); stream.close();
        }
      };
      eventTypes.forEach((type) => stream.addEventListener(type, receive as EventListener));
      stream.onerror = () => {
        if (!finished && stream.readyState === EventSource.CLOSED) {
          setLoading(false);
          setChatEntries((current) => current.map((entry) => entry.id === assistantEntryId
            ? { ...entry, content: "The live task connection closed before a final response arrived.", waiting: false }
            : entry));
        }
      };
    } catch (caught) {
      const failure = caught instanceof Error ? caught.message : "Request failed";
      setError(failure);
      setChatEntries((current) => current.map((entry) => entry.id === assistantEntryId
        ? { ...entry, content: failure, waiting: false, warnings: ["The request could not be started."] }
        : entry));
      setLoading(false);
    }
  }

  async function upload(file: File) {
    setError("");
    const form = new FormData(); form.append("file", file);
    try {
      const response = await fetch(`${api}/api/files/upload`, { method: "POST", body: form });
      if (!response.ok) throw new Error(`Upload returned ${response.status}`);
      const uploaded = await response.json();
      setAttachments((current) => [...current, { name: uploaded.name, path: uploaded.path }]);
    } catch (caught) { setError(caught instanceof Error ? caught.message : "Upload failed"); }
  }

  return <main className="shell">
    <aside className="sidebar">
      <div className="brand"><span>S</span> SovereignAI</div>
      <button className="newTask" onClick={() => { setResult(null); setMessage(""); setAttachments([]); setLiveEvents([]); setChatEntries([]); setError(""); }}>+ New task</button>
      <nav><a className="active">Workbench</a><a>Conversations</a><a>Files</a><a>Knowledge base</a><a>Generated artifacts</a><Link href="/sovereignty">Sovereignty monitor</Link><Link href="/metrics">Evaluation metrics</Link></nav>
      <div className="localBadge"><i /> AIR-GAPPED MODE<br/><small>Local services only</small></div>
    </aside>
    <section className="workspace">
      <header><div><small>INDUSTRIAL AI WORKBENCH</small><h1>What should we accomplish?</h1></div><span className="classification">INTERNAL</span></header>
      <div className="conversation">
        {chatEntries.length ? <div className="chatThread">
          {chatEntries.map((entry) => entry.role === "user"
            ? <article className="messageRow userMessage" key={entry.id}><div className="messageBubble"><b>You</b><p>{entry.content}</p>{entry.attachments?.map((name) => <small className="sentAttachment" key={name}>Attached: {name}</small>)}</div><span className="avatar userAvatar">P</span></article>
            : <article className="messageRow assistantMessage" key={entry.id}><span className="avatar">S</span><div className="messageBubble"><b>SovereignAI</b>{entry.waiting
              ? <div className="waitingResponse"><span className="waitingDots"><i/><i/><i/></span><p>{eventLabels[liveEvents.at(-1)?.type ?? ""] ?? "Waiting for the local model…"}</p><small>Local CPU inference can take a few minutes.</small></div>
              : <><p>{entry.content}</p>{entry.warnings?.map((warning) => <em key={warning}>{warning}</em>)}</>}</div></article>)}
          <div ref={conversationEnd}/>
        </div> : <div className="empty"><div className="mark">S</div><h2>Confidential work stays here.</h2><p>Analyze industrial documents, search internal knowledge, execute controlled tools, and generate finished deliverables—all inside your infrastructure.</p></div>}
        {error && !chatEntries.length && <p className="error">{error}</p>}
      </div>
      <form onSubmit={submit} className="composer">
        <textarea value={message} onChange={(event) => setMessage(event.target.value)} onKeyDown={(event: KeyboardEvent<HTMLTextAreaElement>) => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); event.currentTarget.form?.requestSubmit(); } }} placeholder="Message SovereignAI…" />
        {attachments.map((attachment) => <div className="attachment" key={attachment.path}>File: {attachment.name}<button type="button" onClick={() => setAttachments((current) => current.filter((item) => item.path !== attachment.path))}>x</button></div>)}
        <div><label className="attachButton" title="Attach files">+<input multiple type="file" accept=".txt,.md,.pdf,.docx,.xlsx,.csv,.json,.png,.jpg,.jpeg,.tif,.tiff" onChange={(event) => { Array.from(event.target.files ?? []).forEach((file) => void upload(file)); }} /></label><span>Enter to send · Shift+Enter for a new line</span><button className="send" disabled={loading || !message.trim()}>{loading ? "Waiting…" : "Run task →"}</button></div>
      </form>
    </section>
    <aside className="inspector">
      <h3>Agent execution</h3>
      <section><label>MODEL</label><b>{result?.routing.model_id ?? "Automatic routing"}</b><p>{result?.routing.reason ?? "A local model will be selected for the task."}</p>{result && <div className="metric"><span>Confidence</span><b>{Math.round(result.routing.confidence * 100)}%</b></div>}<div className="metric"><span>Local readiness</span><b>{models.length ? `${models.filter((model) => model.availability === "READY").length}/${models.length} roles` : "Checking"}</b></div><select value={modelOverride} onChange={(event) => setModelOverride(event.target.value)} aria-label="Model override"><option value="">Automatic</option>{models.map((model) => <option key={model.id} value={model.id} disabled={model.availability !== "READY"}>{model.role}: {model.display_name} — {model.availability}</option>)}</select>{models.map((model) => <div className="modelRuntime" key={model.id}><span className={model.availability === "READY" ? "readyDot" : "warnDot"}/><div><b>{model.role}</b><small>{model.model_tag} · {model.availability}</small></div></div>)}</section>
      <section><label>PLAN &amp; LIVE EVENTS</label>{result ? <ol className="plan">{result.plan.steps.map((step) => <li key={step.id} className={step.status}>{step.status === "completed" ? "✓" : step.status === "failed" ? "!" : "○"} {step.title}</li>)}</ol> : liveEvents.length ? <ol className="plan liveEvents">{liveEvents.slice(-10).map((item, index) => <li key={`${item.timestamp}-${index}`} className={item.type === "task_failed" ? "failed" : "completed"}>{item.type.replaceAll("_", " ")}</li>)}</ol> : <p className="muted">Submit a task to see a verifiable execution plan.</p>}</section>
      <section><label>AI SAFETY</label><div className="metric"><span>Decision</span><b>{result?.governance.decision ?? "Pending"}</b></div><div className="metric"><span>Privacy risk</span><b>{result?.governance.privacy_risk ?? "Low"}</b></div><div className="metric"><span>Grounding</span><b>{result?.governance.grounding_score != null ? `${Math.round(result.governance.grounding_score * 100)}%` : "N/A"}</b></div><div className="metric"><span>External requests</span><b>0</b></div></section>
      {result?.sources.length ? <section><label>SOURCES</label>{result.sources.map((source, index) => <details key={`${source.file}-${index}`}><summary>{source.file}</summary><p>Page {source.page ?? "—"} · Section {source.section ?? "—"}</p>{source.text && <p>{source.text}</p>}</details>)}</section> : null}
      {result?.artifacts.length ? <section><label>ARTIFACTS</label>{result.artifacts.map((artifact) => <a className="artifactLink" href={`${api}${artifact.url}`} key={artifact.id}>Download {artifact.name}</a>)}</section> : null}
    </aside>
  </main>;
}
