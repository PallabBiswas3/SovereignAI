"use client";

import { FormEvent, useEffect, useRef, useState } from "react";
import type { TaskLifecycleInfo } from "./TaskLifecycle";

type Props = {
  taskId: string;
  revision: number;
  questions: TaskLifecycleInfo["required_inputs"];
  disabled: boolean;
  apiFetch: (path: string, init?: RequestInit) => Promise<Response>;
  onContinue: (taskId: string) => void;
  onBusy: (busy: boolean) => void;
};

function inputError(code: string): string {
  const messages: Record<string, string> = {
    RESUME_STALE_REVISION: "Saved progress changed. Refresh saved progress before replying.",
    TASK_NOT_WAITING_FOR_INPUT: "This task is no longer waiting for an answer. Refresh saved progress.",
    TASK_INPUT_INVALID_DIAGNOSTIC_JSON: "Upload valid diagnostic JSON with contract_version, domain, inputs and run_context.",
    TASK_INPUT_DIAGNOSTIC_SCOPE_CHANGED: "The diagnostic domain must match the task's requested analysis.",
    TASK_INPUT_FIELDS_MISMATCH: "Answer each requested question and attach each requested file.",
    TASK_INPUT_FILE_CHANGED: "The uploaded file changed. Upload it again before continuing.",
    TASK_INPUT_EMPTY: "Answers cannot be blank.",
    TASK_INPUT_POLICY_BLOCKED: "This reply conflicts with the task's input policy. Review your answer.",
  };
  return messages[code] ?? "The reply could not be accepted. Check your current access, answers and files, then refresh saved progress if needed.";
}

export default function TaskInputReply({ taskId, revision, questions, disabled, apiFetch, onContinue, onBusy }: Props) {
  const [answers, setAnswers] = useState<Record<string, string>>({});
  const [files, setFiles] = useState<Record<string, File>>({});
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const mounted = useRef(true);
  useEffect(() => {
    mounted.current = true;
    return () => { mounted.current = false; onBusy(false); };
  }, [onBusy]);
  if (!questions.length) return null;

  async function submit(event: FormEvent) {
    event.preventDefault();
    if (busy || disabled) return;
    setBusy(true); onBusy(true); setError("");
    try {
      const attachments: Record<string, string> = {};
      for (const question of questions.filter(item => item.kind === "attachment")) {
        if (!files[question.key]) throw new Error("Attach each requested file before continuing.");
        const form = new FormData();
        form.append("checkpoint_revision", String(revision));
        form.append("input_key", question.key);
        form.append("file", files[question.key]);
        const response = await apiFetch(`/api/tasks/${encodeURIComponent(taskId)}/inputs/uploads`, { method: "POST", body: form });
        const payload = await response.json();
        if (!mounted.current) return;
        if (!response.ok) throw new Error(inputError(String(payload.detail?.code ?? "")));
        attachments[question.key] = payload.upload_id;
      }
      const response = await apiFetch(`/api/tasks/${encodeURIComponent(taskId)}/inputs`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ checkpoint_revision: revision, answers, attachments }),
      });
      const payload = await response.json();
      if (!mounted.current) return;
      if (!response.ok) throw new Error(inputError(String(payload.detail?.code ?? "")));
      onContinue(payload.task_id);
    } catch (caught) {
      if (mounted.current) setError(caught instanceof Error ? caught.message : "Could not continue this task.");
    } finally {
      if (mounted.current) { setBusy(false); onBusy(false); }
    }
  }

  return <form className="agentTrace" aria-label="Supply missing task inputs" onSubmit={submit}>
    <h3>Continue this task</h3>
    <p>Supply the requested inputs to continue the original task from its saved progress.</p>
    <fieldset disabled={disabled || busy}>
      {questions.map(question => <div key={question.key}>
        <label htmlFor={`task-input-${question.key}`}>{question.prompt}</label>
        {question.kind === "attachment"
          ? <input id={`task-input-${question.key}`} type="file" required
              accept={question.key === "diagnostic_input" ? ".json" : ".txt,.md,.pdf,.docx,.xlsx,.csv,.json,.png,.jpg,.jpeg,.tif,.tiff"}
              onChange={event => setFiles(current => {
                const updated = { ...current };
                const file = event.target.files?.[0];
                if (file) updated[question.key] = file; else delete updated[question.key];
                return updated;
              })} />
          : <textarea id={`task-input-${question.key}`} required maxLength={4000} rows={3}
              value={answers[question.key] ?? ""}
              onChange={event => setAnswers(current => ({ ...current, [question.key]: event.target.value }))} />}
      </div>)}
      <button type="submit">{busy ? "Validating inputs..." : "Submit inputs and continue"}</button>
    </fieldset>
    {error && <p role="alert" className="error">{error}</p>}
  </form>;
}
