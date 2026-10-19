"use client";

import { useEffect, useRef, useState } from "react";
import { VoiceRecorder, type VoiceState } from "./recorder";

type Props = {
  disabled: boolean;
  apiFetch: (path: string, init?: RequestInit) => Promise<Response>;
  onTranscript: (text: string) => void;
  onBusy: (busy: boolean) => void;
};

export default function VoiceInput(props: Props) {
  const callbacks = useRef(props);
  callbacks.current = props;
  const [view, setView] = useState<{ state: VoiceState; detail: string; elapsed: number }>({ state: "idle", detail: "", elapsed: 0 });
  const controller = useRef<VoiceRecorder | null>(null);
  useEffect(() => {
    const recorder = new VoiceRecorder({
      fetch: (path, init) => callbacks.current.apiFetch(path, init),
      state: (state, detail, elapsed) => {
        setView({ state, detail, elapsed });
        callbacks.current.onBusy(["requesting", "recording", "processing"].includes(state));
      },
      transcript: (text) => callbacks.current.onTranscript(text),
    });
    controller.current = recorder;
    return () => { recorder.cancel(false); controller.current = null; };
  }, []);
  useEffect(() => { if (props.disabled) controller.current?.cancel(); }, [props.disabled]);
  const busy = ["requesting", "recording", "processing"].includes(view.state);
  return <span className="voiceInput">
    {!busy && <button type="button" disabled={props.disabled} aria-label="Start voice recording" title="Record a voice message" onClick={() => void controller.current?.start()}>
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true"><rect x="9" y="2" width="6" height="12" rx="3"/><path d="M5 10v2a7 7 0 0 0 14 0v-2M12 19v3m-4 0h8"/></svg>
    </button>}
    {view.state === "recording" && <button type="button" onClick={() => controller.current?.stop()}>Stop</button>}
    {busy && <button type="button" onClick={() => controller.current?.cancel()}>Cancel voice</button>}
    <span role={view.state === "error" ? "alert" : "status"} aria-live="polite" className={view.state === "error" ? "error" : view.state === "recording" ? "voiceRecording" : "voiceStatus"}>
      {view.state === "requesting" ? "Waiting for microphone permission…" : view.state === "recording"
        ? `● Recording ${Math.floor(view.elapsed / 60)}:${String(view.elapsed % 60).padStart(2, "0")}`
        : view.state === "processing" ? "Processing audio locally…" : view.detail}
    </span>
  </span>;
}
