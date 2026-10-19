export type VoiceState = "idle" | "requesting" | "recording" | "processing" | "ready" | "error";
type Limits = { max_duration_seconds: number; max_upload_bytes: number; timeout_seconds: number; mime_types: string[] };
type Options = {
  fetch: (path: string, init?: RequestInit) => Promise<Response>;
  state: (state: VoiceState, detail: string, elapsed: number) => void;
  transcript: (text: string) => void;
};

const errors: Record<string, string> = {
  ASR_MODEL_UNAVAILABLE: "The local speech model is unavailable. Follow the voice setup instructions.",
  INVALID_AUDIO: "The recording could not be decoded. Please record again.",
  EMPTY_AUDIO: "The recording was empty or too short. Please record again.",
  NO_SPEECH: "No speech was recognized. Please try again.",
  AUDIO_TOO_LONG: "The recording exceeded the duration limit. Please use a shorter question.",
  AUDIO_TOO_LARGE: "The recording exceeded the upload limit. Please use a shorter question.",
  VOICE_BUSY: "Local speech recognition is busy. Please try again shortly.",
  TRANSCRIPTION_TIMEOUT: "Local speech recognition timed out. Try a shorter recording.",
  TRANSCRIPTION_FAILED: "Local speech recognition failed. Please try again.",
};

/** One user-initiated recording at a time; stale permission/network results are ignored. */
export class VoiceRecorder {
  private options: Options;
  private state: VoiceState = "idle";
  private generation = 0;
  private stream?: MediaStream;
  private recorder?: MediaRecorder;
  private abort?: AbortController;
  private timer?: ReturnType<typeof setInterval>;
  private started = 0;
  private elapsed = 0;

  constructor(options: Options) { this.options = options; }

  private update(state: VoiceState, detail = "") {
    this.state = state;
    this.options.state(state, detail, this.elapsed);
  }

  private release() {
    clearInterval(this.timer);
    this.stream?.getTracks().forEach((track) => { track.onended = null; track.stop(); });
    this.stream = undefined;
  }

  cancel(notify = true) {
    ++this.generation;
    this.abort?.abort();
    if (this.recorder) {
      this.recorder.onstop = null;
      this.recorder.ondataavailable = null;
      this.recorder.onerror = null;
      if (this.recorder.state !== "inactive") this.recorder.stop();
      this.recorder = undefined;
    }
    this.release();
    this.elapsed = 0;
    this.state = "idle";
    if (notify) this.update("idle");
  }

  stop() {
    if (this.state !== "recording" || !this.recorder) return;
    this.update("processing");
    clearInterval(this.timer);
    this.recorder.stop();
    this.release();
  }

  async start() {
    if (["requesting", "recording", "processing"].includes(this.state)) return;
    const generation = ++this.generation;
    this.elapsed = 0;
    this.update("requesting");
    let configTimer: ReturnType<typeof setTimeout> | undefined;
    try {
      if (!globalThis.isSecureContext || !navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === "undefined") {
        throw new Error("Microphone recording requires a supported browser on localhost or HTTPS.");
      }
      const abort = new AbortController();
      this.abort = abort;
      configTimer = setTimeout(() => abort.abort(), 10000);
      const response = await this.options.fetch("/api/voice/config", { signal: abort.signal });
      if (!response.ok) throw new Error(response.status === 401 || response.status === 403
        ? "Your session cannot use voice input. Sign in with task access." : "Cannot connect to local speech recognition.");
      const limits: Limits = await response.json();
      clearTimeout(configTimer);
      if (generation !== this.generation) return;
      const mime = ["audio/webm;codecs=opus", "audio/ogg;codecs=opus", "audio/mp4"]
        .find((type) => limits.mime_types.includes(type.split(";")[0]) && MediaRecorder.isTypeSupported(type));
      if (!mime) throw new Error("This browser has no supported audio recording format.");
      // Permission prompts cannot be aborted. A cancelled prompt's eventual stream is stopped immediately.
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      if (generation !== this.generation) { stream.getTracks().forEach((track) => track.stop()); return; }
      this.stream = stream;
      const recorder = new MediaRecorder(stream, { mimeType: mime, audioBitsPerSecond: 64000 });
      this.recorder = recorder;
      const chunks: Blob[] = [];
      let bytes = 0;
      recorder.ondataavailable = (event) => {
        if (generation !== this.generation || !event.data.size) return;
        bytes += event.data.size;
        if (bytes > limits.max_upload_bytes) { this.cancel(); this.update("error", errors.AUDIO_TOO_LARGE); return; }
        chunks.push(event.data);
      };
      recorder.onerror = () => { this.cancel(); this.update("error", "The microphone recording failed. Please try again."); };
      stream.getTracks().forEach((track) => { track.onended = () => {
        this.cancel(); this.update("error", "The microphone disconnected. Please try again.");
      }; });
      recorder.onstop = () => {
        if (generation !== this.generation) return;
        this.release();
        this.recorder = undefined;
        this.update("processing");
        const blob = new Blob(chunks, { type: recorder.mimeType || mime });
        chunks.length = 0;
        void this.transcribe(blob, limits, generation);
      };
      recorder.start(250);
      this.started = Date.now();
      this.update("recording");
      this.timer = setInterval(() => {
        this.elapsed = Math.floor((Date.now() - this.started) / 1000);
        this.update("recording");
        // Leave a small encoding/timer margin below the server's hard PCM limit.
        if (Date.now() - this.started >= (limits.max_duration_seconds - 0.5) * 1000) this.stop();
      }, 200);
    } catch (caught) {
      if (generation !== this.generation) return;
      this.release();
      const name = caught instanceof Error ? caught.name : "";
      const detail = name === "NotAllowedError" ? "Microphone permission was denied. Allow access in your browser and try again."
        : name === "NotFoundError" ? "No microphone is available. Connect a microphone and try again."
        : name === "NotReadableError" ? "The microphone is in use or unavailable. Close other recording apps and try again."
        : name === "AbortError" ? "The local voice service did not respond in time."
        : caught instanceof Error ? caught.message : "Could not start microphone recording.";
      this.update("error", detail);
    } finally { clearTimeout(configTimer); }
  }

  private async transcribe(blob: Blob, limits: Limits, generation: number) {
    if (!blob.size) { this.update("error", errors.EMPTY_AUDIO); return; }
    const abort = new AbortController();
    this.abort = abort;
    const timer = setTimeout(() => abort.abort(), (limits.timeout_seconds + 20) * 1000);
    try {
      const response = await this.options.fetch("/api/voice/transcribe", {
        method: "POST", headers: { "Content-Type": blob.type }, body: blob, signal: abort.signal,
      });
      if (generation !== this.generation) return;
      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(response.status === 401 || response.status === 403
          ? "Your session expired or voice access was denied. Sign in again."
          : errors[payload.detail?.code] ?? "Local transcription failed. Please try again.");
      }
      const payload = await response.json();
      if (generation !== this.generation) return;
      if (typeof payload.text !== "string" || !payload.text.trim()) throw new Error(errors.NO_SPEECH);
      this.options.transcript(payload.text.trim());
      this.update("ready", "Transcription ready. Review or edit your message, then send it.");
    } catch (caught) {
      if (generation !== this.generation) return;
      this.update("error", caught instanceof Error && caught.name === "AbortError"
        ? errors.TRANSCRIPTION_TIMEOUT : caught instanceof TypeError
        ? "Cannot reach the local voice service. Check the connection and try again."
        : caught instanceof Error ? caught.message : "Local transcription failed.");
    } finally { clearTimeout(timer); }
  }
}
