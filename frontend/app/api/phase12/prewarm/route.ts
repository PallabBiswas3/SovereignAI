import { NextResponse } from "next/server";

const ollama = "http://127.0.0.1:11434";
const model = "qwen3:4b-instruct";

export async function POST() {
  const started = performance.now();
  try {
    const response = await fetch(`${ollama}/api/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      cache: "no-store",
      body: JSON.stringify({
        model,
        prompt: "/no_think\nReply with exactly READY.",
        stream: false,
        think: false,
        keep_alive: "15m",
        options: {
          num_ctx: 2048,
          num_predict: 1,
          temperature: 0.2,
          num_thread: 5,
          num_batch: 128,
        },
      }),
    });
    const text = await response.text();
    if (!response.ok) {
      return NextResponse.json(
        { error: `Ollama prewarm returned ${response.status}`, detail: text.slice(0, 2000) },
        { status: 502 },
      );
    }
    const payload = JSON.parse(text) as Record<string, unknown>;
    const psResponse = await fetch(`${ollama}/api/ps`, { cache: "no-store" });
    const psPayload = psResponse.ok ? await psResponse.json() as { models?: Array<Record<string, unknown>> } : {};
    const resident = (psPayload.models ?? []).some((item) => {
      const name = String(item.name ?? item.model ?? "");
      return name === model || name === `${model}:latest`;
    });
    return NextResponse.json({
      model,
      resident,
      wall_seconds: Number(((performance.now() - started) / 1000).toFixed(6)),
      load_duration_seconds: Number((Number(payload.load_duration ?? 0) / 1_000_000_000).toFixed(6)),
      response: String(payload.response ?? "").trim(),
      keep_alive: "15m",
      options: { num_ctx: 2048, num_predict: 1, temperature: 0.2, num_thread: 5, num_batch: 128 },
    });
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Phase 12 prewarm failed" },
      { status: 502 },
    );
  }
}
