const DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434";
const DEFAULT_GENERATION_MODEL = "qwen3:4b-instruct";
const DEFAULT_EMBEDDING_MODEL = "nomic-embed-text";
const EMBEDDING_DIMENSION = 768;

type GenerateOptions = {
  json?: boolean;
  timeoutMs?: number;
  maxTokens?: number;
};

function ollamaUrl(): string {
  const configured = process.env.OLLAMA_URL || DEFAULT_OLLAMA_URL;
  const parsed = new URL(configured);
  const hostname = parsed.hostname.toLowerCase();
  const isLocal = hostname === "localhost" || hostname === "127.0.0.1" || hostname === "::1" || !hostname.includes(".");
  if (!isLocal || !["http:", "https:"].includes(parsed.protocol)) {
    throw new Error(`OLLAMA_URL must be a loopback or internal service URL, received ${configured}`);
  }
  return configured.replace(/\/$/, "");
}

async function postOllama<T>(path: string, body: Record<string, unknown>, timeoutMs: number): Promise<T> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(`${ollamaUrl()}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: controller.signal,
    });
    if (!response.ok) {
      const detail = await response.text();
      throw new Error(`Ollama ${path} returned ${response.status}: ${detail.slice(0, 500)}`);
    }
    return (await response.json()) as T;
  } catch (error: any) {
    if (error?.name === "AbortError") {
      throw new Error(`Ollama ${path} timed out after ${timeoutMs / 1000}s`);
    }
    throw error;
  } finally {
    clearTimeout(timer);
  }
}

export async function generateLocalText(prompt: string, options: GenerateOptions = {}): Promise<string> {
  const payload = await postOllama<{ response?: string }>(
    "/api/generate",
    {
      model: process.env.OLLAMA_GENERATION_MODEL || DEFAULT_GENERATION_MODEL,
      prompt,
      stream: false,
      think: false,
      keep_alive: process.env.OLLAMA_KEEP_ALIVE || "15m",
      ...(options.json ? { format: "json" } : {}),
      options: {
        temperature: 0,
        num_ctx: Number(process.env.OLLAMA_CONTEXT_SIZE || 8192),
        ...(options.maxTokens ? { num_predict: options.maxTokens } : {}),
      },
    },
    options.timeoutMs || Number(process.env.OLLAMA_GENERATION_TIMEOUT_MS || 120_000)
  );
  if (typeof payload.response !== "string" || !payload.response.trim()) {
    throw new Error("Ollama returned an empty generation response");
  }
  return payload.response.trim();
}

export async function generateLocalJson<T>(
  prompt: string,
  timeoutMs?: number,
  maxTokens?: number
): Promise<T> {
  const raw = await generateLocalText(prompt, { json: true, timeoutMs, maxTokens });
  const cleaned = raw.replace(/```json|```/g, "").trim();
  return JSON.parse(cleaned) as T;
}

function normalizeEmbedding(values: number[]): number[] {
  if (values.length < EMBEDDING_DIMENSION) {
    throw new Error(`Ollama embedding returned ${values.length} dimensions; expected at least ${EMBEDDING_DIMENSION}`);
  }
  const truncated = values.slice(0, EMBEDDING_DIMENSION);
  const norm = Math.sqrt(truncated.reduce((sum, value) => sum + value * value, 0));
  if (!Number.isFinite(norm) || norm === 0) {
    throw new Error("Ollama embedding returned an invalid zero-norm vector");
  }
  return truncated.map((value) => value / norm);
}

export async function generateLocalEmbeddings(texts: string[]): Promise<number[][]> {
  if (!texts.length) return [];
  const payload = await postOllama<{ embeddings?: number[][] }>(
    "/api/embed",
    {
      model: process.env.OLLAMA_EMBEDDING_MODEL || DEFAULT_EMBEDDING_MODEL,
      input: texts,
      truncate: true,
    },
    Number(process.env.OLLAMA_EMBEDDING_TIMEOUT_MS || 60_000)
  );
  if (!payload.embeddings || payload.embeddings.length !== texts.length) {
    throw new Error(`Ollama returned ${payload.embeddings?.length || 0} embeddings for ${texts.length} inputs`);
  }
  return payload.embeddings.map(normalizeEmbedding);
}

export async function generateLocalEmbedding(text: string): Promise<number[]> {
  return (await generateLocalEmbeddings([text]))[0];
}

export function localModelConfiguration() {
  return {
    provider: "ollama",
    endpoint: ollamaUrl(),
    generationModel: process.env.OLLAMA_GENERATION_MODEL || DEFAULT_GENERATION_MODEL,
    embeddingModel: process.env.OLLAMA_EMBEDDING_MODEL || DEFAULT_EMBEDDING_MODEL,
    embeddingDimension: EMBEDDING_DIMENSION,
    denseRetrievalEnabled: process.env.GRAPHRAG_DENSE_RETRIEVAL_ENABLED?.toLowerCase() === "true",
  };
}
