export type TaskEvent = { type: string; timestamp: string; payload: Record<string, unknown> };

// An interrupted authenticated stream must stop, including CONNECTING errors.
// The task may still run on the server; never infer completion from a disconnect.
export function openTaskStream(url: string, events: string[], receive: (item: TaskEvent) => void,
  failed: (message: string) => void, Source: typeof EventSource = EventSource) {
  const stream = new Source(url, { withCredentials: true });
  let closed = false;
  const close = () => { closed = true; stream.close(); };
  const fail = (message: string) => { if (!closed) { close(); failed(message); } };
  for (const type of new Set([...events, "access_revoked"])) {
    stream.addEventListener(type, ((event: MessageEvent) => {
      if (closed) return;
      try {
        const item = JSON.parse(event.data) as TaskEvent;
        if (item.type !== type || !item.payload || typeof item.payload !== "object") throw Error("Invalid event");
        receive(item);
        if (["task_completed", "task_failed", "task_cancelled", "access_revoked"].includes(type)) close();
      } catch { fail("The task stream returned an invalid event. The server task may still be running."); }
    }) as EventListener);
  }
  stream.onerror = () => fail("The live connection was interrupted. The server task may still be running; check it before retrying.");
  return close;
}
