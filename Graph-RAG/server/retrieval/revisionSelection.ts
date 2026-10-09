/** Keep an accessible current revision beside any retrieved superseded revision. */
export function includeCurrentRevisionCompanions<T extends {
  id: string;
  metadata: Record<string, unknown>;
}>(ranked: T[], authorizedCandidates: T[], limit: number): T[] {
  const selected = ranked.slice(0, limit);
  const byId = new Map(authorizedCandidates.map((item) => [item.id, item]));
  const candidates = [...byId.values()];

  for (let index = 0; index < selected.length; index += 1) {
    const item = selected[index];
    if (item.metadata?.document_status !== "superseded") continue;
    const family = item.metadata.document_family;
    if (typeof family !== "string" || !family.trim()) continue;
    if (selected.some((other) => other.metadata?.document_family === family
      && other.metadata?.document_status === "current")) continue;

    const currentMatches = candidates.filter((candidate) =>
      candidate.metadata?.document_family === family
      && candidate.metadata?.document_status === "current"
      && !selected.some((other) => other.id === candidate.id)
    );
    // Multiple records marked current are an unresolved source-authority
    // conflict. Do not silently choose one by database order.
    if (currentMatches.length !== 1) continue;
    const current = currentMatches[0];
    // Preserve ranked order at the front. If the historical hit occupies the
    // final context slot, replace it with its current counterpart.
    if (index === limit - 1) selected[index] = current;
    else selected.splice(index + 1, 0, current);
    selected.length = Math.min(selected.length, limit);
    index += 1;
  }
  return selected;
}
