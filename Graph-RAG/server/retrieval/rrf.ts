export interface RankedItem<T> {
  id: string;
  score: number;
  value: T;
}

export interface FusedItem<T> extends RankedItem<T> {
  sources: string[];
}

export function reciprocalRankFusion<T>(
  rankings: Array<{ name: string; items: RankedItem<T>[] }>,
  limit = 20,
  k = 60
): FusedItem<T>[] {
  const fused = new Map<string, { score: number; value: T; sources: Set<string> }>();

  for (const ranking of rankings) {
    ranking.items.forEach((item, index) => {
      const current = fused.get(item.id) || {
        score: 0,
        value: item.value,
        sources: new Set<string>(),
      };

      current.score += 1 / (k + index + 1);
      current.sources.add(ranking.name);
      fused.set(item.id, current);
    });
  }

  return Array.from(fused.entries())
    .map(([id, item]) => ({
      id,
      score: item.score,
      value: item.value,
      sources: Array.from(item.sources),
    }))
    .sort((a, b) => b.score - a.score)
    .slice(0, limit);
}
