export interface BM25Document<T> {
  id: string;
  text: string;
  value: T;
}

export interface BM25Result<T> {
  id: string;
  score: number;
  value: T;
}

const tokenize = (text: string): string[] =>
  (text.toLowerCase().match(/[a-z0-9]+/g) || []).filter((token) => token.length > 1);

export function bm25Rank<T>(
  documents: BM25Document<T>[],
  query: string,
  limit = 20,
  k1 = 1.5,
  b = 0.75
): BM25Result<T>[] {
  if (!documents.length) return [];

  const tokenizedDocs = documents.map((document) => tokenize(document.text));
  const avgDocLength = tokenizedDocs.reduce((sum, tokens) => sum + tokens.length, 0) / documents.length;
  const queryTerms = [...new Set(tokenize(query))];

  const documentFrequency = new Map<string, number>();
  for (const term of queryTerms) {
    let count = 0;
    for (const tokens of tokenizedDocs) {
      if (tokens.includes(term)) count += 1;
    }
    documentFrequency.set(term, count);
  }

  return documents
    .map((document, index) => {
      const tokens = tokenizedDocs[index];
      const frequencies = new Map<string, number>();
      for (const token of tokens) frequencies.set(token, (frequencies.get(token) || 0) + 1);

      let score = 0;
      for (const term of queryTerms) {
        const tf = frequencies.get(term) || 0;
        if (!tf) continue;

        const df = documentFrequency.get(term) || 0;
        const idf = Math.log(1 + (documents.length - df + 0.5) / (df + 0.5));
        const denominator = tf + k1 * (1 - b + b * (tokens.length / Math.max(avgDocLength, 1)));
        score += idf * ((tf * (k1 + 1)) / denominator);
      }

      return { id: document.id, score, value: document.value };
    })
    .filter((result) => result.score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, limit);
}
