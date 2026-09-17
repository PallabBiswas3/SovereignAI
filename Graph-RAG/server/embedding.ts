import { generateLocalEmbedding } from "./localModel";

export async function generateEmbedding(text: string): Promise<number[]> {
  return generateLocalEmbedding(text);
}
