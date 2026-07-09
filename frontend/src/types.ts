// ─── API response shapes ────────────────────────────────────────────────────

export interface ChunkSource {
  chunk_id: string;
  page_number: number | null;
  section_header: string | null;
  source_url: string | null;
}

export interface QueryResponse {
  question: string;
  answer: string;
  provider: string;
  supported: boolean;
  gated: boolean;
  source_chunk_ids: string[];
  sources: ChunkSource[];
}

export interface HealthData {
  status: string;
  indexed_chunks: number;
  indexing: boolean;
}

// ─── App-level state ─────────────────────────────────────────────────────────

/** A single query + response pair stored in the chat history. */
export interface QueryResult {
  id: string;
  question: string;
  answer: string;
  provider: string;
  supported: boolean;
  gated: boolean;
  source_chunk_ids: string[];
  sources: ChunkSource[];
  timestamp: string; // e.g. "10:42 AM"
}
