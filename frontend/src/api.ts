import type { HealthData, QueryResponse } from './types';

const API_BASE = (import.meta.env.VITE_API_URL as string | undefined) ?? 'http://localhost:8000';

// ─── /query ──────────────────────────────────────────────────────────────────

export async function queryAPI(question: string): Promise<QueryResponse> {
  const res = await fetch(`${API_BASE}/query`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question }),
  });

  if (res.status === 429) {
    throw new Error('Rate limit reached — please wait a moment and try again.');
  }
  if (res.status === 503) {
    throw new Error('Index not ready. Start the server with documents loaded.');
  }
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error((body as { detail?: string }).detail ?? `HTTP ${res.status}`);
  }

  return res.json() as Promise<QueryResponse>;
}

// ─── /health ─────────────────────────────────────────────────────────────────

export async function fetchHealth(): Promise<HealthData> {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw new Error('Health check failed');
  return res.json() as Promise<HealthData>;
}

// ─── /index ──────────────────────────────────────────────────────────────────

export async function triggerReindex(): Promise<void> {
  const res = await fetch(`${API_BASE}/index`, { method: 'POST' });
  if (!res.ok) throw new Error('Re-index request failed');
}
