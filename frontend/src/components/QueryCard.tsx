import type { QueryResult } from '../types';
import { useTypewriter } from '../hooks/useTypewriter';

interface QueryCardProps {
  query: QueryResult;
  /** True only for the most-recent card — triggers the typewriter animation. */
  animate: boolean;
}

export default function QueryCard({ query, animate }: QueryCardProps) {
  if (query.gated) return <GatedCard query={query} />;
  return <NormalCard query={query} animate={animate} />;
}

// ─── Normal answer card ───────────────────────────────────────────────────────

function NormalCard({ query, animate }: { query: QueryResult; animate: boolean }) {
  const { displayed, complete } = useTypewriter(query.answer, animate);

  return (
    <article className="bg-surface-container border border-outline-variant rounded-lg p-lg animate-fade-up">
      {/* Query header */}
      <div className="flex items-center justify-between mb-sm">
        <span className="font-sans text-mono-label tracking-[0.08em] text-on-surface-variant uppercase">
          User Query
        </span>
        <span className="font-mono text-mono-label text-on-surface-variant">
          {query.timestamp}
        </span>
      </div>

      {/* Question */}
      <p className="font-sans text-body-lg text-on-surface border-l-2 border-outline-variant pl-md mb-lg leading-relaxed">
        {query.question}
      </p>

      {/* System response */}
      <div className="border-t border-outline-variant pt-md mt-md relative">
        <div className="absolute -top-[10px] left-md bg-surface-container px-xs flex items-center gap-xs font-mono text-mono-label text-primary">
          <span className="material-symbols-outlined text-[13px]">psychology</span>
          System Response
        </div>

        {/* Answer text — typewriter or full */}
        <p className="font-sans text-body-md text-on-surface mt-sm leading-[1.75]">
          {displayed}
          {animate && !complete && <span className="cursor-blink" />}
        </p>

        {/* Footer */}
        <div className="mt-lg pt-sm border-t border-outline-variant flex items-center justify-between">
          <div className="flex items-center gap-md">
            <span className="flex items-center gap-xs font-mono text-mono-label text-on-surface-variant">
              <span className="material-symbols-outlined text-[13px]">precision_manufacturing</span>
              {query.provider}
            </span>
            <span className="flex items-center gap-xs font-mono text-mono-label text-on-surface-variant">
              <span className="material-symbols-outlined text-[13px]">library_books</span>
              {query.sources.length} source{query.sources.length !== 1 ? 's' : ''}
            </span>
          </div>

          {/* Supported badge */}
          <SupportedBadge supported={query.supported} />
        </div>
      </div>
    </article>
  );
}

function SupportedBadge({ supported }: { supported: boolean }) {
  return (
    <div
      className={[
        'flex items-center gap-xs px-sm py-xs rounded border font-mono text-mono-label',
        supported
          ? 'bg-surface-variant border-outline-variant text-[#4ade80]'
          : 'bg-surface-variant border-outline-variant text-on-surface-variant',
      ].join(' ')}
    >
      <span className="material-symbols-outlined text-[13px]">
        {supported ? 'check_circle' : 'help'}
      </span>
      {supported ? 'supported' : 'unverified'}
    </div>
  );
}

// ─── Gated (out-of-scope) card ────────────────────────────────────────────────

function GatedCard({ query }: { query: QueryResult }) {
  return (
    <article
      className="bg-surface-container border border-outline-variant rounded-lg p-lg animate-fade-up gate-pulse"
      style={{
        borderLeft: '2px solid #D4742A',
        backgroundColor: 'rgba(212, 116, 42, 0.025)',
      }}
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-sm">
        <div className="flex items-center gap-xs">
          <span
            className="material-symbols-outlined text-[14px]"
            style={{ color: '#D4742A' }}
          >
            lock
          </span>
          <span className="font-sans text-mono-label tracking-[0.08em] text-on-surface-variant uppercase">
            User Query
          </span>
        </div>
        <div className="flex items-center gap-sm">
          <span className="font-mono text-mono-label text-on-surface-variant">
            {query.timestamp}
          </span>
          <span
            className="font-mono text-mono-label px-sm py-xs rounded border"
            style={{
              color: '#D4742A',
              borderColor: '#D4742A',
              backgroundColor: 'rgba(212,116,42,0.1)',
            }}
          >
            ⚠ Out of Scope
          </span>
        </div>
      </div>

      {/* Question — uppercase for out-of-scope (matches Stitch design) */}
      <p
        className="font-sans text-body-lg uppercase tracking-wide text-on-surface border-l-2 pl-md mb-lg leading-relaxed"
        style={{ borderLeftColor: '#D4742A' }}
      >
        {query.question}
      </p>

      {/* Response */}
      <div className="border-t border-outline-variant pt-md mt-md relative">
        <div
          className="absolute -top-[10px] left-md bg-surface-container px-xs flex items-center gap-xs font-mono text-mono-label"
          style={{ color: '#D4742A' }}
        >
          <span className="material-symbols-outlined text-[13px]">psychology</span>
          System Response
        </div>

        <p className="font-sans text-body-md text-on-surface-variant mt-sm leading-[1.75]">
          {query.answer}
        </p>

        {/* Show top chunk for transparency */}
        {query.sources.length > 0 && (
          <p className="font-mono text-[11px] text-on-surface-variant mt-md opacity-60">
            Top chunk: {query.sources[0].chunk_id}
            {query.sources[0].page_number !== null && ` · Page ${query.sources[0].page_number}`}
            {query.sources[0].section_header && ` · ${query.sources[0].section_header}`}
            {' '}(rerank gate fired)
          </p>
        )}
      </div>
    </article>
  );
}
