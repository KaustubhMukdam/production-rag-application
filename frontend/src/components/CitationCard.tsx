import type { ChunkSource } from '../types';

interface CitationCardProps {
  source: ChunkSource;
  /** Used for stagger animation delay. */
  index: number;
}

/** Extracts a short chapter code from a chunk_id like "speech-and-language-processing-ch-17_703" */
function chapterCode(chunkId: string): string {
  const match = chunkId.match(/(?:ch-\d+|appendix-[a-z])/i);
  if (match) return match[0];
  // Fallback: last segment before underscore, max 10 chars
  return chunkId.split('_')[0].slice(-10);
}

export default function CitationCard({ source, index }: CitationCardProps) {
  const code = chapterCode(source.chunk_id);
  const isTop = index === 0; // Top result gets the gold left-border highlight

  return (
    <div
      className="group rounded p-md border border-outline-variant bg-background hover:bg-surface-container-highest transition-colors cursor-default"
      style={{
        borderLeft: `2px solid ${isTop ? '#e6c364' : '#4d4637'}`,
        animationDelay: `${index * 80}ms`,
        animation: 'slideIn 0.3s ease-out both',
      }}
    >
      {/* Chapter + page row */}
      <div className="flex items-start justify-between mb-xs">
        <span className="font-mono text-mono-label text-on-surface-variant group-hover:text-primary transition-colors">
          {code}
        </span>
        {source.page_number !== null && (
          <span className="font-mono text-mono-label text-primary shrink-0 ml-sm">
            Page {source.page_number}
          </span>
        )}
      </div>

      {/* Section header */}
      {source.section_header && (
        <h4 className="font-sans text-body-sm text-on-surface line-clamp-2 mb-xs">
          {source.section_header}
        </h4>
      )}

      {/* Source link */}
      {source.source_url && (
        <a
          href={source.source_url}
          target="_blank"
          rel="noreferrer"
          className="inline-flex items-center gap-xs font-sans text-[12px] text-primary opacity-60 hover:opacity-100 transition-opacity mt-xs"
        >
          View source
          <span className="material-symbols-outlined text-[12px]">open_in_new</span>
        </a>
      )}
    </div>
  );
}
