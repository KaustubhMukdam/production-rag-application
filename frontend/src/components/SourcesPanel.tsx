import type { ChunkSource } from '../types';
import CitationCard from './CitationCard';

interface SourcesPanelProps {
  sources: ChunkSource[];
  /** Total source count from source_chunk_ids (may differ from sources.length) */
  count: number;
}

/**
 * Right-hand sources panel — fixed, 320px wide, visible on xl (1280px+).
 * On smaller screens the sources are accessible from the answer card footer.
 */
export default function SourcesPanel({ sources, count }: SourcesPanelProps) {
  return (
    <aside
      className={[
        'hidden xl:flex flex-col',
        'fixed right-0 top-[48px] bottom-[80px] z-10',
        'w-[320px]',
        'bg-surface-container border-l border-outline-variant',
        'overflow-y-auto',
      ].join(' ')}
    >
      {/* Panel header */}
      <div className="sticky top-0 z-20 px-md py-sm border-b border-outline-variant bg-surface-container">
        <h3 className="font-sans text-body-sm font-medium tracking-[0.08em] text-on-surface-variant uppercase">
          {count > 0 ? `Sources (${count})` : 'Sources'}
        </h3>
      </div>

      {/* Citation cards */}
      <div className="p-sm flex flex-col gap-sm">
        {sources.length === 0 ? (
          <p className="font-sans text-body-sm text-on-surface-variant p-md text-center opacity-50 leading-relaxed">
            Sources will appear here after a query.
          </p>
        ) : (
          sources.map((source, i) => (
            <CitationCard key={source.chunk_id} source={source} index={i} />
          ))
        )}
      </div>
    </aside>
  );
}
