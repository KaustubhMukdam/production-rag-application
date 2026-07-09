import type { HealthData } from '../types';

interface HeaderProps {
  health: HealthData | null;
  onReindex: () => void;
}

export default function Header({ health, onReindex }: HeaderProps) {
  return (
    <header className="shrink-0 h-[48px] flex items-center justify-between px-margin bg-surface-container-low border-b border-outline-variant z-50">
      {/* Brand */}
      <div className="flex items-center gap-sm">
        <h1 className="font-playfair text-heading font-semibold text-primary leading-none">
          SLP3 Ask-My-Docs
        </h1>
        <span className="hidden md:inline font-sans text-body-sm text-on-surface-variant">
          Scholar's Terminal
        </span>
      </div>

      {/* Status + Re-index */}
      <div className="flex items-center gap-lg">
        <HealthPill health={health} />
        <ReindexButton health={health} onReindex={onReindex} />
      </div>
    </header>
  );
}

// ─── Health pill ──────────────────────────────────────────────────────────────

function HealthPill({ health }: { health: HealthData | null }) {
  if (!health) {
    return (
      <div className="flex items-center gap-xs px-sm py-xs rounded border border-outline-variant bg-surface-container">
        <span className="w-2 h-2 rounded-full bg-error" />
        <span className="font-mono text-mono-label text-on-surface-variant">Unavailable</span>
      </div>
    );
  }

  const dotClass = health.indexing
    ? 'w-2 h-2 rounded-full bg-yellow-400 animate-pulse'
    : 'w-2 h-2 rounded-full bg-green-400';

  return (
    <div className="flex items-center gap-xs px-sm py-xs rounded border border-outline-variant bg-surface-container">
      <span className={dotClass} />
      <span className="font-mono text-mono-label text-on-surface-variant">
        {health.indexing
          ? 'Indexing…'
          : `${health.indexed_chunks.toLocaleString()} chunks`}
      </span>
    </div>
  );
}

// ─── Re-index button ──────────────────────────────────────────────────────────

function ReindexButton({
  health,
  onReindex,
}: {
  health: HealthData | null;
  onReindex: () => void;
}) {
  const busy = health?.indexing ?? false;

  return (
    <button
      onClick={onReindex}
      disabled={busy}
      title="Re-index all PDFs"
      className="flex items-center gap-xs px-sm py-xs font-mono text-mono-label text-on-surface-variant border border-transparent rounded hover:border-outline-variant hover:text-primary transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
    >
      <span
        className="material-symbols-outlined text-[14px]"
        style={{ animation: busy ? 'spin 1s linear infinite' : 'none' }}
      >
        {busy ? 'autorenew' : 'refresh'}
      </span>
      {busy ? 'Indexing…' : 'Re-index'}
    </button>
  );
}
