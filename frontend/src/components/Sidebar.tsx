const NAV_ITEMS = [
  { icon: 'menu_book',    label: 'Chapters',  active: false, available: false },
  { icon: 'format_quote', label: 'Citations', active: true,  available: true  },
  { icon: 'list_alt',     label: 'Index',     active: false, available: false },
  { icon: 'history',      label: 'History',   active: false, available: false },
] as const;

/**
 * Left navigation sidebar.
 *  - Hidden on mobile
 *  - 64px icon-only strip on md (768px+)
 *  - 240px full labels on lg (1024px+)
 *
 * Chapters / Index / History are structural placeholders — they are rendered
 * as per the Stitch design but marked "coming soon" on hover. Only Citations
 * (the active query view) is functional in v1.
 */
export default function Sidebar() {
  return (
    <nav
      className={[
        'hidden md:flex flex-col',
        'fixed left-0 top-[48px] z-10',
        'h-[calc(100%-128px)]',
        'w-[64px] lg:w-[240px]',
        'bg-surface-container border-r border-outline-variant',
        'overflow-y-auto overflow-x-hidden transition-all duration-300',
      ].join(' ')}
    >
      {/* Library header — visible only on wide sidebar */}
      <div className="hidden lg:block p-md border-b border-outline-variant">
        <h2 className="font-playfair text-display font-bold text-primary truncate">
          Research Library
        </h2>
        <p className="font-sans text-body-sm text-on-surface-variant mt-xs truncate">
          NLP Textbook v3
        </p>
      </div>

      <ul className="flex-1 py-sm">
        {NAV_ITEMS.map(item => (
          <NavItem key={item.label} {...item} />
        ))}
      </ul>
    </nav>
  );
}

function NavItem({
  icon,
  label,
  active,
  available,
}: {
  icon: string;
  label: string;
  active: boolean;
  available: boolean;
}) {
  return (
    <li>
      <a
        href="#"
        onClick={e => e.preventDefault()}
        title={available ? label : `${label} — coming soon`}
        className={[
          'group flex items-center gap-md px-md py-sm transition-colors relative',
          active
            ? 'bg-surface-container-high border-r-2 border-primary text-primary'
            : available
            ? 'text-on-surface-variant hover:bg-surface-container-highest hover:text-on-surface'
            : 'text-on-surface-variant/40 cursor-default',
        ].join(' ')}
      >
        <span className="material-symbols-outlined shrink-0">{icon}</span>
        <span className="font-sans text-body-sm hidden lg:block whitespace-nowrap flex-1">
          {label}
        </span>

        {/* "Coming soon" pill — only on lg sidebar, only for inactive unavailable items */}
        {!available && (
          <span className="hidden lg:inline-flex font-mono text-[9px] tracking-widest uppercase text-on-surface-variant/40 border border-outline-variant/40 rounded px-[4px] py-[1px]">
            soon
          </span>
        )}
      </a>
    </li>
  );
}
