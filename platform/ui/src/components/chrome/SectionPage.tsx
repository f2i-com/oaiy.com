/**
 * A section of the editor shown as a page in its main area: Data, Queue,
 * Packages and Settings all use this, so they share one layout with the
 * dashboard's own pages.
 *
 *   [sub-nav] | kicker                                  [actions]
 *             | Title
 *             | One line on what the page is for.
 *             | ─────────────────────────────────────────────────
 *             | cards…
 *
 * The sub-nav is there only for a section with pages of its own (Settings).
 * `fill` lays the content out to the page's full height instead of scrolling
 * it, for a page that scrolls inside itself (Data's table).
 */
import type { ReactNode } from 'react';
import type { LucideIcon } from 'lucide-react';

export interface SubNavItem {
  id: string;
  label: string;
  icon: LucideIcon;
  /** A count or a word beside the label (e.g. how many services run). */
  badge?: ReactNode;
  title?: string;
}

export function SubNav({
  label,
  items,
  active,
  onSelect,
}: {
  label: string;
  items: SubNavItem[];
  active: string;
  onSelect: (id: string) => void;
}) {
  return (
    <nav className="oaiy-subnav" aria-label={label}>
      {items.map((item) => {
        const Icon = item.icon;
        const on = item.id === active;
        return (
          <button
            type="button"
            key={item.id}
            className={on ? 'active' : undefined}
            aria-current={on ? 'page' : undefined}
            title={item.title ?? item.label}
            onClick={() => onSelect(item.id)}
          >
            <Icon size={15} />
            <span>{item.label}</span>
            {item.badge !== undefined && item.badge !== null && <em>{item.badge}</em>}
          </button>
        );
      })}
    </nav>
  );
}

export default function SectionPage({
  kicker,
  title,
  description,
  actions,
  nav,
  fill = false,
  children,
  testId,
}: {
  kicker: string;
  title: ReactNode;
  description?: ReactNode;
  actions?: ReactNode;
  nav?: ReactNode;
  fill?: boolean;
  children: ReactNode;
  testId?: string;
}) {
  return (
    <div className={nav ? 'oaiy-page has-nav' : 'oaiy-page'} data-testid={testId}>
      {nav}
      <div className={fill ? 'oaiy-page-main fill' : 'oaiy-page-main'}>
        <header className="oaiy-page-head">
          <div className="oaiy-page-titles">
            <span className="oaiy-kicker">{kicker}</span>
            <h2>{title}</h2>
            {description && <p>{description}</p>}
          </div>
          {actions && <div className="oaiy-page-actions">{actions}</div>}
        </header>
        <div className="oaiy-page-body">{children}</div>
      </div>
    </div>
  );
}

/** A bordered card with a dashed-kicker title, the dashboard's settings card. */
export function Card({
  title,
  count,
  actions,
  children,
  className = '',
  flush = false,
}: {
  title?: ReactNode;
  count?: ReactNode;
  actions?: ReactNode;
  children?: ReactNode;
  className?: string;
  /** No padding around the body (a list that runs to the card's edges). */
  flush?: boolean;
}) {
  return (
    <section className={`oaiy-card${flush ? ' flush' : ''} ${className}`}>
      {(title || actions) && (
        <div className="oaiy-card-head">
          {title && (
            <h3 className="oaiy-card-title">
              {title}
              {count !== undefined && count !== null && <span className="oaiy-card-count">{count}</span>}
            </h3>
          )}
          {actions && <div className="oaiy-card-actions">{actions}</div>}
        </div>
      )}
      {children}
    </section>
  );
}

/** Nothing here yet: an icon, a line, and what to do about it. */
export function EmptyState({ icon, title, children }: { icon?: ReactNode; title: ReactNode; children?: ReactNode }) {
  return (
    <div className="oaiy-empty">
      {icon}
      <p className="oaiy-empty-title">{title}</p>
      {children && <p className="oaiy-empty-text">{children}</p>}
    </div>
  );
}
