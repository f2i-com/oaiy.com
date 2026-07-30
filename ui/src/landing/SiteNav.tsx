import { useTheme } from '../contexts/ThemeContext';

/**
 * SiteNav — ONE navigation bar shared by every marketing page.
 *
 * Previously each page shipped its own `Nav` whose links were that page's
 * section anchors, so the top bar changed shape when you moved between `/` and
 * `/desktop.html` and there was no way to get back without the wordmark. Now
 * the top bar is always the same set of site destinations with the current page
 * marked, and a page's own sections live in the optional sub-nav strip below it.
 *
 * `page` says which top-level destination is current; `sections` is the
 * in-page jump list for that page (omit it and the strip isn't rendered).
 */

export type SitePage = 'overview' | 'desktop';

/**
 * Site-level destinations. `page` is which page the target lives on, so a link
 * can resolve to a bare hash when you're already there and a full cross-page
 * href when you aren't. That distinction matters: BOTH pages have `#how` and
 * `#capabilities` sections, so from desktop.html "How it works" must go to
 * `index.html#how`, not scroll to the desktop page's own how-section.
 *
 * `root` marks the two entries that ARE a page (rather than a section of one) —
 * only those get the active treatment.
 */
const DESTINATIONS: {
  page: SitePage;
  label: string;
  href: string;
  hash?: string;
  root?: boolean;
}[] = [
  { page: 'overview', label: 'Overview', href: 'index.html', hash: '#top', root: true },
  { page: 'overview', label: 'Capabilities', href: 'index.html#capabilities', hash: '#capabilities' },
  { page: 'overview', label: 'How it works', href: 'index.html#how', hash: '#how' },
  { page: 'desktop', label: 'Desktop app', href: 'desktop.html', hash: '#top', root: true },
];

export interface NavSection {
  id: string;
  label: string;
}

export default function SiteNav({
  page,
  sections,
}: {
  page: SitePage;
  sections?: NavSection[];
}) {
  return (
    <header className="site-nav">
      <div className="site-nav-bar">
        <a href="index.html" className="flex items-center select-none" aria-label="OAIY home">
          <span className="lp-wordmark">OAIY</span>
          <span className="lp-tagline ml-3 hidden sm:inline">Orchestrate AI Yourself</span>
        </a>

        <nav className="site-nav-links" aria-label="Site">
          {DESTINATIONS.map((d) => {
            // On the page a link points at, use the bare hash so it scrolls
            // instead of reloading; from elsewhere use the full cross-page href.
            const here = d.page === page;
            const current = here && d.root === true;
            return (
              <a
                key={d.label}
                href={here && d.hash ? d.hash : d.href}
                className={current ? 'active' : undefined}
                aria-current={current ? 'page' : undefined}
              >
                {d.label}
              </a>
            );
          })}
          <ThemeToggle />
          <a href="app.html" className="btn btn-primary btn-sm ml-1">
            Open the app
            <ArrowRight />
          </a>
        </nav>
      </div>

      {sections && sections.length > 0 && (
        <div className="site-subnav">
          <nav aria-label="On this page">
            {sections.map((s) => (
              <a key={s.id} href={`#${s.id}`}>
                {s.label}
              </a>
            ))}
          </nav>
        </div>
      )}
    </header>
  );
}

function ThemeToggle() {
  const { resolvedTheme, setTheme } = useTheme();
  const isDark = resolvedTheme === 'dark';
  return (
    <button
      type="button"
      onClick={() => setTheme(isDark ? 'light' : 'dark')}
      className="oaiy-icon-btn"
      aria-label={isDark ? 'Switch to Paper Circuit (light)' : 'Switch to Prism Lab (dark)'}
      title={isDark ? 'Paper Circuit' : 'Prism Lab'}
    >
      {isDark ? (
        <svg className="h-[18px] w-[18px]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <circle cx="12" cy="12" r="4" strokeWidth="2" />
          <path
            strokeLinecap="round"
            strokeWidth="2"
            d="M12 3v2m0 14v2M5.6 5.6l1.4 1.4m10 10l1.4 1.4M3 12h2m14 0h2M5.6 18.4l1.4-1.4m10-10l1.4-1.4"
          />
        </svg>
      ) : (
        <svg className="h-[18px] w-[18px]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth="2"
            d="M21 12.8A9 9 0 1111.2 3a7 7 0 009.8 9.8z"
          />
        </svg>
      )}
    </button>
  );
}

function ArrowRight() {
  return (
    <svg className="h-3.5 w-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 12h14m-6-6l6 6-6 6" />
    </svg>
  );
}
