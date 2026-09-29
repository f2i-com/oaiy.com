import { useEffect, useMemo, useState } from 'react';
import { inOaiyWindow } from '../lib/oaiyWindow';
import { currentDevice, downloadPlanFor, refinedDevice } from '../lib/downloadsEnv';

/**
 * "Download OAIY Desktop", for the device the page is on.
 *
 * One component for the landing page, the desktop page and the editor, so the same visitor is
 * offered the same file everywhere: the button is the download that suits the device (the Windows
 * installer, the Linux AppImage), "other downloads" is the rest, and a device OAIY Desktop is not
 * built for (a Mac, a phone) is told so and pointed at the web app. What suits the device is
 * lib/downloads.ts; it is worked out in the page from what the browser already knows, and the
 * links were made when the site was built. Nothing here makes a request or looks for a desktop.
 *
 * Variants:
 *   hero     a secondary button of the marketing pages' size, with its notes under it
 *   primary  the same as the main action of its page
 *   compact  a line for the editor's sidebar, under the engine card (nothing for a device with no download)
 *   card     a paragraph and a small button, for the editor's Settings
 *
 * Where there is no download for the device the button is `fallback` instead: by default a link to the
 * desktop page, and on the desktop page itself (where that would go nowhere) the web app.
 *
 * Never shown in OAIY's own window: it is the desktop already.
 */
export type DownloadVariant = 'hero' | 'primary' | 'compact' | 'card';

const ABOUT = { label: 'About OAIY Desktop', href: 'desktop.html' };

export default function DownloadDesktop({ variant = 'hero', fallback = ABOUT }: { variant?: DownloadVariant; fallback?: { label: string; href: string } }) {
  const [device, setDevice] = useState(currentDevice);
  // The user agent cannot tell ARM Linux or 32-bit Windows from x64; the browser can, when asked (no network), so the
  // first draw is the guess and this is the answer.
  useEffect(() => {
    let live = true;
    void refinedDevice(device).then((refined) => {
      if (live) setDevice(refined);
    });
    return () => {
      live = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps -- asked once, from the guess the page was drawn with
  }, []);
  const plan = useMemo(() => downloadPlanFor(device), [device]);
  if (inOaiyWindow()) return null;

  if (variant === 'compact') {
    if (!plan.primary) return null;
    return (
      <p className="oaiy-engine-get">
        <a href={plan.primary.href} data-download={plan.device.os}>Get OAIY Desktop</a> for the Agent and the services on your computer.
      </p>
    );
  }

  const more = plan.others.length > 0 && (
    <details className="oaiy-download-more">
      <summary>Other downloads</summary>
      <ul>
        {plan.others.map((other) => (
          <li key={other.href}><a href={other.href}>{other.label}</a></li>
        ))}
        <li><a href={plan.allDownloads} target="_blank" rel="noopener noreferrer">All downloads on GitHub</a></li>
      </ul>
    </details>
  );
  const all = plan.others.length === 0 && (
    <a className="oaiy-download-all" href={plan.allDownloads} target="_blank" rel="noopener noreferrer">All downloads on GitHub</a>
  );
  const notes = (
    <>
      {plan.note && <p className="oaiy-download-note">{plan.note}</p>}
      {plan.caption && <p className="oaiy-download-note">{plan.caption}</p>}
    </>
  );

  if (variant === 'card') {
    return (
      <div className="oaiy-download oaiy-download-card">
        {plan.primary ? (
          <a className="btn btn-primary btn-sm" href={plan.primary.href} data-download={plan.device.os}>
            <DownloadIcon /> {plan.primary.label}
          </a>
        ) : null}
        {notes}
        {more}
        {all}
      </div>
    );
  }

  const size = 'btn btn-lg';
  return (
    <div className="oaiy-download">
      {plan.primary ? (
        <a className={`${size} ${variant === 'primary' ? 'btn-primary' : 'btn-secondary'}`} href={plan.primary.href} data-download={plan.device.os}>
          <DownloadIcon /> {plan.primary.label}
        </a>
      ) : (
        <a className={`${size} btn-secondary`} href={fallback.href} data-download={plan.device.os}>{fallback.label}</a>
      )}
      {notes}
      {more}
      {all}
    </div>
  );
}

function DownloadIcon() {
  return (
    <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 3v12m0 0l-4-4m4 4l4-4M5 21h14" />
    </svg>
  );
}
