/**
 * Whether this page is OAIY's own window (OAIY Desktop shows the flow editor in it and injects
 * `window.__OAIY_DESKTOP__` before the page runs), as opposed to a browser tab.
 *
 * Anything that is for a browser tab only asks here: the service worker and the install button
 * (pwa/), and the offer to download OAIY Desktop (components/DownloadDesktop.tsx). Any value counts,
 * not only a complete one: a window that has the object is the desktop's, whatever it carries.
 */
export function inOaiyWindow(): boolean {
  return typeof window !== 'undefined' && Boolean((window as Window & { __OAIY_DESKTOP__?: unknown }).__OAIY_DESKTOP__);
}
