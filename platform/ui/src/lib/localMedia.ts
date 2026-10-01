/**
 * Which addresses a picture, a video or a sound may be loaded from, in the editor's nodes.
 *
 * A flow can carry the address of what its nodes show (an Output node's result, an Image Viewer's picture, a saved video), and a
 * flow can come from someone else (a shared one, an imported file). Opening it made the page GET whatever address the author had
 * written, with no run and no Connect: an address on the visitor's own computer or network (`http://127.0.0.1:8188/view?filename=x.mp4`,
 * `http://192.168.1.1/...`), which is a permission prompt in current Chrome and Edge and tells the author what answers there.
 *
 * So a tab that is not linked to OAIY Desktop does not load a picture, video or sound from an address on this computer or its network
 * (shared/capabilities/local.ts says which), and says so on the node instead. OAIY's own window, the desktop shell and a tab that IS linked
 * (Connect, or an engine address of its own) load them as they always did: a flow that ran on their desktop shows what it made there.
 * The page's own origin is not "another address": a local copy of the editor shows its own files.
 */
import { isLocalHostname } from '@oaiy/shared/capabilities/local';
import { looksOnLoad, pageHost } from '@oaiy/shared/capabilities/host';
import { hasSavedDesktopLink } from './desktopLink';

/** What a node shows in place of a picture, video or sound it will not load. */
export const BLOCKED_MEDIA_WORDS = 'Blocked: this address is on your computer or network, and this page is not linked to OAIY Desktop. Connect it in Settings → Services to load it.';

/** Whether the page may load media from `url`. Data and blob addresses, relative ones and the page's own origin are always fine. */
export function mediaUrlBlocked(url: unknown, page: { origin?: string; href?: string } = typeof location !== 'undefined' ? location : {}): boolean {
  if (typeof url !== 'string' || !url.trim()) return false;
  if (looksOnLoad(pageHost()) || hasSavedDesktopLink()) return false;
  let parsed: URL;
  try {
    parsed = new URL(url.trim(), page.href ?? 'http://page.invalid/');
  } catch {
    return false;
  }
  if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') return false;
  if (page.origin && parsed.origin === page.origin) return false;
  return isLocalHostname(parsed.hostname);
}
