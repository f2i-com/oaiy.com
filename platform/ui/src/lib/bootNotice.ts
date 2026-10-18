/**
 * What happened while the page was starting, before anything could show it (a shared link opened, or not): kept
 * for the page to say once it is up (`takeBootNotice`).
 */
export const BOOT_NOTICE_KEY = 'oaiy-boot-notice';

export interface BootNotice {
  kind: 'success' | 'error' | 'info' | 'warning';
  text: string;
}

/** The notice left while starting, once: read and forgotten. */
export function takeBootNotice(): BootNotice | null {
  try {
    const raw = sessionStorage.getItem(BOOT_NOTICE_KEY);
    if (!raw) return null;
    sessionStorage.removeItem(BOOT_NOTICE_KEY);
    const v = JSON.parse(raw) as Partial<BootNotice>;
    return typeof v.text === 'string' ? { kind: v.kind ?? 'info', text: v.text } : null;
  } catch {
    return null;
  }
}
