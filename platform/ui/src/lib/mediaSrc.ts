/**
 * A file's address a page can load into an <img>, <audio> or <video>. The editor (in OAIY's window or a tab) keeps the
 * files a run makes, and the ones picked by hand, in the page: the shim's `get_media_url` gives a blob: URL for one by
 * its path. The desktop shell has its media server (`pathToMediaUrl`). A URL is already one.
 */
import { useEffect, useState } from 'react';
import { pathToMediaUrl } from 'oaiy-core';

/** A file path on this computer (`/…`, `C:\…`, `C:/…`). */
export function isLocalPath(value: string): boolean {
  return /^([A-Za-z]:[\\/]|\/)/.test(value);
}

/** `src` as an address the page can load (a URL as it is). */
export async function loadableSrc(src: string): Promise<string> {
  if (!isLocalPath(src)) return src;
  const w = window as unknown as { __TAURI_INTERNALS__?: { invoke?: (cmd: string, args?: unknown) => Promise<unknown> } };
  try {
    const got = await w.__TAURI_INTERNALS__?.invoke?.('get_media_url', { filePath: src });
    if (typeof got === 'string' && got && got !== src) return got;
  } catch {
    /* the media server's address, below */
  }
  return pathToMediaUrl(src);
}

/** `src` made loadable (`loadableSrc`), kept current as it changes; undefined while it is worked out, or with none. */
export function useLoadableSrc(src: string | undefined): string | undefined {
  const [out, setOut] = useState<string | undefined>(undefined);
  useEffect(() => {
    let live = true;
    setOut(undefined);
    if (src) void loadableSrc(src).then((u) => { if (live) setOut(u); });
    return () => { live = false; };
  }, [src]);
  return out;
}
