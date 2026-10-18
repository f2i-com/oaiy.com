/**
 * What reached an Output node, sorted into what it shows: pictures, video, sound, a 3D model's file, a table (a list of
 * records), an object, or text. The run hands it a node's value as that node made it: a picture as a data: URL, a sound
 * as `{audio, path}` (its URL has no file ending in the web editor: a blob: URL), a 3D model as `{model, path,
 * picture}`, a code step's list of records as the list. The node had read only URLs' endings, so a song showed as the
 * text of its blob: URL and a table as the first record's JSON.
 */
import { mediaUrlBlocked } from '../../../lib/localMedia';
import { isLocalPath } from '../../../lib/mediaSrc';

export type OutputKind = 'image' | 'video' | 'audio' | 'model3d' | 'table' | 'json' | 'text';

export interface OutputItem {
  kind: OutputKind;
  /** The URL or file path that shows it (media), or its text (text). */
  src: string;
  /** A file's name, where it has one. */
  name?: string;
  /** A picture beside it (a 3D model's picture). */
  poster?: string;
  /** A table's records, an object's value. */
  value?: unknown;
}

/** An address the page may load (not one on this computer or network in a tab that is not linked to OAIY). */
export function isSafeUrl(url: string): boolean {
  if (!url || typeof url !== 'string') return false;
  if (mediaUrlBlocked(url)) return false;
  const t = url.trim().toLowerCase();
  return t.startsWith('http://') || t.startsWith('https://') || t.startsWith('data:image/') || t.startsWith('data:video/')
    || t.startsWith('data:audio/') || t.startsWith('blob:');
}

/**
 * A raw base64 picture with no `data:` in front (OpenAI's `b64_json`, passed on as it is), as a data: URL, by its
 * format's first bytes; else null. RIFF ('UklGR') is a picture only with WebP's tag at characters 12-15 ('RUJQ'): WAV
 * and AVI share the prefix.
 */
export function rawBase64ImageDataUrl(value: unknown): string | null {
  if (typeof value !== 'string' || value.length < 64) return null;
  let mime: string | null = null;
  if (value.startsWith('iVBORw0KGgo')) mime = 'image/png';
  else if (value.startsWith('/9j/')) mime = 'image/jpeg';
  else if (value.startsWith('R0lGOD')) mime = 'image/gif';
  else if (value.startsWith('UklGR') && value.slice(12, 16) === 'RUJQ') mime = 'image/webp';
  if (!mime) return null;
  if (!/^[A-Za-z0-9+/=\s]+$/.test(value)) return null;
  return `data:${mime};base64,${value.replace(/\s/g, '')}`;
}

const ENDINGS: Array<[OutputKind, RegExp]> = [
  ['image', /\.(png|jpe?g|gif|webp|bmp|avif)(\?|#|$)/i],
  ['video', /\.(mp4|webm|mov|avi|mkv|m4v)(\?|#|$)/i],
  ['audio', /\.(wav|mp3|ogg|flac|m4a|aac|opus)(\?|#|$)/i],
  ['model3d', /\.(glb|gltf)(\?|#|$)/i],
];

export { isLocalPath };

/** What a URL or a path is by its scheme or its ending; null where neither says (a blob: URL). */
export function kindOfAddress(value: string): OutputKind | null {
  const v = value.trim();
  if (v.startsWith('data:image/')) return 'image';
  if (v.startsWith('data:video/')) return 'video';
  if (v.startsWith('data:audio/')) return 'audio';
  if (v.startsWith('data:model/gltf')) return 'model3d';
  const comfy = /\/view\?filename=([^&]+)/.exec(v);
  const tail = comfy ? decodeURIComponent(comfy[1]) : v;
  for (const [kind, re] of ENDINGS) if (re.test(tail)) return kind;
  return null;
}

/** The last part of a path or URL, as a file's name. */
export function fileName(value: string): string | undefined {
  if (!value || value.startsWith('data:') || value.startsWith('blob:')) return undefined;
  const clean = value.split(/[?#]/)[0];
  const last = clean.split(/[\\/]/).pop();
  return last || undefined;
}

const isRecord = (v: unknown): v is Record<string, unknown> => !!v && typeof v === 'object' && !Array.isArray(v);
const str = (v: unknown): string | undefined => (typeof v === 'string' && v.trim() ? v.trim() : undefined);

/** One value as an item. */
function itemOf(v: unknown): OutputItem | null {
  if (v === null || v === undefined || v === '') return null;
  if (typeof v === 'string') {
    const b64 = rawBase64ImageDataUrl(v);
    if (b64) return { kind: 'image', src: b64 };
    const kind = kindOfAddress(v);
    if (kind && (isSafeUrl(v) || isLocalPath(v) || v.startsWith('data:'))) return { kind, src: v, name: fileName(v) };
    // (text that is JSON: shown as the object it is)
    const t = v.trim();
    if ((t.startsWith('{') && t.endsWith('}')) || (t.startsWith('[') && t.endsWith(']'))) {
      try {
        return itemsOfParsed(JSON.parse(t));
      } catch {
        /* text after all */
      }
    }
    return { kind: 'text', src: v };
  }
  if (typeof v === 'number' || typeof v === 'boolean') return { kind: 'text', src: String(v) };
  if (Array.isArray(v)) return itemsOfParsed(v);
  if (isRecord(v)) {
    const path = str(v.path);
    const name = path ? fileName(path) : undefined;
    // (a node's media value: its kind by its key, whatever its URL says)
    const video = str(v.video);
    if (video) return { kind: 'video', src: video, name: name ?? fileName(video) };
    const audio = str(v.audio);
    if (audio) return { kind: 'audio', src: audio, name: name ?? fileName(audio) };
    const image = str(v.image) ?? str(v.dataUrl);
    if (image) return { kind: 'image', src: rawBase64ImageDataUrl(image) ?? image, name };
    const model = str(v.model) ?? str(v.glb);
    if (model || (path && kindOfAddress(path) === 'model3d')) {
      return { kind: 'model3d', src: str(v.url) ?? model ?? path!, name: name ?? fileName(model ?? ''), poster: str(v.picture) };
    }
    if (path && kindOfAddress(path)) return { kind: kindOfAddress(path)!, src: str(v.url) ?? path, name };
    return { kind: 'json', src: JSON.stringify(v, null, 2), value: v };
  }
  return { kind: 'text', src: String(v) };
}

/** A list read from JSON: a table when it is records, else JSON. */
function itemsOfParsed(v: unknown): OutputItem {
  if (Array.isArray(v) && v.length > 0 && v.every(isRecord) && !v.some((r) => str(r.video) || str(r.audio) || str(r.image))) {
    return { kind: 'table', src: JSON.stringify(v, null, 2), value: v };
  }
  return { kind: 'json', src: JSON.stringify(v, null, 2), value: v };
}

/**
 * What an Output node shows for its value: one item, or several of one kind of media (a gallery). A list of records is
 * one table; a list mixing kinds, or of plain values, is JSON.
 */
export function outputItems(value: unknown): OutputItem[] {
  if (value === null || value === undefined || value === '') return [];
  if (Array.isArray(value)) {
    if (value.length === 0) return [{ kind: 'json', src: '[]', value }];
    const items = value.map(itemOf).filter((i): i is OutputItem => i !== null);
    const media = (k: OutputKind) => k === 'image' || k === 'video' || k === 'audio' || k === 'model3d';
    if (items.length > 0 && items.every((i) => media(i.kind))) return items;
    // (a list of one string: the string; text pieces: shown together)
    if (items.length === 1) return items;
    const whole = itemsOfParsed(value);
    return [whole];
  }
  const one = itemOf(value);
  return one ? [one] : [];
}

/** A table's columns: every record's keys, in the order they first appear. */
export function tableColumns(rows: Array<Record<string, unknown>>): string[] {
  const cols: string[] = [];
  for (const r of rows) for (const k of Object.keys(r)) if (!cols.includes(k)) cols.push(k);
  return cols;
}

/** A cell's value as text. */
export function cellText(v: unknown): string {
  if (v === null || v === undefined) return '';
  if (typeof v === 'string') return v;
  if (typeof v === 'number' || typeof v === 'boolean') return String(v);
  return JSON.stringify(v);
}

/** A line saying what an item is, for the compact card. */
export function summaryOf(items: OutputItem[]): string {
  if (items.length === 0) return '';
  const first = items[0];
  const count = items.length > 1 ? ` (${items.length})` : '';
  switch (first.kind) {
    case 'image': return `Picture${count}`;
    case 'video': return `Video${count}${first.name ? ` · ${first.name}` : ''}`;
    case 'audio': return `Sound${count}${first.name ? ` · ${first.name}` : ''}`;
    case 'model3d': return `3D model${count}${first.name ? ` · ${first.name}` : ''}`;
    case 'table': {
      const rows = first.value as Array<Record<string, unknown>>;
      const cols = tableColumns(rows);
      return `${rows.length} row${rows.length === 1 ? '' : 's'} · ${cols.slice(0, 4).join(', ')}${cols.length > 4 ? ', …' : ''}`;
    }
    case 'json': {
      const v = first.value;
      if (Array.isArray(v)) return `A list of ${v.length}`;
      const keys = isRecord(v) ? Object.keys(v) : [];
      return keys.length ? `{ ${keys.slice(0, 5).join(', ')}${keys.length > 5 ? ', …' : ''} }` : first.src;
    }
    default:
      // (an ellipsis only where the value goes on: "Words: 8" is all of "Words: 8")
      return first.src.length > 120 ? `${first.src.slice(0, 120).trimEnd()}…` : first.src;
  }
}
