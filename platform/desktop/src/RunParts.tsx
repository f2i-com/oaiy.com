import { useEffect, useLayoutEffect, useRef, useState, type CSSProperties } from 'react';
import { Check, Copy, ImageOff } from 'lucide-react';
import { network } from './api';

/**
 * The parts Run history and Flows → On other computers both show a run with: when and how long, who asked for it,
 * what it was asked with and what it answered (its pictures shown, its words readable, the rest as JSON to copy).
 */

/** A run's error as said: the runtime wraps one already beginning "Error:" in another ("Error: Error: Image Gen: …"). */
export function errorText(message: string): string {
  return message.replace(/^(\s*Error:\s*)+/i, '');
}

/** Copy a text, the button saying so for a moment. Icon-only unless it has a `text` to show. */
export function CopyButton({ value, label, text }: { value: string; label: string; text?: string }) {
  const [copied, setCopied] = useState(false);
  const timer = useRef<number | undefined>(undefined);
  useEffect(() => () => window.clearTimeout(timer.current), []);
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(value);
      setCopied(true);
      window.clearTimeout(timer.current);
      timer.current = window.setTimeout(() => setCopied(false), 1600);
    } catch {
      // (no clipboard in a webview without focus: the text stays there to select)
    }
  };
  return (
    <button
      type="button"
      className={text ? 'btn-tiny' : 'btn-tiny btn-icon-tiny'}
      onClick={() => void copy()}
      aria-label={copied ? `${label}: copied` : label}
      title={copied ? 'Copied' : label}
    >
      {copied ? <Check size={12} aria-hidden /> : <Copy size={12} aria-hidden />}
      {text && <span>{copied ? 'Copied' : text}</span>}
    </button>
  );
}

/** How long, in words that fit a column: 270 ms, 3.5 s, 2 min 5 s, 1 h 4 min. */
export function durationText(ms: number): string {
  if (!Number.isFinite(ms) || ms < 0) return '';
  if (ms < 1000) return `${Math.round(ms)} ms`;
  if (ms < 10_000) return `${(ms / 1000).toFixed(1)} s`;
  const s = Math.round(ms / 1000);
  if (s < 60) return `${s} s`;
  if (s < 3600) return `${Math.floor(s / 60)} min${s % 60 ? ` ${s % 60} s` : ''}`;
  const m = Math.round(s / 60);
  return `${Math.floor(m / 60)} h${m % 60 ? ` ${m % 60} min` : ''}`;
}

const sameDay = (a: Date, b: Date) => a.getFullYear() === b.getFullYear() && a.getMonth() === b.getMonth() && a.getDate() === b.getDate();

/** When, as a list reads it: "just now", "4 min ago", a time today, "Yesterday 09:12", "12 Oct 09:12". */
export function whenText(at: number, now = Date.now()): string {
  if (!Number.isFinite(at)) return '—';
  const d = new Date(at);
  const ago = now - at;
  if (ago >= -5_000 && ago < 60_000) return 'just now';
  if (ago >= 0 && ago < 3_600_000) return `${Math.floor(ago / 60_000)} min ago`;
  const time = d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  const today = new Date(now);
  if (sameDay(d, today)) return time;
  if (sameDay(d, new Date(now - 86_400_000))) return `Yesterday ${time}`;
  const date = d.toLocaleDateString([], { day: 'numeric', month: 'short', ...(d.getFullYear() !== today.getFullYear() ? { year: 'numeric' } : {}) });
  return `${date} ${time}`;
}

/** A timestamp in full (its tooltip), seconds included. */
export function fullWhen(at: string | number | undefined | null): string {
  if (at === undefined || at === null) return '—';
  const d = new Date(at);
  return Number.isNaN(d.getTime()) ? String(at) : d.toLocaleString();
}

/**
 * Who asked for a run, by the product it was asked under: the Agent, a trigger, a paired computer (by its name, when
 * it is known here), an app. A product not known here is said as it is.
 */
export function callerText(product: string, computers?: ReadonlyMap<string, string>): string {
  if (product === 'oaiy-app' || product === 'oaiy-agent') return 'The Agent';
  if (product === 'oaiy-desktop') return 'You, from Run history';
  if (product === 'oaiy-flows') return 'An app, through oaiy-flows';
  if (product === 'formlogic') return 'FormLogic';
  if (product.startsWith('oaiy-trigger:')) return `A trigger, on ${product.slice('oaiy-trigger:'.length)}’s events`;
  if (product.startsWith('oaiy-network:')) {
    const name = computers?.get(product.slice('oaiy-network:'.length));
    return name ? `${name}, over OAIY Network` : 'A paired computer, over OAIY Network';
  }
  return product;
}

const PICTURE_PATH = /\.(?:png|jpe?g|webp|gif)$/i;
const PICTURE_DATA = /^data:image\/(?:png|jpe?g|webp|gif);base64,/i;
const OWN_HOSTS = new Set(['127.0.0.1', 'localhost', '[::1]']);

/**
 * A picture this desktop serves, by an address a run names: on this computer, under `/api/` (its engines' files, a
 * worker's through this desktop), at whatever port the desktop had when the run was made (it is read at the path on
 * the desktop now). Never an address elsewhere: that would be fetched from the internet as the page opens. A picture
 * given inline (`data:image/…`) is one too.
 */
export function isPicture(value: string): boolean {
  if (PICTURE_DATA.test(value)) return value.length <= 12_000_000;
  if (value.length > 2048) return false;
  try {
    const u = new URL(value);
    return (u.protocol === 'http:' || u.protocol === 'https:') && OWN_HOSTS.has(u.hostname) && u.pathname.startsWith('/api/') && PICTURE_PATH.test(u.pathname);
  } catch {
    return false;
  }
}

/** The pictures a value names, anywhere in it (at most `max`, each once). */
export function picturesIn(value: unknown, max = 12): string[] {
  const found = new Set<string>();
  const visit = (v: unknown, depth: number) => {
    if (found.size >= max || depth > 8) return;
    if (typeof v === 'string') {
      if (isPicture(v)) found.add(v);
    } else if (Array.isArray(v)) {
      for (const x of v) visit(x, depth + 1);
    } else if (v && typeof v === 'object') {
      for (const x of Object.values(v)) visit(x, depth + 1);
    }
  };
  visit(value, 0);
  return [...found];
}

/** A value as words: a text as it is, a number or yes/no as said, a list of texts one a line; anything else is not words. */
function wordsOf(value: unknown, pictures: string[]): string | null {
  if (typeof value === 'string') return pictures.includes(value) || !value.trim() ? null : value;
  if (typeof value === 'number' || typeof value === 'boolean') return String(value);
  if (Array.isArray(value) && value.length > 0 && value.every((x) => typeof x === 'string' && !pictures.includes(x))) return value.join('\n');
  return null;
}

/** A long text held to a few lines, the rest a click away. */
export function LongText({ text, lines = 6 }: { text: string; lines?: number }) {
  const [all, setAll] = useState(false);
  const [clipped, setClipped] = useState(false);
  const ref = useRef<HTMLParagraphElement>(null);
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el || all) return;
    const measure = () => setClipped(el.scrollHeight > el.clientHeight + 1);
    measure();
    if (typeof ResizeObserver === 'undefined') return;
    const watch = new ResizeObserver(measure);
    watch.observe(el);
    return () => watch.disconnect();
  }, [text, all]);
  return (
    <>
      <p ref={ref} className={all ? 'run-text' : 'run-text clamped'} style={{ '--lines': lines } as CSSProperties}>
        {text}
      </p>
      {(clipped || all) && (
        <button type="button" className="run-text-more" aria-expanded={all} onClick={() => setAll(!all)}>
          {all ? 'Show less' : 'Show all'}
        </button>
      )}
    </>
  );
}

/** A value as JSON, folded away, with a button to copy it. */
export function JsonBlock({ value, summary }: { value: unknown; summary: string }) {
  let json: string;
  try {
    json = JSON.stringify(value, null, 2) ?? String(value);
  } catch {
    json = String(value);
  }
  return (
    <details className="run-json">
      <summary>{summary}</summary>
      <div className="run-json-body">
        <CopyButton value={json} label={`Copy ${summary.toLowerCase()} as JSON`} text="Copy JSON" />
        <pre>{json.length > 40_000 ? `${json.slice(0, 40_000)}\n…` : json}</pre>
      </div>
    </details>
  );
}

/**
 * Named values as fields: each one's words, its pictures, or (neither) its JSON in a line. A run's inputs, or each
 * text field of an answer.
 */
export function Fields({ values }: { values: Record<string, unknown> }) {
  const entries = Object.entries(values);
  if (entries.length === 0) return <p className="form-hint">Nothing.</p>;
  return (
    <dl className="run-fields">
      {entries.map(([name, value]) => {
        const shots = picturesIn(value);
        const words = wordsOf(value, shots);
        let compact = '';
        if (words === null && shots.length === 0) {
          try {
            compact = JSON.stringify(value) ?? '';
          } catch {
            compact = String(value);
          }
        }
        return (
          <div className="run-field" key={name}>
            <dt>{name}</dt>
            <dd>
              {shots.length > 0 && <Pictures addresses={shots} />}
              {words !== null && <LongText text={words} lines={4} />}
              {words === null && shots.length === 0 && (value === '' || value === null || value === undefined ? <span className="run-empty">empty</span> : <code className="run-inline-json">{compact.length > 400 ? `${compact.slice(0, 400)}…` : compact}</code>)}
            </dd>
          </div>
        );
      })}
    </dl>
  );
}

/** What a flow answered: its pictures, its words (a text, or each text field of an answer), and all of it as JSON. */
export function RunAnswer({ answer }: { answer: unknown }) {
  const shots = picturesIn(answer);
  const whole = wordsOf(answer, shots);
  const fields: [string, string][] =
    answer && typeof answer === 'object' && !Array.isArray(answer)
      ? Object.entries(answer as Record<string, unknown>).flatMap(([k, v]): [string, string][] => {
          const w = wordsOf(v, shots);
          return w === null ? [] : [[k, w]];
        })
      : [];
  const structured = answer !== null && typeof answer === 'object';
  if (answer === undefined || answer === null || (typeof answer === 'string' && !answer.trim())) {
    return <p className="form-hint">It answered nothing.</p>;
  }
  return (
    <div className="flow-run-answer">
      {shots.length > 0 && <Pictures addresses={shots} />}
      {whole !== null && (
        <div className="flow-run-text">
          <LongText text={whole} />
        </div>
      )}
      {fields.length > 0 && (
        <dl className="run-fields">
          {fields.map(([k, v]) => (
            <div key={k} className="run-field">
              <dt>{k}</dt>
              <dd>
                <LongText text={v} lines={4} />
              </dd>
            </div>
          ))}
        </dl>
      )}
      {structured && (shots.length > 0 || fields.length > 0) && <JsonBlock value={answer} summary="Everything it answered" />}
      {structured && shots.length === 0 && fields.length === 0 && (
        <div className="run-json-body">
          <CopyButton value={JSON.stringify(answer, null, 2)} label="Copy the answer as JSON" text="Copy JSON" />
          <pre className="flow-run-json">{JSON.stringify(answer, null, 2).slice(0, 4000)}</pre>
        </div>
      )}
    </div>
  );
}

/** Pictures in a row; one clicked shows large across the row, clicked again small. */
export function Pictures({ addresses }: { addresses: string[] }) {
  const [large, setLarge] = useState<string | null>(null);
  return (
    <div className="flow-run-pictures">
      {addresses.map((a) => (
        <Picture key={a} address={a} large={large === a} onToggle={() => setLarge(large === a ? null : a)} />
      ))}
    </div>
  );
}

function Picture({ address, large, onToggle }: { address: string; large: boolean; onToggle: () => void }) {
  const inline = PICTURE_DATA.test(address);
  const [src, setSrc] = useState<string | null>(inline ? address : null);
  const [failed, setFailed] = useState(false);
  useEffect(() => {
    if (inline) return;
    let url: string | null = null;
    let gone = false;
    setFailed(false);
    network
      .picture(address)
      .then((blob) => {
        if (gone) return;
        url = URL.createObjectURL(blob);
        setSrc(url);
      })
      .catch(() => {
        if (!gone) setFailed(true);
      });
    return () => {
      gone = true;
      if (url) URL.revokeObjectURL(url);
    };
  }, [address, inline]);
  if (failed) {
    return (
      <div className="flow-run-picture is-missing" title={address}>
        <ImageOff size={16} aria-hidden />
        <span>Not there any more</span>
      </div>
    );
  }
  if (!src) return <div className="flow-run-picture is-loading" aria-label="Loading a picture" />;
  return (
    <button type="button" className={`flow-run-picture-button${large ? ' is-large' : ''}`} onClick={onToggle} aria-label={large ? 'Show the picture small' : 'Show the picture large'} title={large ? 'Show it small' : 'Show it large'}>
      <img className="flow-run-picture" src={src} alt="A picture the flow made" />
    </button>
  );
}
