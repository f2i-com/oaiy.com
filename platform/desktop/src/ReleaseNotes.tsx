import type { ReactNode } from 'react';
import { openExternal } from './api';

/**
 * A release's notes as GitHub writes them (Markdown), shown as text a person reads: headings, paragraphs, lists,
 * bold, code, and links, which open in the browser. Only the few forms a release's notes use are read; the rest is
 * shown as it is written. Nothing in the notes is ever read as HTML (each piece is a React text node), and only an
 * https address becomes a link.
 */
export default function ReleaseNotes({ text }: { text: string }) {
  return <div className="update-notes">{blocks(text)}</div>;
}

type Block = { kind: 'heading'; level: number; text: string } | { kind: 'list'; ordered: boolean; items: string[] } | { kind: 'para'; text: string };

/** The notes' blocks: a heading, a list (its items), or a paragraph (its lines joined). */
export function parseBlocks(text: string): Block[] {
  const out: Block[] = [];
  let para: string[] = [];
  let list: { ordered: boolean; items: string[] } | null = null;
  const flush = () => {
    if (para.length) out.push({ kind: 'para', text: para.join(' ') });
    if (list) out.push({ kind: 'list', ...list });
    para = [];
    list = null;
  };
  for (const raw of text.replace(/\r\n?/g, '\n').split('\n')) {
    const line = raw.trim();
    // (a blank line, a rule, or a comment GitHub's template leaves: none of them shown)
    if (!line || /^([-*_]\s*){3,}$/.test(line) || /^<!--.*-->$/.test(line)) {
      flush();
      continue;
    }
    const heading = /^(#{1,6})\s+(.*?)\s*#*$/.exec(line);
    if (heading) {
      flush();
      out.push({ kind: 'heading', level: heading[1].length, text: heading[2] });
      continue;
    }
    const item = /^(?:([-*+])|\d+[.)])\s+(.*)$/.exec(line);
    if (item) {
      const ordered = !item[1];
      if (para.length || (list && list.ordered !== ordered)) flush();
      list ??= { ordered, items: [] };
      list.items.push(item[2]);
      continue;
    }
    // (a line under a list item that is not one of its own: the item's text goes on)
    if (list && /^\s/.test(raw)) {
      list.items[list.items.length - 1] += ` ${line}`;
      continue;
    }
    if (list) flush();
    para.push(line);
  }
  flush();
  return out;
}

function blocks(text: string): ReactNode[] {
  return parseBlocks(text).map((b, i) => {
    if (b.kind === 'heading') {
      return (
        <p key={i} className={`update-notes-heading${b.level <= 2 ? ' is-top' : ''}`}>
          {inline(b.text)}
        </p>
      );
    }
    if (b.kind === 'list') {
      const items = b.items.map((t, j) => <li key={j}>{inline(t)}</li>);
      return b.ordered ? <ol key={i}>{items}</ol> : <ul key={i}>{items}</ul>;
    }
    return <p key={i}>{inline(b.text)}</p>;
  });
}

/** A link's address, when it is one to follow: https only (no javascript:, file: or plain http). */
export function safeHref(url: string): string | null {
  try {
    const u = new URL(url);
    return u.protocol === 'https:' ? u.href : null;
  } catch {
    return null;
  }
}

// `[text](address)` (an address may hold one pair of brackets), **bold**, `code`, or a bare https address.
const INLINE = /\[([^\]]+)\]\(((?:[^()\s]|\([^()\s]*\))+)\)|\*\*([^*]+)\*\*|`([^`]+)`|(https:\/\/[^\s<>()]+)/g;

/** A line's text with its links, bold and code: React nodes, never HTML. */
export function inline(text: string): ReactNode[] {
  const out: ReactNode[] = [];
  let at = 0;
  let n = 0;
  for (const m of text.matchAll(INLINE)) {
    if (m.index > at) out.push(text.slice(at, m.index));
    const [whole, label, address, bold, code, bare] = m;
    if (label !== undefined) {
      const href = safeHref(address);
      out.push(href ? <Link key={n++} href={href}>{label}</Link> : label);
    } else if (bold !== undefined) {
      out.push(<strong key={n++}>{inline(bold)}</strong>);
    } else if (code !== undefined) {
      out.push(<code key={n++}>{code}</code>);
    } else {
      // (the sentence's own full stop or bracket after an address is not part of it)
      const address = bare.replace(/[.,;:!?'"]+$/, '');
      const href = safeHref(address);
      out.push(href ? <Link key={n++} href={href}>{address}</Link> : address);
      if (address.length < whole.length) out.push(whole.slice(address.length));
    }
    at = m.index + whole.length;
  }
  if (at < text.length) out.push(text.slice(at));
  return out;
}

/** A link in the notes: opened in the browser, never in OAIY's own window. */
function Link({ href, children }: { href: string; children: ReactNode }) {
  return (
    <a
      href={href}
      target="_blank"
      rel="noreferrer noopener"
      onClick={(e) => {
        e.preventDefault();
        openExternal(href);
      }}
    >
      {children}
    </a>
  );
}
