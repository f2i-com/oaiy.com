/**
 * `@file` reference grammar, mention extraction, and composer autocomplete
 * popup (`@path` / `@"path with spaces"` + `/` slash commands), ported from
 * DeepSeek Harness (`packages/context/file-reference/src/grammar.ts`).
 */
import type { Vfs } from '../../vfs/vfs';
import { clear, h } from '../dom';

export interface ActiveAtToken {
  prefix: string;
  query: string;
  quoted: boolean;
  start: number;
  end: number;
}

export interface FileReferenceCandidate {
  path: string;
  kind: 'file' | 'directory';
}

export interface SlashCommandItem {
  command: string;
  usage: string;
  description: string;
}

export const SLASH_COMMANDS: readonly SlashCommandItem[] = [
  { command: '/plan', usage: '/plan [on|off]', description: 'Toggle collaborative Plan Mode before editing files' },
  { command: '/compact', usage: '/compact', description: 'Prune older tool outputs & summarize earlier turns' },
  { command: '/softn', usage: '/softn [new|check|preview|apps]', description: 'Create, check, preview, or manage SoftN apps' },
  { command: '/internet', usage: '/internet [on|off|allowlist|status]', description: 'Inspect or configure the sandbox network gate' },
  { command: '/clear', usage: '/clear', description: 'Start a fresh conversation (project files stay)' },
  { command: '/help', usage: '/help', description: 'Show available slash commands' },
];

/**
 * Extract an active `@path` or `@"path with spaces` token immediately before
 * `cursorCol`. Ignores `@` inside an email or identifier.
 */
export function activeAtToken(text: string, cursorCol: number): ActiveAtToken | undefined {
  const beforeCursor = text.slice(0, cursorCol);
  const quoted = /(?:^|\s)(@"([^"\r\n]*))$/u.exec(beforeCursor);
  if (quoted?.[1] !== undefined && quoted[2] !== undefined) {
    return {
      prefix: quoted[1],
      query: quoted[2],
      quoted: true,
      start: cursorCol - quoted[1].length,
      end: cursorCol,
    };
  }
  const plain = /(?:^|\s)(@([^\s@]*))$/u.exec(beforeCursor);
  if (plain?.[1] === undefined || plain[2] === undefined) return undefined;
  return {
    prefix: plain[1],
    query: plain[2],
    quoted: false,
    start: cursorCol - plain[1].length,
    end: cursorCol,
  };
}

/**
 * Format a selected file or directory candidate as prompt text (`@path` or
 * `@"path with spaces"`). Quoted directories keep the closing quote open so the
 * user can keep descending.
 */
export function formatFileMention(candidate: FileReferenceCandidate, preserveQuote: boolean): string | undefined {
  const path = candidate.kind === 'directory' ? `${candidate.path}/` : candidate.path;
  if (/[\u0000-\u001f\u007f-\u009f"]/u.test(path)) return undefined;
  const quoted = preserveQuote || /\s/u.test(path);
  if (!quoted) return `@${path}`;
  if (candidate.kind === 'directory') return `@"${path}`;
  return `@"${path}"`;
}

/** Extract unique VFS file paths explicitly mentioned via `@path` or `@"path"` in a prompt. */
export function extractFileMentions(text: string): string[] {
  const seen = new Set<string>();
  const re = /(?:^|\s)@(?:"([^"\r\n]+)"|([^\s@,;:!?()[\]{}]+))/gu;
  let m: RegExpExecArray | null;
  while ((m = re.exec(text)) !== null) {
    const raw = (m[1] ?? m[2] ?? '').replace(/^\.?\//, '').replace(/\/+$/, '');
    if (raw) seen.add(raw);
  }
  return [...seen];
}

/** Fuzzy subsequence score (higher is better; -1 means no match). */
function fuzzyScore(path: string, query: string): number {
  if (!query) return 100 - Math.min(path.length, 90);
  const p = path.toLowerCase();
  const q = query.toLowerCase();
  const base = p.split('/').pop() ?? p;
  if (base === q) return 1000;
  if (base.startsWith(q)) return 800 - base.length;
  if (p.startsWith(q)) return 700 - p.length;
  const idx = p.indexOf(q);
  if (idx >= 0) return 500 - idx - p.length * 0.1;
  // Ordered subsequence match
  let qi = 0;
  let consecutive = 0;
  let score = 100;
  for (let pi = 0; pi < p.length && qi < q.length; pi++) {
    if (p[pi] === q[qi]) {
      qi++;
      consecutive++;
      score += 8 * consecutive;
    } else {
      consecutive = 0;
    }
  }
  return qi === q.length ? score - p.length * 0.2 : -1;
}

/** Search VFS entries matching `@query` (up to `limit` candidates). */
export function searchMentionCandidates(vfs: Vfs, query: string, limit = 10): FileReferenceCandidate[] {
  const walked = vfs.walk('/', { limit: 2500 });
  const scored: Array<{ item: FileReferenceCandidate; score: number }> = [];
  for (const entry of walked.entries) {
    const clean = entry.path.replace(/^\/+/, '');
    if (!clean || clean.startsWith('.oaiy/')) continue;
    const score = fuzzyScore(clean, query);
    if (score >= 0) {
      scored.push({
        item: { path: clean, kind: entry.type === 'dir' ? 'directory' : 'file' },
        score: score + (entry.type === 'file' ? 5 : 0),
      });
    }
  }
  scored.sort((a, b) => b.score - a.score || a.item.path.localeCompare(b.item.path));
  return scored.slice(0, limit).map((s) => s.item);
}

export interface ComposerAutocomplete {
  /** Handle textarea keydown; returns `true` if the popup consumed the key. */
  handleKeyDown: (e: KeyboardEvent) => boolean;
  /** Recompute popup state from the current textarea text & cursor position. */
  refresh: () => void;
  /** Dismiss popup cleanly. */
  close: () => void;
  readonly open: boolean;
}

/**
 * Attach an interactive `@file` + `/` slash-command autocomplete popup to a
 * composer `<textarea>`.
 */
export function attachComposerAutocomplete(
  input: HTMLTextAreaElement,
  anchor: HTMLElement,
  getVfs: () => Vfs | null,
): ComposerAutocomplete {
  const popup = h('div', { class: 'mention-popup', hidden: true, role: 'listbox' });
  anchor.append(popup);

  type Mode =
    | { kind: 'none' }
    | { kind: 'file'; token: ActiveAtToken; items: FileReferenceCandidate[] }
    | { kind: 'slash'; items: SlashCommandItem[] };

  let mode: Mode = { kind: 'none' };
  let activeIndex = 0;

  const close = () => {
    mode = { kind: 'none' };
    activeIndex = 0;
    popup.hidden = true;
    clear(popup);
  };

  const applySelection = (idx: number) => {
    if (mode.kind === 'file') {
      const item = mode.items[idx];
      if (!item) return;
      const formatted = formatFileMention(item, mode.token.quoted);
      if (!formatted) return;
      const suffix = item.kind === 'directory' ? '' : ' ';
      const before = input.value.slice(0, mode.token.start);
      const after = input.value.slice(mode.token.end);
      input.value = `${before}${formatted}${suffix}${after}`;
      const pos = before.length + formatted.length + suffix.length;
      input.setSelectionRange(pos, pos);
      input.dispatchEvent(new Event('input', { bubbles: true }));
      if (item.kind === 'directory') refresh();
      else close();
    } else if (mode.kind === 'slash') {
      const item = mode.items[idx];
      if (!item) return;
      const needsArg = item.command === '/softn' || item.command === '/internet';
      input.value = `${item.command}${needsArg ? ' ' : ''}`;
      input.setSelectionRange(input.value.length, input.value.length);
      input.dispatchEvent(new Event('input', { bubbles: true }));
      close();
    }
  };

  const render = () => {
    clear(popup);
    if (mode.kind === 'none') {
      popup.hidden = true;
      return;
    }
    const items = mode.kind === 'file' ? mode.items : mode.items;
    if (!items.length) {
      popup.hidden = true;
      return;
    }
    if (activeIndex >= items.length) activeIndex = 0;
    popup.hidden = false;

    if (mode.kind === 'file') {
      popup.append(h('div', { class: 'mention-head' }, 'Files in project'));
      mode.items.forEach((item, i) => {
        const row = h(
          'button',
          {
            type: 'button',
            class: `mention-item${i === activeIndex ? ' active' : ''}`,
            role: 'option',
          },
          h('span', { class: `mention-kind ${item.kind}` }, item.kind === 'directory' ? 'DIR' : 'FILE'),
          h('span', { class: 'mention-path' }, item.kind === 'directory' ? `${item.path}/` : item.path),
        );
        row.onmousedown = (e) => {
          e.preventDefault();
          applySelection(i);
        };
        popup.append(row);
      });
    } else {
      popup.append(h('div', { class: 'mention-head' }, 'Commands'));
      mode.items.forEach((item, i) => {
        const row = h(
          'button',
          {
            type: 'button',
            class: `mention-item${i === activeIndex ? ' active' : ''}`,
            role: 'option',
          },
          h('span', { class: 'mention-cmd' }, item.usage),
          h('span', { class: 'mention-desc' }, item.description),
        );
        row.onmousedown = (e) => {
          e.preventDefault();
          applySelection(i);
        };
        popup.append(row);
      });
    }
  };

  const refresh = () => {
    const val = input.value;
    const cursor = input.selectionStart ?? val.length;
    // 1. Slash command autocomplete at start of prompt (single line, no spaces yet).
    if (val.startsWith('/') && !/\s/.test(val.slice(0, cursor)) && !val.includes('\n')) {
      const q = val.slice(0, cursor).toLowerCase();
      const matched = SLASH_COMMANDS.filter((c) => c.command.startsWith(q));
      if (matched.length) {
        mode = { kind: 'slash', items: matched };
        activeIndex = 0;
        render();
        return;
      }
    }
    // 2. `@file` mention autocomplete.
    const tok = activeAtToken(val, cursor);
    if (tok) {
      const vfs = getVfs();
      if (vfs) {
        const items = searchMentionCandidates(vfs, tok.query, 8);
        if (items.length) {
          mode = { kind: 'file', token: tok, items };
          activeIndex = 0;
          render();
          return;
        }
      }
    }
    close();
  };

  const handleKeyDown = (e: KeyboardEvent): boolean => {
    if (mode.kind === 'none') return false;
    const len = mode.items.length;
    if (!len) return false;
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      activeIndex = (activeIndex + 1) % len;
      render();
      return true;
    }
    if (e.key === 'ArrowUp') {
      e.preventDefault();
      activeIndex = (activeIndex - 1 + len) % len;
      render();
      return true;
    }
    if (e.key === 'Tab' || (e.key === 'Enter' && !e.shiftKey)) {
      e.preventDefault();
      applySelection(activeIndex);
      return true;
    }
    if (e.key === 'Escape') {
      e.preventDefault();
      close();
      return true;
    }
    return false;
  };

  input.addEventListener('blur', () => {
    // Allow mousedown selection before closing.
    setTimeout(close, 120);
  });

  return {
    handleKeyDown,
    refresh,
    close,
    get open() {
      return mode.kind !== 'none';
    },
  };
}
