import { escapeHtml } from './dom';

/**
 * Enough Markdown for chat replies: fenced code (with its language and a Copy
 * button), inline code, bold, italics, headings, lists, tables, quotes, rules
 * and links. Everything is escaped first, so a reply can never inject markup:
 * the only tags are the ones written here, and links go only to http(s).
 * The Copy buttons are wired by the chat (one listener for the log), not by
 * anything in this HTML.
 */
export function renderMarkdown(source: string): string {
  const parts = source.split(/```/);
  let html = '';
  parts.forEach((part, i) => {
    if (i % 2 === 1) {
      const newline = part.indexOf('\n');
      const lang = newline > 0 ? part.slice(0, newline).trim() : '';
      const code = newline >= 0 ? part.slice(newline + 1) : part;
      html += codeBlock(lang, code.replace(/\n$/, ''), i === parts.length - 1);
      return;
    }
    html += blocks(escapeHtml(part).split('\n'));
  });
  return html;
}

/** A fenced block: its language and a Copy button over the code (`open`: still being written). */
function codeBlock(lang: string, code: string, open: boolean): string {
  const label = lang.replace(/[^\w+#.-]/g, '').slice(0, 24);
  return `<div class="code-block${open ? ' writing' : ''}"><div class="code-head"><span class="code-lang">${escapeHtml(label || 'code')}</span><button type="button" class="code-copy" title="Copy the code" aria-label="Copy the code">Copy</button></div><pre class="code"${label ? ` data-lang="${escapeHtml(label)}"` : ''}><code>${escapeHtml(code)}</code></pre></div>`;
}

/** A table's separator row: `|---|:--:|`. */
const SEPARATOR = /^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$/;

function cells(row: string): string[] {
  let line = row.trim();
  if (line.startsWith('|')) line = line.slice(1);
  if (line.endsWith('|') && !line.endsWith('\\|')) line = line.slice(0, -1);
  return line.split(/(?<!\\)\|/).map((c) => c.trim());
}

/** Escaped lines as blocks: paragraphs, lists, headings, tables, quotes and rules. */
function blocks(lines: string[]): string {
  let html = '';
  let list: 'ul' | 'ol' | null = null;
  const closeList = () => {
    if (list) html += `</${list}>`;
    list = null;
  };
  for (let n = 0; n < lines.length; n++) {
    const raw = lines[n];
    // A table: a row of cells, then the separator, then rows while they have a pipe.
    if (raw.includes('|') && SEPARATOR.test(lines[n + 1] ?? '') && !SEPARATOR.test(raw)) {
      closeList();
      const head = cells(raw);
      const align = cells(lines[n + 1]).map((c) => (c.startsWith(':') && c.endsWith(':') ? 'center' : c.endsWith(':') ? 'right' : ''));
      const attr = (i: number) => (align[i] ? ` style="text-align:${align[i]}"` : '');
      let table = `<div class="table-wrap"><table><thead><tr>${head.map((c, i) => `<th${attr(i)}>${inline(c)}</th>`).join('')}</tr></thead><tbody>`;
      n += 2;
      while (n < lines.length && lines[n].includes('|') && lines[n].trim()) {
        table += `<tr>${cells(lines[n]).map((c, i) => `<td${attr(i)}>${inline(c)}</td>`).join('')}</tr>`;
        n++;
      }
      n--;
      html += `${table}</tbody></table></div>`;
      continue;
    }
    const bullet = /^\s*[-*+] (.*)$/.exec(raw);
    const numbered = /^\s*(\d+)[.)] (.*)$/.exec(raw);
    const kind = bullet ? 'ul' : numbered ? 'ol' : null;
    if (list && kind !== list) closeList();
    if (kind && !list) {
      html += kind === 'ol' && numbered && numbered[1] !== '1' ? `<ol start="${Number(numbered[1])}">` : `<${kind}>`;
      list = kind;
    }
    if (kind) {
      const item = bullet?.[1] ?? numbered![2];
      const task = /^\[([ xX])\] (.*)$/.exec(item);
      html += task ? `<li class="task${task[1] === ' ' ? '' : ' done'}"><span class="task-box" aria-hidden="true"></span>${inline(task[2])}</li>` : `<li>${inline(item)}</li>`;
      continue;
    }
    const heading = /^(#{1,4}) (.*)$/.exec(raw);
    if (heading) html += `<h${heading[1].length + 2}>${inline(heading[2])}</h${heading[1].length + 2}>`;
    else if (/^\s*([-*_])(\s*\1){2,}\s*$/.test(raw)) html += '<hr>';
    else if (/^&gt; ?/.test(raw)) html += `<blockquote>${inline(raw.replace(/^&gt; ?/, ''))}</blockquote>`;
    else if (raw.trim()) html += `<p>${inline(raw)}</p>`;
  }
  closeList();
  return html;
}

/** Inline code is kept as it is; bold, italics and links apply to the rest. */
function inline(text: string): string {
  return text
    .split(/(`[^`]+`)/)
    .map((piece, i) => (i % 2 === 1 ? `<code>${piece.slice(1, -1)}</code>` : emphasis(piece)))
    .join('');
}

function emphasis(text: string): string {
  return text
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
    .replace(/(^|\W)\*([^*\s][^*]*)\*(?=\W|$)/g, '$1<em>$2</em>')
    .replace(/\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
}
