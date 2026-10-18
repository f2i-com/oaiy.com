import { escapeHtml } from './dom';

/**
 * Enough Markdown for chat replies: fenced code (with its language and a Copy
 * button), inline code, bold, italics, headings, lists, tables, quotes, rules
 * and links. Everything is escaped first, so a reply can never inject markup:
 * the only tags are the ones written here, and links go only to http(s).
 * The Copy buttons are wired by the chat (one listener for the log), not by
 * anything in this HTML.
 */
export function renderMarkdown(source: string, finished = false): string {
  const parts = source.split(/```/);
  let html = '';
  parts.forEach((part, i) => {
    if (i % 2 === 1) {
      const newline = part.indexOf('\n');
      const lang = newline > 0 ? part.slice(0, newline).trim() : '';
      const code = newline >= 0 ? part.slice(newline + 1) : part;
      // (a block left open is still being written, unless the reply is `finished`: stopped part way, it is whole as it is)
      html += codeBlock(lang, code.replace(/\n$/, ''), !finished && i === parts.length - 1);
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

/** Escaped lines as blocks: paragraphs, lists (nested ones too), headings, tables, quotes and rules. */
function blocks(lines: string[]): string {
  let html = '';
  // The lists open, outermost first, each with how far its items are indented; the last item of each is open too
  // (a list nested in it goes inside it).
  const lists: Array<{ kind: 'ul' | 'ol'; indent: number }> = [];
  const closeLists = (deeperThan = -1) => {
    while (lists.length && lists[lists.length - 1].indent > deeperThan) html += `</li></${lists.pop()!.kind}>`;
  };
  const closeList = () => closeLists();
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
    if (kind) {
      const indent = raw.length - raw.trimStart().length;
      // Back out of the lists indented deeper than this item; two spaces past the list it is in, it starts one inside
      // that list's last item; otherwise it is the next item (of a new list, when it changes from bullets to numbers).
      closeLists(indent);
      const top = lists[lists.length - 1];
      if (top && indent < top.indent + 2 && top.kind === kind) html += '</li>';
      else {
        if (top && indent < top.indent + 2) html += `</li></${lists.pop()!.kind}>`;
        html += kind === 'ol' && numbered && numbered[1] !== '1' ? `<ol start="${Number(numbered[1])}">` : `<${kind}>`;
        lists.push({ kind, indent });
      }
      const item = bullet?.[1] ?? numbered![2];
      const task = /^\[([ xX])\] (.*)$/.exec(item);
      // (a task's box is said, done or not, to a screen reader as to the eye)
      html += task ? `<li class="task${task[1] === ' ' ? '' : ' done'}"><span class="task-box" role="img" aria-label="${task[1] === ' ' ? 'to do' : 'done'}"></span>${inline(task[2])}` : `<li>${inline(item)}`;
      continue;
    }
    // An indented line under an item goes on with that item, on a line of its own.
    if (lists.length && /^\s{2,}\S/.test(raw)) {
      html += `<br>${inline(raw.trim())}`;
      continue;
    }
    closeList();
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
  // Links written as links first, kept aside, then an address written bare becomes one too.
  const links: string[] = [];
  const kept = text.replace(/\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g, (_, label: string, url: string) => {
    links.push(`<a href="${url}" target="_blank" rel="noopener noreferrer">${label}</a>`);
    return `\u0000${links.length - 1}\u0000`;
  });
  return kept
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
    .replace(/(^|\W)\*([^*\s][^*]*)\*(?=\W|$)/g, '$1<em>$2</em>')
    .replace(BARE_URL, (url) => {
      // (the text is escaped: an address ends where a quote or a bracket written around it starts)
      const end = url.search(/&(?:quot|#0?39|lt|gt)(?:;|$)/);
      const link = end < 0 ? url : url.slice(0, end);
      return `<a href="${link}" target="_blank" rel="noopener noreferrer">${link}</a>${end < 0 ? '' : url.slice(end)}`;
    })
    .replace(/\u0000(\d+)\u0000/g, (_, i: string) => links[Number(i)]);
}

/** An address written bare: http(s), up to a space or a tag, its last full stop or comma not part of it. */
const BARE_URL = /https?:\/\/[^\s<>"']*[^\s<>"'.,;:!?)\]]/g;

/** Text as nodes, its bare addresses as links (a tool's result, which is shown as written). */
export function linkedText(text: string): Node[] {
  const out: Node[] = [];
  let at = 0;
  for (const m of text.matchAll(BARE_URL)) {
    if (m.index! > at) out.push(document.createTextNode(text.slice(at, m.index)));
    const a = document.createElement('a');
    a.href = m[0];
    a.target = '_blank';
    a.rel = 'noopener noreferrer';
    a.textContent = m[0];
    out.push(a);
    at = m.index! + m[0].length;
  }
  if (at < text.length) out.push(document.createTextNode(text.slice(at)));
  return out;
}
