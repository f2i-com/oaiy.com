import { escapeHtml } from './dom';

/**
 * Just enough Markdown for chat replies: fenced code, inline code, bold,
 * italics, headings, lists and links. Everything is escaped first, so a
 * reply can never inject markup.
 */
export function renderMarkdown(source: string): string {
  const parts = source.split(/```/);
  let html = '';
  parts.forEach((part, i) => {
    if (i % 2 === 1) {
      const newline = part.indexOf('\n');
      const lang = newline > 0 ? part.slice(0, newline).trim() : '';
      const code = newline >= 0 ? part.slice(newline + 1) : part;
      html += `<pre class="code"${lang ? ` data-lang="${escapeHtml(lang)}"` : ''}><code>${escapeHtml(code.replace(/\n$/, ''))}</code></pre>`;
      return;
    }
    const lines = escapeHtml(part).split('\n');
    let list: 'ul' | 'ol' | null = null;
    for (const raw of lines) {
      const bullet = /^\s*[-*] (.*)$/.exec(raw);
      const numbered = /^\s*\d+[.)] (.*)$/.exec(raw);
      const kind = bullet ? 'ul' : numbered ? 'ol' : null;
      if (list && kind !== list) {
        html += `</${list}>`;
        list = null;
      }
      if (kind && !list) {
        html += `<${kind}>`;
        list = kind;
      }
      const text = inline(bullet?.[1] ?? numbered?.[1] ?? raw);
      const heading = /^(#{1,4}) (.*)$/.exec(raw);
      if (kind) html += `<li>${text}</li>`;
      else if (heading) html += `<h${heading[1].length + 2}>${inline(heading[2])}</h${heading[1].length + 2}>`;
      else if (raw.trim()) html += `<p>${text}</p>`;
    }
    if (list) html += `</${list}>`;
  });
  return html;
}

function inline(text: string): string {
  return text
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
    .replace(/(^|\W)\*([^*\s][^*]*)\*(?=\W|$)/g, '$1<em>$2</em>')
    .replace(/\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
}
