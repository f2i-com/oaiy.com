import { describe, expect, it } from 'vitest';
import { renderMarkdown } from '../../src/ui/markdown';

describe("the chat's Markdown", () => {
  it('never lets a reply inject markup', () => {
    const html = renderMarkdown('<img src=x onerror=alert(1)> **<script>x</script>**\n\n| <b>a</b> | b |\n|---|---|\n| <i>1</i> | 2 |\n\n```html"><svg onload=x>\n<div>hi</div>\n```\n[x](javascript:alert(1)) [y](https://example.com/"onmouseover="x)');
    for (const tag of ['<img', '<script', '<svg', '<b>', '<i>', '<div>']) expect(html).not.toContain(tag);
    expect(html).not.toContain('href="javascript');
    expect(html).not.toMatch(/"onmouseover=/);
    expect(html).toContain('&lt;img src=x onerror=alert(1)&gt;');
  });

  it('draws a table, with the alignment its separator asks for', () => {
    const html = renderMarkdown('| City | °C |\n|---|---:|\n| Cairo | 31 |\n| Oslo | 9 |');
    expect(html).toBe('<div class="table-wrap"><table><thead><tr><th>City</th><th style="text-align:right">°C</th></tr></thead><tbody><tr><td>Cairo</td><td style="text-align:right">31</td></tr><tr><td>Oslo</td><td style="text-align:right">9</td></tr></tbody></table></div>');
  });

  it('gives a code block its language and a Copy button, and marks one still being written', () => {
    const done = renderMarkdown('```python\nprint(1)\n```');
    expect(done).toContain('<span class="code-lang">python</span>');
    expect(done).toContain('class="code-copy"');
    expect(done).toContain('<pre class="code" data-lang="python"><code>print(1)</code></pre>');
    expect(done).not.toContain('writing');
    expect(renderMarkdown('Here:\n```js\nlet a')).toContain('code-block writing');
  });

  it('keeps inline code as it is, and draws headings, lists, quotes and rules', () => {
    expect(renderMarkdown('Use `**not bold**` here')).toBe('<p>Use <code>**not bold**</code> here</p>');
    expect(renderMarkdown('## Title\n- a\n- [x] done\n3. three\n> quoted\n---')).toBe('<h4>Title</h4><ul><li>a</li><li class="task done"><span class="task-box" aria-hidden="true"></span>done</li></ul><ol start="3"><li>three</li></ol><blockquote>quoted</blockquote><hr>');
  });
});
