/** A small element builder: h('div.cls', { onclick }, child, 'text'). */
type Child = Node | string | null | undefined | false;
type Props = Record<string, unknown> & { class?: string; style?: string };

export function h<K extends keyof HTMLElementTagNameMap>(tag: K | `${K}.${string}`, props?: Props | Child, ...children: Child[]): HTMLElementTagNameMap[K] {
  const [name, ...classes] = String(tag).split('.');
  const el = document.createElement(name as K);
  if (classes.length) el.className = classes.join(' ');
  let kids = children;
  if (props && (typeof props !== 'object' || props instanceof Node)) kids = [props as Child, ...children];
  else if (props) {
    for (const [key, value] of Object.entries(props)) {
      if (value === undefined || value === null || value === false) continue;
      if (key === 'class') el.className = [el.className, String(value)].filter(Boolean).join(' ');
      else if (key.startsWith('on') && typeof value === 'function') el.addEventListener(key.slice(2), value as EventListener);
      else if (key in el && typeof value !== 'string') (el as unknown as Record<string, unknown>)[key] = value;
      else el.setAttribute(key, value === true ? '' : String(value));
    }
  }
  for (const kid of kids) if (kid !== null && kid !== undefined && kid !== false) el.append(kid);
  return el;
}

export function clear(el: Element): void {
  while (el.firstChild) el.firstChild.remove();
}

export function escapeHtml(text: string): string {
  return text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}
