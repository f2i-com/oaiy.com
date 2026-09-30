/**
 * The one way this origin builds a page: elements made with `createElement`, and every string a provider or a person supplied set as
 * `textContent`, never as markup (design 6, threat 11). There is no `innerHTML` in this origin's code, and its policy would not run an
 * inline script if there were.
 */

export type Child = Node | string | null | undefined | false;

export interface Props {
  class?: string;
  id?: string;
  text?: string;
  attrs?: Record<string, string>;
  on?: Partial<{ [K in keyof HTMLElementEventMap]: (event: HTMLElementEventMap[K]) => void }>;
  disabled?: boolean;
  hidden?: boolean;
  value?: string;
}

export function h<K extends keyof HTMLElementTagNameMap>(tag: K, props: Props = {}, ...children: Child[]): HTMLElementTagNameMap[K] {
  const el = document.createElement(tag);
  if (props.class) el.className = props.class;
  if (props.id) el.id = props.id;
  if (props.text !== undefined) el.textContent = props.text;
  for (const [name, value] of Object.entries(props.attrs ?? {})) el.setAttribute(name, value);
  for (const [type, listener] of Object.entries(props.on ?? {})) el.addEventListener(type, listener as EventListener);
  if (props.disabled) (el as unknown as { disabled: boolean }).disabled = true;
  if (props.hidden) el.hidden = true;
  if (props.value !== undefined) (el as unknown as { value: string }).value = props.value;
  append(el, children);
  return el;
}

export function append(parent: Node, children: readonly Child[]): void {
  for (const child of children) {
    if (child === null || child === undefined || child === false) continue;
    parent.appendChild(typeof child === 'string' ? document.createTextNode(child) : child);
  }
}

/** Replace what `parent` holds. */
export function fill(parent: Element, ...children: Child[]): void {
  parent.replaceChildren();
  append(parent, children);
}
