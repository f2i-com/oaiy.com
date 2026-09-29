/**
 * The project switcher: the searchable picker over the app's own project
 * <select>, which stays the source of truth (the app fills it, and listens
 * for its change). Choosing here sets its value and says it changed; a switch
 * the app turns down (the agent was working) sets it back, and the button
 * follows.
 */
import { Combobox, type ComboItem } from './combobox';
import { h } from './dom';
import { icon } from './icons';

const FRONT_DESK = 'front-desk';
const GROUPS = ['Phone', 'Projects', 'Incognito'];

function iconFor(value: string, incognito: boolean): string {
  return value === FRONT_DESK ? 'phone' : incognito ? 'glasses' : 'folder';
}

export class ProjectPicker {
  readonly element: HTMLElement;
  private readonly combo: Combobox;

  constructor(private readonly select: HTMLSelectElement) {
    this.combo = new Combobox({
      label: 'Projects',
      placeholder: 'Search projects…',
      groups: GROUPS,
      className: 'project-picker',
      title: 'Switch project',
      choose: (value) => {
        this.select.value = value;
        this.select.dispatchEvent(new Event('change', { bubbles: true }));
      },
      button: (current) => (current
        ? [h('span.project-icon', { class: `kind-${current.kind}` }, icon(iconFor(current.id, current.kind === 'incognito'))), h('span.project-name', current.label)]
        : [h('span.project-name.muted', 'Projects')]),
    });
    select.hidden = true;
    select.tabIndex = -1;
    select.setAttribute('aria-hidden', 'true');
    this.element = h('div.project-switch', this.combo.element, select);
    // The app rebuilds the options (a project made, renamed or deleted), and may set the value back.
    new MutationObserver(() => this.render()).observe(select, { childList: true, subtree: true, attributes: true, characterData: true });
    select.addEventListener('change', () => this.render());
    const own = Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, 'value')!;
    Object.defineProperty(select, 'value', {
      configurable: true,
      get: () => own.get!.call(select),
      set: (value: string) => {
        own.set!.call(select, value);
        this.render();
      },
    });
    this.render();
  }

  private render(): void {
    const items: ComboItem[] = [...this.select.options].map((option) => {
      const text = option.textContent ?? option.value;
      const incognito = /^🕶/.test(text) || / \(incognito\)$/.test(text);
      const label = text.replace(/^[^\p{L}\p{N}]+/u, '').replace(/ \(incognito\)$/, '').trim() || option.value;
      const kind = option.value === FRONT_DESK ? 'phone' : incognito ? 'incognito' : 'project';
      return {
        id: option.value,
        label,
        detail: option.value === FRONT_DESK ? option.title || "The phone's agents" : incognito ? 'Incognito: kept only until you leave it' : undefined,
        group: option.value === FRONT_DESK ? 'Phone' : incognito ? 'Incognito' : 'Projects',
        kind,
        title: option.title || label,
        icon: () => icon(iconFor(option.value, incognito)),
      };
    });
    this.combo.set(items, this.select.value || null);
  }
}
