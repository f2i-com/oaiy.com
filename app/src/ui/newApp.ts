/**
 * The "New SoftN app" dialog: a name, what to start from (a small working
 * task list, a blank page, or one of the example apps), and where it goes (a
 * project of its own, or a folder of the open project).
 */
import { h } from './dom';
import { modal } from './modal';

export interface NewAppChoice {
  name: string;
  /** 'starter', 'blank', or an example's slug. */
  start: string;
  where: 'project' | 'folder';
  /** For where === 'folder'. */
  folder: string;
}

export interface Template {
  id: string;
  name: string;
  description: string;
  icon: string;
}

const slug = (name: string) => name.trim().replace(/[^\w.-]+/g, '-').replace(/^-+|-+$/g, '').toLowerCase() || 'app';

export function newAppDialog(options: {
  projectName: string;
  /** Always shown: the starter and a blank page. */
  templates: Template[];
  /** Complete example apps, in a section that opens on demand. */
  examples: Template[];
  /** Why a folder will not do (it holds an app already, say), or null. */
  folderProblem: (folder: string) => string | null;
}): Promise<NewAppChoice | null> {
  const name = h('input', { type: 'text', value: 'My app', spellcheck: false, autocomplete: 'off' }) as HTMLInputElement;
  const folder = h('input', { type: 'text', value: 'my-app', spellcheck: false, autocomplete: 'off' }) as HTMLInputElement;
  let folderEdited = false;
  folder.addEventListener('input', () => (folderEdited = true));
  const error = h('p.modal-error', { role: 'alert' });

  // What to start from: cards that work as radio buttons.
  let start = options.templates[0]?.id ?? 'starter';
  const card = (t: Template) =>
    h(
      'label.template-card',
      h('input', { type: 'radio', name: 'template', value: t.id, checked: t.id === start, onchange: () => {
        start = t.id;
        // A name still at its default follows the example chosen.
        if (name.dataset.edited !== 'yes') {
          name.value = t.id === 'starter' ? 'My app' : t.id === 'blank' ? 'My app' : t.name;
          if (!folderEdited) folder.value = slug(name.value);
        }
      } }),
      h('span.template-icon', t.icon),
      h('span.template-text', h('strong', t.name), h('span.template-description', { title: t.description }, t.description)),
    );
  const examples = h(
    'details.template-examples',
    h('summary', h('span', 'Start from an example app'), h('span.template-count', `${options.examples.length}`)),
    h('div.template-grid.examples-grid', ...options.examples.map(card)),
  ) as HTMLDetailsElement;
  name.addEventListener('input', () => {
    name.dataset.edited = 'yes';
    error.textContent = '';
    if (!folderEdited) folder.value = slug(name.value);
  });

  let where: NewAppChoice['where'] = 'project';
  const folderRow = h('label.modal-field.folder-field', h('span', 'Folder'), folder);
  folderRow.hidden = true;
  const place = (value: NewAppChoice['where'], label: string, hint: string) =>
    h(
      'label.where-option',
      h('input', { type: 'radio', name: 'where', value, checked: value === where, onchange: () => {
        where = value;
        folderRow.hidden = value !== 'folder';
        error.textContent = '';
        if (value === 'folder') folder.focus();
      } }),
      h('span', h('strong', label), h('span.where-hint', hint)),
    );

  return modal<NewAppChoice>({
    title: 'New SoftN app',
    message: 'Pick a starting point. The agent can change it into what you want while you watch it in the preview.',
    wide: true,
    focus: name,
    body: [
      h('label.modal-field', h('span', 'Name'), name),
      h('fieldset.template-start', h('legend', 'Start from'), h('div.template-grid', ...options.templates.map(card)), options.examples.length ? examples : null),
      h(
        'fieldset.where',
        h('legend', 'Where'),
        place('project', 'A new project', 'its own project, with nothing else in it'),
        place('folder', `A folder in "${options.projectName}"`, 'next to what is already there, e.g. to rebuild an existing app'),
        folderRow,
      ),
      error,
    ],
    ok: {
      label: 'Create app',
      value: () => {
        const chosen = name.value.trim();
        if (!chosen) {
          error.textContent = 'Give the app a name.';
          name.focus();
          return undefined;
        }
        const dir = folder.value.trim().replace(/^\/+|\/+$/g, '');
        if (where === 'folder') {
          const problem = !dir ? 'Name the folder for the app.' : dir.split('/').includes('..') ? 'The folder must be inside the project.' : options.folderProblem(dir);
          if (problem) {
            error.textContent = problem;
            folder.focus();
            return undefined;
          }
        }
        return { name: chosen, start, where, folder: dir };
      },
    },
  });
}
