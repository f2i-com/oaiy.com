/**
 * The file tree: the project's folders and files, expandable, with the file
 * operations a person needs by hand (new, rename, delete). It is one tab stop,
 * moved through with the arrow keys: Up and Down, Right to open a folder (or
 * go into it), Left to close it (or go to its folder), Home and End, Enter to
 * open, F2 to rename, Delete to delete.
 */
import { IGNORED_DIRS, normalizePath, type Vfs } from '../vfs/vfs';
import { clear, h } from './dom';
import { fileIcon, icon } from './icons';
import { askText, confirmAction } from './modal';

export class FileTree {
  readonly element = h('nav.tree', { 'aria-label': 'Files' });
  private readonly list = h('div.tree-list');
  private expanded = new Set<string>(['']);
  private selected: string | null = null;
  private renderQueued = false;
  /**
   * A rename or move made here goes through this, so the editor can save what is typed into the file first and keep
   * it open where it goes (main.ts gives it the editor's). On its own it just makes the move.
   */
  moving: (from: string, to: string, move: () => void) => void = (_from, _to, move) => move();
  /** Whose files these are, when they are not the open project's (the Front desk's). */
  private readonly place = h('div.tree-place', { hidden: true });
  private readonly title = h('div.pane-title.tree-head', h('span.pane-kicker', 'Files'));

  constructor(
    private vfs: Vfs,
    private readonly onOpen: (path: string) => void,
    private readonly onNotice: (message: string) => void,
  ) {
    const tool = (glyph: string, title: string, run: () => void) => h('button.tree-tool', { type: 'button', title, 'aria-label': title, onclick: run }, icon(glyph));
    this.title.append(h(
      'span.tree-tools',
      tool('file-plus', 'New file', () => void this.create('file')),
      tool('folder-plus', 'New folder', () => void this.create('dir')),
      tool('collapse', 'Collapse all folders', () => {
        this.expanded = new Set(['']);
        this.render();
      }),
    ));
    this.list.setAttribute('role', 'tree');
    this.list.setAttribute('aria-label', 'Project files');
    this.element.append(this.title, this.place, this.list);
    this.render();
  }

  /** Say whose files these are (the front desk's, say): nothing for the open project's. */
  setPlace(place: string | null): void {
    this.place.hidden = !place;
    this.place.replaceChildren(...(place ? [icon('phone'), h('span.tree-place-text', h('strong', `${place} files`), h('span', "What the phone's agents read"))] : []));
    this.place.title = place ? `The ${place.toLowerCase()}'s files: what the phone's agent reads` : '';
  }

  /** The folders open and the file chosen: given back to `setVfs` when these files are shown again. */
  get shown(): { expanded: string[]; selected: string | null } {
    return { expanded: [...this.expanded], selected: this.selected };
  }

  setVfs(vfs: Vfs, shown?: { expanded: string[]; selected: string | null }): void {
    this.vfs = vfs;
    this.expanded = new Set(['', ...(shown?.expanded ?? [])]);
    this.selected = shown?.selected ?? null;
    this.render();
  }

  /** Re-render soon (coalesces bursts of changes from the agent). */
  refresh(): void {
    if (this.renderQueued) return;
    this.renderQueued = true;
    requestAnimationFrame(() => {
      this.renderQueued = false;
      this.render();
    });
  }

  /** Select a path (its folders opened), or nothing. */
  select(path: string | null): void {
    this.selected = path;
    if (!path) {
      this.refresh();
      return;
    }
    const parts = path.replace(/^\//, '').split('/');
    for (let i = 1; i < parts.length; i++) this.expanded.add(parts.slice(0, i).join('/'));
    this.refresh();
  }

  private targetDir(): string {
    if (!this.selected) return '';
    const key = this.selected.replace(/^\//, '');
    const stat = this.vfs.stat(this.selected);
    if (stat?.type === 'dir') return key;
    const slash = key.lastIndexOf('/');
    return slash < 0 ? '' : key.slice(0, slash);
  }

  /**
   * Where a new file or folder named `value` goes: in the folder chosen (`dir`), or, for a name that starts with "/",
   * from the project root. ".." goes up a folder, as it does in the terminal.
   */
  private newPath(dir: string, value: string): string {
    return `/${normalizePath(value.trim(), `/${dir}`)}`;
  }

  private async create(kind: 'file' | 'dir'): Promise<void> {
    const dir = this.targetDir();
    const name = await askText({
      title: kind === 'file' ? 'New file' : 'New folder',
      message: dir ? `In /${dir}/. A name that starts with / goes from the project root instead.` : 'At the project root.',
      label: kind === 'file' ? 'File name' : 'Folder name',
      placeholder: kind === 'file'
        ? dir ? 'e.g. quote.js, or /notes/todo.md from the root' : 'e.g. index.js, or src/quote.js'
        : dir ? 'e.g. assets, or /docs from the root' : 'e.g. assets',
      ok: 'Create',
      validate: (value) => {
        // (a name ending in "/" names a folder: a file of that name, or one called "notes" for "notes/", is not what was meant)
        if (kind === 'file' && value.endsWith('/')) return `"${value}" ends with /, which names a folder: make it with New folder, or name a file in it (${value}todo.md).`;
        const path = this.newPath(dir, value);
        if (path === '/') return 'That is the project root: give it a name.';
        return this.vfs.exists(path) ? `${path} already exists.` : null;
      },
    });
    if (!name) return;
    const path = this.newPath(dir, name);
    try {
      if (this.vfs.exists(path)) throw new Error(`${path} already exists`);
      if (kind === 'file') {
        this.vfs.writeFile(path, '', { parents: true });
        this.select(path);
        this.onOpen(path);
      } else {
        this.vfs.mkdir(path, true);
        this.expanded.add(path.slice(1));
        this.select(path);
      }
    } catch (error) {
      this.onNotice((error as Error).message);
    }
  }

  private async rename(path: string): Promise<void> {
    const name = await askText({ title: 'Rename', message: 'A path from the project root: change the folder part to move it.', label: 'New path', value: path.slice(1), ok: 'Rename' });
    const to = `/${name?.replace(/^\/+/, '').replace(/\/+$/, '') ?? ''}`;
    if (!name || to === path || to === '/') return;
    try {
      // (not over a file or folder that is there: a rename does not overwrite or merge)
      if (this.vfs.exists(to)) {
        this.onNotice(`${to} is there already: choose another name, or delete that one first.`);
        return;
      }
      // (into a folder that is not there yet: made, as the path asks)
      const parent = to.slice(0, to.lastIndexOf('/'));
      if (parent && !this.vfs.exists(parent)) this.vfs.mkdir(parent, true);
      this.moving(path, to, () => this.vfs.rename(path, to));
      // What was chosen in the tree stays chosen where it went, its folders opened.
      const chosen = this.selected;
      if (chosen === path || chosen?.startsWith(`${path}/`)) this.select(`${to}${chosen.slice(path.length)}`);
    } catch (error) {
      this.onNotice((error as Error).message);
    }
  }

  private async remove(path: string): Promise<void> {
    const dir = this.vfs.stat(path)?.type === 'dir';
    if (!(await confirmAction({ title: dir ? 'Delete folder' : 'Delete file', message: `Delete ${path}${dir ? ' and everything in it' : ''}?`, ok: 'Delete', danger: true }))) return;
    try {
      this.vfs.remove(path, true);
    } catch (error) {
      this.onNotice((error as Error).message);
    }
  }

  /** Move the keyboard focus to a row (it becomes the tree's one tab stop). */
  private focusRow(row: HTMLElement | undefined): void {
    if (!row) return;
    for (const r of this.list.querySelectorAll<HTMLElement>('.tree-row')) r.tabIndex = r === row ? 0 : -1;
    row.focus();
  }

  private render(): void {
    const focused = (document.activeElement as HTMLElement | null)?.closest?.('.tree-row')?.getAttribute('title') ?? null;
    clear(this.list);
    const rows: HTMLElement[] = [];
    const walk = (dir: string, depth: number) => {
      let entries;
      try {
        entries = this.vfs.list(`/${dir}`);
      } catch {
        return;
      }
      entries.sort((a, b) => (a.type === b.type ? (a.name < b.name ? -1 : 1) : a.type === 'dir' ? -1 : 1));
      for (const entry of entries) {
        const key = dir ? `${dir}/${entry.name}` : entry.name;
        const path = `/${key}`;
        const isDir = entry.type === 'dir';
        const open = this.expanded.has(key);
        const kind = fileIcon(entry.name, isDir, open);
        const toggle = () => {
          if (open) this.expanded.delete(key);
          else this.expanded.add(key);
        };
        const row = h(
          'div.tree-row',
          {
            class: [this.selected === path ? 'selected' : '', isDir ? 'dir' : 'file', isDir && IGNORED_DIRS.has(entry.name) ? 'dim' : ''].filter(Boolean).join(' '),
            style: `--depth:${depth}`,
            title: path,
            role: 'treeitem',
            // (read as its name alone: its Rename and Delete buttons are for the pointer, and F2 and Delete do the same)
            'aria-label': entry.name,
            'aria-keyshortcuts': 'F2 Delete',
            tabindex: -1,
            'aria-level': depth + 1,
            'aria-selected': String(this.selected === path),
            ...(isDir ? { 'aria-expanded': String(open) } : {}),
            onkeydown: (e: KeyboardEvent) => {
              if (e.target !== e.currentTarget) return;
              const all = [...this.list.querySelectorAll<HTMLElement>('.tree-row')];
              const i = all.indexOf(e.currentTarget as HTMLElement);
              const go = (to: number) => {
                e.preventDefault();
                this.focusRow(all[Math.max(0, Math.min(all.length - 1, to))]);
              };
              if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                (e.currentTarget as HTMLElement).click();
              } else if (e.key === 'ArrowDown') go(i + 1);
              else if (e.key === 'ArrowUp') go(i - 1);
              else if (e.key === 'Home') go(0);
              else if (e.key === 'End') go(all.length - 1);
              else if (e.key === 'ArrowRight' && isDir) {
                e.preventDefault();
                if (!open) {
                  this.selected = path;
                  toggle();
                  this.render();
                } else go(i + 1);
              } else if (e.key === 'ArrowLeft') {
                e.preventDefault();
                if (isDir && open) {
                  toggle();
                  this.render();
                } else if (dir) this.focusRow(all.find((r) => r.getAttribute('title') === `/${dir}`));
              } else if (e.key === 'F2') {
                e.preventDefault();
                void this.rename(path);
              } else if (e.key === 'Delete') {
                e.preventDefault();
                void this.remove(path);
              }
            },
            onclick: () => {
              this.selected = path;
              if (isDir) toggle();
              else this.onOpen(path);
              this.render();
            },
          },
          h('span.twisty', { 'aria-hidden': 'true' }, isDir ? icon('chevron-right') : ''),
          h('span.tree-icon', { class: `tone-${kind.tone}`, 'aria-hidden': 'true' }, icon(kind.icon)),
          h('span.name', entry.name),
          h(
            'span.row-actions',
            { 'aria-hidden': 'true' },
            h('button.icon', { type: 'button', tabindex: -1, title: 'Rename', 'aria-label': `Rename ${entry.name}`, onclick: (e: Event) => { e.stopPropagation(); void this.rename(path); } }, icon('pencil')),
            h('button.icon', { type: 'button', tabindex: -1, title: 'Delete', 'aria-label': `Delete ${entry.name}`, onclick: (e: Event) => { e.stopPropagation(); void this.remove(path); } }, icon('trash')),
          ),
        );
        rows.push(row);
        if (isDir && open) walk(key, depth + 1);
      }
    };
    walk('', 0);
    if (!rows.length) {
      this.list.append(h(
        'div.tree-empty',
        h('span.tree-empty-icon', icon('folder-open')),
        h('p', 'No files yet'),
        h('p.muted', 'Make one, drop files on the chat, or ask the agent to write some.'),
        h('button', { type: 'button', onclick: () => void this.create('file') }, icon('file-plus'), h('span', 'New file')),
      ));
      return;
    }
    this.list.append(...rows);
    // One tab stop: the row in focus before, the selected one, or the first.
    const home = rows.find((r) => r.getAttribute('title') === focused) ?? rows.find((r) => r.classList.contains('selected')) ?? rows[0];
    home.tabIndex = 0;
    if (focused && home.getAttribute('title') === focused) home.focus();
  }
}
