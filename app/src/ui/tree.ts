/**
 * The file tree: the project's folders and files, expandable, with the file
 * operations a person needs by hand (new, rename, delete).
 */
import { IGNORED_DIRS, type Vfs } from '../vfs/vfs';
import { clear, h } from './dom';

export class FileTree {
  readonly element = h('nav.tree');
  private readonly list = h('div.tree-list');
  private expanded = new Set<string>(['']);
  private selected: string | null = null;
  private renderQueued = false;

  constructor(
    private vfs: Vfs,
    private readonly onOpen: (path: string) => void,
    private readonly onNotice: (message: string) => void,
  ) {
    const actions = h(
      'div.tree-actions',
      h('button.icon', { title: 'New file', onclick: () => this.create('file') }, '+ file'),
      h('button.icon', { title: 'New folder', onclick: () => this.create('dir') }, '+ folder'),
    );
    this.element.append(h('div.pane-title', 'Files'), actions, this.list);
    this.render();
  }

  setVfs(vfs: Vfs): void {
    this.vfs = vfs;
    this.expanded = new Set(['']);
    this.selected = null;
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

  select(path: string): void {
    this.selected = path;
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

  private create(kind: 'file' | 'dir'): void {
    const dir = this.targetDir();
    const name = prompt(kind === 'file' ? 'New file name (e.g. src/index.js)' : 'New folder name', '');
    if (!name) return;
    const path = `/${dir ? `${dir}/` : ''}${name.replace(/^\/+/, '')}`;
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

  private rename(path: string): void {
    const name = prompt('Rename to (a path from the project root)', path.slice(1));
    if (!name || `/${name.replace(/^\/+/, '')}` === path) return;
    try {
      this.vfs.rename(path, `/${name.replace(/^\/+/, '')}`);
    } catch (error) {
      this.onNotice((error as Error).message);
    }
  }

  private remove(path: string): void {
    if (!confirm(`Delete ${path}?`)) return;
    try {
      this.vfs.remove(path, true);
    } catch (error) {
      this.onNotice((error as Error).message);
    }
  }

  private render(): void {
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
        const row = h(
          'div.tree-row',
          {
            class: [this.selected === path ? 'selected' : '', isDir && IGNORED_DIRS.has(entry.name) ? 'dim' : ''].join(' '),
            style: `padding-left:${8 + depth * 14}px`,
            title: path,
            onclick: () => {
              this.selected = path;
              if (isDir) {
                if (open) this.expanded.delete(key);
                else this.expanded.add(key);
              } else this.onOpen(path);
              this.render();
            },
          },
          h('span.twisty', isDir ? (open ? '▾' : '▸') : ''),
          h('span.name', entry.name),
          h(
            'span.row-actions',
            h('button.icon', { title: 'Rename', onclick: (e: Event) => { e.stopPropagation(); this.rename(path); } }, '✎'),
            h('button.icon', { title: 'Delete', onclick: (e: Event) => { e.stopPropagation(); this.remove(path); } }, '✕'),
          ),
        );
        rows.push(row);
        if (isDir && open) walk(key, depth + 1);
      }
    };
    walk('', 0);
    if (!rows.length) rows.push(h('div.empty', 'Empty project'));
    this.list.append(...rows);
  }
}
