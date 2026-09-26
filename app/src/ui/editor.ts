/**
 * The editor pane: CodeMirror over one open file. Typing writes back into the
 * project (debounced); a change the agent makes to the open file reloads it
 * unless there are unsaved keystrokes.
 */
import { css } from '@codemirror/lang-css';
import { html } from '@codemirror/lang-html';
import { javascript } from '@codemirror/lang-javascript';
import { json } from '@codemirror/lang-json';
import { markdown } from '@codemirror/lang-markdown';
import { python } from '@codemirror/lang-python';
import { oneDark } from '@codemirror/theme-one-dark';
import { EditorState, type Extension } from '@codemirror/state';
import { EditorView, basicSetup } from 'codemirror';
import type { Vfs } from '../vfs/vfs';
import { clear, h } from './dom';

function languageFor(path: string): Extension[] {
  const ext = path.slice(path.lastIndexOf('.') + 1).toLowerCase();
  switch (ext) {
    case 'js': case 'mjs': case 'cjs': case 'jsx': return [javascript({ jsx: true })];
    case 'ts': case 'tsx': case 'mts': return [javascript({ typescript: true, jsx: ext === 'tsx' })];
    case 'py': return [python()];
    case 'json': return [json()];
    case 'md': case 'markdown': return [markdown()];
    case 'html': case 'htm': case 'svg': case 'xml': return [html()];
    case 'css': case 'scss': return [css()];
    default: return [];
  }
}

export class EditorPane {
  readonly element = h('section.editor');
  private readonly title = h('div.editor-title', 'No file open');
  private readonly body = h('div.editor-body');
  private view: EditorView | null = null;
  private path: string | null = null;
  private dirty = false;
  private saveTimer: ReturnType<typeof setTimeout> | null = null;
  private applying = false;

  constructor(private vfs: Vfs) {
    this.element.append(this.title, this.body);
    this.showEmpty();
  }

  get openPath(): string | null {
    return this.path;
  }

  setVfs(vfs: Vfs): void {
    this.flush();
    this.vfs = vfs;
    this.close();
  }

  private showEmpty(): void {
    clear(this.body);
    this.body.append(h('div.empty', h('p', 'Open a file from the tree, or ask the agent to write one.')));
  }

  open(path: string): void {
    this.flush();
    if (!this.vfs.isText(path)) {
      this.path = null;
      this.view?.destroy();
      this.view = null;
      clear(this.body);
      this.title.textContent = path;
      const size = this.vfs.stat(path)?.size ?? 0;
      this.body.append(h('div.empty', h('p', `Binary file (${size.toLocaleString()} bytes) — not shown.`)));
      return;
    }
    this.path = path;
    this.dirty = false;
    this.title.textContent = path;
    const doc = this.vfs.readText(path);
    const state = EditorState.create({
      doc,
      extensions: [
        basicSetup,
        oneDark,
        EditorView.lineWrapping,
        ...languageFor(path),
        EditorView.updateListener.of((update) => {
          if (!update.docChanged || this.applying) return;
          this.dirty = true;
          this.title.textContent = `${this.path} •`;
          if (this.saveTimer) clearTimeout(this.saveTimer);
          this.saveTimer = setTimeout(() => this.flush(), 400);
        }),
      ],
    });
    this.view?.destroy();
    clear(this.body);
    this.view = new EditorView({ state, parent: this.body });
  }

  /** Save pending keystrokes to the project now. */
  flush(): void {
    if (this.saveTimer) {
      clearTimeout(this.saveTimer);
      this.saveTimer = null;
    }
    if (!this.dirty || !this.view || !this.path) return;
    try {
      this.vfs.writeFile(this.path, this.view.state.doc.toString(), { parents: true });
      this.dirty = false;
      this.title.textContent = this.path;
    } catch (error) {
      this.title.textContent = `${this.path} — not saved: ${(error as Error).message}`;
    }
  }

  /** The project changed underneath: reload or close the open file. */
  externalChange(path: string | null): void {
    if (!this.path || (path !== null && `/${path}` !== this.path)) return;
    if (!this.vfs.exists(this.path)) {
      this.close();
      return;
    }
    if (this.dirty || !this.view) return;
    const text = this.vfs.readText(this.path);
    if (text === this.view.state.doc.toString()) return;
    this.applying = true;
    this.view.dispatch({ changes: { from: 0, to: this.view.state.doc.length, insert: text } });
    this.applying = false;
  }

  close(): void {
    this.view?.destroy();
    this.view = null;
    this.path = null;
    this.dirty = false;
    this.title.textContent = 'No file open';
    this.showEmpty();
  }
}
