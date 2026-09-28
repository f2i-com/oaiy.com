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
import { formatBytes, mediaElement, mediaKind, type Media } from './media';

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
  /** The file changed outside the editor while it had unsaved typing: nothing is saved until the person chooses. */
  private conflict = false;
  private readonly conflictBar = h('div.editor-conflict', { role: 'alert' });
  /** The image, audio or video file on show, and its element. */
  private mediaPath: string | null = null;
  private media: Media | null = null;

  constructor(private vfs: Vfs) {
    this.conflictBar.hidden = true;
    this.element.append(this.title, this.conflictBar, this.body);
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

  private dropMedia(): void {
    this.media?.dispose();
    this.media = null;
    this.mediaPath = null;
  }

  /** Show an image or play audio or video; SVG can switch to its source. */
  private showMedia(path: string): void {
    this.path = null;
    this.view?.destroy();
    this.view = null;
    this.dropMedia();
    clear(this.body);
    const bytes = this.vfs.readBytes(path);
    const media = mediaElement(path, bytes);
    if (!media) return;
    this.media = media;
    this.mediaPath = path;
    this.title.textContent = path;
    const info = h('span.media-info', formatBytes(bytes.byteLength));
    const bar = h('div.media-bar', info);
    const img = media.element.querySelector('img');
    if (img) {
      img.addEventListener('load', () => (info.textContent = `${img.naturalWidth}×${img.naturalHeight} px · ${formatBytes(bytes.byteLength)}`), { once: true });
      const zoom = h('button', { title: 'Show at actual size or fit to the pane (or click the image)' }, 'Actual size');
      const toggle = () => {
        const actual = media.element.classList.toggle('actual');
        zoom.textContent = actual ? 'Fit' : 'Actual size';
      };
      zoom.addEventListener('click', toggle);
      img.addEventListener('click', toggle);
      bar.append(zoom);
    }
    if (/\.svg$/i.test(path)) bar.append(h('button', { title: 'Edit the SVG source', onclick: () => this.open(path, { source: true }) }, 'Source'));
    this.body.append(h('div.media-view', bar, media.element));
  }

  open(path: string, options: { source?: boolean } = {}): void {
    this.flush();
    this.conflict = false;
    this.conflictBar.hidden = true;
    if (mediaKind(path) && !options.source) {
      this.showMedia(path);
      return;
    }
    this.dropMedia();
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
    if (!this.dirty || !this.view || !this.path || this.conflict) return;
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
    if (this.mediaPath && (path === null || `/${path}` === this.mediaPath)) {
      if (this.vfs.exists(this.mediaPath)) this.showMedia(this.mediaPath);
      else this.close();
      return;
    }
    if (!this.path || (path !== null && `/${path}` !== this.path)) return;
    if (!this.vfs.exists(this.path)) {
      this.close();
      return;
    }
    if (!this.view) return;
    const text = this.vfs.readText(this.path);
    if (text === this.view.state.doc.toString()) return;
    if (this.dirty) {
      this.showConflict();
      return;
    }
    this.applying = true;
    this.view.dispatch({ changes: { from: 0, to: this.view.state.doc.length, insert: text } });
    this.applying = false;
  }

  private showConflict(): void {
    if (this.conflict) return;
    this.conflict = true;
    if (this.saveTimer) {
      clearTimeout(this.saveTimer);
      this.saveTimer = null;
    }
    clear(this.conflictBar);
    this.conflictBar.append(
      h('span', 'This file was changed (by the agent or the terminal) while you were editing it.'),
      h('button', { onclick: () => this.resolve('theirs') }, 'Take the new version'),
      h('button.primary', { onclick: () => this.resolve('mine') }, 'Keep mine'),
    );
    this.conflictBar.hidden = false;
    this.title.textContent = `${this.path} • (changed elsewhere)`;
  }

  private resolve(choice: 'mine' | 'theirs'): void {
    this.conflict = false;
    this.conflictBar.hidden = true;
    if (!this.view || !this.path) return;
    if (choice === 'mine') {
      this.flush();
      return;
    }
    this.dirty = false;
    this.title.textContent = this.path;
    const text = this.vfs.exists(this.path) ? this.vfs.readText(this.path) : '';
    this.applying = true;
    this.view.dispatch({ changes: { from: 0, to: this.view.state.doc.length, insert: text } });
    this.applying = false;
  }

  close(): void {
    this.conflict = false;
    this.conflictBar.hidden = true;
    this.dropMedia();
    this.view?.destroy();
    this.view = null;
    this.path = null;
    this.dirty = false;
    this.title.textContent = 'No file open';
    this.showEmpty();
  }
}
