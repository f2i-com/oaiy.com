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
import { Compartment, EditorState, type Extension } from '@codemirror/state';
import { EditorView, basicSetup } from 'codemirror';
import type { Vfs } from '../vfs/vfs';
import { clear, h } from './dom';
import { editorLook } from './editorTheme';
import { fileIcon, icon } from './icons';
import { formatBytes, mediaElement, mediaKind, type Media } from './media';
import { onTheme, theme } from './theme';

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

/** What the header says about the open file's saving. */
type SaveState = 'clean' | 'saving' | 'saved' | 'conflict' | 'error' | 'binary' | 'media';

export class EditorPane {
  readonly element = h('section.editor');
  /** The open file's path, as a breadcrumb (its text is the path). */
  private readonly pathLabel = h('span.editor-title', 'No file open');
  private readonly fileGlyph = h('span.editor-icon', { 'aria-hidden': 'true' });
  /** A dot while there is typing not saved yet. */
  private readonly dirtyDot = h('span.editor-dirty', { role: 'img', 'aria-label': 'unsaved changes', title: 'Not saved yet', hidden: true });
  private readonly saveState = h('span.editor-state', { 'aria-live': 'polite' });
  private readonly closeButton = h('button.icon.editor-close', { type: 'button', title: 'Close this file', 'aria-label': 'Close this file', onclick: () => this.closeByHand() }, icon('x')) as HTMLButtonElement;
  private readonly title = h('div.editor-head', this.fileGlyph, this.pathLabel, this.dirtyDot, this.saveState, this.closeButton);
  /** The person closed the file (the tree lets go of it). */
  onClose: () => void = () => {};
  private readonly body = h('div.editor-body');
  private view: EditorView | null = null;
  private path: string | null = null;
  private dirty = false;
  private saveTimer: ReturnType<typeof setTimeout> | null = null;
  private savedTimer: ReturnType<typeof setTimeout> | null = null;
  private applying = false;
  /** The file changed outside the editor while it had unsaved typing: nothing is saved until the person chooses. */
  private conflict = false;
  private readonly conflictBar = h('div.editor-conflict', { role: 'alert' });
  /** The image, audio or video file on show, and its element. */
  private mediaPath: string | null = null;
  private media: Media | null = null;
  /** The editor's colours, which follow the page's light or dark. */
  private readonly look = new Compartment();

  constructor(private vfs: Vfs) {
    this.conflictBar.hidden = true;
    this.element.append(this.title, this.conflictBar, this.body);
    this.showEmpty();
    onTheme((t) => this.view?.dispatch({ effects: this.look.reconfigure(editorLook(t)) }));
  }

  get openPath(): string | null {
    return this.path;
  }

  setVfs(vfs: Vfs): void {
    this.flush();
    this.vfs = vfs;
    this.close();
  }

  /** The header for a file (or none): its icon, its path as a breadcrumb, and how its saving stands. */
  private header(path: string | null, state: SaveState = 'clean', note = ''): void {
    clear(this.pathLabel);
    clear(this.fileGlyph);
    this.closeButton.hidden = !path;
    this.title.classList.toggle('empty', !path);
    if (!path) {
      this.pathLabel.textContent = 'No file open';
      this.pathLabel.removeAttribute('title');
    } else {
      const parts = path.replace(/^\/+/, '').split('/');
      parts.forEach((part, i) => {
        this.pathLabel.append(h('span.crumb-sep', '/'), h(i === parts.length - 1 ? 'span.crumb.leaf' : 'span.crumb', part));
      });
      this.pathLabel.title = path;
      const kind = fileIcon(parts[parts.length - 1]);
      this.fileGlyph.className = `editor-icon tone-${kind.tone}`;
      this.fileGlyph.append(icon(kind.icon));
    }
    this.setState(state, note);
  }

  private setState(state: SaveState, note = ''): void {
    if (this.savedTimer) clearTimeout(this.savedTimer);
    this.savedTimer = null;
    this.dirtyDot.hidden = !(state === 'saving' || state === 'conflict');
    this.saveState.dataset.state = state;
    const words: Record<SaveState, string> = { clean: '', saving: 'Saving…', saved: 'Saved', conflict: 'Changed elsewhere', error: `Not saved: ${note}`, binary: note, media: note };
    this.saveState.textContent = words[state];
    this.saveState.title = state === 'saved' ? 'Saved to the project' : state === 'conflict' ? 'This file changed while you were typing: choose which version to keep' : '';
    // "Saved" says so for a moment, then goes.
    if (state === 'saved') this.savedTimer = setTimeout(() => this.setState('clean'), 1800);
  }

  private showEmpty(): void {
    this.header(null);
    clear(this.body);
    const key = (keys: string, what: string) => h('li', h('kbd', keys), h('span', what));
    this.body.append(h(
      'div.editor-empty',
      h('span.editor-empty-icon', icon('file-text')),
      h('h2', 'No file open'),
      h('p', 'Open a file from the list, or ask the agent to write one.'),
      h('ul.editor-keys', key('Ctrl F', 'find in the file'), key('Ctrl /', 'comment a line'), key('Alt ↑ ↓', 'move a line')),
      h('p.editor-empty-note', 'What you type is saved to the project as you type.'),
    ));
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
    this.header(path, 'media', formatBytes(bytes.byteLength));
    const info = h('span.media-info', formatBytes(bytes.byteLength));
    const bar = h('div.media-bar', info);
    const img = media.element.querySelector('img');
    if (img) {
      img.addEventListener('load', () => (info.textContent = `${img.naturalWidth}×${img.naturalHeight} px · ${formatBytes(bytes.byteLength)}`), { once: true });
      const zoom = h('button', { type: 'button', title: 'Show at actual size or fit to the pane (or click the image)' }, 'Actual size');
      const toggle = () => {
        const actual = media.element.classList.toggle('actual');
        zoom.textContent = actual ? 'Fit' : 'Actual size';
      };
      zoom.addEventListener('click', toggle);
      img.addEventListener('click', toggle);
      bar.append(zoom);
    }
    if (/\.svg$/i.test(path)) bar.append(h('button', { type: 'button', title: 'Edit the SVG source', onclick: () => this.open(path, { source: true }) }, 'Source'));
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
      const size = this.vfs.stat(path)?.size ?? 0;
      this.header(path, 'binary', `${size.toLocaleString()} bytes`);
      this.body.append(h('div.editor-empty', h('span.editor-empty-icon', icon('file')), h('h2', 'Binary file'), h('p', `${size.toLocaleString()} bytes: not shown here.`)));
      return;
    }
    this.path = path;
    this.dirty = false;
    this.header(path);
    const doc = this.vfs.readText(path);
    const state = EditorState.create({
      doc,
      extensions: [
        basicSetup,
        this.look.of(editorLook(theme())),
        EditorView.lineWrapping,
        ...languageFor(path),
        EditorView.updateListener.of((update) => {
          if (!update.docChanged || this.applying) return;
          this.dirty = true;
          if (!this.conflict) this.setState('saving');
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
      this.setState('saved');
    } catch (error) {
      this.setState('error', (error as Error).message);
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
      icon('alert'),
      h('span', 'This file was changed (by the agent or the terminal) while you were editing it.'),
      h('button', { type: 'button', onclick: () => this.resolve('theirs') }, 'Take the new version'),
      h('button.primary', { type: 'button', onclick: () => this.resolve('mine') }, 'Keep mine'),
    );
    this.conflictBar.hidden = false;
    this.setState('conflict');
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
    this.setState('clean');
    const text = this.vfs.exists(this.path) ? this.vfs.readText(this.path) : '';
    this.applying = true;
    this.view.dispatch({ changes: { from: 0, to: this.view.state.doc.length, insert: text } });
    this.applying = false;
  }

  /** Closed with its button: what was typed is saved first, and nothing is open after. */
  private closeByHand(): void {
    this.flush();
    this.close();
    this.onClose();
  }

  close(): void {
    this.conflict = false;
    this.conflictBar.hidden = true;
    this.dropMedia();
    this.view?.destroy();
    this.view = null;
    this.path = null;
    this.dirty = false;
    this.showEmpty();
  }
}
