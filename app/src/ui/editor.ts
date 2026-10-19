/**
 * The editor pane: CodeMirror over one open file. Typing writes back into the
 * project (debounced); a change the agent makes to the open file reloads it
 * unless there are unsaved keystrokes. A file renamed or moved (in the tree,
 * by the agent or in the terminal) stays open where it went.
 */
import { css } from '@codemirror/lang-css';
import { html } from '@codemirror/lang-html';
import { javascript } from '@codemirror/lang-javascript';
import { json } from '@codemirror/lang-json';
import { markdown } from '@codemirror/lang-markdown';
import { python } from '@codemirror/lang-python';
import { Compartment, EditorState, type Extension } from '@codemirror/state';
import { EditorView, basicSetup } from 'codemirror';
import type { Vfs, VfsChange } from '../vfs/vfs';
import { clear, h } from './dom';
import { editorLook } from './editorTheme';
import { fileIcon, icon } from './icons';
import { formatBytes, mediaElement, mediaKind, type Media } from './media';
import { onTheme, theme } from './theme';

/** The editor's keys as this computer writes them: on a Mac or an iPad (as CodeMirror tells them apart) Mod is ⌘ and Alt is ⌥. */
const MAC = typeof navigator !== 'undefined' && /Mac|iPhone|iPad|iPod/.test(navigator.platform);
const MOD = MAC ? '⌘' : 'Ctrl';
const ALT = MAC ? '⌥' : 'Alt';

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

/** What a screen reader calls the editor: the file it holds. */
function nameFor(path: string): Extension {
  return EditorView.contentAttributes.of({ 'aria-label': `Editing ${path.replace(/^\/+/, '')}` });
}

/** Whether two files' bytes are the same. */
function sameBytes(a: Uint8Array, b: Uint8Array): boolean {
  if (a.byteLength !== b.byteLength) return false;
  for (let i = 0; i < a.byteLength; i++) if (a[i] !== b[i]) return false;
  return true;
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
  /** The open file's language, and the name it is read out by: both of its path, which a rename changes. */
  private readonly language = new Compartment();
  private readonly name = new Compartment();
  /** The open file was followed to where it was renamed or moved: the tree shows it there. */
  onMove: (path: string) => void = () => {};
  /** The open file's bytes as last read or saved: a copy of it, moved, holds the same. */
  private known: Uint8Array | null = null;
  /**
   * Files written just now (in this task) besides the open one. A file moved is copied to where it goes and then
   * removed, in one go: when the open file is removed, its copy is among these.
   */
  private written: string[] = [];
  private writtenTimer: ReturnType<typeof setTimeout> | null = null;
  /** A rename or move the tree is making now (see move): the open file, or a folder it is in, followed exactly. */
  private moving: { from: string; to: string } | null = null;

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
    this.keepMine();
    this.vfs = vfs;
    this.close();
  }

  /**
   * Leaving a file whose changes clash with what was written meanwhile (the conflict bar up, not answered): the
   * person's text is kept beside it, as `name.mine.ext`, rather than dropped with the editor.
   */
  private keepMine(): void {
    if (!this.conflict || !this.dirty || !this.view || !this.path) return;
    const at = this.path.lastIndexOf('.');
    const slash = this.path.lastIndexOf('/');
    const [stem, ext] = at > slash + 1 ? [this.path.slice(0, at), this.path.slice(at)] : [this.path, ''];
    let target = `${stem}.mine${ext}`;
    for (let n = 2; this.vfs.exists(target); n++) target = `${stem}.mine-${n}${ext}`;
    try {
      this.vfs.writeFile(target, this.view.state.doc.toString(), { parents: true });
    } catch {
      /* the project cannot take it: nothing more to do as the file is left */
    }
    this.dirty = false;
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
      h('ul.editor-keys', key(`${MOD} F`, 'find in the file'), key(`${MOD} /`, 'comment a line'), key(`${ALT} ↑ ↓`, 'move a line')),
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
    this.known = bytes;
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
    this.keepMine();
    this.conflict = false;
    this.conflictBar.hidden = true;
    if (mediaKind(path) && !options.source) {
      this.showMedia(path);
      return;
    }
    this.dropMedia();
    // (read first: a file that looked like text in its first 8 KB and is not after must not leave the old file's text
    // on show under its name, to be saved over it at the next keystroke)
    let doc: string | null = null;
    if (this.vfs.isText(path)) {
      try {
        doc = this.vfs.readText(path);
      } catch {
        doc = null;
      }
    }
    if (doc === null) {
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
    this.known = this.vfs.readBytes(path);
    this.header(path);
    const state = EditorState.create({
      doc,
      extensions: [
        basicSetup,
        this.look.of(editorLook(theme())),
        EditorView.lineWrapping,
        this.language.of(languageFor(path)),
        this.name.of(nameFor(path)),
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
      this.known = this.vfs.readBytes(this.path);
      this.setState('saved');
    } catch (error) {
      this.setState('error', (error as Error).message);
    }
  }

  /**
   * A rename or move of `from` to `to` (the open file, or a folder it is in, or neither), made by `run`: what is typed
   * is saved first, and the open file stays open where it goes, whatever it holds.
   */
  move(from: string, to: string, run: () => void): void {
    this.flush();
    this.moving = { from, to };
    try {
      run();
    } finally {
      this.moving = null;
    }
  }

  /** The project changed underneath: reload the open file, follow it where it was moved, or close it. */
  externalChange(change: VfsChange): void {
    const path = change.type === 'reset' ? null : change.path;
    if (change.type === 'write') this.wrote(change.path);
    if (change.type === 'remove') {
      const to = this.movedTo(change.path);
      if (to) {
        this.follow(to);
        return;
      }
    }
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
    this.known = this.vfs.readBytes(this.path);
    this.applying = true;
    this.view.dispatch({ changes: { from: 0, to: this.view.state.doc.length, insert: text } });
    this.applying = false;
  }

  /** A file written besides the open one, kept until this task is over (see `written`). */
  private wrote(key: string): void {
    const open = this.path ?? this.mediaPath;
    if (!open || `/${key}` === open) return;
    // (a burst of hundreds, an upload or an import, is not kept whole: a move is of one file, or a folder's)
    if (this.written.length < 2000) this.written.push(`/${key}`);
    this.writtenTimer ??= setTimeout(() => {
      this.written = [];
      this.writtenTimer = null;
    }, 0);
  }

  /**
   * Where the open file went, as `key` is removed: null when it was not moved (it was deleted, or `key` is another
   * file). A move the tree makes says where; the agent's or the terminal's is found by the copy written just before,
   * which holds what the file held (an empty file only under its own name: any new file is empty).
   */
  private movedTo(key: string): string | null {
    const open = this.path ?? this.mediaPath;
    if (!open || `/${key}` !== open) return null;
    if (this.moving) {
      const { from, to } = this.moving;
      if (open !== from && !open.startsWith(`${from}/`)) return null;
      const next = `${to}${open.slice(from.length)}`;
      return this.vfs.stat(next)?.type === 'file' ? next : null;
    }
    const known = this.known;
    if (!known) return null;
    const leaf = (p: string) => p.slice(p.lastIndexOf('/') + 1);
    const copies = this.written.filter((p) => {
      try {
        return this.vfs.stat(p)?.type === 'file' && sameBytes(this.vfs.readBytes(p), known);
      } catch {
        return false;
      }
    });
    return copies.find((p) => leaf(p) === leaf(open)) ?? (known.byteLength ? copies[0] ?? null : null);
  }

  /** The open file was renamed or moved to `to`: it stays open there, as it was (what is typed is saved there). */
  private follow(to: string): void {
    if (this.mediaPath) {
      this.showMedia(to);
      this.onMove(to);
      return;
    }
    this.path = to;
    this.header(to, this.conflict ? 'conflict' : this.dirty ? 'saving' : 'clean');
    this.view?.dispatch({ effects: [this.language.reconfigure(languageFor(to)), this.name.reconfigure(nameFor(to))] });
    this.onMove(to);
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
    this.known = this.vfs.exists(this.path) ? this.vfs.readBytes(this.path) : null;
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
    this.keepMine();
    this.conflict = false;
    this.conflictBar.hidden = true;
    this.dropMedia();
    this.view?.destroy();
    this.view = null;
    this.path = null;
    this.dirty = false;
    this.known = null;
    this.showEmpty();
  }
}
