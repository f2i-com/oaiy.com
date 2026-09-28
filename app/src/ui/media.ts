/**
 * Images, audio and video from the project, shown as themselves: in the file
 * viewer, in the chat (attachments, and files the agent presents), with the
 * browser's own players. Each element holds a blob URL that the owner revokes
 * when the element goes away.
 */
import { h } from './dom';

export type MediaKind = 'image' | 'audio' | 'video';

const MEDIA: Record<string, [MediaKind, string]> = {
  png: ['image', 'image/png'], jpg: ['image', 'image/jpeg'], jpeg: ['image', 'image/jpeg'], gif: ['image', 'image/gif'],
  webp: ['image', 'image/webp'], bmp: ['image', 'image/bmp'], svg: ['image', 'image/svg+xml'], avif: ['image', 'image/avif'], ico: ['image', 'image/x-icon'],
  mp3: ['audio', 'audio/mpeg'], wav: ['audio', 'audio/wav'], ogg: ['audio', 'audio/ogg'], oga: ['audio', 'audio/ogg'], opus: ['audio', 'audio/ogg'],
  m4a: ['audio', 'audio/mp4'], aac: ['audio', 'audio/aac'], flac: ['audio', 'audio/flac'], weba: ['audio', 'audio/webm'],
  mp4: ['video', 'video/mp4'], m4v: ['video', 'video/mp4'], webm: ['video', 'video/webm'], ogv: ['video', 'video/ogg'], mov: ['video', 'video/quicktime'],
};

function ext(path: string): string {
  return path.slice(path.lastIndexOf('.') + 1).toLowerCase();
}

export function mediaKind(path: string): MediaKind | null {
  return MEDIA[ext(path)]?.[0] ?? null;
}

export function mediaMime(path: string): string {
  return MEDIA[ext(path)]?.[1] ?? 'application/octet-stream';
}

export function formatBytes(n: number): string {
  if (n < 1024) return `${n} bytes`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}

export interface Media {
  element: HTMLElement;
  /** Revoke the blob URL. */
  dispose(): void;
}

/**
 * An element that shows or plays the file. `compact` is for the chat: a
 * thumbnail for an image, a slim player for audio and video.
 */
export function mediaElement(path: string, bytes: Uint8Array, options: { compact?: boolean; onOpen?: () => void } = {}): Media | null {
  const kind = mediaKind(path);
  if (!kind) return null;
  const url = URL.createObjectURL(new Blob([bytes as BlobPart], { type: mediaMime(path) }));
  const name = path.split('/').pop() ?? path;
  let element: HTMLElement;
  if (kind === 'image') {
    const img = h('img', { src: url, alt: name, title: options.onOpen ? `${path}: click to open` : name }) as HTMLImageElement;
    if (options.onOpen) {
      img.classList.add('clickable');
      img.addEventListener('click', options.onOpen);
    }
    element = options.compact ? h('span.media.media-thumb', img) : h('div.media.media-image', img);
  } else {
    const player = h(kind, { src: url, controls: true, preload: 'metadata', title: name }) as HTMLMediaElement;
    player.addEventListener('error', () => {
      player.replaceWith(h('span.media-error', `This browser cannot play ${name}.`));
    });
    const label = h('span.media-name', name);
    if (options.onOpen) {
      label.classList.add('clickable');
      label.title = `${path}: click to open`;
      label.addEventListener('click', options.onOpen);
    }
    element = h(`div.media.media-${kind}`, { class: options.compact ? 'compact' : '' }, label, player);
  }
  return { element, dispose: () => URL.revokeObjectURL(url) };
}
