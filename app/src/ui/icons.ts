/**
 * The app's icons: a small set of line drawings on a 24-unit grid, drawn in
 * the text's colour (inline SVG, so they follow the theme and need nothing
 * loaded). In the manner of Lucide's.
 */
type Shape =
  | ['path', string]
  | ['circle', number, number, number]
  | ['line', number, number, number, number]
  | ['polyline', string]
  | ['polygon', string]
  | ['rect', number, number, number, number, number?];

const FILE = 'M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z';
const FOLDER = 'M4 20h16a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2h-7.9a2 2 0 0 1-1.7-.9l-.8-1.2A2 2 0 0 0 7.9 3H4a2 2 0 0 0-2 2v13a2 2 0 0 0 2 2z';
const PHONE = 'M22 16.9v3a2 2 0 0 1-2.2 2 19.8 19.8 0 0 1-8.6-3.1 19.5 19.5 0 0 1-6-6A19.8 19.8 0 0 1 2.1 4.2 2 2 0 0 1 4.1 2h3a2 2 0 0 1 2 1.7c.1 1 .4 1.9.7 2.8a2 2 0 0 1-.5 2.1L8.1 9.9a16 16 0 0 0 6 6l1.3-1.3a2 2 0 0 1 2.1-.4c.9.3 1.8.6 2.8.7a2 2 0 0 1 1.7 2z';
const CUBE = 'M21 16V8a2 2 0 0 0-1-1.7l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.7l7 4a2 2 0 0 0 2 0l7-4a2 2 0 0 0 1-1.7z';

const ICONS: Record<string, { shapes: Shape[]; fill?: boolean }> = {
  search: { shapes: [['circle', 11, 11, 7], ['line', 21, 21, 16.7, 16.7]] },
  'chevron-down': { shapes: [['polyline', '6 9 12 15 18 9']] },
  'chevron-right': { shapes: [['polyline', '9 6 15 12 9 18']] },
  'chevron-up': { shapes: [['polyline', '6 15 12 9 18 15']] },
  check: { shapes: [['polyline', '20 6 9 17 4 12']] },
  x: { shapes: [['line', 18, 6, 6, 18], ['line', 6, 6, 18, 18]] },
  more: { shapes: [['circle', 5, 12, 1.4], ['circle', 12, 12, 1.4], ['circle', 19, 12, 1.4]] },
  phone: { shapes: [['path', PHONE]] },
  'phone-off': { shapes: [['path', PHONE], ['line', 22, 2, 2, 22]] },
  message: { shapes: [['path', 'M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z']] },
  flow: { shapes: [['rect', 3, 3, 6, 6, 1.5], ['rect', 15, 15, 6, 6, 1.5], ['path', 'M9 6h4a2 2 0 0 1 2 2v7']] },
  compass: { shapes: [['circle', 12, 12, 10], ['polygon', '16.2 7.8 14.1 14.1 7.8 16.2 9.9 9.9 16.2 7.8']] },
  folder: { shapes: [['path', FOLDER]] },
  'folder-open': { shapes: [['path', 'M6 14l1.5-2.9A2 2 0 0 1 9.2 10H20a2 2 0 0 1 1.9 2.5l-1.5 6a2 2 0 0 1-2 1.5H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h3.9a2 2 0 0 1 1.7.9l.8 1.2a2 2 0 0 0 1.7.9H18a2 2 0 0 1 2 2v2']] },
  'folder-plus': { shapes: [['path', FOLDER], ['line', 12, 10, 12, 16], ['line', 9, 13, 15, 13]] },
  file: { shapes: [['path', FILE], ['polyline', '14 2 14 8 20 8']] },
  'file-plus': { shapes: [['path', FILE], ['polyline', '14 2 14 8 20 8'], ['line', 12, 18, 12, 12], ['line', 9, 15, 15, 15]] },
  'file-text': { shapes: [['path', FILE], ['polyline', '14 2 14 8 20 8'], ['line', 16, 13, 8, 13], ['line', 16, 17, 8, 17], ['line', 10, 9, 8, 9]] },
  collapse: { shapes: [['polyline', '7 20 12 15 17 20'], ['polyline', '7 4 12 9 17 4']] },
  paperclip: { shapes: [['path', 'M21.4 11.1l-9.2 9.2a6 6 0 0 1-8.5-8.5l8.6-8.6a4 4 0 1 1 5.7 5.7l-8.6 8.6a2 2 0 0 1-2.8-2.8l8.5-8.5']] },
  'arrow-up': { shapes: [['line', 12, 19, 12, 5], ['polyline', '5 12 12 5 19 12']] },
  stop: { shapes: [['rect', 6, 6, 12, 12, 2]], fill: true },
  copy: { shapes: [['rect', 9, 9, 13, 13, 2], ['path', 'M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1']] },
  terminal: { shapes: [['polyline', '4 17 10 11 4 5'], ['line', 12, 19, 20, 19]] },
  code: { shapes: [['polyline', '16 18 22 12 16 6'], ['polyline', '8 6 2 12 8 18']] },
  globe: { shapes: [['circle', 12, 12, 10], ['line', 2, 12, 22, 12], ['path', 'M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z']] },
  image: { shapes: [['rect', 3, 3, 18, 18, 2], ['circle', 9, 9, 2], ['path', 'M21 15l-3.1-3.1a2 2 0 0 0-2.8 0L6 21']] },
  video: { shapes: [['rect', 2, 6, 14, 12, 2], ['path', 'M22 8l-6 4 6 4V8z']] },
  music: { shapes: [['path', 'M9 18V5l12-2v13'], ['circle', 6, 18, 3], ['circle', 18, 16, 3]] },
  cube: { shapes: [['path', CUBE], ['polyline', '3.3 7 12 12 20.7 7'], ['line', 12, 22, 12, 12]] },
  package: { shapes: [['path', CUBE], ['polyline', '3.3 7 12 12 20.7 7'], ['line', 12, 22, 12, 12], ['line', 7.5, 4.3, 16.5, 9.4]] },
  app: { shapes: [['rect', 3, 3, 7, 7, 1.5], ['rect', 14, 3, 7, 7, 1.5], ['rect', 14, 14, 7, 7, 1.5], ['rect', 3, 14, 7, 7, 1.5]] },
  calendar: { shapes: [['rect', 3, 4, 18, 18, 2], ['line', 16, 2, 16, 6], ['line', 8, 2, 8, 6], ['line', 3, 10, 21, 10]] },
  user: { shapes: [['circle', 12, 8, 4], ['path', 'M20 21a8 8 0 0 0-16 0']] },
  users: { shapes: [['circle', 9, 8, 4], ['path', 'M17 21a8 8 0 0 0-16 0'], ['path', 'M16 4.1a4 4 0 0 1 0 7.8'], ['path', 'M23 21a8 8 0 0 0-5-7.4']] },
  list: { shapes: [['line', 9, 6, 21, 6], ['line', 9, 12, 21, 12], ['line', 9, 18, 21, 18], ['circle', 4, 6, 1], ['circle', 4, 12, 1], ['circle', 4, 18, 1]] },
  wrench: { shapes: [['path', 'M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.8-3.8a6 6 0 0 1-7.9 7.9l-6.9 6.9a2.1 2.1 0 0 1-3-3l6.9-6.9a6 6 0 0 1 7.9-7.9l-3.8 3.8z']] },
  sparkle: { shapes: [['path', 'M12 3l1.9 5.1L19 10l-5.1 1.9L12 17l-1.9-5.1L5 10l5.1-1.9z'], ['path', 'M19 16l.7 1.8 1.8.7-1.8.7L19 21l-.7-1.8-1.8-.7 1.8-.7z']] },
  pencil: { shapes: [['path', 'M17 3a2.8 2.8 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5z']] },
  trash: { shapes: [['polyline', '3 6 5 6 21 6'], ['path', 'M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6'], ['path', 'M9 6V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2']] },
  glasses: { shapes: [['circle', 6, 15, 4], ['circle', 18, 15, 4], ['path', 'M14 15a2 2 0 0 0-4 0'], ['path', 'M2.5 13L5 7c.7-1.3 1.4-2 3-2'], ['path', 'M21.5 13L19 7c-.7-1.3-1.5-2-3-2']] },
  braces: { shapes: [['path', 'M8 3H7a2 2 0 0 0-2 2v5a2 2 0 0 1-2 2 2 2 0 0 1 2 2v5a2 2 0 0 0 2 2h1'], ['path', 'M16 21h1a2 2 0 0 0 2-2v-5a2 2 0 0 1 2-2 2 2 0 0 1-2-2V5a2 2 0 0 0-2-2h-1']] },
  hash: { shapes: [['line', 4, 9, 20, 9], ['line', 4, 15, 20, 15], ['line', 10, 3, 8, 21], ['line', 16, 3, 14, 21]] },
  table: { shapes: [['rect', 3, 3, 18, 18, 2], ['line', 3, 9, 21, 9], ['line', 3, 15, 21, 15], ['line', 12, 3, 12, 21]] },
  send: { shapes: [['path', 'M22 2L15 22l-4-9-9-4z'], ['line', 22, 2, 11, 13]] },
  layers: { shapes: [['path', 'M12.8 2.2a2 2 0 0 0-1.6 0L2.6 6.1a1 1 0 0 0 0 1.8l8.6 3.9a2 2 0 0 0 1.6 0l8.6-3.9a1 1 0 0 0 0-1.8z'], ['path', 'M22 17.6l-9.2 4.2a2 2 0 0 1-1.6 0L2 17.6'], ['path', 'M22 12.6l-9.2 4.2a2 2 0 0 1-1.6 0L2 12.6']] },
  alert: { shapes: [['circle', 12, 12, 10], ['line', 12, 8, 12, 12.5], ['line', 12, 16, 12.01, 16]] },
  info: { shapes: [['circle', 12, 12, 10], ['line', 12, 16, 12, 11.5], ['line', 12, 8, 12.01, 8]] },
  clock: { shapes: [['circle', 12, 12, 10], ['polyline', '12 6 12 12 16 14']] },
  diamond: { shapes: [['polygon', '12 2.5 21.5 12 12 21.5 2.5 12']], fill: true },
  eye: { shapes: [['path', 'M2 12s3.6-7 10-7 10 7 10 7-3.6 7-10 7S2 12 2 12z'], ['circle', 12, 12, 3]] },
  mic: { shapes: [['rect', 9, 2, 6, 12, 3], ['path', 'M19 10v1a7 7 0 0 1-14 0v-1'], ['line', 12, 18, 12, 22]] },
  refresh: { shapes: [['path', 'M3 12a9 9 0 0 1 15.5-6.3L21 8'], ['polyline', '21 3 21 8 16 8'], ['path', 'M21 12a9 9 0 0 1-15.5 6.3L3 16'], ['polyline', '3 21 3 16 8 16']] },
  flag: { shapes: [['path', 'M4 15s1-1 4-1 5 2 8 2 4-1 4-1V3s-1 1-4 1-5-2-8-2-4 1-4 1z'], ['line', 4, 22, 4, 15]] },
  'arrow-down': { shapes: [['line', 12, 5, 12, 19], ['polyline', '19 12 12 19 5 12']] },
  activity: { shapes: [['polyline', '22 12 18 12 15 21 9 3 6 12 2 12']] },
  settings: { shapes: [['line', 4, 21, 4, 14], ['line', 4, 10, 4, 3], ['line', 12, 21, 12, 12], ['line', 12, 8, 12, 3], ['line', 20, 21, 20, 16], ['line', 20, 12, 20, 3], ['line', 1, 14, 7, 14], ['line', 9, 8, 15, 8], ['line', 17, 16, 23, 16]] },
  cpu: { shapes: [['rect', 4, 4, 16, 16, 2], ['rect', 9, 9, 6, 6, 1], ['line', 9, 1, 9, 4], ['line', 15, 1, 15, 4], ['line', 9, 20, 9, 23], ['line', 15, 20, 15, 23], ['line', 20, 9, 23, 9], ['line', 20, 14, 23, 14], ['line', 1, 9, 4, 9], ['line', 1, 14, 4, 14]] },
  power: { shapes: [['path', 'M18.4 6.6a9 9 0 1 1-12.8 0'], ['line', 12, 2, 12, 12]] },
  server: { shapes: [['rect', 2, 2, 20, 8, 2], ['rect', 2, 14, 20, 8, 2], ['line', 6, 6, 6.01, 6], ['line', 6, 18, 6.01, 18]] },
  plug: { shapes: [['path', 'M12 22v-5'], ['path', 'M9 8V2'], ['path', 'M15 8V2'], ['path', 'M18 8v5a4 4 0 0 1-4 4h-4a4 4 0 0 1-4-4V8z']] },
  link: { shapes: [['path', 'M10 13a5 5 0 0 0 7.5.5l3-3a5 5 0 0 0-7.1-7.1l-1.7 1.7'], ['path', 'M14 11a5 5 0 0 0-7.5-.5l-3 3a5 5 0 0 0 7.1 7.1l1.7-1.7']] },
};

const NS = 'http://www.w3.org/2000/svg';

/** An icon by name, `aria-hidden` (a control says what it does in words). An unknown name draws a dot. */
export function icon(name: string, className = ''): SVGSVGElement {
  const def = ICONS[name] ?? { shapes: [['circle', 12, 12, 3]] as Shape[], fill: true };
  const svg = document.createElementNS(NS, 'svg');
  svg.setAttribute('viewBox', '0 0 24 24');
  svg.setAttribute('class', `icon icon-${name}${className ? ` ${className}` : ''}`);
  svg.setAttribute('aria-hidden', 'true');
  svg.setAttribute('focusable', 'false');
  svg.setAttribute('fill', def.fill ? 'currentColor' : 'none');
  svg.setAttribute('stroke', def.fill ? 'none' : 'currentColor');
  svg.setAttribute('stroke-width', '2');
  svg.setAttribute('stroke-linecap', 'round');
  svg.setAttribute('stroke-linejoin', 'round');
  for (const shape of def.shapes) {
    const [tag, ...args] = shape;
    const el = document.createElementNS(NS, tag);
    const set = (attrs: Record<string, string | number | undefined>) => {
      for (const [k, v] of Object.entries(attrs)) if (v !== undefined) el.setAttribute(k, String(v));
    };
    if (tag === 'path') set({ d: args[0] as string });
    else if (tag === 'circle') set({ cx: args[0], cy: args[1], r: args[2], ...(def.fill ? {} : ICONS[name] && (args[2] as number) <= 1.5 ? { fill: 'currentColor' } : {}) });
    else if (tag === 'line') set({ x1: args[0], y1: args[1], x2: args[2], y2: args[3] });
    else if (tag === 'polyline' || tag === 'polygon') set({ points: args[0] as string });
    else if (tag === 'rect') set({ x: args[0], y: args[1], width: args[2], height: args[3], rx: args[4] });
    svg.append(el);
  }
  return svg;
}

/** Whether there is an icon by this name. */
export function hasIcon(name: string): boolean {
  return name in ICONS;
}

/** A file's icon and its colour's name, by its extension (folders by their state). */
export function fileIcon(name: string, dir = false, open = false): { icon: string; tone: string } {
  if (dir) return { icon: open ? 'folder-open' : 'folder', tone: 'folder' };
  const ext = name.includes('.') ? name.slice(name.lastIndexOf('.') + 1).toLowerCase() : '';
  switch (ext) {
    case 'js': case 'mjs': case 'cjs': case 'jsx': return { icon: 'code', tone: 'js' };
    case 'ts': case 'tsx': case 'mts': return { icon: 'code', tone: 'ts' };
    case 'py': return { icon: 'code', tone: 'py' };
    case 'html': case 'htm': case 'xml': return { icon: 'code', tone: 'html' };
    case 'css': case 'scss': return { icon: 'hash', tone: 'css' };
    case 'json': case 'webmanifest': return { icon: 'braces', tone: 'json' };
    case 'md': case 'markdown': case 'txt': return { icon: 'file-text', tone: 'text' };
    case 'csv': case 'tsv': return { icon: 'table', tone: 'data' };
    case 'png': case 'jpg': case 'jpeg': case 'gif': case 'webp': case 'svg': case 'avif': case 'bmp': case 'ico': return { icon: 'image', tone: 'image' };
    case 'mp4': case 'webm': case 'mov': case 'mkv': return { icon: 'video', tone: 'media' };
    case 'mp3': case 'wav': case 'ogg': case 'flac': case 'm4a': return { icon: 'music', tone: 'media' };
    case 'glb': case 'gltf': case 'obj': case 'stl': return { icon: 'cube', tone: 'model' };
    case 'softn': case 'zip': return { icon: 'package', tone: 'package' };
    case 'ui': case 'logic': return { icon: 'app', tone: 'softn' };
    case 'sh': return { icon: 'terminal', tone: 'text' };
    default: return { icon: 'file', tone: 'plain' };
  }
}
