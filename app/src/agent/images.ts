/**
 * Images for the model's eyes. A model sees an image at a limited size (about
 * 1–1.5 megapixels, long side ~1568 px, for most vision models), so a large
 * image is shown whole but scaled down, and "zooming" means cropping a region
 * of the ORIGINAL pixels and scaling that to the same budget: a smaller
 * region shows more real detail, never an upscaled blur of a thumbnail.
 */

export interface ImagePart {
  mediaType: 'image/png' | 'image/jpeg' | 'image/webp' | 'image/gif';
  /** Base64, no data: prefix. */
  data: string;
  /** Where it came from, for the transcript. */
  label?: string;
}

export interface ImageView {
  image: ImagePart;
  /** The original image's size in pixels. */
  width: number;
  height: number;
  /** The region shown, in original pixels. */
  region: { x: number; y: number; width: number; height: number };
  /** The size the model sees. */
  shownWidth: number;
  shownHeight: number;
}

const IMAGE_EXT: Record<string, string> = {
  png: 'image/png', jpg: 'image/jpeg', jpeg: 'image/jpeg', gif: 'image/gif', webp: 'image/webp',
  bmp: 'image/bmp', svg: 'image/svg+xml', avif: 'image/avif', ico: 'image/x-icon',
};

export function imageMimeFor(path: string): string | null {
  return IMAGE_EXT[path.slice(path.lastIndexOf('.') + 1).toLowerCase()] ?? null;
}

export const DEFAULT_VIEW_SIZE = 1024;
export const MAX_VIEW_SIZE = 2048;

async function decode(bytes: Uint8Array, mime: string): Promise<ImageBitmap> {
  const blob = new Blob([bytes as BlobPart], { type: mime });
  if (mime !== 'image/svg+xml') return createImageBitmap(blob);
  // SVG decodes through an <img>: createImageBitmap refuses vector images.
  const url = URL.createObjectURL(blob);
  try {
    const img = new Image();
    img.src = url;
    await img.decode();
    const w = img.naturalWidth || 1024;
    const h = img.naturalHeight || 1024;
    return await createImageBitmap(img, { resizeWidth: w, resizeHeight: h });
  } finally {
    URL.revokeObjectURL(url);
  }
}

function toBase64(bytes: Uint8Array): string {
  let binary = '';
  for (let i = 0; i < bytes.length; i += 0x8000) binary += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return btoa(binary);
}

/** Image size without keeping the decoded pixels around. */
export async function imageSize(bytes: Uint8Array, mime: string): Promise<{ width: number; height: number }> {
  const bitmap = await decode(bytes, mime);
  const size = { width: bitmap.width, height: bitmap.height };
  bitmap.close();
  return size;
}

export interface ViewOptions {
  /** Region in original pixels; defaults to the whole image. */
  x?: number;
  y?: number;
  width?: number;
  height?: number;
  /** The longest side the model sees (default 1024, at most 2048). */
  maxSize?: number;
  /** Draw a labelled coordinate grid (in original pixels) to aim the next zoom. */
  grid?: boolean;
  label?: string;
}

/** Crop (in original pixels), scale to the budget, encode for the model. */
export async function viewImage(bytes: Uint8Array, mime: string, options: ViewOptions = {}): Promise<ImageView> {
  const bitmap = await decode(bytes, mime);
  try {
    const W = bitmap.width;
    const H = bitmap.height;
    const clamp = (v: number, lo: number, hi: number) => Math.min(Math.max(v, lo), hi);
    const x = clamp(Math.round(options.x ?? 0), 0, W - 1);
    const y = clamp(Math.round(options.y ?? 0), 0, H - 1);
    const w = clamp(Math.round(options.width ?? W - x), 1, W - x);
    const h = clamp(Math.round(options.height ?? H - y), 1, H - y);
    const budget = clamp(Math.round(options.maxSize ?? DEFAULT_VIEW_SIZE), 64, MAX_VIEW_SIZE);
    // Never upscale beyond 4x: past that, a zoom shows pixels, not detail.
    const scale = Math.min(budget / Math.max(w, h), 4);
    const outW = Math.max(1, Math.round(w * scale));
    const outH = Math.max(1, Math.round(h * scale));
    const canvas = new OffscreenCanvas(outW, outH);
    const ctx = canvas.getContext('2d')!;
    ctx.imageSmoothingEnabled = scale < 1;
    ctx.imageSmoothingQuality = 'high';
    // A transparent image reads better on white than on black.
    if (mime !== 'image/jpeg') {
      ctx.fillStyle = '#ffffff';
      ctx.fillRect(0, 0, outW, outH);
    }
    ctx.drawImage(bitmap, x, y, w, h, 0, 0, outW, outH);
    if (options.grid) drawGrid(ctx, { x, y, w, h }, scale, outW, outH);
    // Photos as JPEG (smaller), everything else as PNG (crisp text and lines).
    const photo = mime === 'image/jpeg' || mime === 'image/webp' || mime === 'image/avif';
    const blob = await canvas.convertToBlob(photo ? { type: 'image/jpeg', quality: 0.88 } : { type: 'image/png' });
    const data = toBase64(new Uint8Array(await blob.arrayBuffer()));
    return {
      image: { mediaType: photo ? 'image/jpeg' : 'image/png', data, label: options.label },
      width: W,
      height: H,
      region: { x, y, width: w, height: h },
      shownWidth: outW,
      shownHeight: outH,
    };
  } finally {
    bitmap.close();
  }
}

function drawGrid(ctx: OffscreenCanvasRenderingContext2D, r: { x: number; y: number; w: number; h: number }, scale: number, outW: number, outH: number): void {
  // About eight lines across, on round numbers of original pixels.
  const raw = Math.max(r.w, r.h) / 8;
  const magnitude = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 5, 10].map((m) => m * magnitude).find((s) => s >= raw) ?? raw;
  ctx.save();
  ctx.lineWidth = 1;
  ctx.font = `${Math.max(10, Math.round(outW / 90))}px sans-serif`;
  ctx.textBaseline = 'top';
  const label = (text: string, px: number, py: number) => {
    const m = ctx.measureText(text);
    ctx.fillStyle = 'rgba(0,0,0,0.65)';
    ctx.fillRect(px, py, m.width + 4, parseInt(ctx.font, 10) + 3);
    ctx.fillStyle = '#ffeb3b';
    ctx.fillText(text, px + 2, py + 1);
  };
  ctx.strokeStyle = 'rgba(255,0,128,0.55)';
  for (let gx = Math.ceil(r.x / step) * step; gx < r.x + r.w; gx += step) {
    const px = Math.round((gx - r.x) * scale) + 0.5;
    ctx.beginPath();
    ctx.moveTo(px, 0);
    ctx.lineTo(px, outH);
    ctx.stroke();
    label(`x=${gx}`, px + 2, 2);
  }
  for (let gy = Math.ceil(r.y / step) * step; gy < r.y + r.h; gy += step) {
    const py = Math.round((gy - r.y) * scale) + 0.5;
    ctx.beginPath();
    ctx.moveTo(0, py);
    ctx.lineTo(outW, py);
    ctx.stroke();
    // The first row's label would sit on the first column's: move it down.
    label(`y=${gy}`, 2, py < parseInt(ctx.font, 10) + 6 ? py + parseInt(ctx.font, 10) + 6 : py + 2);
  }
  ctx.restore();
}

/** An uploaded image, prepared to go straight to the model with a message. */
export async function imageForMessage(bytes: Uint8Array, mime: string, label: string): Promise<{ part: ImagePart; width: number; height: number }> {
  const view = await viewImage(bytes, mime, { maxSize: 1568, label });
  return { part: view.image, width: view.width, height: view.height };
}
