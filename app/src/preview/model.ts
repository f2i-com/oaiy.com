/**
 * 3D models in a project: the .glb and .gltf files the preview can show, and
 * the files one of them gets. The model is drawn in the preview's model frame
 * (public/modelview/, built from scripts/modelview/viewer.js) with three.js.
 */
import type { Vfs } from '../vfs/vfs';

/** Folders whose files are nobody's model. */
const IGNORED = new Set(['node_modules', '.git', 'dist', 'build', '.cache']);
/** A .gltf's folder is sent with it (the buffers and images it names), up to this much. */
const GLTF_FOLDER_MAX = 256 * 1024 * 1024;

export const isModelPath = (path: string): boolean => /\.(glb|gltf)$/i.test(path);

/** The 3D models of the project (SoftN apps' assets included), shallowest first. */
export function findModels(vfs: Vfs): string[] {
  const models: string[] = [];
  for (const entry of vfs.walk('/', { limit: 20_000 }).entries) {
    if (entry.type !== 'file' || !isModelPath(entry.path)) continue;
    const path = entry.path.replace(/^\/+/, '');
    if (path.split('/').slice(0, -1).some((s) => IGNORED.has(s) || s.startsWith('.'))) continue;
    models.push(path);
  }
  return models.sort((a, b) => a.split('/').length - b.split('/').length || a.localeCompare(b)).slice(0, 200);
}

/** A model path as the agent or the person wrote it. */
export function resolveModel(vfs: Vfs, asked: string): { ok: true; path: string } | { ok: false; reason: string } {
  const path = asked.trim().replace(/^\/+/, '');
  if (!isModelPath(path)) return { ok: false, reason: `/${path} is not a 3D model (.glb or .gltf)` };
  if (!vfs.exists(`/${path}`)) {
    const models = findModels(vfs);
    return { ok: false, reason: `/${path} does not exist${models.length ? ` (3D models: ${models.slice(0, 12).join(', ')})` : ''}` };
  }
  return { ok: true, path };
}

/** The files a model needs, keyed by project path: a .glb holds everything; a .gltf gets its folder. */
export function modelFiles(vfs: Vfs, model: string): Record<string, Uint8Array> {
  const out: Record<string, Uint8Array> = { [model]: vfs.readBytes(`/${model}`) };
  if (!/\.gltf$/i.test(model)) return out;
  const folder = model.includes('/') ? `${model.slice(0, model.lastIndexOf('/'))}/` : '';
  let total = out[model].byteLength;
  for (const [raw, data] of vfs.files()) {
    const path = raw.replace(/^\/+/, '');
    if (path === model || (folder && !path.startsWith(folder)) || /\.(html?|ui|logic|py)$/i.test(path)) continue;
    if (total + data.byteLength > GLTF_FOLDER_MAX) continue;
    total += data.byteLength;
    out[path] = data;
  }
  return out;
}

/** A model for people: its project path. */
export const modelLabel = (path: string): string => `/${path}`;
