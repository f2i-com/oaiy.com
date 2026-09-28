/*
 * bot.computer's 3D model viewer, in the live preview's model frame
 * (public/modelview/index.html). scripts/modelview/build.mjs bundles it with
 * three.js (MIT) into one classic script, public/modelview/model-viewer.js.
 *
 * The model (a .glb, or a .gltf with the files it names) arrives from
 * bot.computer over postMessage as bytes. The viewer:
 *   - shows it fitted to the view, in soft studio light on a neutral
 *     background, and the person turns it with the mouse (orbit controls);
 *   - answers the same requests as bot-bridge.js does in the other frames:
 *     "inspect" gives the model's facts as text (triangles, vertices, size,
 *     colours and textures), and "screenshot" draws it from four sides
 *     (front, right, back, top) in a 2x2 grid, or from one angle asked for,
 *     rendered here rather than copied off the page.
 *
 * The frame is sandboxed with an opaque origin: it reaches nothing of
 * bot.computer's, and only posts messages to its parent. The model's own
 * references stay inside the files it was sent with.
 */
import {
  ACESFilmicToneMapping,
  Box3,
  CanvasTexture,
  DirectionalLight,
  GridHelper,
  HemisphereLight,
  LoadingManager,
  NeutralToneMapping,
  PerspectiveCamera,
  PMREMGenerator,
  Scene,
  SRGBColorSpace,
  Sphere,
  Vector3,
  WebGLRenderer,
} from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';

const parentWindow = window.parent;
const post = (message, transfer) => parentWindow.postMessage({ __botComputer: true, ...message }, '*', transfer || []);

// --- errors ---------------------------------------------------------------------

const problem = (message) => post({ type: 'bot:problem', level: 'error', message: String(message).slice(0, 2000), at: Date.now() });
window.addEventListener('error', (event) => problem(event.error instanceof Error ? `${event.error.name}: ${event.error.message}` : event.message));
window.addEventListener('unhandledrejection', (event) => problem(`Unhandled promise rejection: ${event.reason instanceof Error ? event.reason.message : event.reason}`));

// --- the stage ----------------------------------------------------------------------

/** How wide the camera sees, up and down, in degrees: narrow enough to keep shapes true. */
const FOV = 30;
/** The four sides a screenshot shows, as yaw and pitch in degrees (yaw 0 looks at the front, +Z; 90 at the right, +X). */
const SIDES = [
  { label: 'Front (+Z)', yaw: 0, pitch: 10 },
  { label: 'Right (+X)', yaw: 90, pitch: 10 },
  { label: 'Back (-Z)', yaw: 180, pitch: 10 },
  { label: 'Top (+Y)', yaw: 0, pitch: 90 },
];

const status = document.getElementById('status');
const hint = document.getElementById('hint');
const say = (text, error = false) => {
  status.textContent = text;
  status.hidden = !text;
  status.classList.toggle('error', error);
};

const renderer = new WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.outputColorSpace = SRGBColorSpace;
// Khronos' neutral tone mapping keeps a model's base colours as they are
// (ACES would shift them), which is what a look at the model is for.
renderer.toneMapping = typeof NeutralToneMapping === 'number' ? NeutralToneMapping : ACESFilmicToneMapping;
document.body.prepend(renderer.domElement);

const scene = new Scene();
scene.background = backdrop();
// Soft studio light: a lit room reflected in the model, a sky-and-ground fill, and one key light.
const pmrem = new PMREMGenerator(renderer);
scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
scene.environmentIntensity = 0.8;
scene.add(new HemisphereLight(0xffffff, 0x8a8f99, 0.6));
const key = new DirectionalLight(0xffffff, 1.4);
scene.add(key);
scene.add(key.target);

const camera = new PerspectiveCamera(FOV, window.innerWidth / window.innerHeight, 0.01, 100);
const controls = new OrbitControls(camera, renderer.domElement);
controls.addEventListener('change', () => draw());
controls.addEventListener('start', () => (hint.hidden = true));

/** A light, vertical wash of grey: neutral behind any colour, and light enough for dark models. */
function backdrop() {
  const canvas = document.createElement('canvas');
  canvas.width = 2;
  canvas.height = 256;
  const ctx = canvas.getContext('2d');
  const gradient = ctx.createLinearGradient(0, 0, 0, 256);
  gradient.addColorStop(0, '#f1f2f4');
  gradient.addColorStop(1, '#c6cad1');
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, 2, 256);
  const texture = new CanvasTexture(canvas);
  texture.colorSpace = SRGBColorSpace;
  return texture;
}

/** What is shown: the model, where it sits and how far to stand back from it. */
let shown = null;

function draw() {
  renderer.setScissorTest(false);
  renderer.setViewport(0, 0, window.innerWidth, window.innerHeight);
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.render(scene, camera);
}

window.addEventListener('resize', () => {
  renderer.setSize(window.innerWidth, window.innerHeight);
  draw();
});

/**
 * Puts `cam` around the model, looking at its centre from `yaw` and `pitch`
 * (degrees), as close as it can stand with every corner of the model's box in
 * view (and a margin), so each view shows the model as large as it fits.
 */
function aim(cam, yaw, pitch, aspect) {
  const { centre, radius, box } = shown;
  const y = (yaw * Math.PI) / 180;
  const p = (Math.max(-90, Math.min(90, pitch)) * Math.PI) / 180;
  // Straight down (or up) the view's "up" is -Z (or +Z): the front is at the bottom of a top view.
  if (Math.abs(pitch) >= 89.5) cam.up.set(0, 0, pitch > 0 ? -1 : 1);
  else cam.up.set(0, 1, 0);
  const back = new Vector3(Math.sin(y) * Math.cos(p), Math.sin(p), Math.cos(y) * Math.cos(p));
  const right = new Vector3().crossVectors(cam.up, back).normalize();
  const up = new Vector3().crossVectors(back, right);
  const tanV = Math.tan((FOV * Math.PI) / 360) / 1.08;
  const tanH = tanV * aspect;
  let distance = radius * 0.5;
  for (let i = 0; i < 8; i++) {
    const corner = new Vector3(i & 1 ? box.max.x : box.min.x, i & 2 ? box.max.y : box.min.y, i & 4 ? box.max.z : box.min.z).sub(centre);
    const depth = corner.dot(back);
    distance = Math.max(distance, depth + Math.abs(corner.dot(right)) / tanH, depth + Math.abs(corner.dot(up)) / tanV);
  }
  cam.aspect = aspect;
  cam.near = Math.max(distance / 100, distance - radius * 4);
  cam.far = distance + radius * 4;
  cam.position.copy(centre).addScaledVector(back, distance);
  cam.lookAt(centre);
  cam.updateProjectionMatrix();
}

// --- the model ----------------------------------------------------------------------

const MIME = { png: 'image/png', jpg: 'image/jpeg', jpeg: 'image/jpeg', webp: 'image/webp', ktx2: 'image/ktx2', bin: 'application/octet-stream' };
const dirOf = (path) => (path.includes('/') ? path.slice(0, path.lastIndexOf('/')) : '');
/** A reference in a .gltf (relative to its folder) as a project path. */
const resolve = (ref, base) => {
  let path = String(ref).replace(/[?#].*$/, '');
  try {
    path = decodeURIComponent(path);
  } catch {
    /* as written */
  }
  const parts = base ? base.split('/') : [];
  for (const part of path.split('/')) {
    if (!part || part === '.') continue;
    if (part === '..') parts.pop();
    else parts.push(part);
  }
  return parts.join('/');
};

/** Reads the model from its bytes (and, for a .gltf, the files it names). */
function parse(path, files) {
  const refused = [];
  const urls = [];
  const manager = new LoadingManager();
  // A .gltf's buffers and images come from the files it was sent with; nothing else loads.
  manager.setURLModifier((url) => {
    if (/^(data|blob):/i.test(url)) return url;
    const wanted = resolve(url.replace(/^[a-z]+:\/\/[^/]*\//i, ''), dirOf(path));
    const bytes = files[wanted];
    if (!bytes) {
      refused.push(url);
      return 'data:,';
    }
    const made = URL.createObjectURL(new Blob([bytes], { type: MIME[wanted.split('.').pop().toLowerCase()] || 'application/octet-stream' }));
    urls.push(made);
    return made;
  });
  const loader = new GLTFLoader(manager);
  const bytes = files[path];
  const data = /\.gltf$/i.test(path) ? new TextDecoder().decode(bytes) : bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
  return new Promise((done, fail) => {
    loader.parse(
      data,
      '',
      (gltf) => done({ gltf, refused }),
      (error) => fail(new Error(`${error && error.message ? error.message : error}${refused.length ? ` (missing: ${refused.join(', ')})` : ''}`)),
    );
  }).finally(() => setTimeout(() => urls.forEach((u) => URL.revokeObjectURL(u)), 10_000));
}

const TEXTURE_SLOTS = [
  ['map', 'base colour'],
  ['normalMap', 'normal'],
  ['roughnessMap', 'roughness'],
  ['metalnessMap', 'metalness'],
  ['aoMap', 'occlusion'],
  ['emissiveMap', 'emissive'],
];

/** The model's facts, for the agent: what it is made of and how big it is. */
function measure(root, gltf, path, bytes) {
  let meshes = 0;
  let triangles = 0;
  let vertices = 0;
  let coloured = 0;
  let uvs = 0;
  const materials = new Set();
  const textures = new Set();
  root.traverse((node) => {
    if (!node.isMesh || !node.geometry) return;
    meshes++;
    const geometry = node.geometry;
    const position = geometry.attributes.position;
    if (!position) return;
    vertices += position.count;
    triangles += Math.floor((geometry.index ? geometry.index.count : position.count) / 3);
    if (geometry.attributes.color) coloured++;
    if (geometry.attributes.uv) uvs++;
    for (const material of Array.isArray(node.material) ? node.material : [node.material]) {
      if (!material) continue;
      materials.add(material);
      for (const [slot, name] of TEXTURE_SLOTS) if (material[slot]) textures.add(name);
    }
  });
  const box = new Box3().setFromObject(root);
  const size = box.getSize(new Vector3());
  const centre = box.getCenter(new Vector3());
  const fits = box.min.x >= -0.501 && box.min.y >= -0.501 && box.min.z >= -0.501 && box.max.x <= 0.501 && box.max.y <= 0.501 && box.max.z <= 0.501;
  const n = (v) => v.toLocaleString('en-US');
  const f = (v) => (Math.abs(v) < 0.005 ? '0.00' : v.toFixed(2));
  const kinds = [...materials].map((m) => `${m.type.replace(/^Mesh|Material$/g, '')}${m.side === 2 ? ', double-sided' : ''}`);
  const lines = [
    `3D model /${path} (${/\.gltf$/i.test(path) ? 'glTF' : 'glTF binary'}, ${bytes >= 1e6 ? `${(bytes / 1e6).toFixed(1)} MB` : `${Math.max(1, Math.round(bytes / 1e3))} KB`})`,
    `Mesh: ${n(meshes)} mesh${meshes === 1 ? '' : 'es'}, ${n(triangles)} triangles, ${n(vertices)} vertices`,
    `Size: ${f(size.x)} × ${f(size.y)} × ${f(size.z)} (x × y × z), centred at (${f(centre.x)}, ${f(centre.y)}, ${f(centre.z)}); ${fits ? 'it fits the unit cube (-0.5 to 0.5 on each axis)' : `from (${f(box.min.x)}, ${f(box.min.y)}, ${f(box.min.z)}) to (${f(box.max.x)}, ${f(box.max.y)}, ${f(box.max.z)})`}`,
    `Colour: ${coloured ? `vertex colours${coloured < meshes ? ` (on ${coloured} of ${meshes} meshes)` : ''}` : 'no vertex colours'}; ${textures.size ? `textures: ${[...textures].join(', ')}` : 'no textures'}${uvs ? '' : ', no texture coordinates'}`,
    `Materials: ${materials.size} (${[...new Set(kinds)].join('; ') || 'none'})`,
    `Animations: ${gltf.animations.length ? gltf.animations.map((a) => a.name || 'unnamed').join(', ') : 'none'}`,
  ];
  return { text: lines.join('\n'), centre, radius: box.getBoundingSphere(new Sphere()).radius || 0.5, box };
}

/** Shows the model, fitted to the view from a three-quarter angle, over a faint floor grid. */
function show(root, facts) {
  if (shown) scene.remove(shown.root, shown.grid);
  const { centre, radius, box } = facts;
  const extent = Math.max(box.max.x - box.min.x, box.max.z - box.min.z, radius);
  const grid = new GridHelper(extent * 2.5, 10, 0x9aa0aa, 0xb4b9c1);
  grid.position.set(centre.x, box.min.y, centre.z);
  grid.material.transparent = true;
  grid.material.opacity = 0.55;
  scene.add(root, grid);
  key.position.set(centre.x + radius * 3, centre.y + radius * 5, centre.z + radius * 4);
  key.target.position.copy(centre);
  shown = { root, grid, centre, radius, box, text: facts.text };
  aim(camera, 35, 20, window.innerWidth / window.innerHeight);
  controls.target.copy(centre);
  controls.minDistance = radius * 0.2;
  controls.maxDistance = radius * 20;
  controls.update();
  draw();
}

// --- screenshots ----------------------------------------------------------------------

/**
 * Four sides of the model in a 2x2 grid at the frame's size (or one view,
 * from `yaw` and `pitch`, at the whole size), labelled, as a PNG. All drawn
 * in one go, and the person's own view put back before the page repaints.
 */
async function shoot(view) {
  if (!shown) throw new Error(failed || 'no model is shown');
  const width = window.innerWidth;
  const height = window.innerHeight;
  const single = view && (typeof view.yaw === 'number' || typeof view.pitch === 'number');
  const views = single ? [{ label: `yaw ${view.yaw ?? 0}°, pitch ${view.pitch ?? 0}°`, yaw: view.yaw ?? 0, pitch: view.pitch ?? 0, x: 0, y: 0, w: width, h: height }] : quadrants(width, height);
  const out = document.createElement('canvas');
  out.width = width;
  out.height = height;
  const ctx = out.getContext('2d');
  const cam = camera.clone();
  renderer.setScissorTest(true);
  for (const v of views) {
    // WebGL counts rows from the bottom.
    const glY = height - v.y - v.h;
    renderer.setViewport(v.x, glY, v.w, v.h);
    renderer.setScissor(v.x, glY, v.w, v.h);
    aim(cam, v.yaw, v.pitch, v.w / v.h);
    renderer.render(scene, cam);
  }
  const source = renderer.domElement;
  ctx.drawImage(source, 0, 0, source.width, source.height, 0, 0, width, height);
  draw();
  ctx.font = '600 13px system-ui, -apple-system, "Segoe UI", sans-serif';
  ctx.textBaseline = 'middle';
  for (const v of views) {
    const text = v.label;
    const w = ctx.measureText(text).width + 16;
    ctx.fillStyle = 'rgba(255, 255, 255, 0.85)';
    ctx.beginPath();
    ctx.roundRect(v.x + 8, v.y + 8, w, 22, 11);
    ctx.fill();
    ctx.fillStyle = '#1d2330';
    ctx.fillText(text, v.x + 16, v.y + 19);
  }
  if (!single) {
    ctx.fillStyle = 'rgba(40, 46, 58, 0.5)';
    ctx.fillRect(views[1].x - 1, 0, 1, height);
    ctx.fillRect(0, views[2].y - 1, width, 1);
  }
  const blob = await new Promise((done, fail) => out.toBlob((b) => (b ? done(b) : fail(new Error('the model could not be drawn'))), 'image/png'));
  return { png: await blob.arrayBuffer(), width, height, pageHeight: height, scrollY: 0, cut: false, about: shown.text };
}

/** The four sides' places in the grid: front and right above, back and top below. */
function quadrants(width, height) {
  const w = Math.floor(width / 2);
  const h = Math.floor(height / 2);
  const cells = [
    [0, 0, w, h],
    [w, 0, width - w, h],
    [0, h, w, height - h],
    [w, h, width - w, height - h],
  ];
  return SIDES.map((side, i) => ({ ...side, x: cells[i][0], y: cells[i][1], w: cells[i][2], h: cells[i][3] }));
}

// --- talking to bot.computer ------------------------------------------------------------

let port = null;
/** Why the model could not be shown, if it could not. */
let failed = '';

async function start(message) {
  const path = String(message.path || '');
  const files = message.files || {};
  const bytes = files[path];
  const errors = [];
  let text = '';
  say(`Loading /${path}…`);
  try {
    if (!bytes) throw new Error(`/${path} was not sent`);
    const { gltf, refused } = await parse(path, files);
    const root = gltf.scene || (gltf.scenes && gltf.scenes[0]);
    if (!root) throw new Error('the file has no scene to show');
    const facts = measure(root, gltf, path, bytes.byteLength);
    if (refused.length) facts.text += `\nMissing files it names: ${refused.join(', ')}`;
    show(root, facts);
    text = facts.text;
    say('');
    hint.hidden = false;
  } catch (error) {
    const reason = `The model /${path} could not be read: ${error && error.message ? error.message : error}`;
    failed = reason;
    errors.push(reason);
    say(reason, true);
    problem(reason);
  }
  if (port) port.postMessage({ type: 'loaded', ok: errors.length === 0, errors, about: text });
}

window.addEventListener('message', async (event) => {
  const data = event.data;
  if (event.source !== parentWindow || !data) return;
  if (data.type === 'modelview:init') {
    port = event.ports[0] || null;
    start(data);
    return;
  }
  if (data.__botComputer !== true || typeof data.id !== 'number') return;
  let result;
  try {
    if (data.type === 'bot:inspect') result = { ok: true, page: shown ? shown.text : '(no model is shown)' };
    else if (data.type === 'bot:screenshot') result = { ok: true, page: '', shot: await shoot(data.view) };
    else if (data.type === 'bot:act') result = { ok: false, page: shown ? shown.text : '', error: 'a 3D model has nothing to click or type into: preview_screenshot shows it from four sides, or from any angle with yaw and pitch' };
    else result = { ok: false, page: '', error: `unknown request ${data.type}` };
  } catch (error) {
    result = { ok: false, page: '', error: error instanceof Error ? error.message : String(error) };
  }
  post({ type: 'bot:reply', id: data.id, result }, result.shot ? [result.shot.png] : []);
});

// The viewer's own code is in; nothing else may load code, or reach anything but the bytes it holds.
const csp = document.createElement('meta');
csp.httpEquiv = 'Content-Security-Policy';
csp.content = "default-src 'none'; script-src 'none'; style-src 'unsafe-inline'; img-src data: blob:; connect-src data: blob:; font-src 'none'; form-action 'none'; base-uri 'none'; frame-src 'none'; worker-src 'none'";
document.head.prepend(csp);

post({ type: 'bot:bridge' });
parentWindow.postMessage({ type: 'modelview:ready' }, '*');
