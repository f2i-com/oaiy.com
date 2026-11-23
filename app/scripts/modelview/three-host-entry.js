/**
 * Trusted offline Three.js runtime bundled into `public/webpage/three-host.js`
 * for OAIY Agent's sandboxed webpage preview.
 *
 * Untrusted page scripts still execute strictly inside the Zipp VM under the
 * preview's strict CSP; `page-host.js` exposes these constructors/addons to
 * guest scripts through handle proxies so Three.js scenes (such as outputs from
 * Qwen3.5-4B-3D-Coder and Qwen3.5-4B-GLB-3D-Coder) render live WebGL and export
 * `.glb` files completely offline.
 */
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFExporter } from 'three/addons/exporters/GLTFExporter.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { RoundedBoxGeometry } from 'three/addons/geometries/RoundedBoxGeometry.js';
import * as BufferGeometryUtils from 'three/addons/utils/BufferGeometryUtils.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';

const modules = Object.freeze({
  three: THREE,
  'three/addons/controls/OrbitControls.js': Object.freeze({ OrbitControls, default: OrbitControls }),
  'three/addons/exporters/GLTFExporter.js': Object.freeze({ GLTFExporter, default: GLTFExporter }),
  'three/addons/loaders/GLTFLoader.js': Object.freeze({ GLTFLoader, default: GLTFLoader }),
  'three/addons/environments/RoomEnvironment.js': Object.freeze({ RoomEnvironment, default: RoomEnvironment }),
  'three/addons/geometries/RoundedBoxGeometry.js': Object.freeze({ RoundedBoxGeometry, default: RoundedBoxGeometry }),
  'three/addons/utils/BufferGeometryUtils.js': BufferGeometryUtils,
  'three/addons/postprocessing/EffectComposer.js': Object.freeze({ EffectComposer, default: EffectComposer }),
  'three/addons/postprocessing/RenderPass.js': Object.freeze({ RenderPass, default: RenderPass }),
  'three/addons/postprocessing/UnrealBloomPass.js': Object.freeze({ UnrealBloomPass, default: UnrealBloomPass }),
  'three/addons/postprocessing/OutputPass.js': Object.freeze({ OutputPass, default: OutputPass }),
});

window.__oaiyAgentThree = Object.freeze({ THREE, modules });
