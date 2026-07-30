/**
 * Static module registration for the web build.
 *
 * The desktop loads node modules from a filesystem plugins dir (via Tauri
 * `list_plugins` etc.). The browser has no filesystem, so instead we use
 * Vite's `import.meta.glob` to statically discover the built-in modules from
 * the workspace `oaiy-core/modules/*` — exactly the pattern bundled-modules.ts
 * documents as the consuming app's job — and feed them through the same
 * oaiy-core APIs that `pluginLoader.loadAllPlugins` uses on desktop:
 *   registerBundledModules() → loadBundledModules(loader) → register UI components.
 *
 * Manifests + node JSON are imported eagerly (always browser-safe). Each
 * module's runtime.ts / compiler.ts / ui/index.ts is imported LAZILY inside a
 * per-module try/catch, so a runtime that isn't browser-compatible degrades
 * gracefully — the node still registers (appears in the palette, editable),
 * it just can't execute client-side until wired to an external engine.
 */
import {
  registerBundledModules,
  getModuleLoader,
  loadBundledModules,
  type BundledModule,
} from 'oaiy-core';
import { registerNodeComponent, registerComponentByName, GenericNode } from 'oaiy-ui-components';
import { createLogger } from './utils/logger';

const logger = createLogger('WebModules');

// Eager: JSON is always safe to import up front.
// Modules are vendored at src/bundled-modules/ — see the README in that
// directory for the upstream copy script and why we vendor instead of
// globbing across the oaiy-web/oaiy sibling boundary.
const manifestGlob = import.meta.glob('./bundled-modules/*/module.json', {
  eager: true,
}) as Record<string, { default: any }>;
const nodeGlob = import.meta.glob('./bundled-modules/*/nodes/*.json', {
  eager: true,
}) as Record<string, { default: any }>;

// Lazy: code modules, imported per-module in try/catch.
const runtimeGlob = import.meta.glob('./bundled-modules/*/runtime.ts');
const compilerGlob = import.meta.glob('./bundled-modules/*/compiler.ts');
const uiGlob = import.meta.glob('./bundled-modules/*/ui/index.ts');

// Modules whose runtimes can't possibly work in a browser (real shell,
// native processes, desktop-only Tauri commands) should be listed here
// AND not vendored under bundled-modules/. The vendor previously
// included `core-terminal` (shells out to system shells) and
// `plugin-agent` (desktop-only AI flow editor with Tauri HTTP bridge);
// both were deleted from bundled-modules/ 2026-05-28 so the glob no
// longer picks them up. If a future module needs to be skipped without
// removing it from the vendor, add its id here.
const WEB_INCOMPATIBLE_MODULES = new Set<string>([]);

// Individual node ids the web build hides — usually because they only
// make sense alongside a specific desktop-only engine (video_avatar
// drives the desktop avatar generator; comfyui_free_memory pokes a
// ComfyUI VRAM-eviction endpoint that's a no-op without our desktop
// model-manager around it). Add ids here to hide them from the palette
// without touching the shared module manifests.
const WEB_HIDDEN_NODE_IDS = new Set<string>([
  'video_avatar',
  'comfyui_free_memory',
  // Hidden in the local-first web build because the unified Services
  // path doesn't cover them yet:
  //  - `video_downloader` shells out to ffmpeg/yt-dlp behind the scenes,
  //    which only works in the desktop bundle.
  //  - `speech_to_text` has no service-tag wiring (no Whisper / Deepgram
  //    service preset listed for it). Until either lands, hiding both
  //    avoids users wiring up a node that silently no-ops in the browser.
  'video_downloader',
  'speech_to_text',
  // `music_gen` is structurally identical to a custom HTTP service
  // (text → audio file). Hide it from the palette; users wire up a
  // Music Gen service in Settings → Services using the dedicated
  // template, then drop a Text-to-Speech node + pick that service.
  'music_gen',
  // ---------------------------------------------------------------
  // Typed nodes folded into the generic Service Call (2026-05-28)
  // ---------------------------------------------------------------
  // The web build doesn't ship its own AI engines, so AI LLM / Image
  // Gen / Video Gen / TTS were always going to be "pick a service and
  // call it" anyway. With per-node-type Quick-Add templates landing in
  // Settings → Services (one click → a pre-filled openai-chat /
  // image-gen-generic / video-gen-generic / tts-generic / music-gen
  // service), there's no remaining reason to expose four near-identical
  // typed palette entries. Each becomes a Service: dropped from the
  // palette as a synthetic Service Call with the right preset baked in.
  //
  // Existing saved flows with `type: ai_llm` etc. keep working — the
  // compiler + runtime path is untouched. They just don't appear in
  // the palette for NEW drops.
  'ai_llm',
  'image_gen',
  'video_gen',
  'text_to_speech',
  // ---------------------------------------------------------------
  // ffmpeg-dependent nodes — most now route through ffmpeg.wasm
  // (tauri-shim/ffmpeg.ts) via the run_command shim. Only the ones that
  // genuinely can't work in a browser stay hidden.
  // ---------------------------------------------------------------
  // 2026-05-28 — all ffmpeg-using nodes now route through the ffmpeg.wasm
  // shim. The dedicated `extract_video_frames` Tauri command used by
  // video_frame_extractor is now also shimmed (tauri-shim/ffmpeg.ts ::
  // extractVideoFrames + the matching handler in core.ts), so it's
  // unhidden too. First call on a fresh tab triggers the one-off
  // ffmpeg-core download (~25 MB from unpkg). Subsequent calls reuse the
  // warm wasm instance.
  // Was hidden:
  //   audio_fade, audio_append, audio_mixer, save_audio,
  //   video_append, video_pip, video_captions, extend_videos,
  //   video_frame_extractor
  // ---------------------------------------------------------------
  // Browser-control nodes (browser_session / _page / _extract / _action)
  // USED to be hidden here: in a plain browser the chromium ops
  // (click/type/scroll/screenshot) had no backend and silently failed.
  //
  // As of Phase 4b they're driven by the OAIY Desktop's managed
  // "Playwright Browser" service: core-browser/runtime.ts routes every
  // chromium-session op to `http://127.0.0.1:<port>/session/...` when
  // there's no native Tauri host. With OAIY Desktop + that service
  // running they work end-to-end; without it they throw an *actionable*
  // error ("start the OAIY Desktop / install Playwright Browser")
  // instead of silently no-op'ing. So they're unhidden — strictly
  // better than the old hide-because-broken behaviour.
  //
  // browser_request was always visible — it's a pure HTTP GET/POST/etc.,
  // no browser backend involved, handled directly via fetch.
]);

// Node ids that should render as GenericNode in the web build, ignoring
// any custom React component the shared manifest registers. The desktop
// build keeps the rich custom components (AILLMNode.tsx etc.) because
// they wire in features like the inline preset selector, mmproj file
// picker, model-download buttons — most of which are desktop-only.
// For the web, the "generic + Service picker" path the user asked for
// is the goal: drop the typed node, pick a Service from the dropdown
// (filled from Settings → Services), and the JSON-driven properties
// panel + showIf-conditioned fields handle everything else. This avoids
// any per-component refactor — GenericNode renders straight from the
// node JSON the compiler already consumes.
const WEB_FORCE_GENERIC_NODES = new Set<string>([
  'ai_llm',
  'image_gen',
  'video_gen',
  'text_to_speech',
]);

// Extract the module id from a glob key. Paths look like
// `./bundled-modules/<id>/module.json` or `./bundled-modules/<id>/nodes/...`,
// so we match the segment after `bundled-modules/`. The earlier
// `/modules/` regex matched the OLD `../../../oaiy/packages/oaiy-core/modules/`
// path but returned '' for the new local layout, collapsing every manifest
// into the same '' key (last write wins → only plugin-agent survived).
const idFromPath = (p: string): string | null => {
  const m = p.match(/bundled-modules\/([^/]+)\//);
  return m ? m[1] : null;
};
const importerFor = (
  glob: Record<string, () => Promise<any>>,
  id: string,
): (() => Promise<any>) | undefined => {
  const key = Object.keys(glob).find((k) => {
    const kId = idFromPath(k);
    if (kId === null) {
      logger.error('Unparseable module glob key: ' + k);
      return false;
    }
    return kId === id;
  });
  return key ? glob[key] : undefined;
};

/** Summary of a bundled-module load pass, returned so callers can surface an
 *  empty/failed palette instead of letting it fail silently. */
export interface WebModulesLoadSummary {
  /** Modules that registered successfully. */
  loaded: number;
  /** Total modules discovered (after web-incompatible filtering). */
  total: number;
  /** Per-module load errors reported by loadBundledModules, if any. */
  errors: unknown[];
}

let loadPromise: Promise<WebModulesLoadSummary> | null = null;

export function loadWebBundledModules(): Promise<WebModulesLoadSummary> {
  return (loadPromise ??= doLoad());
}

async function doLoad(): Promise<WebModulesLoadSummary> {
  // Sanity-check the glob resolution — if the relative path from this
  // source file to the monorepo's `oaiy/packages/` ever drifts (e.g.
  // someone moves this folder), the globs silently return zero matches
  // and the palette goes empty. Log the count so the failure mode is
  // visible in DevTools.
  logger.info(`Glob discovered ${Object.keys(manifestGlob).length} manifest(s), ${Object.keys(nodeGlob).length} node JSON(s)`);
  if (Object.keys(manifestGlob).length === 0) {
    logger.error('No modules found — check the import.meta.glob path in bundled-web-modules.ts (likely src moved relative to oaiy/packages)');
  }

  // Group manifests + node defs by module id, skipping web-incompatible ones.
  const manifestById: Record<string, any> = {};
  const skipped: string[] = [];
  for (const [p, mod] of Object.entries(manifestGlob)) {
    const id = idFromPath(p);
    if (id === null) {
      logger.error('Unparseable module glob key: ' + p);
      continue;
    }
    if (WEB_INCOMPATIBLE_MODULES.has(id)) {
      skipped.push(id);
      continue;
    }
    manifestById[id] = mod.default;
  }
  const nodesById: Record<string, any[]> = {};
  const hiddenNodes: string[] = [];
  for (const [p, mod] of Object.entries(nodeGlob)) {
    const id = idFromPath(p);
    if (id === null) {
      logger.error('Unparseable module glob key: ' + p);
      continue;
    }
    if (WEB_INCOMPATIBLE_MODULES.has(id)) continue;
    const nodeDef = mod.default as { id?: string };
    if (nodeDef?.id && WEB_HIDDEN_NODE_IDS.has(nodeDef.id)) {
      hiddenNodes.push(nodeDef.id);
      continue;
    }
    (nodesById[id] ??= []).push(mod.default);
  }
  if (hiddenNodes.length > 0) {
    logger.info(`Hid ${hiddenNodes.length} web-incompatible node(s): ${hiddenNodes.join(', ')}`);
  }
  if (skipped.length > 0) {
    logger.info(`Skipped ${skipped.length} web-incompatible module(s): ${skipped.join(', ')}`);
  }

  const modules: BundledModule[] = [];
  const uiById: Record<string, Record<string, unknown>> = {};
  const compilerLoadFailures: string[] = [];

  for (const id of Object.keys(manifestById)) {
    const manifest = manifestById[id];
    if (!manifest) continue;
    if (manifest.id && manifest.id !== id) {
      logger.error(`Module folder/id divergence: folder '${id}' but manifest.id '${manifest.id}' — uiById/component keying may mismatch`);
    }
    const nodes = nodesById[id] ?? [];

    let runtime: unknown;
    let compiler: unknown;
    const rImp = importerFor(runtimeGlob, id);
    if (rImp) {
      try {
        runtime = (await rImp()).default;
      } catch (e) {
        logger.warn(`runtime.ts failed to load for ${id} — node(s) will register but not execute in-browser`, { error: e });
      }
    }
    const cImp = importerFor(compilerGlob, id);
    if (cImp) {
      try {
        compiler = (await cImp()).default;
      } catch (e) {
        logger.error(`compiler.ts failed to load for ${id}`, { error: e });
        compilerLoadFailures.push(id);
      }
    }
    const uImp = importerFor(uiGlob, id);
    if (uImp) {
      try {
        uiById[manifest.id] = await uImp();
      } catch (e) {
        logger.warn(`ui/index.ts failed to load for ${id} — falling back to GenericNode`, { error: e });
      }
    }

    modules.push({ manifest, nodes, runtime: runtime as never, compiler: compiler as never });
  }

  // Same sequence as pluginLoader.loadAllPlugins on desktop.
  registerBundledModules(modules);
  const loader = getModuleLoader();
  const res = await loadBundledModules(loader, modules);
  // Surface compiler.ts load failures (collected above) through the result's
  // errors list so they aren't buried in a per-module console.error — a module
  // whose compiler can't load won't execute its nodes client-side.
  if (compilerLoadFailures.length > 0) {
    const errs = ((res as { errors?: unknown[] }).errors ??= []);
    for (const id of compilerLoadFailures) {
      errs.push(`compiler.ts failed to load for ${id}`);
    }
  }
  logger.info(`Loaded ${res.loaded}/${modules.length} bundled modules`, {
    skipped: res.skipped,
    errors: res.errors,
  });
  const summary: WebModulesLoadSummary = {
    loaded: res.loaded,
    total: modules.length,
    errors: res.errors ?? [],
  };

  // Register UI components (custom where available, GenericNode fallback).
  // Any nodeType in WEB_FORCE_GENERIC_NODES bypasses its custom React
  // component and registers GenericNode instead — see the comment on
  // WEB_FORCE_GENERIC_NODES for why.
  const forced: string[] = [];
  for (const mod of modules) {
    const id = mod.manifest.id as string;
    const ui = uiById[id];
    const mappings = (mod.manifest.ui?.nodes ?? []) as Array<{ nodeType: string; componentName: string }>;
    if (ui) {
      for (const [name, comp] of Object.entries(ui)) {
        if (comp) registerComponentByName(name, comp as never);
      }
      for (const map of mappings) {
        const comp = ui[map.componentName];
        if (!comp) continue;
        if (WEB_FORCE_GENERIC_NODES.has(map.nodeType)) {
          registerNodeComponent(map.nodeType, GenericNode as never);
          forced.push(map.nodeType);
        } else {
          registerNodeComponent(map.nodeType, comp as never);
        }
      }
    }
    for (const node of mod.nodes) {
      const nodeId = (node as { id: string }).id;
      const hasCustom = mappings.some((m) => m.nodeType === nodeId);
      if (!hasCustom) registerNodeComponent(nodeId, GenericNode as never);
    }
  }
  if (forced.length > 0) {
    logger.info(`Forced GenericNode for ${forced.length} web-generic node(s): ${forced.join(', ')}`);
  }

  return summary;
}
