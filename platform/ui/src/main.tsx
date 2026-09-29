// oaiy-web: install the browser Tauri bridge shims FIRST, before any other
// module loads, so window.__TAURI__ / __TAURI_INTERNALS__ and the aliased
// @tauri-apps/api/core invoke exist before anything calls them.
import './tauri-shim/core';

// Console buffer must install BEFORE any other code so we capture
// import-time errors from dynamicModules and plugin loading. It is
// readable via /api/console for scripted/headless debugging.
import { installConsoleBuffer } from './utils/consoleBuffer';
installConsoleBuffer();

// Register dynamic-options resolvers BEFORE module discovery so any node
// mounted with a `service:list` dropdown gets fresh options on its first
// render. The resolver itself just reads localStorage on demand, so it's
// cheap to register early.
//
// Scheme:
//   service:list                 — every registered service (Service Call uses this).
//   service:list:<nodeType>      — only services tagged for the given node type;
//                                  e.g. `service:list:ai_llm` for the AI LLM dropdown.
import { registerDynamicOptionsResolver, registerNodeNoticeProvider } from 'oaiy-ui-components';
import { serviceOptions } from './lib/serviceOptions';
import { nodeNotice } from './lib/nodeAvailability';
import { currentAvailabilityEnv } from './lib/availabilityEnv';
// Every Service dropdown: OAIY's engine models, the desktop's services (Python
// rigs, ComfyUI …) and the user's own, grouped, with "Add a service…" last.
registerDynamicOptionsResolver('service:list', (rest: string) =>
  serviceOptions(rest, currentAvailabilityEnv(), (nodeType) => {
    void import('./lib/addServiceDialog').then((m) => m.openAddServiceDialog(nodeType));
  }),
);
// A node in a flow whose service is not installed says so on the node.
registerNodeNoticeProvider((nodeType, data) => nodeNotice(nodeType, data, currentAvailabilityEnv()));

// Dynamic module discovery - MUST be imported first before any module access
import './dynamicModules';

// Start the OAIY Desktop detection probe. Polls a fixed localhost
// port for the OAIY Desktop app and re-renders any subscribed
// component when the status flips. See lib/desktopDetection.ts.
import { startDesktopDetection } from './lib/desktopDetection';
import { startDesktopServiceSync } from './lib/desktopServices';
startDesktopDetection();
// Mirror OAIY Desktop's running services into the service dropdowns +
// compilers while it's available (Phase 3). No-op when OAIY Desktop
// isn't running. See lib/desktopServices.ts.
startDesktopServiceSync();

// Ask the browser to keep this site's storage (the project, macros and API keys
// live there) rather than clear it when space is short. Once, in the
// background, and never a failure: see lib/persistentStorage.ts.
import { requestPersistentStorage } from './lib/persistentStorage';
void requestPersistentStorage();

// The editor as an installable app: the browser's offer to install is caught here, before
// anything can miss it, and the service worker is registered (in a production build, and
// not in OAIY's own window). See pwa/index.ts.
import { startPwa } from './pwa';
startPwa();

// Seal any API keys still in plaintext localStorage (the first load after this
// change), in the background: the editor's reads wait for the same promise, and
// a browser that cannot seal keeps them as before. See tauri-shim/secretVault.ts.
import { secretVault } from './tauri-shim/secretVault';
void secretVault.ready();

import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './styles/fonts'
import './index.css'
import App from './App.tsx'

// Global error handlers for errors that React's ErrorBoundary doesn't catch
// (async errors, errors in event handlers, etc.)
window.addEventListener('unhandledrejection', (event) => {
  console.error('[Unhandled Promise Rejection]', event.reason);
  // Prevent the default browser handling (logging to console twice)
  event.preventDefault();
});

window.addEventListener('error', (event) => {
  // Resource-load failures (img/script) fire here with event.error === null — skip
  // those. (No de-dup vs ErrorBoundary: `_reactHandled` was never a real property,
  // so the old guard filtered nothing.)
  if (event.error) {
    console.error('[Uncaught Error]', event.error);
  }
});

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
