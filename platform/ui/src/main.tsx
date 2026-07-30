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
import { registerDynamicOptionsResolver } from 'oaiy-ui-components';
import { listAllServices } from './utils/serviceRegistry';
import { listCompanionServices } from './lib/companionServices';
import {
  filterServicesForNodeType,
  type ServiceNodeTag,
} from 'oaiy-core/modules/core-service/examples';
registerDynamicOptionsResolver('service:list', (rest: string) => {
  // User-saved services first, then any services the OAIY Companion is
  // currently running (Phase 3). Companion entries carry a `companion:`
  // id prefix so they never collide with the user's own.
  const all = [...listAllServices(), ...listCompanionServices()];
  const filtered = filterServicesForNodeType(all, (rest as ServiceNodeTag) || '');
  const isCompanion = (id: string) => id.startsWith('companion:');
  return [
    { value: '', label: rest ? '(none — use the fields below)' : '(none — fill fields inline)' },
    ...filtered.map((s) => ({
      value: s.id,
      label: isCompanion(s.id) ? `${s.name}` : s.name,
      description: isCompanion(s.id)
        ? (s.description || 'Running in the OAIY Companion')
        : s.description || (s.isBuiltIn ? 'Built-in example' : 'Custom service'),
    })),
  ];
});

// Dynamic module discovery - MUST be imported first before any module access
import './dynamicModules';

// Start the OAIY Companion detection probe. Polls a fixed localhost
// port for the desktop companion app and re-renders any subscribed
// component when the status flips. See lib/companionDetection.ts.
import { startCompanionDetection } from './lib/companionDetection';
import { startCompanionServiceSync } from './lib/companionServices';
startCompanionDetection();
// Mirror the companion's running services into the service dropdowns +
// compilers while it's available (Phase 3). No-op when the companion
// isn't running. See lib/companionServices.ts.
startCompanionServiceSync();

import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
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
