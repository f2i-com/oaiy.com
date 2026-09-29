/**
 * The flow editor as an installable app: what main.tsx starts, and what the components read.
 *
 * `startPwa()` is called once, first thing, from main.tsx (the editor only: the landing and
 * desktop pages never import this). It listens for the browser's offer to install and
 * registers the service worker, unless this is OAIY's own window, the dev server, or a page
 * that cannot have a worker. See installController.ts, updateController.ts, register.ts and swCore.ts.
 */
import { useSyncExternalStore } from 'react';
import { createInstallController, type InstallState } from './installController';
import { registerServiceWorker, shouldRegister, loadedFiles, type WorkerContainer } from './register';
import { createUpdateController, type UpdateState } from './updateController';

type OaiyWindow = Window & { __OAIY_DESKTOP__?: unknown };

export const installController = createInstallController();
export const updateController = createUpdateController({ reload: () => window.location.reload() });

/** Whether this window is OAIY's own (the desktop injects __OAIY_DESKTOP__ before the page runs). */
export function inOaiyWindow(): boolean {
  return typeof window !== 'undefined' && Boolean((window as OaiyWindow).__OAIY_DESKTOP__);
}

function standalone(): boolean {
  return (
    window.matchMedia?.('(display-mode: standalone)').matches === true ||
    (navigator as Navigator & { standalone?: boolean }).standalone === true
  );
}

let started = false;

export function startPwa(): void {
  if (started || typeof window === 'undefined') return;
  started = true;
  const desktop = inOaiyWindow();

  installController.start({
    addEventListener: (type: string, listener: (event: never) => void) => window.addEventListener(type, listener as EventListener),
    inOaiyWindow: desktop,
    standalone: standalone(),
  });

  if (
    !shouldRegister({
      production: import.meta.env.PROD,
      inOaiyWindow: desktop,
      secureContext: window.isSecureContext,
      hasServiceWorker: 'serviceWorker' in navigator,
    })
  ) {
    return;
  }
  registerServiceWorker(
    {
      container: navigator.serviceWorker as unknown as WorkerContainer,
      whenLoaded: (fn) => {
        if (document.readyState === 'complete') fn();
        else window.addEventListener('load', fn, { once: true });
      },
      onShown: (fn) => document.addEventListener('visibilitychange', () => document.visibilityState === 'visible' && fn()),
      now: () => Date.now(),
      later: (fn, ms) => void window.setTimeout(fn, ms),
      loaded: () => loadedFiles(performance.getEntriesByType('resource') as PerformanceResourceTiming[], window.location.origin),
      log: (message, error) => console.warn(`[pwa] ${message}`, error),
    },
    updateController,
  );
}

/** The install offer, as the components see it. */
export function useInstallState(): InstallState {
  return useSyncExternalStore(installController.subscribe, installController.getState, installController.getState);
}

/** The update message's state, as the components see it. */
export function useUpdateState(): UpdateState {
  return useSyncExternalStore(updateController.subscribe, updateController.getState, updateController.getState);
}
