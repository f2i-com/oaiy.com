/**
 * What the landing page says about privacy, which depends on how the site was built.
 *
 * The release workflow builds the site with no sharing service (`VITE_API_BASE` unset): the editor
 * runs fully standalone and nothing a person builds or runs is sent anywhere by OAIY. A build made
 * with a service (`VITE_API_BASE` set, see lib/sharingPrefs.ts) has sharing on by default. What the code
 * does then, and what the words below rest on (tests/site-assets.mjs holds each to the source):
 *   - A flow is sent to the service only by `createFlow`, from `createShare`, which only the Share
 *     dialog calls, with its keys taken out and encrypted first if the person set a password. Nothing
 *     sends it again: `updateFlow` and `deleteFlow` in lib/backendDispatcher.ts have no caller, so an
 *     edit is not pushed and "Stop sharing" only makes this browser forget the share.
 *   - The share is remembered (localStorage), and while there is one and sharing is on, the editor
 *     polls the service for runs that others queue, sends a heartbeat, and reports each run's result.
 *     With nothing shared it makes no request at all.
 *   - Opening a `?flow=` link reads that flow from the service (lib/openSharedFlow.ts).
 *   - Sharing can be turned off in Settings.
 * A page that says "nothing is uploaded" for a build like that would be false, so the words are chosen
 * here, from the build.
 */

export interface PrivacyPoint {
  title: string;
  body: string;
}

export interface PrivacyCopy {
  /** The line under the section's heading. */
  sub: string;
  points: PrivacyPoint[];
}

/** The keys' sentence, in both builds: sealing needs a browser that can do it (lib: tauri-shim/secretVault.ts). */
const KEYS = 'API keys are kept in your browser, sealed where the browser supports it, and used only in the requests your flows make.';

const SANDBOX: PrivacyPoint = {
  title: 'Flow code runs in a sandbox',
  body: 'Code nodes execute inside Zipp, a JavaScript engine compiled to WebAssembly with no network, no storage and hard limits on CPU and memory.',
};

/** `hasSharingService`: the site was built with a sharing service (VITE_API_BASE). */
export function privacyCopy(hasSharingService: boolean): PrivacyCopy {
  if (!hasSharingService) {
    return {
      sub: 'Local-first on purpose. This site serves the page and the editor, and nothing you build or run is sent back to it.',
      points: [
        { title: 'Your work stays on your device', body: 'Flows, keys and files live in your browser, or on your computer with OAIY Desktop. Nothing is uploaded to OAIY, and there is no account.' },
        { title: 'Keys stay on your device', body: `${KEYS} They are never sent to OAIY.` },
        { title: 'Prompts go where you point them', body: 'To a model on your own machine, or to the provider whose key you added. They never pass through a server of ours.' },
        SANDBOX,
      ],
    };
  }
  return {
    sub: 'Local-first on purpose. This site serves the page and the editor, and it has a sharing service: a flow reaches it only when you press Share.',
    points: [
      {
        title: 'Your work stays on your device until you share it',
        body: 'Flows, keys and files live in your browser, or on your computer with OAIY Desktop. Sharing is on in this build, and a flow is sent to the sharing service only when you press Share, encrypted first if you set a password. Editing it afterwards does not update that copy, and Stop sharing does not delete it. You can turn sharing off in Settings. There is no account.',
      },
      { title: 'Keys stay on your device', body: `${KEYS} A shared flow is sent without them.` },
      {
        title: 'Prompts go where you point them',
        body: "To a model on your own machine, or to the provider whose key you added. While a flow is shared, the editor asks the sharing service for runs that others queue on it, and when someone does, that run's inputs and its result pass through the service.",
      },
      SANDBOX,
    ],
  };
}
