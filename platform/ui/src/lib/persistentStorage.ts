/**
 * Ask the browser to keep this site's storage.
 *
 * Everything the editor keeps lives in the browser's own storage: the project,
 * its run history, the user's macros and services, the API keys, uploaded
 * files. A browser may clear a site's storage when the disk is short unless the
 * site has asked for persistent storage (Chrome and Edge decide quietly, on how
 * much the site is used or whether it is installed; Firefox asks the person).
 * The Agent asks when it starts (app/src/main.ts); the flow editor asks the same
 * way, so a project is not the first thing lost when space runs out.
 *
 * Three rules, all for the page that calls it:
 *   - Once. A second call gets the first call's answer; the browser is not
 *     asked twice in one page load.
 *   - In the background. It returns a promise nobody has to wait for, and the
 *     answer changes nothing the page does.
 *   - Never a failure. A browser without the API, a page it is not offered to
 *     (an insecure address), a refusal and a throw all answer `false`.
 */

/** The part of `navigator` this reads, so a test can hand in a stand-in. */
export interface StorageNavigator {
  storage?: {
    persist?: () => Promise<boolean>;
    persisted?: () => Promise<boolean>;
  };
}

const asked = new WeakMap<object, Promise<boolean>>();

/**
 * Ask for persistent storage; resolves with whether the site has it. It never
 * rejects. With no argument it asks the real `navigator` (and answers `false`
 * where there is none, as under Node).
 */
export function requestPersistentStorage(
  nav: StorageNavigator | undefined = typeof navigator === 'undefined' ? undefined : navigator,
): Promise<boolean> {
  if (!nav || typeof nav !== 'object') return Promise.resolve(false);
  const earlier = asked.get(nav);
  if (earlier) return earlier;
  const answer = ask(nav);
  asked.set(nav, answer);
  return answer;
}

async function ask(nav: StorageNavigator): Promise<boolean> {
  try {
    const storage = nav.storage;
    if (!storage || typeof storage.persist !== 'function') return false;
    // Already granted: nothing to ask (and Firefox would not ask the person again).
    try {
      if (typeof storage.persisted === 'function' && (await storage.persisted())) return true;
    } catch {
      // Could not tell; ask.
    }
    return (await storage.persist()) === true;
  } catch {
    return false;
  }
}
