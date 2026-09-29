/**
 * Stage the Agent and the flow editor into src-tauri/resources so Tauri ships them.
 *
 * OAIY Desktop shows both in its window and serves them itself (src-tauri/src/
 * embed.rs), from the files of their builds: `app/dist` (the Agent) and
 * `platform/ui/dist` (the flow editor). Both are build outputs (git-ignored), so
 * an installer carries them only if they are copied in at build time:
 *
 *   app/dist          →  src-tauri/resources/app     (opens on index.html)
 *   platform/ui/dist  →  src-tauri/resources/flows   (opens on app.html)
 *
 * tauri.conf.json lists both folders in `bundle.resources`, and Tauri keeps their
 * path, so an installed OAIY finds them in `<install>/resources/app` and
 * `<install>/resources/flows`, where embed.rs looks.
 *
 * This does not build the pages: the Agent takes `npm run build:desktop` in app/
 * (it fetches the SoftN runtime its app preview needs) and the flow editor takes
 * `npm run build` in platform/ui/, and both are minutes of work that a build
 * script should not start on its own. It checks that they were built, copies them,
 * and verifies the copy against the build; a page that is missing, empty,
 * incomplete or different from its source is a failed build, because an installer
 * without it would ship and open on "the page is not built" on the user's machine.
 *
 * It runs first when Tauri builds (`beforeBuildCommand`, `npm run build:bundle`),
 * so `npm run tauri:build` cannot make an installer without the pages.
 */
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { StagingError, stagePages } from './stage-pages-lib.mjs';

const desktop = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const repo = path.resolve(desktop, '..', '..');

try {
  stagePages({ repo, desktop, log: (line) => console.log(`stage-pages: ${line}`) });
} catch (error) {
  if (error instanceof StagingError) {
    console.error(`stage-pages: ${error.message}`);
    process.exit(1);
  }
  throw error;
}
