/**
 * Stage the portable language-model engine into src-tauri/resources/engines so Tauri ships it
 * (see stage-engines-lib.mjs). It runs when Tauri builds (`beforeBuildCommand`, `npm run build:bundle`), after the
 * pages, so `npm run tauri:build` cannot make an installer without it.
 */
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { EngineStagingError, stageEngine } from './stage-engines-lib.mjs';

const desktop = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const repo = path.resolve(desktop, '..', '..');
try {
  stageEngine({ repo, desktop, log: (line) => console.log(`stage-engines: ${line}`) });
} catch (error) {
  if (error instanceof EngineStagingError) {
    console.error(`stage-engines: ${error.message}`);
    process.exit(1);
  }
  throw error;
}
