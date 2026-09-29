// A second worker source, only so tests/sw-build.mjs can show that the build id follows the worker's own code.
import { installWorker } from '../../src/pwa/swCore';

export const other = installWorker;
