/**
 * Assemble the providers origin into a folder a static host can serve as it is (design 3.8): the built pages, with the origins the
 * operator chose written into them, and the `_headers` file of the host rendered from web/hosting/headers/providers.headers.
 *
 *     node scripts/assemble.mjs --providers https://providers.example --agent https://agent.example --flows https://flows.example \
 *                               [--app name=https://other.example ...] [--out dist/providers]
 *
 * Nothing about a domain is in the built pages or the template: this is the one place a deployment says which origins there are,
 * and it refuses an origin that is not exactly an origin (https, or http on this computer for tests) and a list that repeats one.
 * It does not build: run `vite build --config providers/vite.config.ts` first (npm run build).
 *
 * WA-06 grows this into all three hosts. It assembles the providers origin only, which is what the port and the Providers page need.
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { readTemplate, renderHeaders } from './headers.mjs';
import { loadTs } from './load-ts.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
export const PROVIDERS_DIST = path.resolve(HERE, '..', 'providers', 'dist');

const META = /<meta\s+name="oaiy-apps"\s+content=""\s*\/?>/;

/**
 * @param {{ outDir: string, distDir?: string, providers: string, apps: Record<string, string>, headers?: (rendered: string) => string }} options
 *   `apps` maps an app name to its origin (`agent`, `flows`); `headers` may change the rendered `_headers` (a test that needs a variant).
 * @returns {Promise<string>} the folder
 */
export async function assembleProviders({ outDir, distDir = PROVIDERS_DIST, providers, apps, headers }) {
  const { parseAppOrigins, isExactOrigin } = await loadTs('shared/broker/protocol.ts');
  if (!fs.existsSync(path.join(distDir, 'broker.html'))) throw new Error(`there is no build in ${distDir}: run \`npm run build\` first`);
  if (!isExactOrigin(providers)) throw new Error(`the providers origin is not an origin: ${providers}`);
  const content = Object.entries(apps).map(([name, origin]) => `${name}=${origin}`).join(' ');
  const parsed = parseAppOrigins(content);
  if (parsed.size !== Object.keys(apps).length) throw new Error(`the apps are not a list of distinct origins: ${content}`);
  if (parsed.has(providers)) throw new Error('the providers origin is also listed as an app');

  fs.rmSync(outDir, { recursive: true, force: true });
  fs.cpSync(distDir, outDir, { recursive: true });

  for (const file of fs.readdirSync(outDir).filter((f) => f.endsWith('.html'))) {
    const at = path.join(outDir, file);
    const html = fs.readFileSync(at, 'utf8');
    if (!META.test(html)) throw new Error(`${file} has no empty <meta name="oaiy-apps"> to fill in`);
    fs.writeFileSync(at, html.replace(META, `<meta name="oaiy-apps" content="${content}" />`));
  }

  const rendered = renderHeaders(readTemplate('providers'), { providers, apps: [...parsed.keys()] });
  fs.writeFileSync(path.join(outDir, '_headers'), headers ? headers(rendered) : rendered);
  return outDir;
}

function parseArgs(argv) {
  const out = { apps: {} };
  for (let i = 0; i < argv.length; i++) {
    const flag = argv[i];
    const value = argv[++i];
    if (value === undefined) throw new Error(`${flag} needs a value`);
    if (flag === '--providers') out.providers = value;
    else if (flag === '--agent') out.apps.agent = value;
    else if (flag === '--flows') out.apps.flows = value;
    else if (flag === '--app') {
      const eq = value.indexOf('=');
      if (eq <= 0) throw new Error('--app is name=origin');
      out.apps[value.slice(0, eq)] = value.slice(eq + 1);
    } else if (flag === '--out') out.out = value;
    else throw new Error(`unknown option ${flag}`);
  }
  return out;
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try {
    const args = parseArgs(process.argv.slice(2));
    if (!args.providers) throw new Error('--providers is required');
    const out = path.resolve(args.out ?? path.join(HERE, '..', 'dist', 'providers'));
    await assembleProviders({ outDir: out, providers: args.providers, apps: args.apps });
    console.log(`assembled ${out}`);
  } catch (e) {
    console.error(e.message);
    process.exit(2);
  }
}
