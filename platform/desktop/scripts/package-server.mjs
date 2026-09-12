// Stage an offline headless distribution. Requires an empty destination: never
// overwrite an installed server or mix files from two releases.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createHash } from 'node:crypto';

const desktop = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const [serverArg, destinationArg] = process.argv.slice(2);
if (!serverArg || !destinationArg) throw new Error('Usage: package-server.mjs <server-binary> <empty-destination>');
const server = path.resolve(serverArg);
const destination = path.resolve(destinationArg);
const cli = path.join(desktop, 'src-tauri/resources/cli');
for (const required of [server, path.join(cli, 'oaiy.mjs'), path.join(cli, 'node_modules/undici/package.json')]) {
  if (!fs.statSync(required).isFile()) throw new Error(`Required distribution file missing: ${required}`);
}
fs.mkdirSync(destination); // deliberately fails if anything is already there
fs.mkdirSync(path.join(destination, 'resources/node'), { recursive: true });
fs.copyFileSync(server, path.join(destination, path.basename(server)));
fs.cpSync(cli, path.join(destination, 'resources/cli'), { recursive: true });
fs.copyFileSync(process.execPath, path.join(destination, 'resources/node', path.basename(process.execPath)));
fs.writeFileSync(path.join(destination, 'README.txt'), `OAIY headless distribution
Contains the server, CLI, undici and Node ${process.version} (${process.platform}/${process.arch}).
Set OAIY_SERVER_TOKEN to a private random credential before making API requests.
Run ${path.basename(server)}. The default listener is loopback; OAIY_SERVER_BIND=lan
requires a token and disables all Origin-based authentication exceptions.
Core flows run offline without Node/npm on PATH. Optional browser/image nodes
still require their documented services/dependencies. Keep resources beside the server.
`);
function inventory(dir, prefix = '') {
  return fs.readdirSync(dir, { withFileTypes: true }).sort((a, b) => a.name.localeCompare(b.name)).flatMap(entry => {
    const relative = prefix + entry.name;
    const full = path.join(dir, entry.name);
    return entry.isDirectory() ? inventory(full, relative + '/') : [{
      path: relative, sha256: createHash('sha256').update(fs.readFileSync(full)).digest('hex'),
    }];
  });
}
fs.writeFileSync(path.join(destination, 'distribution.json'), JSON.stringify({
  nodeVersion: process.version, platform: process.platform, arch: process.arch, files: inventory(destination),
}, null, 2) + '\n');
console.log(`Staged complete headless distribution: ${destination}`);
