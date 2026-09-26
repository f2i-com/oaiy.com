// The portable desktop app: one executable, nothing to install. The app's
// files (the web app, the Zipp engine, the SoftN runtime) are built into it;
// the "portable" in its name makes it keep its data in bot.computer-data/
// beside itself. It uses the system's WebView2 (part of Windows 10 and 11).
//   npm run desktop:portable
import { execSync } from 'node:child_process';
import { copyFileSync, mkdirSync, readFileSync, statSync } from 'node:fs';
import { join } from 'node:path';

const { version } = JSON.parse(readFileSync('package.json', 'utf8'));
execSync('npx tauri build --no-bundle', { stdio: 'inherit' });
const ext = process.platform === 'win32' ? '.exe' : '';
const arch = process.arch === 'x64' ? 'x64' : process.arch;
const from = join('src-tauri', 'target', 'release', `bot-computer${ext}`);
const dir = join('src-tauri', 'target', 'release', 'bundle', 'portable');
const to = join(dir, `bot.computer_${version}_${arch}-portable${ext}`);
mkdirSync(dir, { recursive: true });
copyFileSync(from, to);
console.log(`\nPortable app: ${to} (${(statSync(to).size / 1e6).toFixed(1)} MB)`);
console.log('It keeps its projects and settings in bot.computer-data/ beside itself.');
