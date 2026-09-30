// The release builds the headless server WITH the web login, CI tests that build, and the systemd unit does not restart a
// server that its own configuration refused.
//
//   node scripts/check-release.mjs          from platform/desktop (check-access.mjs runs it first)
//
// oaiy-server's startup rules (design 4.5.5) make a lan install need an owner login and a proxied one need the sign-in, and
// both are the `web` feature: a build without it says "this build has no web login" and cannot serve either. The release
// workflow built the headless server with `--no-default-features` alone, so what it shipped could not be put on a network.
// This fails when `.github/workflows/release.yml` builds `oaiy-server` without `--features web`, when the release evidence
// does not say so, or when `ci.yml` does not run the tests of that configuration.
//
// It reads text and runs nothing: no cargo, no network, no socket.

import fs from 'node:fs';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';

/** The features named on a command line: `--features web`, `--features=web`, `--features "gui,web"`, `-F web`. */
function featuresOf(command) {
  const out = [];
  for (const m of command.matchAll(/(?:--features|-F)(?:=|\s+)(?:"([^"]*)"|'([^']*)'|(\S+))/g)) {
    out.push(...(m[1] ?? m[2] ?? m[3]).split(/[,\s]+/).filter(Boolean));
  }
  return out;
}

/** What is wrong with the two workflows, each as a line; none means they hold. */
export function problemsOf({ release, ci, unit }) {
  const problems = [];
  // The step that builds the server: from its name to the next step, its `run:` line or block.
  const step = /-\s+name:\s*Build the headless server\b[\s\S]*?(?=\n\s*-\s+name:|$)/.exec(release);
  if (!step) {
    problems.push('release.yml has no step named "Build the headless server"');
  } else {
    const run = /\brun:\s*(?:[|>][-+]?\s*\n)?([\s\S]*)/.exec(step[0]);
    const command = run ? run[1].split('\n').filter((l) => /cargo\s+build/.test(l)).join(' ') : '';
    if (!/cargo\s+build\b/.test(command) || !/--bin[ =]oaiy-server\b/.test(command)) {
      problems.push(`the step "Build the headless server" does not run \`cargo build ... --bin oaiy-server\` (${command.trim() || 'no cargo build in it'})`);
    } else if (!featuresOf(command).includes('web')) {
      problems.push(`release.yml builds the headless server without the web feature (${command.trim()}): a lan or a proxied install needs the web login, and would say so and stop. Add \`--features web\``);
    }
  }
  // The evidence that goes out with the release says how the headless server was built.
  const evidence = /headless:\s*\[([^\]]*)\]/.exec(release);
  if (!evidence) problems.push('release.yml records no features for the headless server in the release evidence');
  else if (!/\bweb\b/.test(evidence[1])) {
    problems.push(`the release evidence says the headless server has the features ${evidence[1].trim()}, without web`);
  }
  // CI runs the tests of the configuration that ships.
  const tests = ci.split('\n').filter((l) => /cargo\s+test\b/.test(l));
  if (!tests.some((l) => /--no-default-features/.test(l) && featuresOf(l).includes('web'))) {
    problems.push('ci.yml does not run `cargo test --no-default-features --features web`: the headless server that ships is not tested');
  }
  // The unit that ships does not restart a server that its own configuration refused (exit 78, EX_CONFIG): the comments of
  // oaiy-server.rs and of `auth::exposure` say so, and without these lines the unit restarted it every 3 seconds.
  if (unit !== undefined) {
    const lines = unit.split('\n').map((l) => l.trim()).filter((l) => l && !l.startsWith('#'));
    if (!lines.includes('RestartPreventExitStatus=78')) {
      problems.push('oaiy-server.service does not say RestartPreventExitStatus=78: a server whose configuration is refused (exit 78) is restarted every RestartSec');
    }
    if (!lines.some((l) => /^ExecStartPre=\S*oaiy-server\s+check\s*$/.test(l))) {
      problems.push('oaiy-server.service does not run `oaiy-server check` before the server (ExecStartPre=), which lists every rule that is broken');
    }
    if (!lines.some((l) => /^StartLimitBurst=\d+$/.test(l)) || !lines.some((l) => /^StartLimitIntervalSec=\d+$/.test(l))) {
      problems.push('oaiy-server.service has no StartLimitBurst= and StartLimitIntervalSec=: a failing ExecStartPre (which RestartPreventExitStatus= does not cover) is restarted for ever');
    }
  }
  return problems;
}

/** The workflows and the systemd unit of this repository. */
export function readWorkflows(root) {
  const read = (...parts) => fs.readFileSync(path.join(root, ...parts), 'utf8');
  return {
    release: read('.github', 'workflows', 'release.yml'),
    ci: read('.github', 'workflows', 'ci.yml'),
    unit: read('platform', 'desktop', 'systemd', 'oaiy-server.service'),
  };
}

const here = path.dirname(fileURLToPath(import.meta.url));
if (process.argv[1] && path.resolve(process.argv[1]) === path.resolve(fileURLToPath(import.meta.url))) {
  const root = path.resolve(here, '..', '..', '..');
  let problems;
  try {
    problems = problemsOf(readWorkflows(root));
  } catch (e) {
    problems = [`cannot read the workflows under ${root}: ${e.message}`];
  }
  if (problems.length) {
    console.error('check-release: the release does not build and test what the startup rules need:\n  ' + problems.join('\n  '));
    process.exit(1);
  }
  console.log('check-release: ok. The headless server is built with --features web, CI tests that build, and the unit does not restart exit 78.');
}
