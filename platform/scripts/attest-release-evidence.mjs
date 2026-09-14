#!/usr/bin/env node
// Finalise release evidence AFTER verification (ecosystem review R3-OAI-01).
//
// The web and desktop build jobs run in parallel with the verification gate
// and write `release-evidence-*.json` with `verification.status: "unverified"`
// and `dependencyAudit.status: "pending"`: a build artifact never claims a
// result the run has not established. The release job, which depends on the
// gate, runs this script over the downloaded evidence files. It refuses unless
// the gate's result is an explicit success, every file names the verified
// revision, and every artifact digest recorded at build time still matches the
// bytes about to be published; only then does it rewrite the files with the
// passing status and the exact verification run (workflow run id, attempt,
// URL, ci.yml job names). Nothing is written until every file has passed.
//
//   VERIFY_RESULT=success REVISION=<sha> node scripts/attest-release-evidence.mjs --dir artifacts
//
// Inputs (environment): VERIFY_RESULT (must be "success"), REVISION,
// GITHUB_RUN_ID, GITHUB_RUN_ATTEMPT, GITHUB_SERVER_URL, GITHUB_REPOSITORY,
// VERIFY_WORKFLOW (default ".github/workflows/ci.yml"), VERIFY_JOBS (comma
// separated; default lists the ci.yml lanes).
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { fileURLToPath } from 'node:url';

const DEFAULT_JOBS = 'revision,web,cli,desktop (linux),desktop (windows)';

export function attest(dir, env = process.env, log = console.log) {
  const refuse = (message) => { throw new Error(message); };
  if (env.VERIFY_RESULT !== 'success') {
    refuse(`verification result is "${env.VERIFY_RESULT ?? ''}", not "success": evidence stays unverified`);
  }
  const revision = env.REVISION;
  if (!revision || !/^[0-9a-f]{40}$/.test(revision)) refuse(`REVISION must be the verified 40-hex commit (got "${revision ?? ''}")`);
  if (!env.GITHUB_RUN_ID) refuse('GITHUB_RUN_ID is required to name the verification run');
  const run = {
    id: String(env.GITHUB_RUN_ID),
    attempt: String(env.GITHUB_RUN_ATTEMPT ?? '1'),
    url: env.GITHUB_SERVER_URL && env.GITHUB_REPOSITORY
      ? `${env.GITHUB_SERVER_URL}/${env.GITHUB_REPOSITORY}/actions/runs/${env.GITHUB_RUN_ID}/attempts/${env.GITHUB_RUN_ATTEMPT ?? '1'}`
      : null,
    workflow: env.VERIFY_WORKFLOW || '.github/workflows/ci.yml',
    jobs: (env.VERIFY_JOBS || DEFAULT_JOBS).split(',').map(s => s.trim()).filter(Boolean),
  };

  const names = fs.readdirSync(dir).filter(name => /^release-evidence-.+\.json$/.test(name)).sort();
  if (names.length === 0) refuse(`no release-evidence-*.json in ${dir}`);

  // Validate everything first; write nothing until every file passes.
  const finalised = [];
  for (const name of names) {
    const file = path.join(dir, name);
    let evidence;
    try { evidence = JSON.parse(fs.readFileSync(file, 'utf8')); }
    catch (error) { refuse(`${name}: not valid JSON (${error.message})`); }
    if (evidence.revision !== revision) refuse(`${name}: records revision ${evidence.revision}, expected ${revision}`);
    const already = evidence.verification?.status === 'verified';
    if (already && evidence.verification?.run?.id === run.id && evidence.verification?.run?.attempt === run.attempt) {
      // Idempotent re-run of the same attempt.
    } else if (already) {
      refuse(`${name}: already attested by run ${evidence.verification.run?.id}/${evidence.verification.run?.attempt}; refusing to re-stamp`);
    } else if (evidence.verification?.status !== 'unverified' || evidence.dependencyAudit?.status !== 'pending') {
      refuse(`${name}: expected verification.status "unverified" and dependencyAudit.status "pending" from the build job (got "${evidence.verification?.status}" / "${evidence.dependencyAudit?.status}")`);
    }
    if (!Array.isArray(evidence.artifacts) || evidence.artifacts.length === 0) refuse(`${name}: lists no artifacts`);
    for (const artifact of evidence.artifacts) {
      if (typeof artifact?.name !== 'string' || !/^[0-9a-f]{64}$/.test(artifact?.sha256 ?? '')) refuse(`${name}: malformed artifact entry ${JSON.stringify(artifact)}`);
      const artifactPath = path.join(dir, artifact.name);
      if (!fs.existsSync(artifactPath)) refuse(`${name}: artifact ${artifact.name} is missing from ${dir}`);
      const actual = crypto.createHash('sha256').update(fs.readFileSync(artifactPath)).digest('hex');
      if (actual !== artifact.sha256) refuse(`${name}: ${artifact.name} digest ${actual} differs from the recorded ${artifact.sha256}`);
    }
    finalised.push({ file, name, evidence });
  }

  const attestedAt = new Date().toISOString();
  for (const { file, name, evidence } of finalised) {
    evidence.verification = {
      status: 'verified',
      result: 'success',
      revision,
      run,
      attestedAt,
    };
    evidence.dependencyAudit = {
      ...(evidence.dependencyAudit ?? {}),
      status: 'pass',
      level: evidence.dependencyAudit?.level ?? 'high',
      revision,
      run: { id: run.id, attempt: run.attempt },
    };
    fs.writeFileSync(file, JSON.stringify(evidence, null, 2) + '\n');
    log(`${name}: verified by run ${run.id}/${run.attempt} at ${revision}, ${evidence.artifacts.length} artifact digests re-checked`);
  }
  return finalised.map(f => f.name);
}

function main() {
  const args = process.argv.slice(2);
  const at = args.indexOf('--dir');
  const dir = at >= 0 ? args[at + 1] : 'artifacts';
  if (!dir) { console.error('usage: attest-release-evidence.mjs --dir <artifacts dir>'); process.exit(2); }
  try {
    attest(path.resolve(dir));
  } catch (error) {
    console.error(`::error::release evidence NOT attested: ${error.message}`);
    process.exit(1);
  }
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main();
}
