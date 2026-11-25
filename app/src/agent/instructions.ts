/**
 * Workspace instruction auto-discovery & live change reconciliation, adapted from
 * DeepSeek Harness (`packages/context/agent-instructions`).
 *
 * Discovers `AGENTS.md`, `OAIY.md`, `CLAUDE.md`, `AGENTS.local.md`, and
 * `.oaiy/instructions.md` at project root (baseline) and in ancestor folders of
 * files touched during a run (nested). Emits instructions as cache-stable user
 * context injections rather than mutating the system prompt, preserving 100%
 * prompt KV prefix caching.
 */
import { normalizePath, type Vfs } from '../vfs/vfs';

export const INSTRUCTION_CANDIDATES = [
  'AGENTS.md',
  'OAIY.md',
  'CLAUDE.md',
  'AGENTS.local.md',
  '.oaiy/instructions.md',
] as const;

const DEFAULT_MAX_BYTES = 24_576;
const DEFAULT_MAX_SOURCE_BYTES = 10_240;

/** Deterministic FNV-1a 64-bit hex digest of trimmed instruction text (dedup & change tracking). */
export function trimmedInstructionDigest(text: string): string {
  const trimmed = text.trim();
  let h = 0xcbf29ce484222325n;
  const prime = 0x100000001b3n;
  const mask = 0xffffffffffffffffn;
  for (let i = 0; i < trimmed.length; i++) {
    h ^= BigInt(trimmed.charCodeAt(i));
    h = (h * prime) & mask;
  }
  return h.toString(16).padStart(16, '0');
}

export interface DiscoveredInstruction {
  path: string;
  content: string;
  digest: string;
  truncated: boolean;
}

function readCandidate(vfs: Vfs, relPath: string, maxSourceBytes: number): DiscoveredInstruction | null {
  const key = normalizePath(relPath);
  if (!key || !vfs.exists(`/${key}`)) return null;
  const st = vfs.stat(`/${key}`);
  if (!st || st.type !== 'file' || !vfs.isText(`/${key}`)) return null;
  const raw = vfs.readText(`/${key}`);
  const trimmed = raw.trim();
  if (!trimmed) return null;
  const truncated = trimmed.length > maxSourceBytes;
  const content = truncated
    ? `${trimmed.slice(0, maxSourceBytes)}\n…[truncated at ${maxSourceBytes.toLocaleString()} chars; read /${key} for the full file]`
    : trimmed;
  return {
    path: key,
    content,
    digest: trimmedInstructionDigest(trimmed),
    truncated,
  };
}

/** Collect ancestor directory prefixes (root `""` first, down to parent of `filePath`). */
function ancestorDirs(filePath: string): string[] {
  const clean = normalizePath(filePath);
  const parts = clean.split('/').filter(Boolean);
  const dirs = [''];
  let cur = '';
  for (let i = 0; i < parts.length - 1; i++) {
    cur = cur ? `${cur}/${parts[i]}` : parts[i]!;
    dirs.push(cur);
  }
  return dirs;
}

/**
 * Tracks discovered workspace instruction files across a conversation so:
 *  - Baseline project root instructions are injected once at turn 0.
 *  - Nested instructions in subdirectories are injected the first time a file there is touched.
 *  - Edited or deleted instruction files emit concise incremental updates on the next step.
 */
export class WorkspaceInstructions {
  private seen = new Map<string, string>(); // path -> digest
  private baselineLoaded = false;

  constructor(
    private readonly maxBytes = DEFAULT_MAX_BYTES,
    private readonly maxSourceBytes = DEFAULT_MAX_SOURCE_BYTES,
  ) {}

  reset(): void {
    this.seen.clear();
    this.baselineLoaded = false;
  }

  /** Record existing root instructions when resuming a saved session without re-injecting them. */
  seedBaseline(vfs: Vfs): void {
    if (this.baselineLoaded) return;
    this.baselineLoaded = true;
    for (const candidate of INSTRUCTION_CANDIDATES) {
      const item = readCandidate(vfs, candidate, this.maxSourceBytes);
      if (item) this.seen.set(item.path, item.digest);
    }
  }

  /** Load root-level workspace instructions once (for the start of a conversation). */
  loadBaseline(vfs: Vfs): string | null {
    if (this.baselineLoaded) return null;
    this.baselineLoaded = true;
    const found: DiscoveredInstruction[] = [];
    const seenDigests = new Set<string>();
    let budget = this.maxBytes;

    for (const candidate of INSTRUCTION_CANDIDATES) {
      const item = readCandidate(vfs, candidate, this.maxSourceBytes);
      if (!item) continue;
      if (seenDigests.has(item.digest)) {
        this.seen.set(item.path, item.digest);
        continue;
      }
      if (item.content.length > budget && found.length > 0) continue;
      seenDigests.add(item.digest);
      this.seen.set(item.path, item.digest);
      found.push(item);
      budget = Math.max(0, budget - item.content.length);
    }

    if (!found.length) return null;
    return formatInstructionsBlock(found, 'Workspace instructions loaded from this project:');
  }

  /**
   * Reconcile instruction files after tools touched `touchedPaths`:
   *  - discovers nested `AGENTS.md` / `OAIY.md` / `CLAUDE.md` in touched subdirectories,
   *  - detects modifications or removals of any previously seen instruction file.
   */
  reconcile(vfs: Vfs, touchedPaths: readonly string[]): string | null {
    const addedOrChanged: DiscoveredInstruction[] = [];
    const removed: string[] = [];

    // 1. Check previously loaded instruction files for updates or deletion.
    for (const [path, prevDigest] of [...this.seen.entries()]) {
      const current = readCandidate(vfs, path, this.maxSourceBytes);
      if (!current) {
        this.seen.delete(path);
        removed.push(path);
      } else if (current.digest !== prevDigest) {
        this.seen.set(path, current.digest);
        addedOrChanged.push(current);
      }
    }

    // Build digest set AFTER step 1 so a renamed file with identical content to a deleted file is not skipped.
    const seenDigests = new Set(this.seen.values());

    // 2. Discover newly created root instructions or nested instruction files in directories touched this step.
    const dirsToScan = new Set<string>(['']);
    for (const rawPath of touchedPaths) {
      const clean = normalizePath(rawPath);
      for (const dir of ancestorDirs(rawPath)) {
        dirsToScan.add(dir);
      }
      if (clean && vfs.stat(`/${clean}`)?.type === 'dir') {
        dirsToScan.add(clean);
      }
    }
    let budget = this.maxBytes;
    for (const dir of dirsToScan) {
      const candidates = dir === '' ? INSTRUCTION_CANDIDATES : (['AGENTS.md', 'OAIY.md', 'CLAUDE.md', 'AGENTS.local.md'] as const);
      for (const name of candidates) {
        const path = dir ? `${dir}/${name}` : name;
        if (this.seen.has(path)) continue;
        const item = readCandidate(vfs, path, this.maxSourceBytes);
        if (!item) continue;
        this.seen.set(path, item.digest);
        if (seenDigests.has(item.digest)) continue;
        if (item.content.length > budget && addedOrChanged.length > 0) continue;
        seenDigests.add(item.digest);
        addedOrChanged.push(item);
        budget = Math.max(0, budget - item.content.length);
      }
    }

    if (!addedOrChanged.length && !removed.length) return null;
    const parts: string[] = [];
    if (addedOrChanged.length) {
      parts.push(formatInstructionsBlock(addedOrChanged, 'Updated or nested workspace instructions:'));
    }
    if (removed.length) {
      parts.push(`Workspace instruction file${removed.length === 1 ? '' : 's'} removed (no longer apply): ${removed.map((p) => `/${p}`).join(', ')}`);
    }
    return parts.join('\n\n');
  }
}

function formatInstructionsBlock(items: readonly DiscoveredInstruction[], heading: string): string {
  const sections = items.map((item) => `--- /${item.path} ---\n${item.content}`);
  return `<workspace-instructions>\n${heading}\n\n${sections.join('\n\n')}\n</workspace-instructions>`;
}
