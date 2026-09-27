/**
 * A project's saved voices. The speech service keeps voices by name for every
 * project at once, so a "Gary" made for one project would block (or be reused
 * by) another. Each project gets its own key: the agent names a voice "Gary",
 * and it is saved on the service as "Gary-<key>". The names the agent uses and
 * the names they are saved under are kept in `.botcomputer/voices.json`, so a
 * later chat in the same project speaks in the same voices.
 */
import type { Vfs } from '../vfs/vfs';
import type { VoiceInfo } from './media';

export const VOICES_FILE = '/.botcomputer/voices.json';

export interface ProjectVoices {
  /** This project's suffix for the voices it saves. */
  key: string;
  /** The agent's name for each voice, and the name it is saved under. */
  voices: Record<string, string>;
}

/** A saved name with another project's key: not offered here. */
const KEYED = /-[0-9a-f]{6}$/;

function newKey(): string {
  const bytes = new Uint8Array(3);
  crypto.getRandomValues(bytes);
  return [...bytes].map((b) => b.toString(16).padStart(2, '0')).join('');
}

export function readProjectVoices(vfs: Vfs): ProjectVoices {
  try {
    const parsed = JSON.parse(vfs.readText(VOICES_FILE)) as Partial<ProjectVoices>;
    if (typeof parsed.key === 'string' && /^[0-9a-f]{6}$/.test(parsed.key)) {
      const voices = Object.fromEntries(Object.entries(parsed.voices ?? {}).filter(([, v]) => typeof v === 'string')) as Record<string, string>;
      return { key: parsed.key, voices };
    }
  } catch {
    /* none yet */
  }
  return { key: newKey(), voices: {} };
}

export function writeProjectVoices(vfs: Vfs, voices: ProjectVoices): void {
  vfs.writeFile(VOICES_FILE, `${JSON.stringify(voices, null, 2)}\n`, { parents: true });
}

/**
 * The name to save a new voice under, for this project: only the letters,
 * digits, spaces, - and _ the service takes, within its 64 bytes.
 */
export function savedName(voices: ProjectVoices, name: string): string {
  let clean = name.replace(/[^\p{L}\p{N} _-]/gu, '').replace(/\s+/g, ' ').replace(/^[- ]+/, '').trim();
  while (new TextEncoder().encode(clean).length > 57) clean = clean.slice(0, -1).trimEnd();
  return `${clean || 'voice'}-${voices.key}`;
}

/** This project's own name for a voice, matched regardless of case. */
export function ownVoice(voices: ProjectVoices, name: string): string | undefined {
  const wanted = name.trim().toLowerCase();
  return Object.keys(voices.voices).find((k) => k.toLowerCase() === wanted);
}

/** The service's name for a voice the agent named (unchanged when it is not one of this project's). */
export function serviceVoice(voices: ProjectVoices, name: string | undefined): string | undefined {
  if (!name) return name;
  const own = ownVoice(voices, name);
  return own ? voices.voices[own] : name.trim();
}

/**
 * The saved voices this project can use, by the names the agent knows: its
 * own under their short names, and voices saved without a project key (made
 * in nrob Studio, say). Other projects' voices are left out.
 */
export function projectVoiceList(voices: ProjectVoices, saved: VoiceInfo[]): VoiceInfo[] {
  const byService = new Map(Object.entries(voices.voices).map(([mine, service]) => [service, mine]));
  return saved.flatMap((v) => {
    const mine = byService.get(v.name);
    if (mine) return [{ ...v, name: mine }];
    return KEYED.test(v.name) ? [] : [v];
  });
}
