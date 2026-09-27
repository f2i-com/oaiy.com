import { afterEach, describe, expect, it, vi } from 'vitest';
import { NetGate } from '../../src/gate/netgate';
import { EMPTY_MEDIA } from '../../src/agent/media';
import { runTool, type ToolContext } from '../../src/agent/tools';
import { Vfs } from '../../src/vfs/vfs';
import { VOICES_FILE, projectVoiceList, readProjectVoices, savedName, serviceVoice, writeProjectVoices } from '../../src/agent/voices';

afterEach(() => vi.unstubAllGlobals());

describe("a project's saved voices", () => {
  it('are saved under its own key and found again by the name the agent used', () => {
    const vfs = new Vfs();
    const project = readProjectVoices(vfs);
    expect(project.key).toMatch(/^[0-9a-f]{6}$/);
    const saved = savedName(project, 'Gary');
    expect(saved).toBe(`Gary-${project.key}`);
    project.voices.Gary = saved;
    writeProjectVoices(vfs, project);
    // A later chat reads the same key and voices from the project.
    const again = readProjectVoices(vfs);
    expect(again).toEqual(project);
    expect(vfs.readText(VOICES_FILE)).toContain(saved);
    expect(serviceVoice(again, 'Gary')).toBe(saved);
    expect(serviceVoice(again, 'gary')).toBe(saved);
    // Stock and unknown names pass through.
    expect(serviceVoice(again, 'alloy')).toBe('alloy');
    expect(serviceVoice(again, undefined)).toBeUndefined();
  });

  it('two projects each get a Gary of their own', () => {
    const a = readProjectVoices(new Vfs());
    const b = readProjectVoices(new Vfs());
    expect(a.key).not.toBe(b.key);
    expect(savedName(a, 'Gary')).not.toBe(savedName(b, 'Gary'));
  });

  it("list this project's voices by their short names, and leave out other projects'", () => {
    const project = { key: 'abc123', voices: { Gary: 'Gary-abc123' } };
    const listed = projectVoiceList(project, [
      { name: 'Gary-abc123', description: 'tired chef' },
      { name: 'Gary-ffee00', description: "another project's" },
      { name: 'Narrator', description: 'made in nrob Studio' },
    ]);
    expect(listed).toEqual([
      { name: 'Gary', description: 'tired chef' },
      { name: 'Narrator', description: 'made in nrob Studio' },
    ]);
  });

  it('are saved under names the service takes', () => {
    const project = { key: 'abc123', voices: {} };
    expect(savedName(project, "Dr. O'Brien")).toBe('Dr OBrien-abc123');
    expect(savedName(project, '  Old   Man  ')).toBe('Old Man-abc123');
    expect(savedName(project, 'Zoë')).toBe('Zoë-abc123');
    expect(savedName(project, '!!!')).toBe('voice-abc123');
    const long = savedName(project, 'é'.repeat(80));
    expect(new TextEncoder().encode(long).length).toBeLessThanOrEqual(64);
  });

  it("are not designed again by accident: a character keeps their voice unless it is replaced", async () => {
    const asked: string[] = [];
    vi.stubGlobal('fetch', vi.fn(async (_url: string, init?: RequestInit) => {
      const body = JSON.parse(String(init?.body)) as { name: string; description: string };
      asked.push(body.description);
      return new Response(JSON.stringify({ name: body.name }), { headers: { 'Content-Type': 'application/json' } });
    }));
    const media = { ...EMPTY_MEDIA, baseUrl: 'http://127.0.0.1:8080', speechModel: 'qwen3-tts' };
    const vfs = new Vfs();
    const ctx: ToolContext = { vfs, gate: new NetGate(), reads: new Map(), shell: { cwd: '/', env: {} }, media: () => media };
    const create = (description: string, replace?: boolean) => runTool({ id: description, name: 'create_voice', input: { name: 'Gary', description, ...(replace ? { replace } : {}) } }, ctx);
    await create('gruff');
    const again = await create('squeaky');
    expect(again.content).toContain('"Gary" is already this project\'s voice');
    expect(asked).toEqual(['gruff']);
    await create('squeaky', true);
    expect(asked).toEqual(['gruff', 'squeaky']);
    const project = readProjectVoices(vfs);
    expect(project.voices).toEqual({ Gary: `Gary-${project.key}` });
  });

  it('made at the same time share the project key, and none is lost', async () => {
    // The service answers in turn, the last voice first.
    const pending: Array<() => void> = [];
    vi.stubGlobal('fetch', vi.fn(async (_url: string, init?: RequestInit) => {
      const body = JSON.parse(String(init?.body)) as { name: string };
      await new Promise<void>((done) => pending.push(done));
      return new Response(JSON.stringify({ name: body.name }), { headers: { 'Content-Type': 'application/json' } });
    }));
    const media = { ...EMPTY_MEDIA, baseUrl: 'http://127.0.0.1:8080', speechModel: 'qwen3-tts' };
    const vfs = new Vfs();
    const ctx: ToolContext = { vfs, gate: new NetGate(), reads: new Map(), shell: { cwd: '/', env: {} }, media: () => media };
    const made = ['Gary', 'Mary'].map((name, i) => runTool({ id: String(i), name: 'create_voice', input: { name, description: 'a voice' } }, ctx));
    while (pending.length < 2) await new Promise((r) => setTimeout(r, 1));
    pending[1]();
    await made[1];
    pending[0]();
    await made[0];
    const project = readProjectVoices(vfs);
    expect(project.voices).toEqual({ Gary: `Gary-${project.key}`, Mary: `Mary-${project.key}` });
  });
});
