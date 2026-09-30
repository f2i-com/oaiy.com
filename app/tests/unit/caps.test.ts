import { describe, expect, it } from 'vitest';
import { readHost, type HostEnv } from '@oaiy/shared/capabilities/host';
import { agentCaps } from '../../src/desktop/caps';
import mainSource from '../../src/main.ts?raw';

/** The Agent's features that are the desktop's: the phone, the calendar's tools, flows as tools, OAIY's own tools and "Set up OAIY". */
const DESKTOPS = ['phone', 'calendar', 'flowTools', 'control', 'setup'] as const;
const PAIRED = { origin: 'http://127.0.0.1:17972' };

const at = (env: HostEnv) => readHost(env);
const OAIY_WINDOW = at({ location: { hostname: 'oaiy.localhost', protocol: 'http:' }, __OAIY_DESKTOP__: { origin: PAIRED.origin, token: 't' } });
const SHELL = at({ location: { hostname: 'botcomputer.localhost', protocol: 'http:' } });
const TAB = at({ location: { hostname: 'agent.example.org', protocol: 'https:' } });
const DEV = at({ location: { hostname: 'localhost', protocol: 'http:' } });

const on = (caps: ReturnType<typeof agentCaps>) => DESKTOPS.filter((id) => caps.features[id]);

describe("what the Agent can do, from where it is and the desktop it is paired with", () => {
  it("with the desktop host every control that existed still exists: OAIY's own window has the phone, the calendar, flows, OAIY's tools and setup", () => {
    expect(on(agentCaps(OAIY_WINDOW, PAIRED))).toEqual([...DESKTOPS]);
    expect(agentCaps(OAIY_WINDOW, PAIRED).mode).toBe('paired');
  });

  it('the desktop shell and a tab have them once they are paired, and none before: the pairing chip is what they show', () => {
    for (const host of [SHELL, TAB, DEV]) {
      expect(on(agentCaps(host, PAIRED))).toEqual([...DESKTOPS]);
      expect(on(agentCaps(host, null))).toEqual([]);
      expect(agentCaps(host, null).mode).toBe('standalone');
    }
  });

  it("a page that is paired again with another desktop keeps them, and one that forgets its pairing loses them at once (the caps follow the desktop, they are not remembered)", () => {
    expect(on(agentCaps(TAB, { origin: 'http://127.0.0.1:17972' }))).toEqual([...DESKTOPS]);
    expect(on(agentCaps(TAB, { origin: 'http://192.168.1.50:17972' }))).toEqual([...DESKTOPS]);
    expect(on(agentCaps(TAB, null))).toEqual([]);
  });

  it("OAIY's own window has them on with no pairing at all: that is why each control also asks for the desktop it talks to, and the caps can only hide", () => {
    expect(on(agentCaps(OAIY_WINDOW, null))).toEqual([...DESKTOPS]);
  });

  it('the Agent asks for every one of them where it shows or offers the control', () => {
    for (const id of DESKTOPS) expect(mainSource, id).toContain(`canDo('${id}')`);
    // Not by a test of its own: the desktop's presence is asked of the caps, once (desktop/caps.ts).
    expect(mainSource).toContain('agentCaps(HOST, desktop)');
    expect(mainSource).not.toContain('desktop ? [...flowTools');
  });
});
