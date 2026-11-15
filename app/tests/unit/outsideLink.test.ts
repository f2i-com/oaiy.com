import { describe, expect, it } from 'vitest';
import { OPEN_SETTLE_MS, OUTSIDE_LINK_LIMIT, OpenGate, outsideLinkOf, shownName } from '../../src/preview/outsideLink';

describe('outsideLinkOf: what a previewed SoftN app hands its page when a link in it goes elsewhere', () => {
  it('reads a web page, a mail address and a phone number, as the browser reads each', () => {
    expect(outsideLinkOf('https://example.com/a?b=1#c')).toEqual({ href: 'https://example.com/a?b=1#c', kind: 'web', name: 'example.com' });
    expect(outsideLinkOf('http://example.com:8080')).toEqual({ href: 'http://example.com:8080/', kind: 'web', name: 'example.com:8080' });
    expect(outsideLinkOf('mailto:someone@example.com?subject=Hi')).toEqual({ href: 'mailto:someone@example.com?subject=Hi', kind: 'mail', name: 'someone@example.com' });
    expect(outsideLinkOf('tel:+61400000000')).toEqual({ href: 'tel:+61400000000', kind: 'phone', name: '+61400000000' });
  });

  it("names a host as the browser reads it, so one made to look like another site's is seen as itself", () => {
    expect(outsideLinkOf('https://bücher.example/')?.name).toBe('xn--bcher-kva.example');
    expect(outsideLinkOf('https://example.com.evil.example/login')?.name).toBe('example.com.evil.example');
    // A mail address's domain too: a Cyrillic "а" in "apple.com" is not shown as an "a".
    expect(outsideLinkOf('mailto:support@%D0%B0pple.com')?.name).toBe('support@xn--pple-43d.com');
    // Every address a mail goes to is named, however many.
    const many = Array.from({ length: 12 }, (_, i) => `person${i}@example.com`).join(',');
    expect(outsideLinkOf(`mailto:${many},hidden@evil.example`)?.name).toContain('hidden@evil.example');
  });

  it("names every address a mail goes to: its query's to, cc and bcc as well as the path's", () => {
    const mail = outsideLinkOf('mailto:support@bank.example?to=billing@evil.example&CC=a@evil.example&bcc=spy@evil.example&subject=Refund');
    expect(mail?.name).toBe('support@bank.example, billing@evil.example, a@evil.example, spy@evil.example');
    expect(outsideLinkOf('mailto:?to=someone%2Btag@example.com')?.name).toBe('someone+tag@example.com');
    for (const value of ['mailto:a@example.com?cc=%0Ab@evil.example', 'mailto:a@example.com?bcc=not-an-address', 'mailto:?subject=Hi']) {
      expect(outsideLinkOf(value), value).toBeNull();
    }
  });

  it('shows a long name by its end, which says where it goes', () => {
    const host = `${'a'.repeat(60)}.${'b'.repeat(60)}.example.com`;
    expect(shownName(host)).toMatch(/^….*\.example\.com$/);
    expect(shownName(host).length).toBe(80);
    expect(shownName('example.com')).toBe('example.com');
  });

  it('takes nothing else', () => {
    for (const value of [
      undefined, null, 7, {}, '', 'not an address', '/about.html', '//example.com/x',
      'javascript:alert(1)', 'data:text/html,x', 'blob:null/1', 'file:///C:/x', 'ftp://example.com/',
      'https://bank.example.com@evil.example/', 'https://user:pass@example.com/', 'mailto:', 'tel:',
      `https://example.com/${'a'.repeat(OUTSIDE_LINK_LIMIT)}`, 'https://example.com/\u0007', 'mailto:%0Asomeone@example.com',
      // What turns text round, takes no space, or is a control of the second range, written or %-encoded.
      'mailto:moc.elppa@%E2%80%AEevil.example', 'mailto:support@app%E2%80%8Ble.com', 'mailto:a%C2%85b@example.com', 'tel:+61%E2%80%AE400',
      'mailto:support@ex\u00adample.com', 'mailto:not-an-address', 'mailto:@example.com', 'mailto:someone@',
      // A host longer than a name server answers for.
      `https://${'a.'.repeat(130)}example/`,
    ]) {
      expect(outsideLinkOf(value), String(value).slice(0, 40)).toBeNull();
    }
  });

  it("offers no address on the page's own host, whatever its scheme or port, and others as before", () => {
    const own = 'http://tauri.localhost';
    expect(outsideLinkOf('http://tauri.localhost/index.html', own)).toBeNull();
    expect(outsideLinkOf('http://TAURI.localhost:80/x', own)).toBeNull();
    expect(outsideLinkOf('https://tauri.localhost/x', own)).toBeNull();
    expect(outsideLinkOf('http://tauri.localhost:8080/x', own)).toBeNull();
    expect(outsideLinkOf('https://example.com/', own)?.name).toBe('example.com');
    expect(outsideLinkOf('mailto:someone@tauri.localhost', own)?.kind).toBe('mail');
    expect(outsideLinkOf('https://example.com/', 'null')?.name).toBe('example.com');
  });
});

describe('OpenGate: the bar opens nothing a press meant for the app reaches', () => {
  it('opens a click that began once the address had been shown a while', () => {
    const gate = new OpenGate();
    gate.arm(1000);
    gate.press(1000 + OPEN_SETTLE_MS);
    expect(gate.click(1000 + OPEN_SETTLE_MS + 80, false)).toBe(true);
  });

  it('opens nothing for a press begun too soon, and starts the wait over from it', () => {
    const gate = new OpenGate();
    gate.arm(1000);
    // A double click meant for the app: its second press lands on the bar as it appears.
    gate.press(1100);
    expect(gate.click(1150, false)).toBe(false);
    // That press and its click each start the wait over: ready OPEN_SETTLE_MS after the click, at 1750.
    gate.press(1749);
    expect(gate.click(1800, false)).toBe(false);
    // The last click (1800) started it over again: a press from 2400 on opens.
    gate.press(1800 + OPEN_SETTLE_MS + 10);
    expect(gate.click(1800 + OPEN_SETTLE_MS + 60, false)).toBe(true);
  });

  it('times a click from the keyboard or a screen reader as its own press', () => {
    const gate = new OpenGate();
    gate.arm(0);
    expect(gate.click(OPEN_SETTLE_MS - 1, true)).toBe(false);
    expect(gate.click(2 * OPEN_SETTLE_MS, true)).toBe(true);
  });

  it('times a click from the keyboard from when Open was given the focus as well', () => {
    // The bar takes the focus from the app's frame: a Tab and a Space typed for the app reached Open and opened.
    const gate = new OpenGate();
    gate.arm(0);
    gate.focus(5000);
    expect(gate.click(5050, true)).toBe(false);
    expect(gate.click(5050 + 2 * OPEN_SETTLE_MS, true)).toBe(true);
    // A pointer's click is timed from its own press, whatever focus its press gave Open.
    gate.arm(0);
    gate.focus(OPEN_SETTLE_MS + 10);
    gate.press(OPEN_SETTLE_MS + 10);
    expect(gate.click(OPEN_SETTLE_MS + 60, false)).toBe(true);
  });

  it("waits as long as it is told for an address that took another's place", () => {
    const gate = new OpenGate();
    gate.arm(0, 2 * OPEN_SETTLE_MS);
    gate.press(OPEN_SETTLE_MS + 50);
    expect(gate.click(OPEN_SETTLE_MS + 100, false)).toBe(false);
    gate.press(4 * OPEN_SETTLE_MS);
    expect(gate.click(4 * OPEN_SETTLE_MS + 50, false)).toBe(true);
  });

  it('waits again for each newer address', () => {
    const gate = new OpenGate();
    gate.arm(0);
    gate.arm(5000);
    gate.press(5100);
    expect(gate.click(5150, false)).toBe(false);
  });

  it('opens nothing before any address is shown', () => {
    expect(new OpenGate().click(1e9, true)).toBe(false);
  });
});
