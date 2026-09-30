import { describe, expect, it } from 'vitest';
import { inPlainSight, type RestorePreview } from './api';
import { isInvisible, visibleText } from './visibleText';

const TAGS = Array.from({ length: 25 }, (_, i) => String.fromCodePoint(0xe0041 + (i % 26))).join('');

const preview = (over: Partial<RestorePreview> = {}): RestorePreview => ({
  inspectId: 'i1',
  fileName: 'a.oaiybackup',
  createdAt: '2026-09-30T00:00:00Z',
  appVersion: '0.1.0',
  platform: 'windows',
  includesKeys: false,
  categories: [],
  lacks: [],
  partial: [],
  excluded: [],
  redo: [],
  totalFiles: 1,
  totalBytes: 10,
  classes: [],
  items: [],
  notRestored: [],
  keys: { inBackup: false },
  notes: [],
  ...over,
});

describe('the characters a person cannot see are made visible', () => {
  it('says each run of them as what it is and how many there are', () => {
    expect(visibleText(`Ask how the visit went${TAGS}`)).toBe('Ask how the visit went[25 invisible characters: U+E0041 U+E0042 U+E0043 U+E0044 U+E0045 U+E0046 …]');
    expect(visibleText('a​b‮c')).toBe('a[1 invisible character: U+200B]b[1 invisible character: U+202E]c');
    expect(visibleText('a​‍‌b')).toBe('a[3 invisible characters: U+200B U+200D U+200C]b');
  });

  it('leaves a text that has none as it is, and a line break, a tab and ordinary text with them', () => {
    for (const text of ['', 'Prefers texts', 'line one\nline two\tand a tab\r\n', 'café 你好 \u{1F468}', 'a b']) expect(visibleText(text)).toBe(text);
    expect(isInvisible(0x09) || isInvisible(0x0a) || isInvisible(0x0d)).toBe(false);
    expect(isInvisible(0x200b) && isInvisible(0xe0041) && isInvisible(0x202e) && isInvisible(0x2028) && isInvisible(0xfe0f)).toBe(true);
  });

  it('is the same on every text of a dry run that came from a backup, whoever sent it', () => {
    const seen = inPlainSight(
      preview({
        lacks: [`lack${TAGS}`],
        partial: ['partial​'],
        redo: ['redo‮'],
        notes: [`note${TAGS}`],
        items: [{ class: 'agentData', name: `agent/brief${TAGS}`, title: `The brief${TAGS}`, what: `Says: "Ask how the visit went${TAGS}"` }],
        notRestored: [{ name: `n${TAGS}`, why: `w${TAGS}` }],
        excluded: [{ pattern: `p${TAGS}`, reason: `r${TAGS}`, redo: `d${TAGS}` }],
      }),
    );
    const all = JSON.stringify(seen);
    expect([...all].some((ch) => isInvisible(ch.codePointAt(0) ?? 0) && ch !== '\n')).toBe(false);
    expect(seen.items[0].what).toContain('[25 invisible characters: U+E0041');
    expect(seen.items[0].title).toContain('The brief[25 invisible characters');
    expect(seen.excluded[0].redo).toContain('[25 invisible characters');
    expect(seen.notRestored[0].why).toContain('[25 invisible characters');
    expect(seen.redo[0]).toBe('redo[1 invisible character: U+202E]');
  });
});
