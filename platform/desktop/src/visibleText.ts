/**
 * Text a person reads in the dry run of a restore, with the characters they cannot see made visible.
 *
 * A backup can hold words a model reads that are made to say one thing to a person and another to a model: tag characters, zero-width
 * and direction controls. The desktop makes them visible where it builds each description; this does the same for every text that
 * reaches the panel from it, so that nothing is drawn as it is written whatever sends it. The rule is the desktop's (`visible` in
 * `backup/parts.rs`): each run of them is said as `[3 invisible characters: U+E0041 U+E0042 U+E0043]`, the first six named.
 */

/** Whether a character is one a person cannot see (or that is drawn as nothing). A line break, a tab and a carriage return are not. */
export function isInvisible(codePoint: number): boolean {
  return (
    (codePoint >= 0x00 && codePoint <= 0x08) ||
    codePoint === 0x0b ||
    codePoint === 0x0c ||
    (codePoint >= 0x0e && codePoint <= 0x1f) ||
    (codePoint >= 0x7f && codePoint <= 0x9f) ||
    codePoint === 0xad ||
    codePoint === 0x034f ||
    codePoint === 0x061c ||
    codePoint === 0x115f ||
    codePoint === 0x1160 ||
    codePoint === 0x17b4 ||
    codePoint === 0x17b5 ||
    (codePoint >= 0x180b && codePoint <= 0x180f) ||
    (codePoint >= 0x200b && codePoint <= 0x200f) ||
    (codePoint >= 0x2028 && codePoint <= 0x202e) ||
    (codePoint >= 0x2060 && codePoint <= 0x206f) ||
    codePoint === 0x3164 ||
    (codePoint >= 0xfe00 && codePoint <= 0xfe0f) ||
    codePoint === 0xfeff ||
    codePoint === 0xffa0 ||
    (codePoint >= 0xfff0 && codePoint <= 0xfff8) ||
    (codePoint >= 0x1bca0 && codePoint <= 0x1bca3) ||
    (codePoint >= 0x1d173 && codePoint <= 0x1d17a) ||
    (codePoint >= 0xe0000 && codePoint <= 0xe0fff)
  );
}

/** `text` with each run of invisible characters said as what it is and how many there are. */
export function visibleText(text: string): string {
  let out = '';
  let run: number[] = [];
  const flush = () => {
    if (run.length === 0) return;
    const named = run
      .slice(0, 6)
      .map((c) => `U+${c.toString(16).toUpperCase().padStart(4, '0')}`)
      .join(' ');
    out += `[${run.length} invisible character${run.length === 1 ? '' : 's'}: ${named}${run.length > 6 ? ' …' : ''}]`;
    run = [];
  };
  for (const ch of text) {
    const code = ch.codePointAt(0) ?? 0;
    if (isInvisible(code)) {
      run.push(code);
    } else {
      flush();
      out += ch;
    }
  }
  flush();
  return out;
}
