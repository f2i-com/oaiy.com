/** A port's kind of value: its dot's colour on a card, and its name in words. */

/**
 * What a port carries, as its dot's colour (one set for every node, the Macro node's): text green, numbers amber,
 * pictures pink, video blue, sound purple, files orange, lists and objects cyan, anything grey. See `.oaiy-port.port-*`
 * in the editor's stylesheet.
 */
export function portTypeClass(type: string | undefined): string {
  const t = String(type ?? 'any').toLowerCase().replace(/\[\]$/, '');
  if (t === 'string' || t === 'text') return 'port-text';
  if (t === 'number' || t === 'boolean' || t === 'integer') return 'port-number';
  if (t === 'image') return 'port-image';
  if (t === 'video') return 'port-video';
  if (t === 'audio') return 'port-audio';
  if (t === 'file' || t === 'path') return 'port-file';
  if (t === 'array' || t === 'object' || t === 'json') return 'port-data';
  return 'port-any';
}

/** A port's kind of value in words, for its tooltip. */
export function typeWords(type: string | undefined): string {
  const t = String(type ?? 'any').toLowerCase();
  const list = t.endsWith('[]');
  const base = list ? t.slice(0, -2) : t;
  const word = base === 'string' ? 'text' : base === 'boolean' ? 'yes or no' : base === 'any' ? 'anything' : base === 'array' ? 'a list' : base === 'object' ? 'an object' : base;
  return list ? `a list of ${word}` : word;
}

