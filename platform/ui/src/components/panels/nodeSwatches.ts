/**
 * A node kind's colour as a swatch: the stripe beside a node's name in the
 * inspector. These are the node kinds' own colours (as on the canvas), not
 * the editor's chrome, so they are literal Tailwind colours rather than theme
 * tokens (tests/css-tokens.mjs leaves this file alone for that reason).
 *
 * Static literals: Tailwind only emits classes it sees whole during its
 * content scan, so an interpolated `bg-${color}-500` would purge to nothing
 * for any colour not written out elsewhere.
 */
export const SWATCH_CLASS: Record<string, string> = {
  amber: 'bg-amber-500', blue: 'bg-blue-500', cyan: 'bg-cyan-500',
  emerald: 'bg-emerald-500', gray: 'bg-gray-500', green: 'bg-green-500',
  indigo: 'bg-indigo-500', orange: 'bg-orange-500', pink: 'bg-pink-500',
  purple: 'bg-purple-500', red: 'bg-red-500', slate: 'bg-slate-500',
  teal: 'bg-teal-500', violet: 'bg-violet-500',
};
