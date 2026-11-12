/**
 * A chip's words begin with a capital, on every page: "Running", "Timed out", "Needs setup". A state's word is often
 * kept in lower case because sentences use it too ("it timed out"); a chip shows it through this.
 */
export const capital = (words: string): string => (words ? words.charAt(0).toUpperCase() + words.slice(1) : words);
