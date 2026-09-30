/**
 * Words the Providers page and the embedded modal say about keys, in one place so that both say the same and a test can hold them to what
 * is true. (They are plain strings: they are set as text, never as markup.)
 */
import type { ProviderSummary } from '@oaiy/shared/providers/types';

/** What a provider's row says about its key. A key that is stored and cannot be opened says so, and what to do: it is not "stored". */
export function keyText(summary: Pick<ProviderSummary, 'hasKey' | 'keyUnreadable'>): string {
  if (summary.keyUnreadable) return 'key unreadable: re-enter it';
  return summary.hasKey ? 'key stored' : 'no key';
}
