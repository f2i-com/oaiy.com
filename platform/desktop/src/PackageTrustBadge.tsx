import type { PackageTrust, PackageTrustState } from './api';

/**
 * A small badge for what OAIY knows about a plugin's package: who signed it, or why it
 * is not trusted. The wording is the host's (`plugins/trust.rs`) where there is a
 * reason, and it goes in the tooltip so the card stays short.
 */

const CLASS: Record<PackageTrustState, string> = {
  verified: 'badge badge-ok',
  // Nothing is wrong with these, and nothing is vouched for either.
  'trusted-local': 'badge badge-neutral',
  'unsigned-dev': 'badge badge-neutral',
  // These are not started.
  unsigned: 'badge badge-err',
  quarantined: 'badge badge-err',
};

export function trustLabel(trust: PackageTrust): string {
  switch (trust.state) {
    case 'verified':
      return trust.publisher ? `verified · ${trust.publisher}` : 'verified';
    case 'trusted-local':
      return 'trusted by you';
    case 'unsigned-dev':
      return 'unsigned (dev)';
    case 'unsigned':
      return 'unsigned';
    case 'quarantined':
      return 'quarantined';
  }
}

function trustTitle(trust: PackageTrust): string {
  if (trust.state === 'verified') {
    const by = trust.publisher ? `Signed by ${trust.publisher}` : 'Signed by a publisher this app trusts';
    const key = trust.keyId ? ` (key ${trust.keyId})` : '';
    const release = trust.version ? `, release ${trust.version}` : '';
    return `${by}${key}${release}. Every file is as signed.`;
  }
  return trust.reason ?? trustLabel(trust);
}

export default function PackageTrustBadge({ trust }: { trust: PackageTrust }) {
  return (
    <span className={CLASS[trust.state]} title={trustTitle(trust)} data-trust={trust.state}>
      {trustLabel(trust)}
    </span>
  );
}
