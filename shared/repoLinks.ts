/**
 * Where OAIY's public repository and its releases are: the part of platform/ui/src/landing/repoLinks.ts that the download
 * offer needs (shared/downloads.ts), moved here so it has one home. That file re-exports these and keeps the folders the
 * landing pages link to.
 */

/** The public repository. OAIY is Apache-2.0, so the source link is part of the pitch. */
export const REPO_URL = 'https://github.com/f2i-com/oaiy.com';

/** The branch the folder links follow. */
export const REPO_BRANCH = 'main';

/** The newest release: the installers are its files. */
export const RELEASES_URL = `${REPO_URL}/releases/latest`;

/** Every release, for a file the pages do not offer themselves. */
export const RELEASES_ALL_URL = `${REPO_URL}/releases`;

/**
 * A file of one release. A release is published under the name of the tag that was pushed
 * (`tag_name: ${{ github.ref_name }}` in release.yml), which is `0.1.0` or `v0.1.0` (the workflow takes
 * both), so the address uses the tag as it is; the installers' names carry the version without the `v`
 * (`oaiy-desktop-<version>-windows-x64-setup.exe`), which shared/downloads.ts works out from the same tag.
 */
export function releaseAssetUrl(tag: string, file: string): string {
  return `${REPO_URL}/releases/download/${tag}/${file}`;
}
