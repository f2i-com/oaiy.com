/**
 * Where the site points into this repository.
 *
 * The site is `platform/ui` of a repository whose root also holds the Agent,
 * the crates and the docs, so a folder a page links to is a path from the
 * repository root, not from `ui/`. Two links named `desktop` and
 * `api/service-library` when this was a repository of its own, and went to
 * nothing when it was merged into this one. Every such link is built from
 * here, so it is written once and tests/landing.mjs can check that each folder
 * is really in the repository.
 */

/** The public repository. OAIY is Apache-2.0, so the source link is part of the pitch. */
export const REPO_URL = 'https://github.com/f2i-com/oaiy.com';

/** The branch the folder links follow. */
export const REPO_BRANCH = 'main';

/** The newest release: the installers are its files. */
export const RELEASES_URL = `${REPO_URL}/releases/latest`;

/** The folders of the repository the pages link to, as paths from its root. */
export const REPO_FOLDERS = {
  /** OAIY Desktop: its README is the installation documentation. */
  desktop: 'platform/desktop',
  /** The ready-made service templates the desktop page's library lists. */
  serviceLibrary: 'platform/api/service-library',
} as const;

/** A folder of the repository, on GitHub. */
export function repoFolderUrl(folder: keyof typeof REPO_FOLDERS): string {
  return `${REPO_URL}/tree/${REPO_BRANCH}/${REPO_FOLDERS[folder]}`;
}
