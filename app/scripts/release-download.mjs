/**
 * A file of a GitHub release, for the two fetchers (fetch-zipp.mjs, fetch-softn.mjs).
 *
 * Two routes, tried in this order:
 *   1. GitHub's API: the release's asset list (api.github.com/repos/<owner>/<repo>/releases/tags/<tag>), then the
 *      asset's own address there. On 2026-11-08 the plain download addresses answered 504 for every f2i-com release
 *      while this route served the same files.
 *   2. the plain download address (github.com/<owner>/<repo>/releases/download/<tag>/<name>), as before.
 *
 * The route changes where the bytes come from, never what is accepted: the callers check every file against the
 * digests they pin, exactly as they did.
 *
 * A token from GITHUB_TOKEN or GH_TOKEN (a workflow passes one; it lifts the API's limit of 60 anonymous requests an
 * hour) is sent to GitHub's own API host and nowhere else: the asset's address redirects to GitHub's file storage,
 * and that redirect is followed by hand, without the token.
 */
const API_HOST = 'api.github.com';
const TIMEOUT_MS = 120_000;

const apiHeaders = (accept, env) => {
  const token = env.GITHUB_TOKEN || env.GH_TOKEN;
  return { Accept: accept, 'User-Agent': 'oaiy-agent-fetch', 'X-GitHub-Api-Version': '2022-11-28', ...(token ? { Authorization: `Bearer ${token}` } : {}) };
};

const onApi = (url) => {
  try {
    const parsed = new URL(url);
    return parsed.protocol === 'https:' && parsed.host === API_HOST;
  } catch {
    return false;
  }
};

async function bytesOf(response, url) {
  if (!response.ok) throw new Error(`GET ${url}: HTTP ${response.status}`);
  return new Uint8Array(await response.arrayBuffer());
}

/** The asset's bytes through the API. Throws with what went wrong (the caller then tries the plain address). */
async function throughApi({ repository, tag, name, fetchImpl, env }) {
  const listUrl = `https://${API_HOST}/repos/${repository}/releases/tags/${tag}`;
  const list = await fetchImpl(listUrl, { headers: apiHeaders('application/vnd.github+json', env), signal: AbortSignal.timeout(TIMEOUT_MS) });
  if (!list.ok) throw new Error(`GET ${listUrl}: HTTP ${list.status}`);
  const asset = ((await list.json()).assets ?? []).find((a) => a?.name === name);
  if (!asset) throw new Error(`the ${tag} release lists no ${name}`);
  // (an address anywhere but the API's own host gets no token: it is not asked this way at all)
  if (!onApi(asset.url)) throw new Error(`the ${tag} release gives ${name} an address that is not GitHub's API`);
  // The API answers with the bytes, or with a redirect to GitHub's file storage: followed here, without the token.
  const first = await fetchImpl(asset.url, { headers: apiHeaders('application/octet-stream', env), redirect: 'manual', signal: AbortSignal.timeout(TIMEOUT_MS) });
  if (first.status >= 300 && first.status < 400) {
    const location = first.headers.get('location');
    if (!location) throw new Error(`GET ${asset.url}: HTTP ${first.status} without a location`);
    const next = new URL(location, asset.url).href;
    return bytesOf(await fetchImpl(next, { redirect: 'follow', ...(onApi(next) ? { headers: apiHeaders('application/octet-stream', env) } : {}), signal: AbortSignal.timeout(TIMEOUT_MS) }), next.split('?')[0]);
  }
  return bytesOf(first, asset.url);
}

/**
 * The bytes of `name` in the release `tag` of `repository` ("owner/repo"): the API's asset address first, the plain
 * download address second. A failure names both. `fetchImpl` and `env` are for the tests.
 */
export async function downloadReleaseFile({ repository, tag, name, fetchImpl = globalThis.fetch, env = process.env }) {
  let apiError;
  try {
    return await throughApi({ repository, tag, name, fetchImpl, env });
  } catch (error) {
    apiError = error;
  }
  const url = `https://github.com/${repository}/releases/download/${tag}/${name}`;
  try {
    return await bytesOf(await fetchImpl(url, { redirect: 'follow', signal: AbortSignal.timeout(TIMEOUT_MS) }), url);
  } catch (error) {
    throw new Error(`through GitHub's API: ${apiError.message}; then ${error.message}`);
  }
}
