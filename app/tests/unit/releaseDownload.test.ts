// scripts/release-download.mjs: where the two fetchers (fetch-zipp.mjs, fetch-softn.mjs) get a release's file from.
// GitHub's API asset address first, the plain download address second; a token only to GitHub's API host.
import { describe, expect, it } from 'vitest';
// @ts-expect-error a plain .mjs script, with no types of its own
import { downloadReleaseFile } from '../../scripts/release-download.mjs';
import FETCH_ZIPP from '../../scripts/fetch-zipp.mjs?raw';
import FETCH_SOFTN from '../../scripts/fetch-softn.mjs?raw';
import VITE_CONFIG from '../../vite.config.ts?raw';

const REPO = 'f2i-com/softn.com';
const TAG = 'v0.0.32';
const NAME = 'softn-formlogic-runtime-v0.0.32.zip';
const LIST = `https://api.github.com/repos/${REPO}/releases/tags/${TAG}`;
const ASSET = `https://api.github.com/repos/${REPO}/releases/assets/4242`;
const STORED = 'https://release-assets.githubusercontent.com/github-production-release-asset/1/abc?sig=secret';
const PLAIN = `https://github.com/${REPO}/releases/download/${TAG}/${NAME}`;
const BYTES = new Uint8Array([80, 75, 3, 4]);

type Call = { url: string; authorization: string | undefined; redirect: string | undefined };

/** A fetch that answers by address and records what each request carried. */
function network(answers: Record<string, () => Response>) {
  const calls: Call[] = [];
  const fetchImpl = async (url: string, init: { headers?: Record<string, string>; redirect?: string } = {}): Promise<Response> => {
    calls.push({ url, authorization: init.headers?.Authorization, redirect: init.redirect });
    const answer = answers[url];
    if (!answer) throw new TypeError(`fetch failed: ${url}`);
    return answer();
  };
  return { calls, fetchImpl };
}

const list = () => new Response(JSON.stringify({ assets: [{ name: 'other.zip', url: 'https://api.github.com/repos/x/y/releases/assets/1' }, { name: NAME, url: ASSET }] }), { status: 200 });
const redirect = () => new Response(null, { status: 302, headers: { location: STORED } });
const bytes = () => new Response(BYTES, { status: 200 });

describe("a release's file, for the fetchers", () => {
  it("comes through GitHub's API first: the asset list, the asset's address, then where that redirects", async () => {
    const { calls, fetchImpl } = network({ [LIST]: list, [ASSET]: redirect, [STORED]: bytes });
    const got = await downloadReleaseFile({ repository: REPO, tag: TAG, name: NAME, fetchImpl, env: {} });
    expect([...got]).toEqual([...BYTES]);
    expect(calls.map((c) => c.url)).toEqual([LIST, ASSET, STORED]);
    // the plain address was not needed
    expect(calls.some((c) => c.url === PLAIN)).toBe(false);
  });

  it("sends a token to GitHub's API host and to nothing else: the redirect to file storage is followed by hand, without it", async () => {
    for (const env of [{ GITHUB_TOKEN: 'ghs_secret' }, { GH_TOKEN: 'ghs_secret' }]) {
      const { calls, fetchImpl } = network({ [LIST]: list, [ASSET]: redirect, [STORED]: bytes });
      await downloadReleaseFile({ repository: REPO, tag: TAG, name: NAME, fetchImpl, env });
      expect(calls.find((c) => c.url === LIST)?.authorization).toBe('Bearer ghs_secret');
      expect(calls.find((c) => c.url === ASSET)).toMatchObject({ authorization: 'Bearer ghs_secret', redirect: 'manual' });
      expect(calls.find((c) => c.url === STORED)?.authorization).toBeUndefined();
    }
    // and with no token, none is made up
    const { calls, fetchImpl } = network({ [LIST]: list, [ASSET]: bytes });
    await downloadReleaseFile({ repository: REPO, tag: TAG, name: NAME, fetchImpl, env: {} });
    expect(calls.every((c) => c.authorization === undefined)).toBe(true);
  });

  it('falls back to the plain download address when the API cannot give it, and that request carries no token', async () => {
    const cases: Array<[string, Record<string, () => Response>]> = [
      ['the API is down', { [PLAIN]: bytes }],
      ['the API is rate limited', { [LIST]: () => new Response('rate limited', { status: 403 }), [PLAIN]: bytes }],
      ['the release lists no such file', { [LIST]: () => new Response(JSON.stringify({ assets: [] }), { status: 200 }), [PLAIN]: bytes }],
      ['the asset answers an error', { [LIST]: list, [ASSET]: () => new Response('gone', { status: 504 }), [PLAIN]: bytes }],
      ['the asset is listed at an address that is not the API', { [LIST]: () => new Response(JSON.stringify({ assets: [{ name: NAME, url: 'https://example.test/asset' }] }), { status: 200 }), [PLAIN]: bytes }],
    ];
    for (const [name, answers] of cases) {
      const { calls, fetchImpl } = network(answers);
      const got = await downloadReleaseFile({ repository: REPO, tag: TAG, name: NAME, fetchImpl, env: { GITHUB_TOKEN: 'ghs_secret' } });
      expect([...got], name).toEqual([...BYTES]);
      expect(calls.at(-1), name).toMatchObject({ url: PLAIN, authorization: undefined });
      // (the token never went anywhere but the API)
      expect(calls.filter((c) => c.authorization).every((c) => new URL(c.url).host === 'api.github.com'), name).toBe(true);
    }
  });

  it('names both routes when neither gives the file', async () => {
    const { fetchImpl } = network({ [LIST]: () => new Response('', { status: 504 }), [PLAIN]: () => new Response('', { status: 504 }) });
    await expect(downloadReleaseFile({ repository: REPO, tag: TAG, name: NAME, fetchImpl, env: {} })).rejects.toThrow(`through GitHub's API: GET ${LIST}: HTTP 504; then GET ${PLAIN}: HTTP 504`);
  });

  it('is what both fetchers use, each still holding what arrives to its pinned digest', () => {
    expect(FETCH_ZIPP).toContain("downloadReleaseFile({ repository: 'f2i-com/zipp.org', tag: RELEASE, name })");
    expect(FETCH_SOFTN).toContain("downloadReleaseFile({ repository: 'f2i-com/softn.com', tag: RELEASE, name: ZIP })");
    // neither asks the plain address itself any more
    expect(FETCH_ZIPP).not.toContain('releases/download');
    expect(FETCH_SOFTN).not.toContain('releases/download');
    // the checks that decide what is accepted are where they were
    expect(FETCH_ZIPP).toContain("if (sha256(sumsBytes) !== SUMS_SHA256) throw new Error(");
    expect(FETCH_ZIPP).toContain('if (sums.get(MODULE) !== sha256(wasm)) throw new Error(');
    expect(FETCH_SOFTN).toContain('if (sha256(zip) !== ZIP_SHA256) throw new Error(');
  });
});

describe("the service worker's cache name", () => {
  it("is made from the installed SoftN runtime's stamp too, so a move to another runtime on the same ZIPP replaces the cache", () => {
    expect(VITE_CONFIG).toContain("for (const file of ['zipp/zipp_wasm.js', 'zipp/SOURCE.json', 'softn/SOURCE.json']) {");
  });
});
