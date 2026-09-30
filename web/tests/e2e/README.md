# The web app's browser tests

Real origins in a real browser, against the folders the assembler makes (`web/scripts/assemble.mjs`), with a fake provider that is as awkward
as the real ones. From `web/`:

```sh
npm run test:e2e        # builds the providers origin, then every case in cases/ (Chromium, and Firefox where a case asks)
npm run test:mutations  # breaks the code on purpose, one thing at a time, and requires the tests to fail for each
```

The unit tests (`npm test`, which the CI gate runs) are in `../unit`; these need Chromium and Firefox and a few minutes, and are run by
hand. They never touch the desktop app: the harness's browser refuses the product's own ports (17972 among them) and records the attempt.

## What the leak scans can and cannot show

`leakscan.mjs` looks for a key in the places a page can keep bytes (its storage, what it was sent over the port, its memory) and in the
shapes a leak usually takes (as it is, reversed, base64, hex, percent-encoded, as character codes, any 20 characters of it, in text and in
the bytes of typed arrays). Every scan is shown to find what it claims to (`../unit/leakscan.test.mjs`, `cases/harness.test.mjs`), and the
mutation runner puts two leaks the scans must find into the holder (H1, a `Uint8Array`; H2, the key reversed).

A scan that finds nothing is not a proof. It does not find a key that is:

- **shifted**: every character moved, XORed or enciphered;
- **split**: in runs shorter than 20 characters, or over several messages, storage entries or objects that are never joined where a scan
  looks. A page that is sent the key 14 characters at a time holds no 20 of them together;
- **in a Blob or a File**: their bytes are in neither the heap snapshot nor anything reachable from `window` as text, and the buffer scan
  reads only ArrayBuffers and typed arrays;
- **held only by a closure**, in a **worker's** heap, or in another process.

What holds is the rule the scans check, not the scans: nothing the port says depends on a key (`web/providers/src/fixed.ts` says why even an
error's words are not passed on). Two things the rule does not cover, and no test here can:

- a provider you give a key to can see it, and the body of a **success** (a chat stream, or any path fetched, `/models` included) is passed
  to the app that asked as the provider sent it. A provider that echoes its `Authorization` header into a 2xx body hands the key to the
  app. The Providers page says so;
- while a call is being made the key is in the holder's memory.

## Layout

| Path | What |
|---|---|
| `harness.mjs`, `hosts.mjs` | the browser, and the hosts that serve the assembled folders by `Host` with the header templates applied |
| `fake-provider.mjs` | OpenAI and Anthropic stand-ins with the awkward behaviours (an opaque 401 on chat, paged lists, a redirect, a slow stream) |
| `leakscan.mjs` | the scans above |
| `fixtures/shell/` | the page that stands in for an app: it embeds the providers frame and speaks the port protocol |
| `cases/` | the tests: E1 same-site, E2 isolation, E3 keys never cross, E4 streaming, E6 failure classes, the headers, the Providers page, the harness itself |
| `mutations.mjs` | the mutation checks: `--only <word>` runs the ones whose name has the word; a run that is cut off puts the files back on the next start |
