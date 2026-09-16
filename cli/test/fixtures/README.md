The localhost certificate and private key are public test fixtures, used only by
`cli-http-exit.mjs` for a local HTTPS server. They have no production use or authority.
The test trusts this certificate only in its child process with `NODE_EXTRA_CA_CERTS`;
TLS verification remains enabled and the operating system trust store is unchanged.

`realm-probe.json` is a one-block flow that reports what flow code can see of the
process running it (`typeof process`, `require`, `Buffer`, `globalThis.__TAURI__`) and
whether it can read `OAIY_SERVER_TOKEN`. `zipp-cli.mjs` runs it with that variable set;
it is also the quickest hand check that a built CLI runs flows on ZIPP:
`node dist/oaiy.mjs run test/fixtures/realm-probe.json`.
