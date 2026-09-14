The localhost certificate and private key are public test fixtures, used only by
`cli-http-exit.mjs` for a local HTTPS server. They have no production use or authority.
The test trusts this certificate only in its child process with `NODE_EXTRA_CA_CERTS`;
TLS verification remains enabled and the operating system trust store is unchanged.
