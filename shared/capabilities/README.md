# shared/capabilities

What a page can do where it is, decided in one place for the Agent (`app/`) and the flow editor (`platform/ui/`).

| File | What it decides |
|---|---|
| `host.ts` | Which window a page is in: one of OAIY's own windows, the desktop shell, or a tab in a browser. Read once, from the page's own globals, and without a request. |
| `features.ts`, `derive.ts` | Which features are on, from the window and the links its person saved (a paired desktop, an OAIY server). A feature that needs OAIY Desktop is hidden where none is behind the page. |
| `local.ts` | What counts as this computer or its network (the guard on the flow editor's media addresses). |

## The guard on local addresses, and what it cannot see

A tab that is not linked to OAIY Desktop does not load a picture, a video or a sound from an address on this computer or its network
(`platform/ui/src/lib/localMedia.ts` asks `isLocalHostname`), and a flow that comes in from outside arrives without the run outputs those
nodes show. OAIY's own window, the desktop shell and a linked tab load them as they always did.

`isLocalHostname` reads the address **as written**: loopback, the private ranges, link-local, carrier-grade NAT (Tailscale), the IPv6
forms that carry an IPv4 address (mapped, NAT64, 6to4, Teredo), the local-use NAT64 prefix, names that are only ever local (`.local`,
`.internal`, `.lan`, `.home.arpa`, `*.localhost`) and single-label names (`printer`). It does not look a name up and it does not follow a
redirect, so two things are **known limits**, on purpose:

- **A public address that redirects to this network** (`https://public.example.org/x.png` answering `302 Location: http://192.168.1.5/…`)
  looks public here. The browser follows the redirect.
- **A public name that resolves to a private address** (`192.168.1.5.nip.io`, or any name in a person's own DNS that points inside) looks
  public here.

For both, the browser's own question (Chrome's Local Network Access: "this site wants to connect to devices on your network") is what still
gates the request. The words in the flow editor's Connect card say the same (`platform/ui/src/lib/connectWords.ts`), and
`platform/ui/tests/local-media.mjs` writes the two down as cases so the day one changes is a decision.

The browser tests hold a page to the same list with **their own** implementation (`web/tests/e2e/local-ranges.mjs`), and
`web/tests/unit/local-ranges.test.mjs` makes the two agree over the IPv4 space and the IPv6 forms, so a range one leaves out is a failure
there and not a page that passes unseen.
