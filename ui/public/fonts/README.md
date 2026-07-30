# Bundled font licences

The web app self-hosts Inter and JetBrains Mono (imported in `src/index.css` from
the `@fontsource-variable/*` packages, so the bundler emits the woff2 files as
hashed assets). It used to `<link>` them from `fonts.googleapis.com`, which sent
every page load to Google — in a product whose landing page sells "nothing leaves
your device".

Both are **SIL Open Font License 1.1**, not Apache-2.0 like the rest of this
repo. The OFL requires its text and copyright notice to travel with the font
binaries, and the emitted assets carry no licence of their own — so these two
files sit in `public/` and are copied verbatim into `dist/` alongside them.

| Font | Role | Upstream | Licence |
|---|---|---|---|
| Inter | UI, body, display (weight + tracking carry the display voice) | [rsms/inter](https://github.com/rsms/inter) | [`inter.OFL.txt`](inter.OFL.txt) |
| JetBrains Mono | data, code, readouts | [JetBrains/JetBrainsMono](https://github.com/JetBrains/JetBrainsMono) | [`jetbrains-mono.OFL.txt`](jetbrains-mono.OFL.txt) |

Both are **variable** weight, which the design system depends on — it asks for
580/620/650/720 and a static cut would snap those to the nearest available.

The OAIY Desktop self-hosts its own pair separately; see
`desktop/public/fonts/README.md`.
