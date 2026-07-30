# Bundled fonts

Two self-hosted web fonts so OAIY Desktop renders identically offline — no
Google Fonts request, which also means no third-party call from an app whose
whole pitch is that nothing leaves your device.

Both are **SIL Open Font License 1.1**, not Apache-2.0 like the rest of this
repo. That's fine — OFL is permissive and imposes nothing on the surrounding
code — but the OFL requires its text and copyright notice to travel with the
font binaries, which is what the `.OFL.txt` files next to them are for.

| File | Font | Upstream | License |
|---|---|---|---|
| `public-sans.woff2` | Public Sans (UI / body / the OAIY wordmark) | [uswds/public-sans](https://github.com/uswds/public-sans) | [`public-sans.OFL.txt`](public-sans.OFL.txt) |
| `jetbrains-mono.woff2` | JetBrains Mono (data / code / readouts) | [JetBrains/JetBrainsMono](https://github.com/JetBrains/JetBrainsMono) | [`jetbrains-mono.OFL.txt`](jetbrains-mono.OFL.txt) |

Both are variable weight (`font-weight: 300 700` in `src/styles.css`), which the
design system depends on — it asks for 580/620/650/720, and a static font would
snap those to the nearest cut.

If you ever swap or subset these, keep the matching `.OFL.txt` alongside and
don't reuse the reserved font names for a modified version (OFL §3).
