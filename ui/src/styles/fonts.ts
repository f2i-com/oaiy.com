/**
 * Self-hosted web fonts.
 *
 * The app used to `<link>` Inter and JetBrains Mono from fonts.googleapis.com,
 * so every page load told Google who was reading — in a product whose landing
 * page sells "nothing leaves your device", and whose desktop companion already
 * self-hosts the same two families from disk.
 *
 * **Why this is a .ts file and not an `@import` in index.css.** That was the
 * first attempt and it produced 14 `@font-face` rules that resolved to nothing:
 * Tailwind v4 handles `@import` itself and inlines the file contents *without
 * rebasing the relative `url()`s inside them*. So `url(./files/inter-…woff2)`
 * survived into the built stylesheet still relative to the package, pointing at
 * `dist/assets/files/…`, which does not exist — and because no rule matched a
 * used family, the browser never even attempted a fetch. Zero woff2 requests,
 * zero warnings, every page quietly rendering in Segoe UI.
 *
 * Reaching the CSS through the JS graph hands it to Vite's own asset pipeline
 * instead, which rewrites each `url()` to a hashed emitted file. Verified by
 * asserting the pages actually load the faces rather than trusting that they do
 * — a font that fails to load falls back and still *looks* deliberate, which is
 * what made the first attempt so quiet.
 *
 * These declare the families as `Inter Variable` and `JetBrains Mono Variable`
 * (fontsource's naming for variable cuts), which is why the `--font-*` stacks in
 * index.css lead with those exact names.
 *
 * Variable weight is load-bearing, not a nicety: the design system asks for
 * 580/620/650/720, and a static cut snaps each to the nearest shipped weight,
 * flattening the typographic hierarchy the shell depends on.
 *
 * Both are SIL OFL 1.1 rather than Apache-2.0. The licence texts ship in
 * public/fonts/ because the emitted woff2 assets carry none of their own; see
 * NOTICE and public/fonts/README.md.
 *
 * Only the upright weight axis is imported. The packages also ship italic and
 * optical-size axes, which nothing here uses and which would roughly double the
 * emitted assets. Subsetting is left to the browser: each face carries a
 * `unicode-range`, so a latin-only page fetches only the latin file.
 */
import '@fontsource-variable/inter';
import '@fontsource-variable/jetbrains-mono';
