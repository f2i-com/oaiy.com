# Curated ZIPP third-party notices

ZIPP's web-python release bundle ships `LICENSE-APACHE` but no notices for the
code it compiles in from elsewhere: the Unicode data (Unicode licence) behind
its Python identifier and `\N{...}` name tables. OAIY installs that engine into
`ui/vendor/zipp-wasm-python/` for its CLI, Desktop and headless server, and
whatever redistributes it has to carry those notices, so OAIY keeps them here.
The JavaScript-only web bundle compiles in none of it.

Releases before v0.0.21 also compiled in a fork of the RustPython parser (MIT),
and these notices carried its licence too. v0.0.21 parses Python with ZIPP's
own `crates/zipp-pyparse`, which contains no RustPython code (its README
records what it shares with RustPython's AST by design), and the fork is gone
from the repository, so the notices name the Unicode data alone. Installing an
older release against them warns, like a newer one does.

`THIRD_PARTY_LICENSES.txt` is made from one file of a zipp.org release tag,
read with `git show <tag>:<path>`: `LICENSE-UNICODE` at the repository root. It
is the line `ZIPP engine: Apache-2.0. See the source repository for its
complete notices.`, then `Unicode data:` and that file as it is with its own
final newline. The two parts are joined by `\n\n`, trailing whitespace is
trimmed, and one `\n` ends the file. That is the recipe softn.com's
`packages/@softn/core/scripts/build-zipp-wasm.mjs` writes a Python build's
notices with, so this file and Softn's curated copy are the same bytes. Git
keeps it LF on every OS (`.gitattributes`), because its digest is recorded.

`SOURCE.json` records the tag the file was generated from, the source path and
the file's SHA-256. `scripts/regen-zipp-notices.mjs` writes both, or compares
them with a tag's sources:

```bash
node scripts/regen-zipp-notices.mjs ../zipp.org vX.Y.Z           # regenerate from that tag
node scripts/regen-zipp-notices.mjs ../zipp.org vX.Y.Z --check   # or compare
```

`scripts/fetch-zipp-release.mjs` installs this copy into
`ui/vendor/zipp-wasm-python/` when the release bundle has no
`THIRD_PARTY_LICENSES.txt` of its own, and records it in that folder's
`SOURCE.json` as `notices.source: "oaiy-curated"` with its digest. It refuses a
copy that is not the file `SOURCE.json` here records. A release newer or older
than the recorded tag installs with a warning to regenerate (an annotation in
GitHub Actions), because it may compile in something this copy does not name.
Once a ZIPP release ships the notices in its bundle, the install takes those
instead (`notices.source: "zipp-release"`) and this copy is no longer used.

A curated file cannot prove itself complete against ZIPP's real dependency
graph. When the warning appears, check what the release compiles in,
regenerate from its tag and reinstall; `--check` refuses an install whose
notices are not this file.
