# Vendored core modules

The 11 modules under this directory are **vendored copies** of
`oaiy/packages/oaiy-core/modules/*` from the upstream desktop monorepo
(`oaiy-app/oaiy`). They live here so the oaiy-web repo is self-contained:
cloning + `npm install` + `vite build` works without needing the sibling
`oaiy/` checkout.

Two upstream modules are intentionally **not** vendored because they
can't function in a browser at all:
- `core-terminal` — shells out to a system shell.
- `plugin-agent` — desktop-only AI flow editor that depends on Tauri.

## Why vendor instead of glob across the sibling

The desktop app and oaiy-web ship from separate repos. The earlier setup
had `bundled-web-modules.ts` doing
`import.meta.glob('../../../oaiy/packages/...')`, which:

1. Required developers to clone both repos as siblings just to build.
2. Broke when the sibling moved or was missing — silently produced
   an empty palette.
3. Made oaiy-web depend on the upstream desktop being checked out at a
   compatible revision.

Vendoring trades a refresh step for a hermetic build.

## What's wired up

- `bundled-web-modules.ts` — eager-globs `./bundled-modules/*/module.json`
  + `./bundled-modules/*/nodes/*.json`, lazy-imports each module's
  `runtime.ts` / `compiler.ts` / `ui/index.ts`. It also handles React
  component registration from each module's `ui/index.ts` (this was
  previously a separate `moduleUI.ts`, now removed).
- `tsconfig.json` — path alias `"oaiy-core/modules/*"` →
  `"./src/bundled-modules/*"` (must come before the broader `oaiy-core/*`
  alias).
- `vite.config.ts` — `manualChunks` routes anything matching
  `/src/bundled-modules/` to the `oaiy-modules` chunk.

## Import rewrites

Upstream modules use relative paths like
`from '../../src/module-types'` to reach `oaiy-core/src/*`. After
copying, those relative paths no longer resolve, so every vendored file
that imported via `'../../src/'` was rewritten to `'oaiy-core/src/'`
(which still resolves to the upstream package via the existing
`oaiy-core` alias).

## Refreshing the vendor

When upstream module code changes, re-copy + re-rewrite:

```pwsh
# from oaiy.com/ui/
Remove-Item -Recurse -Force src/bundled-modules/<module-name>
Copy-Item -Recurse ../../oaiy/packages/oaiy-core/modules/<module-name> src/bundled-modules/
# Rewrite the imports:
(Get-ChildItem -Recurse src/bundled-modules/<module-name> -Filter *.ts) |
  ForEach-Object {
    (Get-Content $_.FullName) -replace
      "from '\.\./\.\./src/", "from 'oaiy-core/src/" |
      Set-Content $_.FullName
  }
```

Then re-run `npx tsc --noEmit` + `npx vite build` to catch any new
divergence (e.g. an upstream module that started importing a Node-only
API the web shim doesn't cover).
