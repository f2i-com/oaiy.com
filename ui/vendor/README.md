# Vendored OAIY packages

This directory makes `oaiy-web` a **standalone project** — it has no dependency
on the sibling `oaiy` monorepo. Both packages here are the **source** of the
shared OAIY libraries, vendored in and consumed via the aliases in
`ui/vite.config.ts` (build) and the `paths` in `ui/tsconfig.json` (typecheck).

| Folder | Upstream | Imported as |
|---|---|---|
| `oaiy-core/src` | `oaiy/packages/oaiy-core/src` | `oaiy-core`, `oaiy-core/src/*` |
| `oaiy-ui-components/src` | `oaiy/packages/oaiy-ui-components/src` | `oaiy-ui-components`, `oaiy-ui-components/*` |

Notes:
- `oaiy-core/modules/*` imports resolve to `ui/src/bundled-modules/*` (the web
  build's own module copies used for codegen + execution), **not** here.
- Tests (`__tests__`, `*.test.ts`) were intentionally excluded — they pull no
  runtime weight.
- `oaiy-core`'s runtime deps (`acorn`, `cheerio`, `uuid`) are already in
  `ui/package.json`; `oaiy-ui-components` depends only on `oaiy-core` + peers
  (`react`, `@xyflow/react`, `@monaco-editor/react`).

## Local edits living here

`oaiy-core/src/runtime.ts` + `oaiy-core/src/queue/JobManager.ts` carry the
project-constant wiring (`setProjectConstants` + `ctx.getConstant`) that lets
the web runtime resolve API-key constants (e.g. `OPENAI_API_KEY`) at run time —
see the "BYOK from the wizard" fix.

## Re-syncing from upstream (optional)

If you ever want to pull newer upstream source, re-copy `src/` from the
matching package and re-apply any local edits noted above:

```sh
cp -r ../../../oaiy/packages/oaiy-core/src        ./oaiy-core/src
cp -r ../../../oaiy/packages/oaiy-ui-components/src ./oaiy-ui-components/src
# then remove __tests__ and re-apply the runtime/JobManager edits above
```
