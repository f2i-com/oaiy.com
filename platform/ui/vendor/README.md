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

From the flow-editor review (2026-10-18), to carry over on a re-sync (or to
upstream into `oaiy/packages/*`):

- `oaiy-core/src/types.ts`: `GraphNode` has `parentId`, `extent` and `style`
  (`width`/`height`), so a flow keeps its groups (Ctrl+G) when saved.
- `oaiy-core/src/compiler.ts`: `compile()` leaves out `group` nodes (and any
  edge touching one) before compiling: a box on the canvas, not a step.
- `oaiy-core/src/constants.ts` + `oaiy-core/src/queue/JobManager.ts`: the
  queue's history keeps its newest `MAX_FULL_JOB_HISTORY` (10) runs whole; older
  ones keep their timing, status and last `SLIM_JOB_HISTORY_LOGS` (50) logs
  (50 whole runs held every picture they made in memory).
- `oaiy-core/src/module-types.ts` and
  `oaiy-ui-components/src/nodes/fields/SelectField.tsx`: an option's `action`
  is given a `choose(value)` to pick a value afterwards ("Add a service…"
  picks the service added); SelectField passes its `onChange`.
- `oaiy-ui-components/src/nodes/fields/SelectField.tsx` (the stored value that
  is not listed) and `oaiy-ui-components/src/nodes/GenericNode.tsx` (the card's
  summary): a paired worker's model (`engine:<kind>:node.<id>/<model>`) is
  shown as "<model> on a paired computer", not its id.

From the flow-editor polish (2026-10-18), likewise:

- `oaiy-core/src/runtime.ts`: "Registered dynamic module: …" goes to the
  developer's console (`runtimeLogger.debug`), not the run's log: every run
  registered every module, a dozen lines before the flow's first step.
- `oaiy-core/src/runtime.ts` (`Workflow Error: …`) and
  `oaiy-core/src/queue/JobManager.ts` (the failed job's `Error: …` line): the
  message is written with the `Error: ` prefixes an error collects on its way up
  removed, so the log says "Error: Browser blocked…", not
  "Error: Error: Error: Browser blocked…"; and the job's line is left out when
  the runtime's "Workflow Error: …" with the same message is the job's last
  line (the same paragraph twice, one under the other).
- `oaiy-ui-components/src/components/CollapsibleNodeWrapper.tsx`: a port's
  handle is keyed by its kind and id (`target:image`), not its id alone: Image
  View takes in and gives out `image`, two siblings with one key.

From the node-by-node polish (2026-10-18, second pass), likewise:

- `oaiy-ui-components/src/components/CollapsibleNodeWrapper.tsx`:
  - a card shows its run state from the node's own `_status` when it is not
    given one (cards made from a definition, Output, Image View and the input
    cards never showed which step ran or failed);
  - its icon is its kind's, from a resolver the host registers (the palette's
    set), unless the card says the icon is its own (`ownIcon`: a service's
    emoji);
  - a port's name is its definition's ("Prompt", not a card's "prompt" or
    "desc"; a handle the definition lacks keeps its label, and `ownLabel` keeps
    a name the person gave: a macro's or a template's variables); its dot is
    coloured by its kind of value (`portTypeClass`), and its name is in the
    card's own ink (`labelColor` is no longer shown);
  - the kind's word over the name is the 700 ink on the light header (the 600
    was about 3:1).
- `oaiy-ui-components/src/components/portTypes.ts` (new): `portTypeClass` and
  `typeWords` (the colours are `.oaiy-port.port-*` in `ui/src/index.css`).
- `oaiy-ui-components/src/registry/nodeIcons.ts` (new):
  `registerNodeIconResolver` / `resolveNodeIcon` (`ui/src/bundled-web-modules.ts`
  registers `lib/nodeIcons.tsx`).
- `oaiy-ui-components/src/hooks/useNodeDefinition.ts` (new): `definitionOf` /
  `useNodeDefinition`, GenericNode's lookup for any card.
- `oaiy-ui-components/src/index.ts`: exports the above.
- `oaiy-ui-components/src/nodes/GenericNode.tsx`: the compact card names an
  engine kind's default ("the default picture model", not "image") and OAIY
  Voice; "Advanced options" in sentence case.
- `oaiy-ui-components/src/nodes/fields/SelectField.tsx`: the empty choice says
  "Choose…".

From running every way of running a flow end to end (2026-10-18; regression
cases in `cli/test/flow-nodes.ts`), likewise:

- `oaiy-core/src/runtime/BuiltinModules.ts`: `Utility.*` and `Agent.*` take
  their arguments spread (`Agent.set(key, value)`), as host.call passes them;
  they read `args[0]` of one array, so the Memory node and the loop's history
  stored the key's second letter under its first.
- `oaiy-core/src/queue/JobManager.ts`: a job's `error` is the message without
  the `Error: ` prefixes it collects on its way up (a desktop run said
  "Error: Error: …"); a job is `aborted` when its own signal fired, not when
  its error's text says "aborted" (a request's own timeout).
- `oaiy-core/src/compiler.ts`:
  - `getSourceVar`: a macro node's output handle is that output's variable
    (`node_<id>_out_<output>`), not the macro's whole answer.
  - `branchTest` / `gateOnBranch`: a node wired straight to a condition's
    `true`/`false` is on that branch whatever else is wired to it; a node after
    branch steps runs when one of its inputs ran, or when one always runs (an
    optional step's merge). In a loop, an iteration whose branch was not taken
    resets the node's values (it kept the last iteration's). Every declaration
    inside a gate becomes an assignment (`let node_X_out` was matched inside
    `let node_X_out_result`).
  - `orderWithLoops`: each loop is generated as one step (start, body, end)
    in dependency order, so two loops whose nodes interleave both run.
  - `findInnerLoopNodes`: a loop's body is what follows its start and not its
    end, side branches included (they ran before the loop, or once after it).
  - `generateLoopCode`: For Each over null runs no iteration and over a
    non-array one (it threw, or ran none); a count wired in is that count, 0
    included, capped at `MAX_WORKFLOW_LOOP_ITERATIONS`; a loop on an untaken
    branch does not run and its results are null.
  - `workflow_context` starts empty (only `__macro_inputs__` carried in): it
    was the inputs object, so an input named `__output__` set the answer.
- `oaiy-core/src/compiler/utils.ts`: `escapeString` escapes U+2028 and U+2029
  (they end a `//` comment a compiler writes a setting into).
- `oaiy-core/src/value-utils.ts`: `extractDeepValue` copies values whole: it
  replaced an object of one or two keys named like `text`, `result` or `url` by
  that key's value (`{ text, lang }` came back as the text).
- `oaiy-core/src/runtime.ts`: `policyFetch` sends a FormData, Blob or bytes as
  their bytes (`body_base64` to `http_request`) and search params as a form, and
  passes the caller's signal (`performHttpRequest`'s `extra`); "Workflow Error"
  is logged unless the run's signal fired; `__system.finish_part` collects an
  answer sent in parts.
- `oaiy-core/src/workflow-trampoline.ts`: an `await` of a promise or a plain
  value resumes with that value; an answer past 1 MiB code units goes in
  `__system.finish_part` calls before `__system.finish` (one call past the
  4 MiB host.call limit failed the run as it ended).
- `oaiy-core/src/zipp-executor.ts` and `oaiy-core/src/untrusted-executor.ts`:
  `__system.finish_part`, as above.

From a hands-on review of the flow editor (2026-10-19; cases in
`ui/tests/flow-editor-review.mjs`), likewise:

- `oaiy-core/src/node-errors.ts` (new): a step's own failure names its node in
  the message (`nodeErrorPrefix`, `[oaiy-node:<id>] `; an error leaves a flow's
  script as text, whatever engine runs it), and a node that does not compile is
  a `CompilationError` with its id (`nodeCompileError`). Its own file:
  `oaiy-core/src/errors` is also a folder, which a bundler may take it for.
- `oaiy-core/src/compiler.ts`: a module's `CompilationError` with a node id
  (a Logic Block whose code does not parse) is passed on as it is, not wrapped
  in "Node … failed to compile in module compiler …". A loop's item previews,
  history lengths, stop-condition checks and results preview are
  `console.debug` (out of the run's log: a results preview was a picture's
  base64).
- `oaiy-core/src/runtime.ts`: `runWorkflow` marks the node a compile error or a
  marked error names (`onNodeStatus(id, 'error')`) and takes the mark (and
  ZIPP's " (in <script>)") off the message; its lines say "--- The flow
  started ---", "--- The flow finished ---" and "The flow failed: …"; a
  module's `error` status once the run's signal has fired is `stopped`;
  "Registered module …" is `runtimeLogger.debug`.
- `oaiy-core/src/types.ts` (`NodeStatus`, with `stopped`; exported from
  `index.ts`), `oaiy-core/src/module-types.ts`, `oaiy-core/src/queue/types.ts`:
  a node's run status may be `stopped`; a job keeps `failedNodeId`, the first
  node marked failed (a canvas that hears of a job after it failed still marks
  it). `oaiy-core/src/queue/JobManager.ts` sets it, and says "Stopping the
  flow…" and "The flow was stopped".
- `oaiy-ui-components/src/components/CollapsibleNodeWrapper.tsx`: a `stopped`
  status (grey, dashed); what the host says of a node (`useNodeNoticeDetail`)
  is on every card: a warning mark in its header, its short form on the folded
  card, the whole of it on the expanded one.
- `oaiy-ui-components/src/hooks/useNodeNotice.ts`: a notice may come with a
  short form (`{ text, short }`); `useNodeNoticeDetail`, `getNodeNoticeDetail`.
- `oaiy-ui-components/src/registry/serviceNames.ts` (new):
  `registerServiceDescriber` / `describeServiceId`, the host's words for a
  service id. `nodes/GenericNode.tsx` (the card's summary, "minicpm-v on
  Studio PC", "the default language model") and `nodes/fields/SelectField.tsx`
  (a stored value not listed) use it; GenericNode's notice box moved to the
  wrapper.
- `oaiy-ui-components/src/hooks/useDynamicOptions.ts`: no Tauri event listener
  is asked of the web build's stand-in (`__TAURI_INTERNALS__.__oaiyShim`), which
  warned in the console on every page with a dropdown.
- `oaiy-ui-components/src/index.ts`: exports the above.

From a visual polish of the flow editor (2026-10-19, after the dashboard's
sidebar redesign), likewise:

- `oaiy-ui-components/src/components/CollapsibleNodeWrapper.tsx`:
  - a card's kind colour is a class, `kind-<colour>` (the colours it knows,
    `KINDS`; any other is `slate`), not the Tailwind border, header, icon,
    word and hover classes per colour (`colorMap`, gone): `ui/src/index.css`
    gives each kind `--kind` and `--kind-ink`, and draws the card on the
    editor's surface with a hairline of that colour, a faint wash on its
    header, its icon a tinted tile and its kind's word in that ink. Five kinds
    were five saturated boxes; the palette drew the same kinds another way.
  - its run state is `.oaiy-node-status <status>` (a dot, a pill with its word
    on an expanded card), coloured in `ui/src/index.css` from the theme's
    tokens: the accent while it runs (it was amber, inside an accent ring), not
    `STATUS_META`'s Tailwind classes, which now hold only the words.
  - a failed check (`validationIssues`) is marked as a notice is
    (`oaiy-node-warn`), not a Tailwind amber dot of its own.
  - the card is a Tailwind `group`: the hand-made cards' resize handles
    (`group-hover:opacity-100`) show while the card is hovered; with no group
    around them they never showed at all.

## Re-syncing from upstream (optional)

If you ever want to pull newer upstream source, re-copy `src/` from the
matching package and re-apply any local edits noted above:

```sh
cp -r ../../../oaiy/packages/oaiy-core/src        ./oaiy-core/src
cp -r ../../../oaiy/packages/oaiy-ui-components/src ./oaiy-ui-components/src
# then remove __tests__ and re-apply the runtime/JobManager edits above
```

## ZIPP engines (installed, not committed)

| Folder | Holds | Used by |
|---|---|---|
| `zipp-wasm/` | `zipp-<v>-javascript.wasm` of a zipp.org release, v0.0.26 or later (JavaScript only, 1 MiB stack) | the browser flow sandbox (`src/workers/zipp-untrusted-worker.ts`) |
| `zipp-wasm-python/` | `zipp-<v>-javascript-python.wasm` of the same release (JavaScript and Python, 16 MiB stack) | installed for the CLI, Desktop and headless server; nothing loads it yet but `tests/zipp-executor.mjs` |
| `zipp-notices/` | committed: the Unicode notices the release does not ship, and `SOURCE.json` naming the ZIPP release they were generated from (see its `README.md`) | installed beside `zipp-wasm-python/` |

Both engine folders are gitignored and generated by
`node scripts/fetch-zipp-release.mjs` (ui's `dev`, `typecheck`, `test`,
`test:zipp` and `build` hooks run it with `--ensure`). Each holds the module as
`zipp_wasm_bg.wasm` and the bindings it carries (its `zipp.glue` section) as
`zipp_wasm.js`, byte for byte; the release embed kit's `zipp-engine.d.ts` (as
`zipp_wasm.d.ts`) and `LICENSE-APACHE`; `PROFILE.json`, what the engine's
`zippProfile()` reports; ZIPP's top-level `SHA256SUMS` as `RELEASE-SHA256SUMS`;
and `SOURCE.json` recording the release, commit, variant and digests. Nothing in
them is edited by hand: `--check` refuses any file that is not the release's.
See `TESTING.md` ("ZIPP engines") for the modes.
