# Service Library

Example **companion ServiceTemplate** files — local services you download and
import into the OAIY Companion (install script + run command + health check).
Served to the desktop page (`/desktop.html`) so visitors can browse + download.

## Adding an example

Just **drop a `*.json` file in this folder** — it appears in the listing
automatically (no rebuild, no manifest). The API reads the folder live:

- `GET /api/service-library` — lists every file with parsed metadata
  (`name`, `description`, `icon`, `category`, `count`, `size`, `downloadUrl`).
- `GET /api/service-library/{file}` — downloads one file (path-traversal-guarded).

Override the folder with `SERVICE_LIBRARY_DIR` (absolute path) in `api/.env`.

## File shape (companion ServiceTemplate)

| field | notes |
|---|---|
| `id` | unique id — also the on-disk filename for the template |
| `name`, `description`, `category` | display + grouping |
| `defaultPort` | port the service listens on; available everywhere as `${port}` |
| `install` | `{"kind":"none"}` or `{"kind":"script","windows","unix"}` — a per-OS install script |
| `run` | `{command, args[], env, cwd}` — how to launch; a bare command resolves against the companion bin dir then PATH |
| `health` | `{url, timeoutSecs}` — readiness probe; `url` supports `${port}` |
| `docsUrl` | optional link |
| `files` | `{filename: contents}` — bundled scripts written to the scripts dir (self-contained package) |

**Placeholders** (in `run` + `health`): `${port}`, `${dataDir}`, `${binDir}`,
`${modelsDir}`, `${modelDirs}`.
**Env vars** (in install scripts): `OAIY_DATA_DIR`, `OAIY_VENVS_DIR`,
`OAIY_BIN_DIR`, `OAIY_MODELS_DIR`, `OAIY_SCRIPTS_DIR`.

> Note: this is the COMPANION format (runs local processes). The web app's
> Service Call nodes use a different, simpler format (an HTTP endpoint + a body
> template) — added in the app under Settings → Services.
