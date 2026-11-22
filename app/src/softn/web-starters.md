# Build a SoftN Container v2 app in OAIY

Choose the stack the user asks for. Vanilla JavaScript is the smallest starting point. React and Vue use the same HTML, CSS and JavaScript project layout. For apps that need a server API and relational database, use the **Full-Stack SQLite** starter (`fullstack-sqlite`), which runs an embedded backend and **XDB SQLite** completely offline inside the sandboxed Container v2 runtime.

## Start a project

Call `softn_examples` with `name` set to `vanilla-js`, `react`, `vue`, or `fullstack-sqlite` and `install_to` set to a project folder. The tool returns the new app's path. Read its `app.js`, `index.html`, `manifest.json` (and `backend/server.js` / `db/schema.sql` for full-stack apps) before editing. For example, to build a full-stack SQLite app or a React tracker in the workspace:

```json
{"name":"fullstack-sqlite","install_to":"apps"}
```

```json
{"name":"react","install_to":"apps"}
```

Each starter has `"engine": 2` and `"main": "index.html"` in its manifest. Keep these fields when changing the app's name or adding files. Load code with `<script type="module" src="./app.js"></script>` and place a `<div id="app"></div>` before it. Relative imports resolve to files inside the project.

## Vanilla JavaScript

Edit the DOM with browser APIs. For example, this entry creates a button and counts clicks:

```js
const button = document.createElement('button');
let count = 0;
button.textContent = 'Count 0';
button.addEventListener('click', () => {
  button.textContent = `Count ${++count}`;
});
document.getElementById('app').appendChild(button);
```

## React

The React starter includes React and its DOM renderer in `vendor/react.js`. Edit components and hooks directly in `app.js`. This form runs without a JSX transform:

```js
import { React, createRoot } from './vendor/react.js';

function Counter() {
  const [count, setCount] = React.useState(0);
  return React.createElement('button', {
    onClick: () => setCount(value => value + 1)
  }, `Count ${count}`);
}

createRoot(document.getElementById('app')).render(React.createElement(Counter));
```

## Vue

The Vue starter includes its runtime in `vendor/vue.js`. Edit its component or add more JavaScript modules using `ref` and `h`:

```js
import { createApp, h, ref } from './vendor/vue.js';

createApp({
  setup() {
    const count = ref(0);
    return () => h('button', {
      onClick: () => { count.value += 1; }
    }, `Count ${count.value}`);
  }
}).mount('#app');
```

## Full-Stack Backends & XDB SQLite

SoftN Container v2 supports embedded full-stack backends (`javascript`, `typescript`, `python`, `php`, `rust`) and pure-Rust **XDB SQLite** (`xdb-wasm.wasm`) right inside OAIY's sandboxed preview — no external server required!

Declare `backend` and `databases` in `manifest.json`:

```json
{
  "name": "My App",
  "version": "1.0.0",
  "engine": 2,
  "main": "index.html",
  "backend": {
    "runtime": "javascript",
    "entry": "backend/server.js",
    "routes": ["/api/"]
  },
  "databases": [
    {
      "name": "app",
      "engine": "sqlite",
      "file": "app.db",
      "seedSql": "db/schema.sql",
      "persist": true
    }
  ]
}
```

### Embedded JavaScript / TypeScript backend (`backend/server.js`)

Define a synchronous `handleRequest(req)` (where `req` has `method`, `path`, `query`, `headers`, `body`) and return `{ status, headers?, json? | body? }`. Use the built-in `sqlite.open('app.db')` handle (`query`, `execute`, `all`, `first`, `get`, `run`, `seedJson`, `close`):

```js
function handleRequest(req) {
  const db = sqlite.open('app.db');
  if (req.method === 'GET' && req.path === '/api/notes') {
    return { status: 200, json: { notes: db.all('SELECT id, title FROM notes ORDER BY id DESC') } };
  }
  if (req.method === 'POST' && req.path === '/api/notes') {
    const { title } = JSON.parse(req.body || '{}');
    const res = db.run('INSERT INTO notes (title) VALUES (?)', [title]);
    const note = db.first('SELECT id, title FROM notes WHERE id = ?', [res.last_insert_rowid]);
    return { status: 201, json: { note } };
  }
  return { status: 404, json: { error: 'Not found' } };
}
```

### Embedded Python backend (`backend/server.py`)

Set `"runtime": "python"` and `"entry": "backend/server.py"`, define `handle_request(req)`, and call `sqlite.open("app.db")` (`all`, `first`, `get`, `run`, `query`, `execute`, `seed_json`, `close`):

```python
def handle_request(req):
    db = sqlite.open("app.db")
    if req["method"] == "GET" and req["path"] == "/api/notes":
        return {"status": 200, "json": {"notes": db.all("SELECT id, title FROM notes ORDER BY id DESC")}}
    return {"status": 404, "json": {"error": "Not found"}}
```

### Frontend-only XDB SQLite (`softn:sqlite`)

Even without a backend, any Container v2 frontend module can import `softn:sqlite` directly:

```js
import sqlite from 'softn:sqlite';
const db = await sqlite.open('app.db');
await db.run('CREATE TABLE IF NOT EXISTS notes (id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT NOT NULL)');
await db.run('INSERT INTO notes (title) VALUES (?)', ['Hello XDB SQLite']);
const rows = await db.all('SELECT id, title FROM notes ORDER BY id');
```

### SQLite Agent Tools

Use OAIY's direct XDB SQLite tools to inspect and manage app databases without guessing schemas:
- `sqlite_list_databases` — list `.db` files, `manifest.databases` entries, and `.sql` / `.json` seeds in an app
- `sqlite_schema` — inspect tables, columns, constraints, indexes, and row counts via XDB SQLite WASM
- `sqlite_query` — run read-only `SELECT` / `PRAGMA` / `EXPLAIN` queries against an app's database
- `sqlite_execute` — run DDL/DML (`CREATE`, `ALTER`, `INSERT`, `UPDATE`, `DELETE`) and persist updated `.db` bytes in the app

## Preview, verify, and share

Keep the framework module and its notices with the project. Use `softn_check` with the app folder to validate files, backend config, SQLite schemas/seeds, and live rendering. Use `softn_inspect` to read the live UI, `softn_interact` to click buttons, fill forms, and test real backend/SQLite CRUD flows, `preview_viewport` for responsive sizing, and `preview_screenshot` to visually verify the rendered app! Export the folder as a `.softn` archive to reopen it in SoftN Runtime, Studio or Builder.

SoftN Studio and Builder also support JSX and Vue single-file components. Their exports carry both the original files and prepared JavaScript. Keep both when importing one into OAIY. After editing a `.jsx` or `.vue` file, reopen the source in Studio or Builder and export again to refresh the prepared modules. The OAIY starters above use plain `app.js`, so edits preview immediately.

For an existing classic app, keep its manifest and component syntax. Its JavaScript or Python writing guide is available through `softn_docs` with the app's folder.
