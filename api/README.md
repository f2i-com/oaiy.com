# oaiy-api

PHP 8.1+ / Slim 4 / MySQL backend for shared flows + AI-dispatched runs.

The job description is short: **store a JSON blob keyed by two random hashes; queue runs that the user's browser picks up and executes; relay results back**. That's it. No model serving, no auth beyond the hashes themselves.

See `../README.md` at the repo root for the architecture overview and the full API reference. This file is the deploy + ops cheat sheet.

## Requirements

- PHP **8.1** or newer (typed properties, readonly, ENUMs).
- Either:
  - **PDO SQLite** (default — bundled with every PHP build, no server needed), OR
  - **PDO MySQL** + a MySQL **5.7+** / MariaDB **10.3+** server.
- **Composer** for dependency install.

## Install

```bash
cd api
composer install
cp .env.example .env       # optional — defaults are sane (SQLite, var/oaiy.sqlite)
php bin/migrate.php        # creates the SQLite file + applies schema
```

That's it for SQLite — no database server to install or configure.

### Switching to MySQL

Edit `.env`:

```
DB_DRIVER=mysql
DB_HOST=127.0.0.1
DB_NAME=oaiy_web
DB_USER=root
DB_PASS=yourpass
```

Then create the database and apply the migration:

```bash
mysql -u root -p -e 'CREATE DATABASE oaiy_web CHARACTER SET utf8mb4'
php bin/migrate.php
```

`bin/migrate.php` picks the right SQL file (`migrations/001_initial.sqlite.sql` or `migrations/001_initial.sql` for MySQL) based on `DB_DRIVER` and applies it idempotently.

## Run (dev)

```bash
php -S 0.0.0.0:8080 -t public/
```

Then `curl http://localhost:8080/` should respond:

```json
{ "name": "oaiy-api", "docs": "..." }
```

## Deploy

### Apache

The `.htaccess` files in `./` and `./public/` route everything through `public/index.php`. Make sure `mod_rewrite` is on and the vhost's `<Directory>` block allows `AllowOverride All`.

DocumentRoot should point at `…/api/public`. The `vendor/` and `migrations/` folders should NOT be web-accessible — only `public/` is meant to be reachable.

### nginx

Point `try_files` at `index.php`:

```nginx
server {
  listen 80;
  server_name api.oaiy.com;
  root /var/www/oaiy/api/public;
  index index.php;

  location / {
    try_files $uri $uri/ /index.php$is_args$args;
  }
  location ~ \.php$ {
    fastcgi_pass unix:/run/php/php8.1-fpm.sock;
    fastcgi_index index.php;
    include fastcgi.conf;
  }
}
```

### Shared host (WAMP/MAMP/cPanel)

Drop the whole `api/` folder somewhere outside the web root, then have the web root point at `api/public/`. If you can't move folders (cPanel typically forces `public_html/`), set the DocumentRoot to `public_html/api/public/` and use the included `.htaccess` files to route — the api/`.htaccess` redirects bare-domain hits at `public/index.php` for setups where you can't move the DocumentRoot.

## Environment

All config lives in `.env`. See `.env.example` for the full list:

| Var | Default | Purpose |
|---|---|---|
| `DB_DRIVER` | `sqlite` | Database backend: `sqlite` (zero-setup) or `mysql`. |
| `DB_SQLITE_PATH` | `var/oaiy.sqlite` | SQLite file path (relative to `api/`, or absolute). Used when `DB_DRIVER=sqlite`. |
| `DB_HOST` | `127.0.0.1` | MySQL host (when `DB_DRIVER=mysql`). |
| `DB_PORT` | `3306` | MySQL port. |
| `DB_NAME` | `oaiy_web` | Database name (must exist; SQL migration creates tables, not the DB). |
| `DB_USER` | `root` | MySQL user. |
| `DB_PASS` | *(empty)* | MySQL password. |
| `UI_BASE` | `http://localhost:5175` | Frontend origin — used to compose `urls.view` / `urls.edit` in `POST /flows` responses. |
| `CORS_ALLOW_ORIGIN` | `*` | Comma-separated origin allow-list, or `*` to allow anything. |
| `POLL_TIMEOUT` | `20` | Seconds the browser dispatcher's long-poll request waits before returning `null`. |
| `MAX_QUEUED_RUNS` | `10` | Per-flow cap on `status='queued'` rows. POSTing past it returns 429. |
| `RUN_TTL` | `900` | Seconds before a run stuck in `running` is reset to a terminal `error` (floored at 60). |
| `APP_DEBUG` | `false` | When `true`, 500s return DB/stack detail. Leave `false`/unset in production. |
| `SERVICE_LIBRARY_DIR` | `service-library` | Folder the companion service library is served from (`GET /api/service-library`). |

## Layout

```
api/
├── composer.json
├── .env.example
├── .htaccess              # for bare-domain → public/ rewrites
├── public/
│   ├── .htaccess          # the actual Apache router
│   └── index.php          # front controller — 5 lines, all wiring lives in src/App.php
├── migrations/
│   └── 001_initial.sql    # `flows` + `runs` tables
└── src/
    ├── App.php            # Slim app factory + route table + CORS middleware
    ├── Db.php             # PDO singleton
    ├── Hash.php           # Crockford-base32 hash generator
    └── Controllers/
        ├── FlowsController.php       # CRUD + status + manifest
        ├── RunsController.php        # external-caller enqueue + poll
        └── DispatchController.php    # browser-side long-poll + heartbeat
```

## Migrations

There's exactly one migration today: `migrations/001_initial.sql`. Apply it once after creating the database. Future schema changes will land as `002_*.sql`, `003_*.sql`, … and you'll need to apply them by hand (no migration runner — the schema's small enough that this is fine).

## Operations

- **Garbage collection** — `POST /api/flows` is unauthenticated, so the `flows`/`runs` tables grow without bound unless reclaimed. Cron **`php bin/gc.php`** (e.g. hourly): it purges terminal runs older than `RUN_RETENTION_DAYS` (7) and abandoned flows — both idle ones (`last_seen` older than `FLOW_IDLE_DAYS`, 30) **and never-connected drive-by flows** (`last_seen` NULL + `created_at` older than `FLOW_NEVER_SEEN_DAYS`, 7). It is driver-aware (works on the default SQLite). _(The old manual `DELETE … WHERE last_seen < NOW()-INTERVAL 30 DAY` query was MySQL-only AND never reclaimed drive-by flows, whose `last_seen` stays NULL.)_
- **Logs** — Slim's error handler returns JSON to the caller AND emits PHP's standard error to whatever log destination your php-fpm/Apache is configured for. Tail that for backend errors.
- **Long-poll workers** — `GET /api/flows/{hash}/runs/pending` blocks an FPM worker for `POLL_TIMEOUT` (~20s); each online flow tab pins ~1 worker continuously. Give it its OWN FPM pool (sized for expected concurrent online flows) separate from the data endpoints, or a handful of tabs/idle slowloris polls can exhaust `pm.max_children` and starve enqueue/poll/result. The built-in `php -S` dev server is single-process — one long-poll blocks everything; use it for solo dev only.

## Hardening checklist before going public

- [ ] Set `CORS_ALLOW_ORIGIN` to your own domain (or specific tools you trust). `*` is the dev-friendly default.
- [ ] Cap `MAX_QUEUED_RUNS` lower if you expect untrusted callers.
- [ ] Run the API behind HTTPS — both hashes and the owner_token cookie cross the wire on every request.
- [ ] Set the owner_token cookie via your application code with `Secure; HttpOnly; SameSite=Strict` (the current code returns it in the POST response; if you want it auto-stored, set the cookie server-side in `FlowsController::create`).
- [ ] **Add a per-IP rate limit** at the nginx/Cloudflare layer for `POST /api/flows` and `POST /api/flows/*/runs` — these are unauthenticated/edit-hash-only write paths and without a limit the tables grow under drive-by abuse (the app caps each payload but not the request rate).
- [ ] **Isolate the long-poll route** `…/runs/pending` on its own FPM pool + `limit_conn` per IP, and disable proxy/fastcgi buffering for it (see Operations). Schedule `php bin/gc.php` so storage is reclaimed.
