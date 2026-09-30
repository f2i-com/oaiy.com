# OAIY Relay

A small server you run yourself. It stores and forwards small opaque items in per-device mailboxes, so the OAIY desktop, a
phone and a provider such as FormLogic can exchange live data without the provider carrying it. It is written in plain PHP
(no Composer, no framework), keeps its state in SQLite (or MySQL and MariaDB), needs no cron and no background process, and
runs on ordinary shared hosting as well as on a small VPS. Live audio never touches it.

The wire protocol is [`platform/protocol/relay/v1/`](../protocol/relay/v1/README.md). This folder is one implementation of it.

## What you need

| | |
|---|---|
| PHP | 8.0 or later (8.2 recommended). Tested on 8.0.30 and 8.4.15. |
| Extensions | `sodium` (bundled with PHP since 7.2), `json`, `hash`, and `pdo_sqlite` (the default store) or `pdo_mysql`. `openssl` and `curl` matter only to the optional FCM sender, which is not part of this build. |
| Disk | A writable `data/` folder on a **local** filesystem (SQLite's write-ahead log is unsafe on NFS and similar). |
| HTTPS | A publicly trusted certificate on a hostname of its own (a subdomain such as `relay.example.com`, not a path on a site that hosts anything else). The desktop and the phone use the bundled web roots and cannot use a private CA. |
| Workers | Each waiting poll holds one PHP worker for up to 20 seconds. Measure what your host allows with the [host probe](probe/host-probe.php) and [`docs/relay-hosting-matrix.md`](../../docs/relay-hosting-matrix.md) before you commit to a plan. |

## Layout

```
oaiy-relay/
  public/           the document root, and the ONLY folder the web server may serve
    index.php       the front controller for /v1/*
    install.php     the web installer (answers 404 unless you arm it, see below)
    .htaccess       Apache and LiteSpeed rules: front controller, Authorization pass-through, deny the rest
  src/              the code (every file starts with: defined('OAIY_RELAY') or exit;)
  bin/              install.php, doctor.php, relay.php (administration), gc.php  (each starts with a command-line guard)
  probe/            host-probe.php: the one-file host probe
  tests/            the test runner and the tests (not part of a release zip)
  data/             created by the installer; config, database, secrets. Outside public/
    .htaccess       written by the installer: deny everything (Apache 2.4 and 2.2)
    web.config      written by the installer: deny everything (IIS)
  .htaccess         deny everything (Apache 2.4 and 2.2): this folder is never a document root
  web.config        deny everything (IIS)
  VERSION
```

`data/`, `src/` and `bin/` are protected by living **above** the document root. Upload the whole folder somewhere
outside your web root (for example `~/oaiy-relay/`) and point the site's document root at `~/oaiy-relay/public`. The
installer and the doctor treat any other layout as a failure.

If the document root is the relay folder by mistake, three things stand between the web and your secrets, and none of them
replaces the right layout. The `.htaccess` and `web.config` at the root and in `data/` refuse every request on Apache, LiteSpeed
and IIS, but only where the server honours them (Apache needs `AllowOverride` for `.htaccess`; nginx ignores both, so its
configuration below denies the folders itself). Every PHP file in `src/` and `bin/` exits at once when it is requested. And the
installer and the doctor ask the web for a canary in `data/` and refuse or fail when they get it.

## Installing on cPanel-style hosting

1. Choose PHP 8.0 or later in the host's PHP selector and make sure `sodium` and `pdo_sqlite` are enabled (both usually are).
2. Upload and extract the relay **above** the web root, for example `~/oaiy-relay/`.
3. Create a subdomain (`relay.example.com`) whose document root is `~/oaiy-relay/public`, and turn on AutoSSL.
4. Install, in one of two ways.
   - With a terminal: `php bin/install.php --url=https://relay.example.com`
   - Without one: create a file named `INSTALL_ENABLED` in `~/oaiy-relay/` (next to `public/`, not inside it) with the file
     manager. Put **a token you choose**, at least 16 characters, in it. Open `https://relay.example.com/install.php`, type
     the token and the relay's public address, tick call features only if you mean to (see below) and press Install.
5. The installer prints (or the page shows) **paths, never secrets**: `data/first-key.txt` (the one-time desktop enrolment key)
   and `data/admin-token.txt` (for the status page and the doctor). Open them in the file manager or over SSH, paste the key into
   OAIY (Connections, Remote access) within an hour, and delete both files. The web installer deletes `INSTALL_ENABLED` itself.
6. Run `php bin/doctor.php --url=https://relay.example.com --admin-token-file=data/admin-token.txt` if you have a shell, and fix
   anything red. OAIY's "Test this relay" then measures the worker pool, streaming and body limits on this host.

A lost first key is re-armed, never re-issued over the network: create `INSTALL_ENABLED` again and run `php bin/install.php
--rekey` (or open `install.php?mode=rekey` and use the token). It writes a new `data/first-key.txt` and changes nothing else.

### The installer's rules

- It never prints or sends a token, key or secret: not to the screen, not in an HTTP response or header, not in an error.
  It writes them to files (mode 0600, `data/` mode 0700) and names the files.
- It refuses to write a secret where the web can read it, and it decides that in one place for the command line installer, the
  web installer and a re-key alike: `data/` inside `public/`, `data/` inside the document root the web server itself reports
  (the web installer knows it), or a canary file put in `data/` that is served over the web. **The canary probe is on by
  default**: the command line asks at the `--url` address (or at `--probe-url` if you name another), the web installer at the
  public address you type, a re-key at the address in `config.json`. Refusal happens before any question is asked and before
  anything is written, so a document root that is the relay folder leaves no key behind. An address that cannot be reached is
  never a silent pass: the installer says the exposure of `data/` was **not checked** and to run the doctor once the site is up.
  `--no-probe` (and the web form's "do not ask" box) skips the probe on purpose, and says so. It refuses a second run.
- The web installer works only while `INSTALL_ENABLED` exists and the relay is not installed; a wrong or missing token is a
  bare 403 after a 250 ms delay; every other request is a bare 404.
- The call-features question (default **no**) is asked on a terminal, or set with `--call-features=yes|no`, or by the checkbox
  on the web installer: *"Enabling call features lets whoever administers this host read caller names and call captions, and act
  as your phone on call control, until Noise sealing ships. Enable only on a host you administer."*

`bin/install.php` options: `--url=` (required, `https://host`, no path), `--call-features=yes|no`, `--yes` (never ask),
`--db=sqlite|mysql`, `--dsn=`, `--db-user=`, `--db-pass-file=PATH` (the password comes from a file, never the command line),
`--probe-url=URL` or `--no-probe`, `--rekey`. Set `OAIY_RELAY_DATA` to keep `data/` elsewhere (the command line only). Exit codes:
0 done, 1 refused or failed, 2 usage.

## Installing on a VPS

Recommended: a small VPS, nginx (or Caddy) in front of PHP-FPM with a pool of its own for the relay, `pm.max_children` sized
from RAM (about 30 MB a child is the assumption in the capacity tables), the same installer as above, and coturn beside it once
call features arrive. A minimal pool:

```ini
[relay]
user = relay
group = relay
listen = /run/php/php-relay.sock
listen.owner = www-data
listen.group = www-data
pm = static
pm.max_children = 8
request_terminate_timeout = 60s
php_admin_value[memory_limit] = 64M
php_admin_value[display_errors] = off
```

### nginx

Not run through nginx in this repository's tests: check it with the doctor before trusting it.

```nginx
server {
    listen 443 ssl http2;
    server_name relay.example.com;
    root /home/relay/oaiy-relay/public;          # only public/ is served; data/, src/, bin/ are outside it

    # A stalled request body must not pin a PHP worker for long (the doctor probes this).
    client_header_timeout 20s;
    client_body_timeout   20s;
    client_max_body_size  1m;                    # the relay's request limit is 1 MiB

    # The API. Pass the Authorization header explicitly: FastCGI stacks may drop it otherwise.
    location ^~ /v1/ {
        include fastcgi_params;
        fastcgi_pass unix:/run/php/php-relay.sock;
        fastcgi_param SCRIPT_FILENAME $document_root/index.php;
        fastcgi_param SCRIPT_NAME     /index.php;
        fastcgi_param HTTP_AUTHORIZATION $http_authorization;
        fastcgi_read_timeout 60s;                # a hold is at most 20 s (up to 300 s if you raise wait.max)
        fastcgi_buffering off;                   # answers that stream (the calibration probe) must not be buffered
    }

    # The web installer: only while INSTALL_ENABLED exists. Remove this block after installing if you like.
    location = /install.php {
        include fastcgi_params;
        fastcgi_pass unix:/run/php/php-relay.sock;
        fastcgi_param SCRIPT_FILENAME $document_root/install.php;
    }

    location = /status.html { }                  # the static status page

    # Belt and braces. With root at public/ these folders do not exist under the root; this block is what still protects them if
    # someone later points root at the relay folder itself (nginx ignores .htaccess and web.config). It must come before the
    # catch-all, and it matches at any depth and in any case, and the encoded and dot-dot spellings nginx has already resolved.
    location ~* ^/(data|src|bin|probe|tests)(/|$) { return 404; }
    location ~* ^/(\.htaccess|web\.config|INSTALL_ENABLED|VERSION)$ { return 404; }

    # Everything else does not exist.
    location ~ /\. { deny all; }
    location /     { return 404; }
}
```

The doctor's exposure probes (`/data/relay.sqlite`, `/data/secrets/relay.key`, `/src/Db.php`, `/bin/doctor.php` and more) are the
check that this works on your server; a probe that is not answered with 403 or 404 is a failure.

Do not put the relay behind a proxy that terminates TLS and that you do not control (a proxied CDN hostname sees every bearer
token and every unsealed lane). If you must, list its addresses in `client_ip.trusted_proxies` and set `client_ip.header`, or
every client will share one rate-limit address. The proxy must also drop request headers that have an underscore in their
name (`X_Forwarded_For`): PHP folds that spelling into the same variable as `X-Forwarded-For`, so a client that sends both
could choose its own address. nginx drops them by default (`underscores_in_headers off`). Where PHP reports header names as sent
(`php -S`, Apache) the relay reads only the hyphenated name itself; on FastCGI it cannot tell, so the proxy has to.

## Apache and LiteSpeed: `public/.htaccess`

Not run through Apache in this repository's tests: read it as a starting point and let the doctor judge it on your host. It:

- turns directory listings off, denies dotfiles and every file type except `.php` and `.html`;
- passes the `Authorization` header to PHP (`CGIPassAuth On` on Apache 2.4.13 or later, and the `SetEnvIf` / `RewriteRule`
  forms for older stacks). Without it every request is a uniform `401`, which looks like a revoked device;
- returns 404 for anything outside `/v1/`, `/status.html` and `/install.php`, and sends `/v1/*` to `index.php`.

The folder above it carries its own `.htaccess` (and `data/` gets one from the installer) that refuses everything, in Apache 2.4
syntax (`Require all denied`) and 2.2 syntax (`Order deny,allow` and `Deny from all`), each inside an `IfModule` for the module
that understands it. It matters only when the document root is wrong, and then only if the server lets `.htaccess` files apply
(`AllowOverride`). `web.config` does the same on IIS. The doctor tells you which of them held.

## Configuration: `data/config.json`

Read on every request. Validation fails closed: a wrong type or a value that would widen a limit past the protocol's maximum
is an error, not a default. Keys this build understands (unknown keys are ignored):

| Key | Default | Meaning |
|---|---|---|
| `public_url` | required | The externally visible base, `https://host` with no path. Every URL in a response is built from it (the `Host` header never is). `http` is accepted only for a loopback host, which is how the tests run |
| `db.driver`, `db.dsn`, `db.user`, `db.pass` | `sqlite`, none | The database. SQLite lives in `data/relay.sqlite`; `mysql` needs a DSN |
| `db.journal` | `wal` | `wal`, or `truncate` on a network filesystem. The installer sets it from `/proc/self/mountinfo` |
| `wait.max` | 20 | Longest a poll is held, seconds (at most 300, and lowered to the measured hold minus 5 by the calibration) |
| `wait.gap_ms` | 250 | The poll gap rule |
| `wait.fallback_s` | 5 | Short-poll interval after a refused hold |
| `presence_window` | 60 | Seconds a poll start counts as online (effective value is at least `wait.max` + 5) |
| `capacity.workers` | unset | The worker pool. Unset means 5 until the calibration measures it |
| `limits.desktops` | 2 | Desktop devices this relay admits |
| `limits.rosterMax` | 16 | Phones in one roster |
| `limits.mailboxItems`, `limits.mailboxBytes`, `limits.bulkShare` | 512, 8 MiB, 0.75 | Per-inbox quotas and the share the bulk lanes may use |
| `limits.batchItems`, `limits.batchBytes`, `limits.hdrBytes` | 64, 1 MiB, 512 | Post batch and header bounds |
| `limits.lookupWait`, `limits.lookupHeld` | 8, 4 | Waiting lookups (providers only) |
| `limits.lanes.<lane>.body`, `.ttl.default`, `.ttl.max` | the protocol's | Narrow a lane's size cap or lifetime; never widen |
| `call.enabled` | false | Call features (ring lane and, later, admission and signalling) |
| `token_pepper` | none | Optional HMAC pepper (16 characters or more) for token hashes; set it before issuing tokens |
| `wake.mode`, `wake.safety_ms` | `file`, 2000 | `file` (shard files, with a database fetch every `safety_ms`) or `db` (poll the database) |
| `client_ip.header`, `client_ip.trusted_proxies` | none | Honoured only when `REMOTE_ADDR` is a trusted proxy |
| `cors.extra_origins` | none | Extra https origins listed in `info` |

Accepted and validated now, used by later parts of the relay: `apps`, `compat.sse`, `turn.*`, `stun.*`, `push.fcm.*`,
`limits.slotBytes`, `limits.sigItems`, `limits.sigSenderShare`.

## The doctor

`php bin/doctor.php [--url=https://relay.example.com] [--admin-token-file=PATH] [--json] [--no-slow-body] [--skip-web]`

It checks PHP, extensions, the ini values that matter (memory_limit 64M or more, post_max_size, disabled functions), the data
folder (outside `public/`, modes, config, schema, journal mode against the filesystem type), that a wake shard written here is
visible to a second PHP process within 50 ms, and the keys. With `--url` it also asks the public address what only the web can
answer: it requests `/data/relay.sqlite`, `/data/secrets/admission.hmac`, `/data/secrets/relay.key`, a dot-dot path into
`data/`, `/bin/doctor.php`, `/src/Db.php`, `/install.php` (after installation) and `/.env` and fails on anything but 403 or 404;
it sends a dummy bearer through the real web stack and fails loudly when the header does not arrive; with `--admin-token-file`
it compares what the **web** PHP sees (ini values, functions, `REMOTE_ADDR`, forwarding headers) with the command line's; and
it opens a request whose body never finishes and reports whether the host closes it. The admin token comes from a file, is only
sent in an `Authorization` header and is never printed. Exit 0 when nothing failed (warnings are advice), 1 on a failure, 2 on
usage. `--slow-body-watch=SECONDS` (default 5) sets how long the slow-body probe watches.

## Devices, enrolment and the roster

A device is a desktop, a phone or a provider, and each holds one capability token (`oaiyrt1.<id>.<secret>`, 63 characters) that
names only that device. The database keeps the token's id and a hash of its secret, never the token. Every way a token can fail
is the same `401`; a device that was revoked hears `401 revoked`, but only after its secret verified, so a stranger learns
nothing. Twenty wrong tokens in a minute from one address, or twenty wrong secrets for one token id in an hour, are refused for a
while (a correct token is never refused by the address counter).

- **Enrolment.** `php bin/relay.php key desktop` (or `provider`) mints a single-use key, valid for an hour by default, and writes
  it to `data/keys/<id>.txt`; the desktop redeems it with `POST /v1/enroll`. The key carries a secret from which both the key id
  and a signing key are derived; the relay stores only the derived **public** key, so a copy of the database cannot redeem
  anything. Redeeming means signing the exact request body, so the secret is never sent. Five wrong proofs burn a key.
- **Phones** are paired through the desktop, never by the relay alone (the pairing routes are a later part of the relay, not
  in this build). The desktop lists, renames, limits and revokes them with `GET/POST /v1/devices…`, and pushes its
  authoritative roster with `POST /v1/roster`. A phone that the roster leaves out is revoked at once. A phone cannot change
  its own keys.
- **Revoking** a device is immediate and complete: its token stops working, its inbox and pending items are deleted, and a
  request it is holding open ends with `401 revoked` within a quarter of a second.
- **Token rotation:** `POST /v1/tokens/rotate` returns a new token; the old one works for ten more minutes, and a second
  rotation inside that time is `409`.

### `php bin/relay.php`, the administration commands

Run on the host, with shell access to `data/` (which is the relay's trust root). None of it is reachable from the network, and
nothing prints a secret unless you pass `--print`.

| Command | What it does |
|---|---|
| `key desktop\|provider [--ttl=SECONDS] [--name=NAME] [--print]` | Mint an enrolment key. Written to a 0600 file by default; `--print` writes it to the terminal |
| `devices` | List devices (no tokens, no keys) |
| `revoke <deviceId>` | Revoke a device; revoking a desktop revokes its phones too |
| `roster-reset <desktopId> [appId]` | Forget a stored roster so that the next push starts clean |
| `reset limits` / `reset epoch` | Clear rate limits and token locks / start a new epoch (every client resets its cursor) |
| `status` | Counts and facts, no secrets |
| `backup [--out=FILE]` | A consistent copy of the SQLite database (`VACUUM INTO`); never overwrites |
| `restore FILE --yes` | Put a backup back. Refuses another relay's file, a newer schema or a damaged file, keeps the old database, and starts a new epoch |
| `export DIR` / `import DIR --yes` | Move a relay: database, config, keys and a checksummed manifest; `import` refuses an installed relay |
| `gc [--force] [--vacuum]` | Run the garbage collector now |

Exit codes: 0 done, 1 failed, 2 usage. Set `OAIY_RELAY_DATA` to name a data folder elsewhere.

### Calibration

The relay does not know how many PHP workers it may use. OAIY's "Test this relay" measures it with four routes that only the
desktop (or the admin token) may call: `GET /v1/admin/hold`, `GET /v1/admin/stream-probe`, `POST /v1/admin/echo` and
`POST /v1/admin/capacity`. A measurement can only lower a limit (workers, the largest body, the longest hold), never raise one
past the configuration, and `info` then advertises what is in force.

## Garbage collection without cron

The relay clears out expired items, old metadata, spent tickets and stale signal files itself, at most once a minute, from a
health or status request (or after a poll when the host can answer first). If your host has cron, `*/5 * * * * php
~/oaiy-relay/bin/gc.php` is optional: `php bin/gc.php [--force] [--vacuum]` runs the same pass, and `--vacuum` (weekly at
most, because it locks the database) compacts the file.

## Tests

```
php tests/run.php                  every test
php tests/run.php --filter=poll    the tests whose name contains "poll"
php tests/run.php --file=poll|holds  the tests of tests/cases/poll.php and holds.php
php tests/run.php --slow           also the slow ones (real waits and holds; a minute or more)
php tests/run.php --stop           stop at the first failure
php tests/run.php --list           the test names
OAIY_TEST_DB=mysql php tests/run.php     the same tests on a throwaway MySQL 8.x (or =mariadb for MariaDB)
```

`OAIY_TEST_DB` starts a MySQL or MariaDB server of its own for the run: `mysqld` from `C:/wamp64/bin` or the one named by
`OAIY_TEST_MYSQLD` / `OAIY_TEST_MARIADBD`, with `--no-defaults`, a data directory under the test's temporary folder and a port
the operating system chose. It is shut down and deleted when the run ends and never touches another MySQL on the machine.
Tests that are about SQLite alone (backup, restore, export) skip themselves there.

Dependency free: nothing to install. The runner enables `sodium` for a run on a PHP that ships it disabled (it never edits
`php.ini`), works in temporary directories it deletes, starts `php -S` on free loopback ports of its own, and never touches a
web server's document root, a real data folder or any port that is not its own. `php -S` serves one request at a time on
Windows, so tests that need overlapping requests start several servers on one data folder. The test clock is a PHP constant
that only `tests/prepend.php` defines; the release zip leaves `tests/` out.

### What has and has not been tested

Run, on Windows 11, with PHP 8.0.30 and 8.4.15, over `php -S` and the command line: the installer (CLI and web), the doctor's
checks (with synthetic facts and against real `php -S` servers, including a server whose document root is the relay folder), the
`gc.php` command, the secrecy of every installer output (searched for the actual secrets), and the exposure of `bin/` and `src/`
through the web. The relay's own rules (authentication, items, the poll, holds, quotas, garbage collection) are tested on
SQLite, and the same suite runs unchanged on MySQL 8.4.7 and MariaDB 11.4.9 (`OAIY_TEST_DB`), which also has its own tests of
the schema (case-sensitive ids, `MEDIUMTEXT` bodies, strict mode) and of concurrent posts.

**Not run:**

- `public/.htaccess` was **not** run through Apache or LiteSpeed. The nginx snippet above was **not** run through nginx or PHP-FPM.
  Both are written from the design and from general knowledge of those servers; the doctor's exposure and Authorization probes
  are the check to use on your host.
- No real shared host, cPanel account, VPS, CDN or proxy was tried.
- File modes (0600 and 0700) and the ownership checks are only asserted on POSIX systems, and this was developed on Windows,
  where the tests skip them. The `/proc/self/mountinfo` reader is tested with synthetic text, not on an NFS mount.
- The doctor's MySQL checks (`max_allowed_packet`, `max_user_connections`) and a MySQL install by the installer were not
  exercised against a MySQL server. The relay itself, with its test suite, was (next item).
- MySQL and MariaDB were tested on a **local throwaway server only** (MySQL 8.4.7 and MariaDB 11.4.9 on Windows, over loopback,
  one server process, root without a password). Not tried: MySQL 5.7, MariaDB 10.x, a shared host's MySQL with its own
  `max_user_connections`, `sql_mode` or collation defaults, a remote server, or TLS to the database.
- The installer's terminal prompt was not driven on a real terminal (its answer parser is tested).
- The web installer was tested over plain http on loopback, not over TLS.
