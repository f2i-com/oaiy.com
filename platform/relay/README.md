# OAIY Relay

A small server you run yourself. It stores and forwards small opaque items in per-device mailboxes, so the OAIY desktop, a
phone and a provider such as FormLogic can exchange live data without the provider carrying it. It is written in plain PHP
(no Composer, no framework), keeps its state in SQLite (or MySQL and MariaDB), needs no cron and no background process, and
runs on ordinary shared hosting as well as on a small VPS. Live audio never touches it.

The wire protocol is [`platform/protocol/relay/v1/`](../protocol/relay/v1/README.md). This folder is one implementation of it.

## What you need

| | |
|---|---|
| PHP | 8.0 or later; **8.2 or later is recommended** (PHP 8.0 and 8.1 are end of life, and the doctor warns on both). Tested on 8.0.30 and 8.4.15. |
| Extensions | `sodium` (bundled with PHP since 7.2), `json`, `hash`, and `pdo_sqlite` (the default store) or `pdo_mysql`. `openssl` and `curl` matter only to the optional FCM sender, which is not part of this build. |
| Disk | A writable `data/` folder on a **local** filesystem (SQLite's write-ahead log is unsafe on NFS and similar). The hold markers in `data/holds/` are stamped with PHP's clock and not the filesystem's, so a `data/` whose filesystem clock is off PHP's by minutes still counts its holds (a marker is stale 5 seconds past its cap, or when stamped more than a minute ahead). |
| HTTPS | A publicly trusted certificate on a hostname of its own (a subdomain such as `relay.example.com`, not a path on a site that hosts anything else). The desktop and the phone use the bundled web roots and cannot use a private CA. |
| Workers | Each waiting poll holds one PHP worker for up to 20 seconds; at most three polls that wait of one credential run at once (the fourth is `429`, `Retry-After: 1`, before the database is read), so one phone cannot pin a small pool. Measure what your host allows with the [host probe](probe/host-probe.php) and [`docs/relay-hosting-matrix.md`](../../docs/relay-hosting-matrix.md) before you commit to a plan. |

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

1. Choose PHP 8.2 or later in the host's PHP selector (8.0 works but is end of life) and make sure `sodium` and `pdo_sqlite` are enabled (both usually are).
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

**On Apache or LiteSpeed (cPanel is both), three things about `public/.htaccess`** (the section below has the detail): the host's
`AllowOverride` must cover **AuthConfig, FileInfo, Options and Indexes** (or `All`), or every request is a 500; **`mod_rewrite`** must be
on, or `/v1/` is a 404; and the file **replaces any `Require` of a folder above `public/`**, so a password or an address limit that you put
on a parent folder does not protect the relay (put it in a `<Location>` block of the server's configuration instead). The doctor says
the first two when it sees them.

A lost first key is re-armed, never re-issued over the network: create `INSTALL_ENABLED` again and run `php bin/install.php
--rekey` (or open `install.php?mode=rekey` and use the token). It writes a new `data/first-key.txt` and changes nothing else.

### The installer's rules

- It never prints or sends a token, key or secret: not to the screen, not in an HTTP response or header, not in an error.
  It writes them to files (mode 0600, `data/` mode 0700) and names the files. Files are created under a umask that leaves exactly
  the mode asked for, so a host with a lax umask never has a secret readable by others, even for an instant. The SQLite database
  (whose `-wal`, `-shm` and `-journal` files take its mode), the log and the backups are owner-only too, and the doctor checks all
  of them on a POSIX system.
- It refuses to write a secret where the web can read it, and it decides that in one place for the command line installer, the
  web installer and a re-key alike: `data/` inside `public/`, `data/` inside the document root the web server itself reports
  (the web installer knows it), or a canary file put in `data/` that is served over the web. **The canary probe is on by
  default**: the command line asks at the `--url` address (or at `--probe-url` if you name another), the web installer at the
  public address you type, a re-key at the address in `config.json`. Refusal happens before any question is asked and before
  anything is written, so a document root that is the relay folder leaves no key behind. The canary is asked for at `/data/` and
  at every place a folder above `data/` could serve it (`/relay/data/`, `/site/relay/data/`, up to four folders of `data/`'s own
  path), because a relay unpacked into a folder of an existing site is served under that folder's name and the origin's
  `/data/` says nothing about it; a canary served at any of them is a refusal that names the folder. A site that maps the
  relay to a name its folders do not have (an alias, a rewrite) is asked with `--probe-url=` followed by the address and path
  where the relay folder is served. An address that cannot be reached is
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

Run against a real Apache 2.4 (`tests/cases/htaccess.php`, which starts the httpd it finds on a loopback port and skips where there is
none; 2.2 syntax and LiteSpeed were not run): read it as a starting point and let the doctor judge it on your host.

**What it needs from the server.** `AllowOverride` that covers **AuthConfig, FileInfo, Options and Indexes** (or `All`), and
**`mod_rewrite`**. With less `AllowOverride` (say `FileInfo Options Indexes`) Apache does not allow the `Require` in the file and
answers **500 to every request**; the doctor's `web.authorization` row says so (a 500 that carries no `X-OAIY-Relay` header is the web
server's own). Without `mod_rewrite` the front controller is missing and `/v1/` is a 404, which the doctor says too; what is in
`public/` is refused all the same (the rules below do not depend on it). **Before this was run against Apache the front controller did
not work at all** (the file's own rewrite sent the internal redirect to `index.php` to a 404), and an item id that starts with a dot
(`/v1/items/.a`, which design 4.3 allows) or holds one was refused (`/v1/items/cmd.1`): both are fixed and have tests.

It:

- **allows only what is the relay's**: one `Require expr` that holds only if both the request line as the client sent it and the
  path Apache resolved it to (dot segments taken out: `GET /v1/../README` is `/README`) are `/v1/...`, `/status.html` or
  `/install.php` (the resolved path may also be `/index.php`, where the rewrite sends `/v1/...`). **Everything else is a 403**,
  whatever the server's modules: a `README`, a nested page, a dot folder (`/.git/config`), `index.php` asked for directly, an
  editor's leftover that a careless upload put in `public/`, and a path that starts like the relay's and climbs out of it. (The
  first version of this line read the request line only, and `/v1/../README` was handed out: a test sends it raw now.)
  LiteSpeed's handling of `Require expr` was not tried; if it refuses the line, the doctor's `web.authorization` row says the web
  server itself answered;
- **grants itself** by that same line. Apache reads the `.htaccess` of every folder on the way to a file wherever `AllowOverride`
  covers them, and many hosts set it for all of `/home` or `/var/www`, so the package root's deny-all (below) would otherwise refuse every
  request to the relay, which is what a real Apache 2.4.65 did before this was added. **This replaces any `Require` of a parent folder's
  `.htaccess` or of a `<Directory>` block of the server's configuration that covers `public/`** (Apache's rule: a folder's `Require`
  replaces its parents'), so a site that is password protected or limited to some addresses at a folder above the relay is **not**
  protected for the relay: **say it in a `<Location>` block of the server's configuration, which Apache applies after the `.htaccess` files
  and which holds** (`<Location "/"> Require ip 192.0.2.0/24 </Location>`; `tests/cases/htaccess.php` runs both);
- turns directory listings off;
- passes the `Authorization` header to PHP (`CGIPassAuth On` on Apache 2.4.13 or later, and the `SetEnvIf` / `RewriteRule`
  forms for older stacks). Without it every request is a uniform `401`, which looks like a revoked device;
- sends `/v1/*` to `index.php` (the conditions read `THE_REQUEST`, the request as sent, because after the redirect the URI is
  `/index.php`), and answers 404 for anything else where the server is Apache 2.2.
The folder above it carries its own `.htaccess` (and `data/` gets one from the installer) that refuses everything, in Apache 2.4
syntax (`Require all denied`) and 2.2 syntax (`Order deny,allow` and `Deny from all`), each inside an `IfModule` for the module
that understands it. It matters only when the document root is wrong, and then only if the server lets `.htaccess` files apply
(`AllowOverride`). `web.config` does the same on IIS. The doctor tells you which of them held. On a real Apache 2.4 a site whose
document root holds the relay folder, and one whose document root is the relay folder, answered 403 to `data/`, `src/`, `bin/`,
`VERSION`, `INSTALL_ENABLED` and the dotfiles, and `data/` alone was refused where the package root's file was missing (an upload
that skips dotfiles). The grant in `public/.htaccess` replaces the `Require` rules that were applied to that folder before it (a
`<Directory>` block of the server's configuration and any `.htaccess` above), which is how Apache lets a folder's rules replace its
parents'; a site that limits the relay to some addresses says so in a `<Location>` block, which Apache applies after the `.htaccess`
files.

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
| `call.enabled` | false | Call features: the ring lane, the admission issuer and the Aokie compatibility routes (see below) |
| `call.challenge_s` | 25 | How long an Aokie endpoint challenge lives, 10 to 30 seconds. The shipped phone refuses a challenge that is more than 30 seconds ahead of its own clock or not ahead of it, so a phone clock may be `30 - challenge_s` seconds behind the relay's and `challenge_s - 1` ahead: 25 (FormLogic's and the design's) tolerates 5 behind and 24 ahead, 15 tolerates 15 behind and 14 ahead |
| `apps` | unset (any) | The app ids this relay issues admissions for (a list); another is `403` |
| `compat.sse` | `auto` | `auto`/`on`: offer the framed stream only after the calibration proved the host flushes; `force`: offer it regardless; `off`: never |
| `turn.urls`, `turn.secret`, `turn.ttl`, `turn.relay_only` | none, none, 600, false | The TURN server of your coturn (`turn:` and `turns:` urls, at most 8), its `static-auth-secret` (32 to 4,096 bytes, not a placeholder), the credential lifetime (60 to 3,600 s) and whether every route must use TURN |
| `stun.urls` | none | STUN urls (`stun:`, `stuns:`, at most 8) |
| `limits.sigItems`, `limits.sigSenderShare` | 1024, 0.25 | Frames in one Aokie party mailbox (at most 1,024), and the share of a mailbox and of `limits.mailboxBytes` one phone may use in the plugin's |
| `token_pepper` | none | Optional HMAC pepper (16 characters or more) for token hashes; set it before issuing tokens |
| `wake.mode`, `wake.safety_ms` | `file`, 2000 | `file` (shard files, with a database fetch every `safety_ms`) or `db` (poll the database) |
| `client_ip.header`, `client_ip.trusted_proxies` | none | Honoured only when `REMOTE_ADDR` is a trusted proxy |
| `cors.extra_origins` | none | Extra https origins listed in `info` |
| `gc.one_in` | 20 | About one request in this many checks whether a garbage-collection pass is due (0 to 1,000; 0: none does, and only a health or status request, or a poll that can be answered before the pass on a host that can, runs it) |
| `debug.error_sites` | `false` | When `true`, every `401`, `404` and `5xx` the relay answers is also one line in `data/logs/relay.log` (`event: error_site`) with the route, status, code, the file and line that decided it, the relay's clock and the host's, and the process: no credential, id, header or body. For finding out why an answer was what it was; leave it off otherwise |

Accepted and validated now, used by a later part of the relay: `push.fcm.*` and `limits.slotBytes`.

## The doctor

`php bin/doctor.php [--url=https://relay.example.com] [--admin-token-file=PATH] [--json] [--no-slow-body] [--skip-web]`

It checks PHP, extensions, the ini values that matter (memory_limit 64M or more, post_max_size, disabled functions), the data
folder (outside `public/`, modes, config, schema, journal mode against the filesystem type, and a warning naming
`first-key.txt` or `admin-token.txt` and how old it is while either is still there: each holds a secret you were to read once
and delete), that a wake shard written here is
visible to a second PHP process within 50 ms, and the keys. With `--url` it also asks the public address what only the web can
answer: it requests `/data/relay.sqlite`, `/data/secrets/admission.hmac`, `/data/secrets/relay.key`, a dot-dot path into
`data/`, `/bin/doctor.php`, `/src/Db.php`, `/install.php` (after installation), `/.env`, and what an install leaves behind that
would hand over the relay (`/data/first-key.txt`, `/data/admin-token.txt`, `/data/config.json`, `/data/secrets/admin.json` and the
web installer's `/INSTALL_ENABLED`) and fails on anything but 403 or 404. If `--url` has a path (a relay unpacked into a folder
of an existing site, `https://site/relay`), every probe is made at that path **and** at the site's root, because that is the
commonest way for `/relay/data/` to be served;
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
- **Phones** are paired through the desktop, never by the relay alone (next section). The desktop lists, renames, limits and
  revokes them with `GET/POST /v1/devices…`, and pushes its authoritative roster with `POST /v1/roster`. A phone that the
  roster leaves out is revoked at once. A phone cannot change its own keys.
- **Revoking** a device is immediate and complete: its token stops working, its inbox and pending items are deleted, what it had
  already posted to others is not delivered afterwards (for a phone that includes its frames in the plugin's mailbox), a post that was in flight at the moment of the revocation does not commit after it and makes no mailbox for a revoked device, and a
  request it is holding open ends with `401 revoked` within a quarter of a second.
- **Token rotation:** `POST /v1/tokens/rotate` returns a new token; the old one works for ten more minutes, and a second
  rotation inside that time is `409`.

## Pairing a phone

The relay keeps a **rendezvous** for one pairing: a mailbox that holds the desktop's offer, the phone's response and the owner's
decision, and that hands the phone its token sealed to its own key. It never sees the secret (the QR code or the typed code): the
phone finds the rendezvous by an id (`pid`) derived from it, checks the offer's MAC with a key only the two ends can derive, and
the relay only stores and forwards. The steps and their answers are in the protocol package
([`README.md`](../protocol/relay/v1/README.md), section 10.1, and the recorded ceremony in
[`fixtures/pairing-ceremony.json`](../protocol/relay/v1/fixtures/pairing-ceremony.json)); what the relay adds:

- **The pid is the phone's only credential.** An unknown, an expired and a burned pid answer one identical `404` (same lookup, same
  body), a bad request body is `400` whatever the pid, and a wait for an unknown pid returns at once. Nothing tells a stranger
  which pids exist.
- **Limits** (each has a test): an offer of 4,096 bytes, a response of 8,192, three responses and three rejects per rendezvous,
  60 `GET`s while it is open or answered (reads of an outcome are not counted; they have a bucket of 10 a minute per address and pid), at most 900 seconds of life (600 by default), 16 rendezvous open per desktop, 30 requests a minute per
  client address on the phone's two routes, and at most 4 waiting requests per address. A waiting `GET` is an edge hold: when the
  worker pool is nearly full it is answered at once with `hold.refused` and the phone short-polls.
- **The sealed token.** On approval the relay creates the phone's device, mints its token, seals it with `sodium_crypto_box_seal`
  to the phone's X25519 key (a key of small order is `422` before anything is created) and stores only the sealed box, the token's
  hash and the desktop's signed **receipt**. The plaintext token exists inside that one request. The relay checks the receipt
  against the desktop key in the offer, and the approved keys against the response the phone posted, before it creates anything;
  the phone checks both again, because the relay is not trusted.
- **Races.** Every change of state is a conditional `UPDATE` on a row read under its lock, in one immediate transaction: two
  responders to one pid cannot both win, an approval racing a burn has one winner, and the database agrees with it. The two checks
  that count rows that do not exist yet (16 open rendezvous per desktop; 16 phones per desktop and one active device per phone
  key) take a named gate first, so on MySQL and MariaDB the second of two requests waits for the first to commit and counts what
  it made; an approval also locks the desktop's row and is refused (`401 revoked`) when the desktop was revoked meanwhile. The
  tests race real requests on both servers.
- **Garbage collection** removes a rendezvous ten minutes after the phone read its outcome, and at expiry.

`fixtures/sealed-token.json` holds sealed tokens the real relay produced, with the recipient key and the checks a reader must
make; `php tests/fixtures.php --check` verifies them and `--write` records them again (`--write pairing` or `--write aokie` for one set). `fixtures/rust-check/` opens them with the Rust `crypto_box` crate, and `fixtures/selftest_fixtures.py` shows that the readers refuse damaged copies.

## Call features: admissions, TURN and the Aokie routes

With `call.enabled` on (the installer asks; the default is no, and everything below answers `403 feature_disabled` until it is
on) the relay does what FormLogic does for the shipped Aokie plugin and phone, so they can signal a call through it:

- **Admissions.** `POST /v1/admission` (alias `/v1/aokie-companion/admission`) mints the 90 second bearer for the plugin when the
  desktop's token asks and for a phone when the phone asks for itself, with the phone's checks (its own device and key, a live
  desktop, a roster that lists it, `state_read`) and short-lived ICE credentials. Nothing an admission does changes what the relay
  stores. The admission secret is `data/secrets/admission.hmac`, made by the installer, never in an answer, a log or the status page.
- **Signalling.** `GET .../relay/challenge`, `POST` and `GET .../relay/frames` and `GET .../relay/stream` under
  `/v1/aokie-companion/`: a mailbox per party (the plugin, and each phone), frames that live 120 seconds and are never
  acknowledged, and the stream the shipped carriers open. Every request re-reads the device row, so removing a phone from the
  desktop's roster or revoking it ends its access at its next request and its held stream within a fraction of a second.
- **Errors** on these routes have FormLogic's three members, `{"error":true,"code","message"}`, and a wait is the `Retry-After` header.

The shipped plugin and phone are not unchanged clients in every respect. The relay works around what it can (a post to a phone that
is gone is a success, a display name is cut to 120 bytes, frames are sent in ASCII, a request with no `supportedTransports` is read
as `relay`), and the rest needs a change in them: a 403, 404 or 503 from any route makes the plugin re-bootstrap, `Retry-After` is
ignored, a phone has no route to discover a personal relay, and the challenge's margin is narrow (`call.challenge_s`). The list is
"Design defects" 9 and 10 in the [protocol README](../protocol/relay/v1/README.md).

**TURN.** Put a coturn beside the relay (a phone on a carrier network needs it: the status page warns when none is configured) with
`use-auth-secret`, `static-auth-secret=<the same string as turn.secret>` and a `realm`, and list its urls in `turn.urls`. Every
admission carries a credential: `username = <expiry>:<opaque id>` (the expiry is rounded down to a window of a sixth of `turn.ttl`, at most 100 seconds, so one endpoint keeps one username, and one coturn `user-quota` key, for a window instead of getting a new one with every admission), `credential = base64(HMAC-SHA1(secret, username))`, the id a
keyed hash of the endpoint (FormLogic's), so no device id reaches coturn's log. A bad `turn.*` or `stun.*` section stops the relay
at the next request instead of issuing credentials that the plugin's decoder would refuse.

**The stream and the calibration.** The framed stream needs a host that sends the first bytes of a response before it ends, which
a shared host's proxy may not (output buffering, FastCGI, a CDN). The relay writes the preamble at once, a keepalive every two
seconds and always ends with an `end` event, but whether the bytes reach the client is the host's business: OAIY's "Test this relay"
runs `GET /v1/admin/stream-probe`, and only a passing probe makes an admission advertise `relay` (and `info` list
`compat.sse-framed-poll`). On a host where it fails the shipped plugin is answered `422` and call signalling does not work there;
`compat.sse: "force"` overrides the probe at your own risk. A stream takes one worker of the pool for up to 20 seconds (or
`wait.max`, if lower), one per party; when the pool is at its hard limit a stream is refused with `503` and a frames wait becomes
a short poll.

The rules and their numbers are in the protocol package (`README.md` section 10.6 and interpretations 36 to 45), and the answers
of the real relay to the plugin's and the phone's requests are recorded under
[`fixtures/aokie/`](../protocol/relay/v1/fixtures/aokie/README.md), with the rules of the shipped decoders transcribed and applied
to them (`php tests/fixtures.php --check` and the conformance suite run that).

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
past the configuration, and `info` then advertises what is in force. A calibration hold pins a worker like any other, so a credential
may have 16 of them running at once and has a budget of hold time: 600 seconds at once and a fifth of a second back for every second
(twelve minutes an hour), after which `GET /v1/admin/hold` is `429 rate_limited` with the seconds to wait. (A cap that followed the
pool, its workers plus two, never tripped on a small one: the requests beyond the workers wait in the web server, where nothing counts
them, so a credential could pin every worker for as long as it liked.) **What a measurement costs.** One 35 second hold per worker: a
pool of 5 is 175 of the 600 seconds, which come back in about fifteen minutes, so a second measurement a quarter of an hour later is
whole; a pool of 17 workers or more spends all 600 (17 holds are 595 seconds), and the budget is back after about fifty minutes: a second
measurement before that is `429 rate_limited` with the seconds to wait. That is the price of a bound that a stolen credential cannot
get round; set `capacity.workers` in `config.json` (the table above) on a host whose pool you know, and nothing needs measuring.

## Garbage collection without cron

The relay clears out expired items, old metadata, spent tickets and stale signal files itself, at most once a minute, from a
health or status request (or after a poll when the host can answer first). If your host has cron, `*/5 * * * * php
~/oaiy-relay/bin/gc.php` is optional: `php bin/gc.php [--force] [--vacuum]` runs the same pass, and `--vacuum` (weekly at
most, because it locks the database) compacts the file. A request's pass has a budget of 50 ms and keeps to it inside the deletes as
well as between them (a hundred rows at a time): a backlog of a hundred thousand rows of metadata, or a limiter table far over its
bound, is worked off over many passes and never by one request that waits for all of it.

**How fast a backlog goes.** Where the host can answer first (PHP-FPM and LiteSpeed: `fastcgi_finish_request`) the pass runs after the
answer with a budget of 5 seconds, about 60,000 rows of metadata a pass. Where it cannot (mod_php, CGI, `php -S`) a pass is paid for by a
health or status request, or by about one request in `gc.one_in` (20) of any kind, and has the 50 ms budget: **about 600 rows a pass and
a pass a minute at most, so a backlog of a hundred thousand rows takes about three hours there.** A backlog does not grow without bound
on such a host (the limiter table has a hard cap, and items expire), but one that is already there is cleared slowly: run
`php bin/gc.php --force` once by hand (it has no budget), or put it in cron as above, and the host that has no cron gets the slow path
only. A test keeps the budget honest between chunks of a backlog (`4.18.6 a backlog is cut at the budget between chunks of a hundred`).

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
`php.ini`), works in temporary directories it deletes, starts `php -S` on free loopback ports of its own (and, for the `.htaccess`
tests only, an Apache `httpd` with a configuration of its own, bound to `127.0.0.1`, when `OAIY_TEST_HTTPD` names one or WAMP's is
found; those tests skip otherwise), and never touches a web server's document root, a real data folder or any port that is not
its own. `php -S` serves one request at a time on
Windows, so tests that need overlapping requests start several servers on one data folder. The test clock is a PHP constant
that only `tests/prepend.php` defines; the release zip leaves `tests/` out.

### What has and has not been tested

Run, on Windows 11, with PHP 8.0.30 and 8.4.15, over `php -S` and the command line: the installer (CLI and web), the doctor's
checks (with synthetic facts and against real `php -S` servers, including a server whose document root is the relay folder), the
`gc.php` command, the secrecy of every installer output (searched for the actual secrets), and the exposure of `bin/` and `src/`
through the web. The relay's own rules (authentication, items, the poll, holds, quotas, garbage collection) are tested on
SQLite, and the same suite runs unchanged on MySQL 8.4.7 and MariaDB 11.4.9 (`OAIY_TEST_DB`), which also has its own tests of
the schema (case-sensitive ids, `MEDIUMTEXT` bodies, strict mode) and of concurrent posts.

The pairing rendezvous and the sealed token (RL-06), and the admission issuer, the Aokie routes and the stream (RL-07) have their
own tests: every rule of the design's sections 4.10 and 4.14 is named by one, the two are run against fleets of `php -S` servers
for the races (two responders to one pid, an approval racing a burn, a newer stream replacing an older one within a step, a
revocation ending a held stream), the pool tests (`tests/cases/pool.php`: a pool of W workers, meaning W `php -S` servers behind
`tests/pool_front.php`, a front that queues a request until a worker is free as PHP-FPM does, is put in front of the relay while one
phone opens fifty streams, frames waits or native polls at once, and again every two seconds (with a new admission each time for the
streams and frames waits; a phone's and a desktop's token for the polls), on 5 and on 8 workers,
and `/v1/health` is timed: design 9.2's test; it needs no Python and runs with `--slow`) and against a fake clock for every lifetime, the bearers and credentials are checked against the
design's vectors and against FormLogic's own known answers, the sealed tokens of a pairing were opened with the Rust `crypto_box` crate 0.9.1 (`fixtures/rust-check`, which also shows that its `unseal` alone does not refuse a small-order ephemeral key), and mutation testing broke the rules that carry safety one at a time (220 changes, 207 caught; the other 13 are equivalent: a second guard makes the first redundant, or the platform's own library refuses the same thing). The fixes that followed two independent reviews were each mutation-checked in place the same way: 173 changes, 160 caught, and the 13 that survive are equivalent and named in the commit messages (a cheap pre-check in front of an atomic one, a redundant second guard, a shared lock that only MySQL and MariaDB can show, a dead address asked at every path, a file chmod-ed at once after it is made, a spelling of the same cap that cannot be valid base64url, an access check that Apache does not repeat for an internal redirect, a directory listing that a Require refuses first). Some of the older logs of the first round were not kept, so its count (118 changes, 9 survivors) is from the specifications and not from every log.

**Not run:**

- **No Aokie code was run.** The plugin's and the phone's decoders were read, not compiled: the fixtures are checked by a Python
  transcription of their rules (`fixtures/aokie/aokie_decoders.py`), which can be wrong where the Rust differs from my reading.
  A Rust contract test against the same fixtures is listed in `fixtures/aokie/README.md` and was not written.
- **No coturn was run**, so no real allocation was made with these credentials; the credential is the HMAC that coturn's
  `use-auth-secret` computes, and it agrees with FormLogic's own computed answers and with Python's `hmac`.
- The framed stream was tested against `php -S` on Windows only. How a real host's web server, PHP-FPM, LiteSpeed or a CDN buffers
  it, and how they report a client that hung up, was not tried: the calibration's probe is what decides, on your host.

- `public/.htaccess` and the deny-all files were run through a real Apache 2.4.65 (WAMP's, on Windows) and 2.4.58 (Ubuntu 24.04 under
  WSL 2), without PHP: which files are refused and served under each layout, that `/v1/...` (item ids with dots and a leading dot
  included) reaches `index.php` (Apache then hands out the file, which is how the test sees it), what `AllowOverride` and a missing
  `mod_rewrite` do, and what a parent's `Require` and a `<Location>` do (`tests/cases/htaccess.php`); **not** with PHP behind it (that
  the front controller then runs, and the `Authorization` pass-through), not on Apache 2.2 and not on LiteSpeed. The nginx snippet above was **not** run through nginx or
  PHP-FPM. Those are written from the design and from general knowledge of those servers; the doctor's exposure and Authorization
  probes are the check to use on your host.
- No real shared host, cPanel account, VPS, CDN or proxy was tried.
- File modes (0600 and 0700) and the ownership checks are only asserted on POSIX systems, and this was developed on Windows,
  where the tests skip them. The one that needs no database (`Fs::createPrivate`, which makes the SQLite file owner-only: a missing
  file is created, a wider one narrowed, under the loosest umask) was run under Ubuntu's PHP 8.3 in WSL 2; the install-level ones
  (the database, its `-wal` and `-shm`, the log and every secret after an install under `umask(0)`) need `pdo_sqlite`, which that
  PHP lacks, so they have not been run on Linux. The `/proc/self/mountinfo` reader is tested with synthetic text, not on an NFS mount.
- The doctor's MySQL checks (`max_allowed_packet`, `max_user_connections`) and a MySQL install by the installer were not
  exercised against a MySQL server. The relay itself, with its test suite, was (next item).
- The shared row locks of the revocation and roster races are spelled `LOCK IN SHARE MODE` (`Db::forShare`): MySQL 8.0 and later call
  that deprecated in favour of `FOR SHARE` (8.4.7 takes both), and MariaDB 11.4.9 refuses `FOR SHARE` (a syntax error: tried), so the one
  spelling both accept is used. A MySQL that drops the old one needs its own in that one function.
- MySQL and MariaDB were tested on a **local throwaway server only** (MySQL 8.4.7 and MariaDB 11.4.9 on Windows, over loopback,
  one server process, root without a password). Not tried: MySQL 5.7, MariaDB 10.x, a shared host's MySQL with its own
  `max_user_connections`, `sql_mode` or collation defaults, a remote server, or TLS to the database.
- The installer's terminal prompt was not driven on a real terminal (its answer parser is tested).
- The web installer was tested over plain http on loopback, not over TLS.
