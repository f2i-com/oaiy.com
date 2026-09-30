# Relay hosting matrix

What a host must do before the OAIY Relay (`platform/relay/`) can run on it, how to find out in ten minutes, and what has
been measured so far. Nothing in a repository can settle these facts: they belong to the host's web server, PHP handler
and firewall, so the answer for your host is the probe's answer, not this page's.

## Run the probe first

`platform/relay/probe/host-probe.php` is one file with no dependencies (PHP 7.2 or later for the probe itself; the relay
needs 8.0).

1. Open the file and set `OAIY_PROBE_TOKEN` to a random string of at least 16 characters (`openssl rand -hex 16`).
2. Upload it to a folder the host serves as PHP, on the hostname you mean to use for the relay if you can.
3. Open `https://your-host/host-probe.php`, type the token, press "Run all" (about a minute; tick the box for the 35 second
   hold). Or open `https://your-host/host-probe.php#token=...&auto=1`: a URL fragment is never sent to the server or logged.
4. Press "Copy the report" and paste it where you keep your notes. The report holds no token and no header value.
5. Delete the file. It also stops answering 24 hours after it was last modified, and answers `404` and nothing else until a
   token is set.

The probe answers only requests that carry the token in an `X-Probe-Token` header, compares it in constant time, never
echoes a credential (the Authorization check reports yes or no), and runs behind a nonce-based Content-Security-Policy.

## What each check decides

| Check | Why it matters | If it fails |
|---|---|---|
| PHP 8.0 or later; `sodium`; `pdo_sqlite` (or `pdo_mysql`) | The relay signs and verifies with libsodium and stores items in SQLite or MySQL | Pick another PHP version in the host's selector, enable the extension, or use another host |
| **Authorization header reaches PHP** | Every request carries the device token in it. When a FastCGI-family stack strips it, every request is a uniform `401`, which looks like a revoked device | Add `CGIPassAuth On` (Apache 2.4.13 or later) or `RewriteRule .* - [E=HTTP_AUTHORIZATION:%{HTTP:Authorization}]` to `.htaccess`; on nginx `fastcgi_param HTTP_AUTHORIZATION $http_authorization;` |
| **Streaming**: first byte at once, chunks a second apart (event stream with and without `X-Accel-Buffering`, and plain text) | The shipped Aokie carriers need response headers within 10 seconds and read the stream as it arrives | The relay does not offer the compatibility stream on this host. Phones and the plugin need poll-mode carriers (a later package), or a host that flushes |
| Methods GET, POST, PUT, PATCH, DELETE, OPTIONS | The relay only ever needs GET, POST and OPTIONS (every mutating route has a POST form) | Nothing: the aliases are optional |
| Bodies from 32 KiB to 1 MiB | Lane limits go up to 384 KiB; a WAF or `post_max_size` below that lowers what the relay may advertise | The calibration lowers the advertised lane limits to what the host accepts |
| Held requests of 5, 20 and 35 seconds | A poll is held for up to 20 seconds | The calibration lowers `wait.max` to the measured hold minus 5 seconds |
| **Worker pool** (N parallel 8 second holds against one short request) | A held poll pins one PHP worker for its whole hold | Fewer than 8 workers: fewer polling phones, a shorter `wait.max`, push, or a small VPS. The page can only measure this over HTTP/2 or HTTP/3 |
| `fastcgi_finish_request` / `litespeed_finish_request` | Without one, garbage collection and push sends run only from health and status requests, in slices of 50 ms | None needed; the relay copes |
| SQLite and its write-ahead log, second connection reads | The default store. A network filesystem cannot use the write-ahead log | The relay falls back to `journal_mode=TRUNCATE`, or use MySQL |
| Filesystem type, ownership, resident size of a PHP process | Capacity tables assume about 30 MB per worker | Size the pool from the real figure |
| `REMOTE_ADDR`, forwarding headers | Rate limits are per client address. Behind a proxy or CDN every client shares one address until `client_ip` names the trusted proxy | Set `client_ip.header` and `client_ip.trusted_proxies` |

## Measured

Only what was run is in this table. A blank cell is untested, not "fine".

| Host | Measured by | PHP | Authorization | Streaming | Methods | Bodies to 1 MiB | Hold 20 s | Pool | finish_request | SQLite (WAL) |
|---|---|---|---|---|---|---|---|---|---|---|
| PHP built-in server (`php -S`), Windows 11, loopback | `platform/relay/tests/cases/probe.php`, the page script run under Node against it | 8.0.30 and 8.4.15 (8.0.30 needs `-d extension=sodium`) | reaches PHP: `HTTP_AUTHORIZATION`, `getallheaders`, `apache_request_headers` | first byte in a few milliseconds, chunks at 1.0, 2.0, 3.0 s, with and without `X-Accel-Buffering`, as event stream and plain text | all six | all six sizes exact | held 20.0 s | **1 worker**: `php -S` serves one request at a time on Windows (`PHP_CLI_SERVER_WORKERS` does nothing there), so a short request waits behind a hold. Two servers on one folder do overlap. Not a fact about any real host | none | 3.33.0 (8.0.30) and 3.49.2 (8.4.15): WAL accepted, second connection reads |

The page itself was run under Node with a small stand-in for the browser's DOM against a real `php -S` server, in real time
(the token is entered, all checks run, the report is produced), and its script is syntax-checked. It has not been run in a
browser on a real host.

## Not yet measured

These are the hosts the relay is meant for. Each row is filled in from a probe report from that host, and until then every
cell is unknown. What is written under "Expected" is general knowledge about the stack, not a finding.

| Host type | Expected (unverified) | Result |
|---|---|---|
| cPanel shared hosting, Apache with an FPM/LSAPI handler | `Authorization` often stripped unless `.htaccess` passes it; a small per-domain pool (five children is a figure recalled from cPanel's documentation); request timeouts of 30 to 100 seconds; buffering possible behind a proxy or compression module | untested |
| cPanel with LiteSpeed | buffers streamed output unless told not to; different PHP handler (`litespeed_finish_request`) | untested |
| Small VPS, nginx or Caddy in front of PHP-FPM with a pool of its own | headers pass with `fastcgi_param`; flushing works with `X-Accel-Buffering: no`; pool sized by RAM | untested |
| Behind Cloudflare or another proxy that terminates TLS | the proxy sees every token; buffering and timeouts are the proxy's | untested; not recommended |

## Adding a row

Run the probe, paste the report under a new heading, and fill the row from the report. The report's `checks` names each row
of the page (`info`, `ini`, `sqlite`, `fs`, `memory`, `auth`, `methods`, `body`, `flush-sse`, `flush-sse-noaccel`,
`flush-plain`, `hold`, `pool`), each with `status` (`ok`, `warn`, `fail`) and a `detail` sentence, and its `data` holds the
raw figures (`maxBody`, `maxHold`, `workers`, `streamOk`).
