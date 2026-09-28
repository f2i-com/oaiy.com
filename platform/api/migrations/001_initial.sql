-- oaiy-web :: api :: initial schema (v1)
--
-- Three tables, all on InnoDB so we get FKs + row-level locking on the
-- runs queue. utf8mb4 for emoji-clean flow titles.

SET NAMES utf8mb4;
SET time_zone = '+00:00';

CREATE TABLE IF NOT EXISTS flows (
  id           BIGINT UNSIGNED   AUTO_INCREMENT PRIMARY KEY,
  -- 22-char crockford base32 (110 bits = 22 × 5) — long enough that brute-force
  -- enumeration is impractical without a side channel.
  -- Credential columns: binary, case-SENSITIVE collation (ascii_bin) so MySQL
  -- matches them EXACTLY — like SQLite's default BINARY — instead of the table's
  -- accent/case-folding utf8mb4_unicode_ci, which would widen the set of inputs that
  -- equal a stored capability token. Values are uppercase ASCII Crockford/hex.
  hash_view    CHAR(22) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  hash_edit    CHAR(22) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  -- Random 32-byte token returned in the create() RESPONSE (not a cookie). Lets the
  -- original creator delete: anyone holding the edit hash can edit, but only the
  -- owner_token holder can permanently destroy. (Existing MySQL deployments: ALTER
  -- TABLE flows MODIFY each column to ascii_bin — generated values are ASCII, so no
  -- row changes meaning.)
  owner_token  CHAR(32) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  title        VARCHAR(200)      NOT NULL DEFAULT '',
  flow_json    JSON              NOT NULL,
  created_at   TIMESTAMP         NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at   TIMESTAMP         NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  -- last_seen tracks when a browser dispatcher last hit /heartbeat — used
  -- by /flows/{hash}/status so external callers can tell whether the
  -- user is online to execute runs.
  last_seen    TIMESTAMP         NULL DEFAULT NULL,
  UNIQUE KEY ux_flows_hash_view (hash_view),
  UNIQUE KEY ux_flows_hash_edit (hash_edit)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS runs (
  id           BIGINT UNSIGNED   AUTO_INCREMENT PRIMARY KEY,
  flow_id      BIGINT UNSIGNED   NOT NULL,
  -- queued → running → done|error|cancelled.
  -- The browser-side dispatcher claims a row by UPDATE … WHERE
  -- status='queued' LIMIT 1, atomically flipping to 'running'.
  status       ENUM('queued', 'running', 'done', 'error', 'cancelled') NOT NULL DEFAULT 'queued',
  inputs       JSON              NOT NULL,
  result       JSON              NULL DEFAULT NULL,
  error        TEXT              NULL DEFAULT NULL,
  -- A short note from the caller (e.g. ChatGPT) describing why this run
  -- was queued — surfaced in the per-run approval toast.
  reason       VARCHAR(500)      NOT NULL DEFAULT '',
  created_at   TIMESTAMP         NOT NULL DEFAULT CURRENT_TIMESTAMP,
  started_at   TIMESTAMP         NULL DEFAULT NULL,
  finished_at  TIMESTAMP         NULL DEFAULT NULL,
  KEY ix_runs_flow_status (flow_id, status, created_at),
  CONSTRAINT fk_runs_flow FOREIGN KEY (flow_id) REFERENCES flows(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
