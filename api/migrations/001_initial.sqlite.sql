-- oaiy-web :: api :: initial schema (v1) — SQLite variant
--
-- Mirrors 001_initial.sql for MySQL with SQLite-specific idiom:
--   * INTEGER PRIMARY KEY AUTOINCREMENT instead of BIGINT AUTO_INCREMENT
--   * TEXT for everything that's VARCHAR/JSON/ENUM in MySQL — SQLite is
--     dynamically typed; we keep the JSON in TEXT and parse in PHP, and
--     the status ENUM becomes a TEXT + CHECK constraint.
--   * updated_at maintained via trigger because SQLite has no ON UPDATE.
--   * Foreign-key cascades work IF you've turned FKs on at connect time
--     (Db.php does `PRAGMA foreign_keys = ON`).

CREATE TABLE IF NOT EXISTS flows (
  id           INTEGER PRIMARY KEY AUTOINCREMENT,
  hash_view    TEXT    NOT NULL UNIQUE,
  hash_edit    TEXT    NOT NULL UNIQUE,
  owner_token  TEXT    NOT NULL,
  title        TEXT    NOT NULL DEFAULT '',
  flow_json    TEXT    NOT NULL,
  created_at   TEXT    NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at   TEXT    NOT NULL DEFAULT CURRENT_TIMESTAMP,
  last_seen    TEXT
);

-- ON UPDATE CURRENT_TIMESTAMP equivalent.
CREATE TRIGGER IF NOT EXISTS flows_updated_at
AFTER UPDATE ON flows FOR EACH ROW
WHEN NEW.updated_at = OLD.updated_at
BEGIN
  UPDATE flows SET updated_at = CURRENT_TIMESTAMP WHERE id = OLD.id;
END;

CREATE TABLE IF NOT EXISTS runs (
  id           INTEGER PRIMARY KEY AUTOINCREMENT,
  flow_id      INTEGER NOT NULL REFERENCES flows(id) ON DELETE CASCADE,
  status       TEXT    NOT NULL DEFAULT 'queued'
                       CHECK (status IN ('queued', 'running', 'done', 'error', 'cancelled')),
  inputs       TEXT    NOT NULL,
  result       TEXT,
  error        TEXT,
  reason       TEXT    NOT NULL DEFAULT '',
  created_at   TEXT    NOT NULL DEFAULT CURRENT_TIMESTAMP,
  started_at   TEXT,
  finished_at  TEXT
);

CREATE INDEX IF NOT EXISTS ix_runs_flow_status
  ON runs(flow_id, status, created_at);
