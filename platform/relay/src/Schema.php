<?php
declare(strict_types=1);

namespace Oaiy\Relay;

defined('OAIY_RELAY') or exit;

/**
 * The database schema (Appendix B) for SQLite and for MySQL or MariaDB. All times are Unix seconds (rl.w is
 * milliseconds); every numeric column is an INTEGER or BIGINT, never text; every hash is hex text.
 *
 * Differences from Appendix B, each on purpose: hashes are lower-case hex text (CHAR(64)) rather than BLOB so that
 * no driver has a binary-parameter quirk; devices has presence_changed_at; enroll_keys has created_at; and the idempotency key of an item is
 * (mailbox, lane, sender, id), so that one sender's ids cannot collide with another's (version 2; Appendix B keys it by (mailbox, lane, id)).
 * A difference on purpose too, and a recorded one: Interpretation 24 of the protocol package.
 * Forward-only migrations: the version lives in meta.schema_version, an older database is brought up to date by Db::migrate() on the
 * first request that meets it, and the code refuses to run against a newer one.
 */
final class Schema
{
    public const VERSION = 2;

    /** @return list<string> */
    public static function ddl(string $driver): array
    {
        return $driver === 'mysql' ? self::mysql() : self::sqlite();
    }

    /**
     * The statements that take a database from version $from to $from + 1.
     * @return list<string>
     */
    public static function migration(string $driver, int $from): array
    {
        if ($from === 1) { // the idempotency key gains the sender
            return $driver === 'mysql'
                ? ['ALTER TABLE items DROP INDEX items_dedupe, ADD UNIQUE KEY items_dedupe (mailbox, lane, sender, id)']
                : ['DROP INDEX IF EXISTS items_dedupe', 'CREATE UNIQUE INDEX IF NOT EXISTS items_dedupe ON items(mailbox, lane, sender, id)'];
        }
        return [];
    }

    /** @return list<string> */
    private static function sqlite(): array
    {
        return [
            'CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v INTEGER, s TEXT)',
            'CREATE TABLE IF NOT EXISTS devices (
                id TEXT PRIMARY KEY, role TEXT NOT NULL, name TEXT NOT NULL, ver TEXT NOT NULL, caps TEXT NOT NULL,
                ed25519 TEXT, x25519 TEXT, thumbprint TEXT, owner_desktop TEXT, app_id TEXT, peer_thumbprint TEXT,
                grants TEXT NOT NULL, flags TEXT NOT NULL, origins TEXT,
                created_at INTEGER NOT NULL, keys_changed_at INTEGER, last_poll_at INTEGER, last_seen_at INTEGER,
                revoked_at INTEGER, presence_changed_at INTEGER, push_kind TEXT, push_token TEXT)',
            'CREATE INDEX IF NOT EXISTS devices_role ON devices(role, revoked_at)',
            'CREATE INDEX IF NOT EXISTS devices_owner ON devices(owner_desktop, app_id, revoked_at)',
            'CREATE TABLE IF NOT EXISTS tokens (
                id TEXT PRIMARY KEY, device_id TEXT NOT NULL REFERENCES devices(id) ON DELETE CASCADE,
                secret_hash TEXT NOT NULL, created_at INTEGER NOT NULL, not_after INTEGER, revoked_at INTEGER,
                last_used_at INTEGER, grace_until INTEGER)',
            'CREATE INDEX IF NOT EXISTS tokens_device ON tokens(device_id)',
            'CREATE TABLE IF NOT EXISTS tokid_fail (
                id TEXT NOT NULL, addr TEXT NOT NULL, fails INTEGER NOT NULL, first_at INTEGER NOT NULL,
                locked_until INTEGER, PRIMARY KEY (id, addr))',
            'CREATE TABLE IF NOT EXISTS enroll_keys (
                kid TEXT PRIMARY KEY, role TEXT NOT NULL, pub TEXT NOT NULL, name TEXT, exp INTEGER NOT NULL,
                used_at INTEGER, fails INTEGER NOT NULL DEFAULT 0, created_at INTEGER NOT NULL)',
            'CREATE TABLE IF NOT EXISTS mailboxes (
                id TEXT PRIMARY KEY, next_seq INTEGER NOT NULL, live_items INTEGER NOT NULL, live_bytes INTEGER NOT NULL,
                bulk_items INTEGER NOT NULL, bulk_bytes INTEGER NOT NULL, created_at INTEGER NOT NULL)',
            'CREATE TABLE IF NOT EXISTS items (
                mailbox TEXT NOT NULL, seq INTEGER NOT NULL, lane TEXT NOT NULL, id TEXT NOT NULL, sender TEXT NOT NULL,
                re TEXT, rp TEXT, hdr TEXT NOT NULL, body TEXT, body_hash TEXT NOT NULL, size INTEGER NOT NULL,
                subject_id TEXT, grants TEXT, state INTEGER NOT NULL DEFAULT 0,
                at INTEGER NOT NULL, exp INTEGER NOT NULL, delivered_at INTEGER, acked_at INTEGER,
                PRIMARY KEY (mailbox, seq))',
            'CREATE UNIQUE INDEX IF NOT EXISTS items_dedupe ON items(mailbox, lane, sender, id)',
            'CREATE INDEX IF NOT EXISTS items_re ON items(mailbox, re) WHERE re IS NOT NULL',
            'CREATE INDEX IF NOT EXISTS items_exp ON items(exp)',
            'CREATE INDEX IF NOT EXISTS items_sender ON items(sender, lane, id)',
            'CREATE TABLE IF NOT EXISTS slots (
                dev TEXT NOT NULL, name TEXT NOT NULL, body TEXT NOT NULL, ct TEXT NOT NULL, etag TEXT NOT NULL,
                readers TEXT NOT NULL, at INTEGER NOT NULL, exp INTEGER NOT NULL, PRIMARY KEY (dev, name))',
            'CREATE TABLE IF NOT EXISTS replyboxes (
                rid TEXT PRIMARY KEY, dev TEXT NOT NULL, provider TEXT NOT NULL, sub TEXT NOT NULL,
                secret_hash TEXT NOT NULL, jti TEXT NOT NULL, created_at INTEGER NOT NULL, exp INTEGER NOT NULL,
                done INTEGER NOT NULL DEFAULT 0, ai_n INTEGER NOT NULL DEFAULT 0, ai_in_n INTEGER NOT NULL DEFAULT 0,
                posted_bytes INTEGER NOT NULL DEFAULT 0)',
            'CREATE TABLE IF NOT EXISTS tickets_used (jti TEXT PRIMARY KEY, provider TEXT NOT NULL, exp INTEGER NOT NULL)',
            'CREATE TABLE IF NOT EXISTS pairings (
                pid TEXT PRIMARY KEY, desktop_dev TEXT NOT NULL, app_id TEXT NOT NULL, offer TEXT NOT NULL,
                mac TEXT NOT NULL, desktop_thumb TEXT NOT NULL, state TEXT NOT NULL, response TEXT,
                rejects INTEGER NOT NULL DEFAULT 0, responses INTEGER NOT NULL DEFAULT 0, gets INTEGER NOT NULL DEFAULT 0,
                phone_dev TEXT, sealed_token TEXT, receipt TEXT, created_at INTEGER NOT NULL, exp INTEGER NOT NULL,
                read_at INTEGER)',
            'CREATE TABLE IF NOT EXISTS roster (
                desktop_dev TEXT NOT NULL, app_id TEXT NOT NULL, revision INTEGER NOT NULL, hash TEXT NOT NULL,
                thumbprints TEXT NOT NULL, updated_at INTEGER NOT NULL, PRIMARY KEY (desktop_dev, app_id))',
            'CREATE TABLE IF NOT EXISTS push_jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT, device_id TEXT NOT NULL, payload TEXT NOT NULL,
                expires_at INTEGER NOT NULL, attempts INTEGER NOT NULL DEFAULT 0, next_at INTEGER NOT NULL)',
            'CREATE TABLE IF NOT EXISTS rl (k TEXT PRIMARY KEY, w INTEGER NOT NULL, n INTEGER NOT NULL)',
        ];
    }

    /** @return list<string> */
    private static function mysql(): array
    {
        $ascii = 'CHARACTER SET ascii COLLATE ascii_bin';
        $utf = 'CHARACTER SET utf8mb4 COLLATE utf8mb4_bin';
        $eng = 'ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin';
        return [
            "CREATE TABLE IF NOT EXISTS meta (k VARCHAR(64) $ascii NOT NULL PRIMARY KEY, v BIGINT NULL, s VARCHAR(255) $ascii NULL) $eng",
            "CREATE TABLE IF NOT EXISTS devices (
                id VARCHAR(40) $ascii NOT NULL PRIMARY KEY, role VARCHAR(16) $ascii NOT NULL,
                name VARCHAR(255) $utf NOT NULL, ver VARCHAR(64) $utf NOT NULL, caps TEXT $utf NOT NULL,
                ed25519 VARCHAR(64) $ascii NULL, x25519 VARCHAR(64) $ascii NULL, thumbprint VARCHAR(64) $ascii NULL,
                owner_desktop VARCHAR(40) $ascii NULL, app_id VARCHAR(64) $ascii NULL, peer_thumbprint VARCHAR(64) $ascii NULL,
                grants TEXT $ascii NOT NULL, flags TEXT $ascii NOT NULL, origins TEXT $ascii NULL,
                created_at BIGINT NOT NULL, keys_changed_at BIGINT NULL, last_poll_at BIGINT NULL, last_seen_at BIGINT NULL,
                revoked_at BIGINT NULL, presence_changed_at BIGINT NULL, push_kind VARCHAR(16) $ascii NULL, push_token TEXT $ascii NULL,
                KEY devices_role (role, revoked_at), KEY devices_owner (owner_desktop, app_id, revoked_at)) $eng",
            "CREATE TABLE IF NOT EXISTS tokens (
                id VARCHAR(16) $ascii NOT NULL PRIMARY KEY, device_id VARCHAR(40) $ascii NOT NULL,
                secret_hash CHAR(64) $ascii NOT NULL, created_at BIGINT NOT NULL, not_after BIGINT NULL, revoked_at BIGINT NULL,
                last_used_at BIGINT NULL, grace_until BIGINT NULL, KEY tokens_device (device_id),
                CONSTRAINT tokens_device_fk FOREIGN KEY (device_id) REFERENCES devices(id) ON DELETE CASCADE) $eng",
            "CREATE TABLE IF NOT EXISTS tokid_fail (
                id VARCHAR(16) $ascii NOT NULL, addr VARCHAR(48) $ascii NOT NULL, fails INT NOT NULL, first_at BIGINT NOT NULL,
                locked_until BIGINT NULL, PRIMARY KEY (id, addr)) $eng",
            "CREATE TABLE IF NOT EXISTS enroll_keys (
                kid VARCHAR(16) $ascii NOT NULL PRIMARY KEY, role VARCHAR(16) $ascii NOT NULL, pub CHAR(64) $ascii NOT NULL,
                name VARCHAR(255) $utf NULL, exp BIGINT NOT NULL, used_at BIGINT NULL, fails INT NOT NULL DEFAULT 0,
                created_at BIGINT NOT NULL) $eng",
            "CREATE TABLE IF NOT EXISTS mailboxes (
                id VARCHAR(160) $ascii NOT NULL PRIMARY KEY, next_seq BIGINT UNSIGNED NOT NULL, live_items INT NOT NULL,
                live_bytes BIGINT NOT NULL, bulk_items INT NOT NULL, bulk_bytes BIGINT NOT NULL, created_at BIGINT NOT NULL) $eng",
            "CREATE TABLE IF NOT EXISTS items (
                mailbox VARCHAR(160) $ascii NOT NULL, seq BIGINT UNSIGNED NOT NULL, lane VARCHAR(16) $ascii NOT NULL,
                id VARCHAR(128) $ascii NOT NULL, sender VARCHAR(160) $ascii NOT NULL, re VARCHAR(128) $ascii NULL,
                rp VARCHAR(40) $ascii NULL, hdr TEXT $utf NOT NULL, body MEDIUMTEXT $utf NULL, body_hash CHAR(64) $ascii NOT NULL,
                size INT NOT NULL, subject_id VARCHAR(64) $ascii NULL, grants TEXT $ascii NULL, state TINYINT NOT NULL DEFAULT 0,
                at BIGINT NOT NULL, exp BIGINT NOT NULL, delivered_at BIGINT NULL, acked_at BIGINT NULL,
                PRIMARY KEY (mailbox, seq), UNIQUE KEY items_dedupe (mailbox, lane, sender, id), KEY items_re (mailbox, re),
                KEY items_exp (exp), KEY items_sender (sender, lane, id)) $eng",
            "CREATE TABLE IF NOT EXISTS slots (
                dev VARCHAR(40) $ascii NOT NULL, name VARCHAR(64) $ascii NOT NULL, body MEDIUMTEXT $utf NOT NULL,
                ct VARCHAR(16) $ascii NOT NULL, etag VARCHAR(40) $ascii NOT NULL, readers TEXT $ascii NOT NULL,
                at BIGINT NOT NULL, exp BIGINT NOT NULL, PRIMARY KEY (dev, name)) $eng",
            "CREATE TABLE IF NOT EXISTS replyboxes (
                rid VARCHAR(32) $ascii NOT NULL PRIMARY KEY, dev VARCHAR(40) $ascii NOT NULL, provider VARCHAR(40) $ascii NOT NULL,
                sub VARCHAR(64) $utf NOT NULL, secret_hash CHAR(64) $ascii NOT NULL, jti VARCHAR(64) $ascii NOT NULL,
                created_at BIGINT NOT NULL, exp BIGINT NOT NULL, done TINYINT NOT NULL DEFAULT 0, ai_n INT NOT NULL DEFAULT 0,
                ai_in_n INT NOT NULL DEFAULT 0, posted_bytes BIGINT NOT NULL DEFAULT 0) $eng",
            "CREATE TABLE IF NOT EXISTS tickets_used (jti VARCHAR(64) $ascii NOT NULL PRIMARY KEY, provider VARCHAR(40) $ascii NOT NULL, exp BIGINT NOT NULL) $eng",
            "CREATE TABLE IF NOT EXISTS pairings (
                pid VARCHAR(32) $ascii NOT NULL PRIMARY KEY, desktop_dev VARCHAR(40) $ascii NOT NULL, app_id VARCHAR(64) $ascii NOT NULL,
                offer MEDIUMTEXT $utf NOT NULL, mac VARCHAR(64) $ascii NOT NULL, desktop_thumb VARCHAR(64) $ascii NOT NULL,
                state VARCHAR(16) $ascii NOT NULL, response MEDIUMTEXT $utf NULL, rejects INT NOT NULL DEFAULT 0,
                responses INT NOT NULL DEFAULT 0, gets INT NOT NULL DEFAULT 0, phone_dev VARCHAR(40) $ascii NULL,
                sealed_token TEXT $ascii NULL, receipt TEXT $utf NULL, created_at BIGINT NOT NULL, exp BIGINT NOT NULL,
                read_at BIGINT NULL) $eng",
            "CREATE TABLE IF NOT EXISTS roster (
                desktop_dev VARCHAR(40) $ascii NOT NULL, app_id VARCHAR(64) $ascii NOT NULL, revision BIGINT UNSIGNED NOT NULL,
                hash VARCHAR(64) $ascii NOT NULL, thumbprints TEXT $ascii NOT NULL, updated_at BIGINT NOT NULL,
                PRIMARY KEY (desktop_dev, app_id)) $eng",
            "CREATE TABLE IF NOT EXISTS push_jobs (
                id BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY, device_id VARCHAR(40) $ascii NOT NULL, payload TEXT $utf NOT NULL,
                expires_at BIGINT NOT NULL, attempts INT NOT NULL DEFAULT 0, next_at BIGINT NOT NULL) $eng",
            "CREATE TABLE IF NOT EXISTS rl (k VARCHAR(128) $ascii NOT NULL PRIMARY KEY, w BIGINT NOT NULL, n BIGINT NOT NULL) $eng",
        ];
    }
}
