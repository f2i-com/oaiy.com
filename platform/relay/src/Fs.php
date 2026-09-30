<?php
declare(strict_types=1);

namespace Oaiy\Relay;

defined('OAIY_RELAY') or exit;

/** Filesystem facts the relay needs: the type of the mount a directory is on, and whether write-ahead logging is safe there. */
final class Fs
{
    /** Filesystems on which SQLite's write-ahead log is unsafe (it needs shared memory that a network filesystem cannot give). */
    private const NETWORK = ['nfs', 'nfs4', 'cifs', 'smb', 'smb2', 'smb3', 'smbfs', 'ceph', 'glusterfs', '9p', 'afs', 'lustre', 'gpfs'];

    /** The filesystem type of the mount holding $dir from /proc/self/mountinfo; null when it cannot be read. */
    public static function type(string $dir, ?string $mountinfo = null): ?string
    {
        $real = @realpath($dir);
        $info = $mountinfo ?? @file_get_contents('/proc/self/mountinfo');
        if ($real === false || !is_string($info)) {
            return null;
        }
        $real = str_replace('\\', '/', $real);
        $best = null;
        $bestLen = -1;
        foreach (explode("\n", $info) as $line) {
            $parts = explode(' - ', $line, 2);
            if (count($parts) !== 2) {
                continue;
            }
            $left = explode(' ', $parts[0]);
            $right = explode(' ', $parts[1]);
            if (count($left) < 5 || $right[0] === '') {
                continue;
            }
            $mount = str_replace(['\\040', '\\011', '\\012', '\\134'], [' ', "\t", "\n", '\\'], $left[4]);
            $prefix = rtrim($mount, '/');
            if (($real === $mount || strpos($real . '/', $prefix . '/') === 0) && strlen($mount) > $bestLen) {
                $best = $right[0];
                $bestLen = strlen($mount);
            }
        }
        return $best;
    }

    public static function isNetwork(?string $type): bool
    {
        return $type !== null && (in_array($type, self::NETWORK, true) || strncmp($type, 'fuse.', 5) === 0 || $type === 'fuse');
    }

    /**
     * The SQLite journal mode to use for a database in $dir: 'wal' unless the mount is a network filesystem, or its type
     * could not be read on a system that has /proc (then the safe 'truncate').
     */
    public static function journalFor(string $dir, ?string $mountinfo = null): string
    {
        $type = self::type($dir, $mountinfo);
        if (self::isNetwork($type)) {
            return 'truncate';
        }
        if ($type === null && ($mountinfo !== null || is_readable('/proc/self/mountinfo'))) {
            return 'truncate';
        }
        return 'wal';
    }

    /**
     * Make sure a file exists and only its owner can read or write it. The file is created under a umask of 0177, so it is
     * never, even for an instant, readable by others whatever the host's umask is, and an existing one is chmod-ed. Used for
     * files that hold data rather than secrets one by one (the SQLite database: SQLite gives its -wal, -shm and -journal
     * files the mode of the database file, so creating that file right is what keeps the others right).
     */
    public static function createPrivate(string $path): void
    {
        if (!is_file($path)) {
            $old = umask(0177);
            try {
                @touch($path);
            } finally {
                umask($old);
            }
        }
        @chmod($path, 0600);
    }

    /**
     * A path with symlinks resolved, for a path that may not exist yet: the nearest existing ancestor is resolved and the
     * missing rest appended (a fresh install has no data/ folder yet, and that is exactly when it must be checked).
     */
    private static function resolveLoose(string $path): ?string
    {
        $cur = str_replace('\\', '/', $path);
        $tail = [];
        for ($i = 0; $i < 64; $i++) {
            $real = @realpath($cur);
            if ($real !== false) {
                $real = rtrim(str_replace('\\', '/', $real), '/');
                return $tail ? $real . '/' . implode('/', array_reverse($tail)) : $real;
            }
            $parent = dirname($cur);
            $name = basename($cur);
            if ($parent === $cur || $name === '..' || $name === '.') {
                return null;
            }
            $tail[] = $name;
            $cur = $parent;
        }
        return null;
    }

    /** True when $inner is $outer or inside it (both resolved; symlinks followed; $inner may not exist yet). */
    public static function isInside(string $inner, string $outer): bool
    {
        $a = self::resolveLoose($inner);
        $b = self::resolveLoose($outer);
        if ($a === null || $b === null) {
            return false;
        }
        $a = rtrim($a, '/') . '/';
        $b = rtrim($b, '/') . '/';
        if (stripos(PHP_OS, 'WIN') === 0) {
            $a = strtolower($a);
            $b = strtolower($b);
        }
        return strpos($a, $b) === 0;
    }
}
