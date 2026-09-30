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

    /** True when $inner is $outer or inside it (both resolved; symlinks followed). */
    public static function isInside(string $inner, string $outer): bool
    {
        $a = @realpath($inner);
        $b = @realpath($outer);
        if ($a === false || $b === false) {
            return false;
        }
        $a = rtrim(str_replace('\\', '/', $a), '/') . '/';
        $b = rtrim(str_replace('\\', '/', $b), '/') . '/';
        if (stripos(PHP_OS, 'WIN') === 0) {
            $a = strtolower($a);
            $b = strtolower($b);
        }
        return strpos($a, $b) === 0;
    }
}
