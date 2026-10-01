<?php
declare(strict_types=1);

namespace OaiyTest;

/**
 * What a test run leaves running. A test that starts a server (php -S, a pool front, a database server, Apache) stops it when it
 * ends; a server that is not stopped outlives the run, holds its port and its files, and is found by the next run as a leak that
 * nobody can account for. This lists what a run owns: the processes below the runner's own in the process tree, and any process, whoever
 * its parent is, whose command line names this run's test root (a server started with a configuration of its own under it, or with the marker
 * Server::phpFlags() puts on every php process the tests start, so a daemon that detached from its parent and was adopted by init is
 * still seen), so that the runner can name the test
 * that left one behind (after every test on a POSIX host, where listing costs a few milliseconds; on Windows, where it costs a second,
 * at the end of the run, or after every test with OAIY_TEST_PROCS=each) and so that the end of the run fails when anything is left.
 *
 * It only ever reports and kills processes that are descendants of the runner or carry this run's own test root (a random name made for this run)
 * in their command line, by the pid it found in the table: nothing else on the machine is looked at beyond the table's own rows, and a process
 * that is neither is never touched. What it cannot see is a process that is neither a descendant nor marked (a daemon started with a
 * command line that does not name the root): every launcher in the harness names it.
 */
final class Procs
{
    /** @var array<int,true> pids meant to outlive the test that started them, and be stopped at the end of the run (the shared database server) */
    private static array $kept = [];
    /** The test root of the run, remembered at its start: Tmp::cleanup() forgets its own, and the end of the run still has to find what names it. */
    private static string $watched = '';
    /** True once a listing could not be made: the check was skipped, and the runner says so rather than report a clean run. */
    public static bool $blind = false;

    /** Remember this run's test root for live(): the runner says it once, after Tmp::init(). */
    public static function watch(string $root): void
    {
        self::$watched = $root;
    }

    /** Say that this process is meant to live until the end of the run (its owner stops it with Tmp::onCleanup). */
    public static function keep(int $pid): void
    {
        if ($pid > 0) {
            self::$kept[$pid] = true;
        }
    }

    /** The end of the run: what was meant to live until now has had its chance to stop, and is judged like the rest. */
    public static function releaseKept(): void
    {
        self::$kept = [];
    }

    /** Whether a check after every test is cheap here: it is where `ps` is, and it is not where listing takes a second. */
    public static function eachTest(): bool
    {
        $env = getenv('OAIY_TEST_PROCS');
        if ($env === 'each') {
            return true;
        }
        if ($env === 'end') {
            return false;
        }
        return stripos(PHP_OS, 'WIN') !== 0;
    }

    /**
     * Every live process of the machine as rows: pid => [ppid, command]. A zombie (exited, not yet reaped by its parent) is not live.
     * The helper that makes the listing is left out. Returns null where the table cannot be read (the check is then skipped, not passed).
     *
     * @return array<int,array{0:int,1:string}>|null
     */
    private static function table(): ?array
    {
        $win = stripos(PHP_OS, 'WIN') === 0;
        if ($win) {
            $ps = 'Get-CimInstance Win32_Process | ForEach-Object { "{0}|{1}|{2}" -f $_.ProcessId, $_.ParentProcessId, (($_.CommandLine -replace "[\r\n]+", " ")) }';
            $cmd = ['powershell.exe', '-NoProfile', '-NonInteractive', '-Command', $ps];
        } else {
            $cmd = ['ps', '-A', '-ww', '-o', 'pid=,ppid=,stat=,args='];
        }
        $p = @proc_open($cmd, [0 => ['pipe', 'r'], 1 => ['pipe', 'w'], 2 => ['pipe', 'w']], $pipes);
        if (!is_resource($p)) {
            return null;
        }
        fclose($pipes[0]);
        $helper = (int)proc_get_status($p)['pid'];
        $out = (string)stream_get_contents($pipes[1]);
        fclose($pipes[1]);
        fclose($pipes[2]);
        proc_close($p);
        $rows = [];
        foreach (preg_split('/\R/', $out) ?: [] as $line) {
            if ($win) {
                if (!preg_match('/^(\d+)\|(\d+)\|(.*)$/', trim($line), $m)) {
                    continue;
                }
                $rows[(int)$m[1]] = [(int)$m[2], $m[3]];
            } else {
                if (!preg_match('/^\s*(\d+)\s+(\d+)\s+(\S+)\s+(.*)$/', $line, $m) || $m[3][0] === 'Z') {
                    continue;
                }
                $rows[(int)$m[1]] = [(int)$m[2], $m[4]];
            }
        }
        unset($rows[$helper]);
        return $rows === [] ? null : $rows;
    }

    /**
     * The live processes below this one in the process tree (children, grandchildren and so on), without those that are meant to
     * outlive a test (keep()) and what they started.
     *
     * @return array<int,string> pid => command
     */
    public static function below(): array
    {
        $rows = self::table();
        if ($rows === null) {
            self::$blind = true;
            return [];
        }
        return self::descendants($rows);
    }

    /**
     * What this run owns now: the processes below the runner's own, and every other process whose command line names this run's test root
     * (Tmp::root(): an Apache whose configuration is under it, a php that was started with the marker, a database server whose data
     * directory is), whatever its parent is. Without those that are meant to outlive a test (keep()).
     *
     * @return array<int,string> pid => command
     */
    public static function live(): array
    {
        $rows = self::table();
        if ($rows === null) {
            self::$blind = true;
            return [];
        }
        $found = self::descendants($rows);
        $root = self::$watched !== '' ? self::$watched : Tmp::root();
        if ($root !== '') {
            $needle = self::norm($root);
            $me = getmypid();
            foreach ($rows as $pid => [, $cmd]) {
                if ($pid !== $me && !isset(self::$kept[$pid]) && !isset($found[$pid]) && strpos(self::norm($cmd), $needle) !== false) {
                    $found[$pid] = $cmd;
                }
            }
        }
        return $found;
    }

    /** Forward slashes and lower case, so that a command line from the process table and a path from PHP compare. */
    private static function norm(string $s): string
    {
        return strtolower(str_replace('\\', '/', $s));
    }

    /**
     * @param array<int,array{0:int,1:string}> $rows
     * @return array<int,string> pid => command
     */
    private static function descendants(array $rows): array
    {
        $kids = [];
        foreach ($rows as $pid => [$ppid]) {
            $kids[$ppid][] = $pid;
        }
        $found = [];
        $queue = [getmypid()];
        while ($queue) {
            $at = array_shift($queue);
            foreach ($kids[$at] ?? [] as $child) {
                if (isset(self::$kept[$child]) || isset($found[$child])) {
                    continue;
                }
                $found[$child] = $rows[$child][1];
                $queue[] = $child;
            }
        }
        return $found;
    }

    /**
     * Wait up to $seconds for the processes this run owns (live()) that are not in $before to end by themselves (a stop takes a moment to
     * show), and return those that did not.
     *
     * @param array<int,string> $before
     * @return array<int,string> pid => command
     */
    public static function leftSince(array $before, float $seconds = 3.0): array
    {
        $deadline = microtime(true) + $seconds;
        do {
            $left = array_diff_key(self::live(), $before);
            if ($left === []) {
                return [];
            }
            usleep(150000);
        } while (microtime(true) < $deadline);
        return $left;
    }

    /** Whether a process is alive (a zombie is not). */
    public static function alive(int $pid): bool
    {
        if ($pid <= 0) {
            return false;
        }
        if (stripos(PHP_OS, 'WIN') === 0) {
            $out = [];
            @exec('tasklist /FI "PID eq ' . $pid . '" /FO CSV /NH 2>&1', $out);
            return strpos(implode("\n", $out), '"' . $pid . '"') !== false;
        }
        $stat = @file_get_contents('/proc/' . $pid . '/stat');
        if ($stat !== false) {
            $i = strrpos($stat, ')');
            return $i === false || ($stat[$i + 2] ?? '') !== 'Z';
        }
        if (is_dir('/proc/self')) {
            return false; // Linux, and no such process
        }
        $out = [];
        @exec('ps -p ' . $pid . ' -o stat= 2>/dev/null', $out);
        return $out !== [] && trim($out[0]) !== '' && trim($out[0])[0] !== 'Z';
    }

    /** End a process, and what it started, by its pid: politely, or ($force) at once. */
    public static function end(int $pid, bool $force = false): void
    {
        if ($pid <= 0) {
            return;
        }
        if (stripos(PHP_OS, 'WIN') === 0) {
            @exec('taskkill /PID ' . $pid . ' /T /F 2>&1');
            return;
        }
        @exec('kill -' . ($force ? 'KILL' : 'TERM') . ' ' . $pid . ' 2>&1');
    }

    /** Wait up to $seconds for a process to be gone. */
    public static function gone(int $pid, float $seconds): bool
    {
        $deadline = microtime(true) + $seconds;
        do {
            if (!self::alive($pid)) {
                return true;
            }
            usleep(50000);
        } while (microtime(true) < $deadline);
        return !self::alive($pid);
    }
}
