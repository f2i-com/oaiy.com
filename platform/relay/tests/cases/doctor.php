<?php
declare(strict_types=1);

use Oaiy\Relay\Doctor;
use Oaiy\Relay\Fs;
use Oaiy\Relay\HttpProbe;
use Oaiy\Relay\Installer;
use OaiyTest\Server;
use OaiyTest\Tmp;

// Helpers (inst_scratch, inst_cli, inst_installed, inst_secrets, inst_no_secrets) live in tests/cases/installer.php.

/** @param list<array{name:string,level:string,message:string}> $rows */
function doc_level(array $rows, string $name): string
{
    foreach ($rows as $r) {
        if ($r['name'] === $name) {
            return $r['level'];
        }
    }
    fail('no check named ' . $name . ' in: ' . implode(', ', array_map(static fn(array $r): string => $r['name'], $rows)));
}

/** @param list<array{name:string,level:string,message:string}> $rows */
function doc_msg(array $rows, string $name): string
{
    foreach ($rows as $r) {
        if ($r['name'] === $name) {
            return $r['message'];
        }
    }
    fail('no check named ' . $name);
}

/** @param list<array{name:string,level:string,message:string}> $rows @return list<string> */
function doc_failures(array $rows): array
{
    $f = [];
    foreach ($rows as $r) {
        if ($r['level'] === 'fail') {
            $f[] = $r['name'] . ': ' . $r['message'];
        }
    }
    return $f;
}

/** Provision a relay in a temp data folder, in process. */
function doc_data(): string
{
    $data = Tmp::dir('docdata') . '/data';
    Installer::provision($data, ['public_url' => 'https://relay.example.com', 'journal' => 'wal']);
    // Windows cannot delete a SQLite file while a connection object is still alive in a reference cycle.
    Tmp::after('gc_collect_cycles');
    return $data;
}

// ------------------------------------------------------------------------------------------------ synthetic facts

test('4.18.1 doctor: PHP older than 8.0 fails, 8.0 and 8.1 work but warn that they are end of life and name 8.2, 8.2 and later pass', function () {
    eq('fail', doc_level(Doctor::phpVersion('7.4.33'), 'php.version'));
    eq('fail', doc_level(Doctor::phpVersion('5.6.40'), 'php.version'));
    eq('fail', doc_level(Doctor::phpVersion('7.99.0'), 'php.version'));
    foreach (['8.0.0', '8.0.30', '8.1.0', '8.1.34'] as $v) {
        eq('warn', doc_level(Doctor::phpVersion($v), 'php.version'), $v);
        contains('end of life', doc_msg(Doctor::phpVersion($v), 'php.version'));
        contains('8.2', doc_msg(Doctor::phpVersion($v), 'php.version'));
        contains('PHP ' . substr($v, 0, 3), doc_msg(Doctor::phpVersion($v), 'php.version'));
    }
    foreach (['8.2.0', '8.2.30', '8.3.6', '8.4.15', '9.0.0'] as $v) {
        eq('ok', doc_level(Doctor::phpVersion($v), 'php.version'), $v);
        not_contains('end of life', doc_msg(Doctor::phpVersion($v), 'php.version'));
    }
    contains('8.0', doc_msg(Doctor::phpVersion('7.4.33'), 'php.version'));
    contains('8.2', doc_msg(Doctor::phpVersion('7.4.33'), 'php.version'));
});

test('4.18.1 doctor: the run on this PHP reports its version as ok on 8.2 or later and as a warning, never a failure, on 8.0 and 8.1', function () {
    $data = doc_data();
    $rows = Doctor::run(['dataDir' => $data, 'web' => false]);
    eq(version_compare(PHP_VERSION, '8.2.0', '>=') ? 'ok' : 'warn', doc_level($rows, 'php.version'));
    eq([], doc_failures($rows));
});

test('4.18.1 doctor: a missing sodium, json, hash or database driver fails; a missing openssl or curl only warns', function () {
    $all = ['sodium' => true, 'json' => true, 'hash' => true, 'pdo_sqlite' => true, 'pdo_mysql' => false, 'openssl' => true, 'curl' => true];
    $rows = Doctor::extensions($all);
    eq('ok', doc_level($rows, 'php.extensions'));
    eq('ok', doc_level($rows, 'php.extensions.optional'));
    foreach (['sodium', 'json', 'hash'] as $e) {
        $rows = Doctor::extensions(array_merge($all, [$e => false]));
        eq('fail', doc_level($rows, 'php.extensions'), $e);
        contains($e, doc_msg($rows, 'php.extensions'));
    }
    $rows = Doctor::extensions(array_merge($all, ['pdo_sqlite' => false]));
    eq('fail', doc_level($rows, 'php.extensions'), 'no driver at all');
    eq('ok', doc_level(Doctor::extensions(array_merge($all, ['pdo_sqlite' => false, 'pdo_mysql' => true])), 'php.extensions'), 'MySQL alone is enough');
    $rows = Doctor::extensions(array_merge($all, ['openssl' => false, 'curl' => false]));
    eq('ok', doc_level($rows, 'php.extensions'));
    eq('warn', doc_level($rows, 'php.extensions.optional'));
    contains('FCM', doc_msg($rows, 'php.extensions.optional'));
});

test('4.18.7 doctor: memory_limit below 64M fails, post_max_size below 2M warns, unlimited and unreadable are handled', function () {
    foreach (['32M' => 'fail', '63M' => 'fail', '65535K' => 'fail', '64M' => 'ok', '128M' => 'ok', '1G' => 'ok', '-1' => 'ok', '67108864' => 'ok'] as $v => $want) {
        eq($want, doc_level(Doctor::memoryLimit((string)$v), 'ini.memory_limit'), (string)$v);
    }
    eq('warn', doc_level(Doctor::memoryLimit(null), 'ini.memory_limit'));
    eq('warn', doc_level(Doctor::memoryLimit('lots'), 'ini.memory_limit'));
    foreach (['1M' => 'warn', '2047K' => 'warn', '2M' => 'ok', '8M' => 'ok', '0' => 'ok', '-1' => 'ok'] as $v => $want) {
        eq($want, doc_level(Doctor::postMax((string)$v), 'ini.post_max_size'), (string)$v);
    }
    // The label says which SAPI the value came from.
    eq('fail', doc_level(Doctor::memoryLimit('32M', 'web'), 'ini.memory_limit (web)'));
});

test('4.18.7 doctor: ini shorthand is parsed the way PHP reads it', function () {
    eq(1048576, Doctor::bytes('1M'));
    eq(1048576, Doctor::bytes('1m'));
    eq(2048, Doctor::bytes('2K'));
    eq(1073741824, Doctor::bytes('1G'));
    eq(100, Doctor::bytes('100'));
    eq(0, Doctor::bytes('0'));
    eq(-1, Doctor::bytes('-1'));
    eq(null, Doctor::bytes(null));
    eq(null, Doctor::bytes(''));
    eq(null, Doctor::bytes('1.5M'));
    eq(null, Doctor::bytes('M'));
});

test('4.18.7 doctor: disable_functions containing usleep fails, set_time_limit and the finish_request functions only warn', function () {
    $avail = ['usleep' => true, 'set_time_limit' => true, 'fastcgi_finish_request' => true, 'litespeed_finish_request' => false];
    $rows = Doctor::functions('', $avail);
    eq('ok', doc_level($rows, 'ini.disable_functions'));
    eq('ok', doc_level($rows, 'ini.finish_request'));
    $rows = Doctor::functions('exec, usleep, shell_exec', array_merge($avail, ['usleep' => false]));
    eq('fail', doc_level($rows, 'ini.disable_functions'));
    contains('usleep', doc_msg($rows, 'ini.disable_functions'));
    $rows = Doctor::functions('exec', array_merge($avail, ['usleep' => false]));
    eq('fail', doc_level($rows, 'ini.usleep'), 'usleep missing without being listed is still a failure');
    $rows = Doctor::functions('set_time_limit', array_merge($avail, ['set_time_limit' => false]));
    eq('ok', doc_level($rows, 'ini.disable_functions'));
    eq('warn', doc_level($rows, 'ini.set_time_limit'));
    $rows = Doctor::functions('', array_merge($avail, ['fastcgi_finish_request' => false, 'litespeed_finish_request' => false]));
    eq('warn', doc_level($rows, 'ini.finish_request'));
    contains('50 ms', doc_msg($rows, 'ini.finish_request'));
    eq('ok', doc_level(Doctor::functions('', array_merge($avail, ['fastcgi_finish_request' => false, 'litespeed_finish_request' => true])), 'ini.finish_request'));
});

test('4.18.7 doctor: output compression warns, a short max_execution_time without set_time_limit warns, output_buffering is only reported', function () {
    $rows = Doctor::execution('30', '4096', '0', true);
    eq('ok', doc_level($rows, 'ini.zlib.output_compression'));
    eq('ok', doc_level($rows, 'ini.output_buffering'));
    eq('ok', doc_level($rows, 'ini.max_execution_time'));
    foreach (['1', 'On', 'on', 'true', '4096'] as $z) {
        eq('warn', doc_level(Doctor::execution('30', '0', $z, true), 'ini.zlib.output_compression'), $z);
    }
    foreach (['', '0', 'Off', 'off'] as $z) {
        eq('ok', doc_level(Doctor::execution('30', '0', $z, true), 'ini.zlib.output_compression'), 'off: ' . $z);
    }
    eq('warn', doc_level(Doctor::execution('20', '0', '0', false), 'ini.max_execution_time'));
    eq('ok', doc_level(Doctor::execution('20', '0', '0', true), 'ini.max_execution_time'), 'set_time_limit can lift it');
    eq('ok', doc_level(Doctor::execution('0', '0', '0', false), 'ini.max_execution_time'), 'unlimited');
});

test('4.18.5 doctor: SQLite in WAL mode on a network filesystem fails, on a local one it passes, an unknown type is said', function () {
    // Facts from a synthetic /proc/self/mountinfo, through the same function the relay uses to read the real one.
    $dir = Tmp::dir('mnt');
    $real = str_replace('\\', '/', (string)realpath($dir));
    $mi = static fn(string $fs): string => "22 1 0:20 / / rw - ext4 /dev/sda1 rw\n36 22 0:32 / " . $real . " rw - " . $fs . " server:/export rw\n";
    foreach (['nfs4', 'nfs', 'cifs', 'smb3', 'fuse.sshfs', '9p', 'ceph', 'glusterfs'] as $fs) {
        $t = Fs::type($dir, $mi($fs));
        eq($fs, $t);
        $rows = Doctor::journal('sqlite', 'wal', 'wal', $t, true);
        eq('fail', doc_level($rows, 'db.journal'), $fs);
        contains($fs, doc_msg($rows, 'db.journal'));
        contains('truncate', doc_msg($rows, 'db.journal'));
        eq('ok', doc_level(Doctor::journal('sqlite', 'truncate', 'truncate', $t, true), 'db.journal'), "$fs with TRUNCATE");
    }
    foreach (['ext4', 'xfs', 'btrfs', 'overlay', 'tmpfs'] as $fs) {
        $t = Fs::type($dir, $mi($fs));
        eq('ok', doc_level(Doctor::journal('sqlite', 'wal', 'wal', $t, true), 'db.journal'), $fs);
    }
    // /proc readable but the mount not found: WAL is a guess. /proc unreadable (Windows): "unknown", not an alarm.
    eq('warn', doc_level(Doctor::journal('sqlite', 'wal', 'wal', null, true), 'db.journal'));
    eq('ok', doc_level(Doctor::journal('sqlite', 'wal', 'wal', null, false), 'db.journal'));
    contains('unknown', doc_msg(Doctor::journal('sqlite', 'wal', 'wal', null, false), 'db.journal'));
    // The database is in another mode than the config says.
    eq('warn', doc_level(Doctor::journal('sqlite', 'wal', 'delete', 'ext4', true), 'db.journal'));
    // MySQL has no journal mode to judge.
    eq('ok', doc_level(Doctor::journal('mysql', '', null, null, false), 'db.journal'));
});

test('4.18.8 doctor: a secret file readable by group or other fails, a wide data folder warns, Windows is not judged', function () {
    $items = static fn(int $secret, int $dir): array => [
        ['label' => 'data/', 'mode' => $dir, 'dir' => true, 'secret' => false],
        ['label' => 'data/secrets/relay.key', 'mode' => $secret, 'dir' => false, 'secret' => true],
        ['label' => 'data/secrets/gone', 'mode' => null, 'dir' => false, 'secret' => true],
    ];
    eq('ok', doc_level(Doctor::modes($items(0600, 0700), true), 'data.modes'));
    foreach ([0644, 0640, 0604, 0666, 0777] as $m) {
        $rows = Doctor::modes($items($m, 0700), true);
        eq('fail', doc_level($rows, 'data.modes'), sprintf('%04o', $m));
        contains('relay.key', doc_msg($rows, 'data.modes'));
    }
    foreach ([0755, 0750, 0775] as $m) {
        eq('warn', doc_level(Doctor::modes($items(0600, $m), true), 'data.modes'), sprintf('%04o', $m));
    }
    eq('fail', doc_level(Doctor::modes($items(0644, 0755), true), 'data.modes'), 'a failure outranks a warning');
    eq('ok', doc_level(Doctor::modes($items(0666, 0777), false), 'data.modes'), 'not judged where modes mean nothing');
});

test('4.18.8 doctor: the mode check also covers the SQLite database with its -wal, -shm and -journal files, the log and the backups, and names each that is too open', function () {
    $data = Tmp::dir('modes') . '/data';
    $files = ['config.json', 'first-key.txt', 'admin-token.txt', 'relay.sqlite', 'relay.sqlite-wal', 'relay.sqlite-shm', 'relay.sqlite-journal', 'logs/relay.log', 'logs/relay.log.1',
        'backups/relay-1.sqlite', 'secrets/relay.key'];
    foreach ($files as $f) {
        @mkdir(dirname($data . '/' . $f), 0700, true);
        file_put_contents($data . '/' . $f, 'x');
        @chmod($data . '/' . $f, 0644); // too open, on a system that has modes; Windows reports 0666, which is too open as well
    }
    $labels = array_column(Doctor::dataModeItems($data), 'label');
    foreach ($files as $f) {
        ok(in_array('data/' . $f, $labels, true), "data/$f is looked at");
    }
    $rows = Doctor::modes(Doctor::dataModeItems($data), true);
    eq('fail', doc_level($rows, 'data.modes'));
    foreach ($files as $f) {
        contains('data/' . $f . ' is ', doc_msg($rows, 'data.modes'));
    }
    // A file that is not there is not a finding; a stray file that is not one of the relay's is not looked at.
    unlink($data . '/relay.sqlite-shm');
    file_put_contents($data . '/notes.txt', 'x');
    $labels = array_column(Doctor::dataModeItems($data), 'label');
    ok(!in_array('data/relay.sqlite-shm', $labels, true) && !in_array('data/notes.txt', $labels, true));
});

test('4.18.8 doctor (POSIX): an install under the loosest umask still leaves the database, its -wal and -shm, the log and every secret owner-only, and the doctor agrees', function () {
    if (!inst_posix()) {
        skip('file modes are not enforced on Windows; the same files are checked above by the list the doctor judges');
    }
    $old = umask(0); // a host whose umask lets everything through: every file would be 0666 without the code's own care
    try {
        [$root, $data] = inst_installed();
    } finally {
        umask($old);
    }
    $mode = static function (string $f) use ($data): string {
        clearstatcache(true, $data . '/' . $f);
        return sprintf('%04o', fileperms($data . '/' . $f) & 0777);
    };
    foreach (['relay.sqlite', 'config.json', 'first-key.txt', 'admin-token.txt', 'secrets/relay.key', 'secrets/admission.hmac', 'secrets/admin.json'] as $f) {
        eq('0600', $mode($f), $f);
    }
    eq('0700', $mode(''), 'data/');
    // Use the database: SQLite creates -wal and -shm now, with the mode of the database file.
    $umask = umask(0);
    try {
        $cfg = Oaiy\Relay\Config::load($data);
        $db = Oaiy\Relay\Db::open($cfg);
        $db->write(fn($d) => $d->exec("UPDATE meta SET v = v WHERE k = 'schema_version'"));
        $seen = 0;
        foreach (['relay.sqlite-wal', 'relay.sqlite-shm'] as $f) {
            if (is_file($data . '/' . $f)) { // absent when the install chose the truncate journal (a filesystem WAL is unsafe on)
                $seen++;
                eq('0600', $mode($f), $f);
            }
        }
        ok($seen === 2 || (($cfg->toArray()['db']['journal'] ?? 'wal') !== 'wal'), 'in WAL mode both extra files exist while the database is open');
        // The log, created by the relay under the same umask.
        Oaiy\Relay\Log::setFile($data . '/logs/relay.log');
        Oaiy\Relay\Log::write('info', 'test');
        eq('0600', $mode('logs/relay.log'), 'logs/relay.log');
        // A backup.
        [$code, $out, $err] = inst_cli($root, 'relay.php', ['backup'], ['OAIY_RELAY_DATA' => $data]);
        eq(0, $code, "$out $err");
        $backups = glob($data . '/backups/*');
        ok($backups !== [], 'the backup was made');
        foreach ($backups as $b) {
            clearstatcache(true, $b);
            eq('0600', sprintf('%04o', fileperms($b) & 0777), basename($b));
        }
        $rows = Doctor::run(['dataDir' => $data, 'web' => false]);
        eq('ok', doc_level($rows, 'data.modes'), doc_msg($rows, 'data.modes'));
        // And the doctor notices when the database is opened up.
        chmod($data . '/relay.sqlite', 0644);
        $rows = Doctor::run(['dataDir' => $data, 'web' => false]);
        eq('fail', doc_level($rows, 'data.modes'));
        contains('data/relay.sqlite is 0644', doc_msg($rows, 'data.modes'));
    } finally {
        Oaiy\Relay\Log::setFile(null);
        umask($umask);
        $db = null;
        Tmp::after('gc_collect_cycles');
    }
});

test('4.18.8 doctor: the one-time key and the admin token left in data/ are a warning that names the file and its age, and nothing left is a pass', function () {
    eq('ok', doc_level(Doctor::leftovers([]), 'data.leftovers'));
    $rows = Doctor::leftovers([['name' => 'first-key.txt', 'age' => 90]]);
    eq('warn', doc_level($rows, 'data.leftovers'));
    contains('data/first-key.txt', doc_msg($rows, 'data.leftovers'));
    contains('90 s old', doc_msg($rows, 'data.leftovers'));
    contains('one-time enrolment key', doc_msg($rows, 'data.leftovers'));
    contains('--rekey', doc_msg($rows, 'data.leftovers'));
    not_contains('admin-token', doc_msg($rows, 'data.leftovers'));
    $rows = Doctor::leftovers([['name' => 'admin-token.txt', 'age' => 3 * 3600 + 5]]);
    eq('warn', doc_level($rows, 'data.leftovers'));
    contains('data/admin-token.txt', doc_msg($rows, 'data.leftovers'));
    contains('3 h old', doc_msg($rows, 'data.leftovers'));
    contains('admin token', doc_msg($rows, 'data.leftovers'));
    $both = Doctor::leftovers([['name' => 'first-key.txt', 'age' => 5], ['name' => 'admin-token.txt', 'age' => 5]]);
    contains('first-key.txt', doc_msg($both, 'data.leftovers'));
    contains('admin-token.txt', doc_msg($both, 'data.leftovers'));
    // Ages read naturally.
    foreach ([[0, '0 s'], [119, '119 s'], [120, '2 min'], [3599, '59 min'], [7199, '119 min'], [7200, '2 h'], [86400, '24 h'], [172799, '47 h'], [172800, '2 days'], [-5, '0 s']] as [$s, $want]) {
        eq($want, Doctor::age($s), (string)$s);
    }
    // No secret in the message: only names and ages.
    not_contains('oaiy://', doc_msg($both, 'data.leftovers'));
    not_contains('oaiyadm1', doc_msg($both, 'data.leftovers'));
});

test('4.18.8 doctor: a fresh install still has its two files and the doctor says so; once they are deleted it passes', function () {
    $data = doc_data();
    $rows = Doctor::run(['dataDir' => $data, 'web' => false]);
    eq('warn', doc_level($rows, 'data.leftovers'));
    contains('first-key.txt', doc_msg($rows, 'data.leftovers'));
    contains('admin-token.txt', doc_msg($rows, 'data.leftovers'));
    eq([], doc_failures($rows), 'a warning, not a failure');
    // A file made an hour ago is reported as an hour old.
    touch($data . '/first-key.txt', time() - 3 * 3600);
    contains('3 h old', doc_msg(Doctor::run(['dataDir' => $data, 'web' => false]), 'data.leftovers'));
    unlink($data . '/first-key.txt');
    unlink($data . '/admin-token.txt');
    eq('ok', doc_level(Doctor::run(['dataDir' => $data, 'web' => false]), 'data.leftovers'));
});

test('4.18.8 doctor: every exposure probe must answer 403 or 404, anything else (200, redirect, error, no answer) fails', function () {
    $mk = static function (int $status) {
        $r = [];
        foreach (Doctor::EXPOSURE as $p) {
            $r[] = ['path' => $p, 'status' => $status, 'error' => $status === 0 ? 'cannot connect' : null];
        }
        return $r;
    };
    foreach ([403, 404] as $s) {
        $rows = Doctor::exposure($mk($s));
        eq(count(Doctor::EXPOSURE), count($rows));
        eq([], doc_failures($rows), (string)$s);
    }
    foreach ([200, 204, 301, 302, 401, 400, 500, 503, 0] as $s) {
        $rows = Doctor::exposure($mk($s));
        eq(count(Doctor::EXPOSURE), count(doc_failures($rows)), "status $s fails every probe");
    }
    // One reachable path among protected ones is named.
    $r = $mk(404);
    $r[3]['status'] = 200;
    $rows = Doctor::exposure($r);
    eq(1, count(doc_failures($rows)));
    eq('fail', doc_level($rows, 'web.exposure /data/../data/relay.sqlite'));
    // The design's list, then what an install leaves behind that would hand over the relay: the one-time key, the admin token,
    // the config, the admin token's record and the web installer's own token.
    eq(['/data/relay.sqlite', '/data/secrets/admission.hmac', '/data/secrets/relay.key', '/data/../data/relay.sqlite', '/bin/doctor.php', '/src/Db.php', '/install.php', '/.env',
        '/data/first-key.txt', '/data/admin-token.txt', '/data/config.json', '/data/secrets/admin.json', '/INSTALL_ENABLED'], Doctor::EXPOSURE);
});

test('4.9.1 doctor: the dummy bearer must arrive: authHeaderSeen true passes, false fails loudly with the fix, a non-answer fails', function () {
    $ok = Doctor::authorizationSeen(['status' => 200, 'body' => '{"ok":true,"time":1,"authHeaderSeen":true}', 'error' => null]);
    eq('ok', doc_level($ok, 'web.authorization'));
    $bad = Doctor::authorizationSeen(['status' => 200, 'body' => '{"ok":true,"time":1,"authHeaderSeen":false}', 'error' => null]);
    eq('fail', doc_level($bad, 'web.authorization'));
    contains('does NOT reach PHP', doc_msg($bad, 'web.authorization'));
    contains('CGIPassAuth', doc_msg($bad, 'web.authorization'));
    contains('HTTP_AUTHORIZATION', doc_msg($bad, 'web.authorization'));
    foreach ([['status' => 500, 'body' => '', 'error' => null], ['status' => 0, 'body' => '', 'error' => 'cannot connect'], ['status' => 200, 'body' => 'not json', 'error' => null],
        ['status' => 200, 'body' => '{"ok":true}', 'error' => null], ['status' => 404, 'body' => '{"authHeaderSeen":true}', 'error' => null]] as $r) {
        eq('fail', doc_level(Doctor::authorizationSeen($r), 'web.authorization'), json_encode($r));
    }
    // A truthy string is not "true".
    eq('fail', doc_level(Doctor::authorizationSeen(['status' => 200, 'body' => '{"authHeaderSeen":"true"}', 'error' => null]), 'web.authorization'));
});

test('4.18.7 doctor: the web SAPI\'s ini values are compared with the command line\'s and a difference is shown', function () {
    $cli = ['memory_limit' => '128M', 'post_max_size' => '8M', 'max_execution_time' => '0', 'output_buffering' => '0', 'zlib.output_compression' => '0', 'disable_functions' => ''];
    eq('ok', doc_level(Doctor::compareIni($cli, $cli), 'web.ini-difference'));
    $web = array_merge($cli, ['memory_limit' => '32M', 'disable_functions' => 'usleep']);
    $rows = Doctor::compareIni($cli, $web);
    eq('warn', doc_level($rows, 'web.ini-difference'));
    contains('memory_limit: CLI "128M", web "32M"', doc_msg($rows, 'web.ini-difference'));
    contains('disable_functions', doc_msg($rows, 'web.ini-difference'));
    // Keys the web did not report are not differences.
    eq('ok', doc_level(Doctor::compareIni($cli, ['memory_limit' => '128M']), 'web.ini-difference'));
});

test('4.7.1 doctor: a non-public REMOTE_ADDR on a public URL and a forwarding header with no client_ip are warned about', function () {
    eq('warn', doc_level(Doctor::remoteAddr('10.0.0.5', false, false, false), 'web.remote-addr'));
    eq('warn', doc_level(Doctor::remoteAddr('127.0.0.1', false, false, false), 'web.remote-addr'));
    eq('ok', doc_level(Doctor::remoteAddr('203.0.113.9', false, false, false), 'web.remote-addr'));
    eq('ok', doc_level(Doctor::remoteAddr('127.0.0.1', true, false, false), 'web.remote-addr'), 'a loopback URL explains it');
    eq('warn', doc_level(Doctor::remoteAddr(null, false, false, false), 'web.remote-addr'));
    $rows = Doctor::remoteAddr('203.0.113.9', false, true, false);
    eq('warn', doc_level($rows, 'web.forwarding'));
    contains('client_ip', doc_msg($rows, 'web.forwarding'));
    foreach (Doctor::remoteAddr('203.0.113.9', false, true, true) as $r) {
        ok($r['name'] !== 'web.forwarding', 'configured: no forwarding warning');
    }
});

test('4.7.2 doctor: a stalled body the host closed passes, one still open warns and names the timeout to set', function () {
    eq('ok', doc_level(Doctor::slowBody(['closed' => true, 'seconds' => 1.2, 'error' => null], 5.0), 'web.slow-body'));
    $rows = Doctor::slowBody(['closed' => false, 'seconds' => 5.0, 'error' => null], 5.0);
    eq('warn', doc_level($rows, 'web.slow-body'));
    contains('RequestReadTimeout', doc_msg($rows, 'web.slow-body'));
    contains('client_body_timeout', doc_msg($rows, 'web.slow-body'));
    contains('20 s', doc_msg($rows, 'web.slow-body'));
    eq('warn', doc_level(Doctor::slowBody(['closed' => false, 'seconds' => 0.0, 'error' => 'cannot connect'], 5.0), 'web.slow-body'));
});

test('4.18.8 doctor: "inside the web root" is judged on resolved paths, also for a folder that does not exist yet, and a shared name prefix is not inside', function () {
    $base = str_replace('\\', '/', (string)realpath(Tmp::dir('paths')));
    mkdir($base . '/public');
    mkdir($base . '/public-old');
    mkdir($base . '/data');
    eq(true, Doctor::isInsideLoose($base . '/public', $base . '/public'), 'the same folder');
    eq(true, Doctor::isInsideLoose($base . '/public/x/y/z', $base . '/public'), 'a missing folder below');
    eq(true, Doctor::isInsideLoose($base . '/public/sub/../data', $base . '/public'), '.. that stays inside');
    eq(true, Doctor::isInsideLoose($base . '/public/./data', $base . '/public'));
    eq(false, Doctor::isInsideLoose($base . '/data', $base . '/public'), 'a sibling');
    eq(false, Doctor::isInsideLoose($base . '/public-old', $base . '/public'), 'a shared name prefix');
    eq(false, Doctor::isInsideLoose($base . '/public-old/x', $base . '/public'));
    eq(false, Doctor::isInsideLoose($base . '/public/../data', $base . '/public'), '.. that leaves');
    eq(false, Doctor::isInsideLoose($base . '/newdata', $base . '/public'), 'a missing sibling');
    eq(false, Doctor::isInsideLoose($base, $base . '/public'), 'the parent is not inside');
    eq(true, Doctor::isInsideLoose(str_replace('/', '\\', $base) . '/public/deep', $base . '/public'), 'backslashes');
});

test('4.18.7 doctor: worst(), text() and json() report the same result', function () {
    $rows = [Doctor::row('a', 'ok', 'fine'), Doctor::row('b', 'warn', 'hmm'), Doctor::row('c', 'ok', 'fine')];
    eq('warn', Doctor::worst($rows));
    eq('fail', Doctor::worst(array_merge($rows, [Doctor::row('d', 'fail', 'no')])));
    eq('ok', Doctor::worst([Doctor::row('a', 'ok', 'fine')]));
    eq('ok', Doctor::worst([]));
    $t = Doctor::text($rows);
    contains('[warn] b: hmm', $t);
    contains('2 ok, 1 warnings, 0 failures', $t);
    $j = json_decode(Doctor::json(array_merge($rows, [Doctor::row('d', 'fail', 'no')])), true);
    eq('fail', $j['result']);
    eq(['ok' => 2, 'warn' => 1, 'fail' => 1], $j['summary']);
    eq(4, count($j['checks']));
});

test('4.18.7 doctor: a message is made safe before printing: URLs and token-like runs are removed', function () {
    $m = Doctor::safe('could not open mysql://user:hunter2@db.example.com/x or https://relay.example.com/data and ' . str_repeat('A', 40));
    not_contains('hunter2', $m);
    not_contains('db.example.com', $m);
    not_contains(str_repeat('A', 40), $m);
    contains('[url]', $m);
});

// ------------------------------------------------------------------------------------------------ real facts

test('4.18.7 doctor: a fresh install has no failure; every group of checks ran', function () {
    $data = doc_data();
    $rows = Doctor::run(['dataDir' => $data, 'web' => false]);
    eq([], doc_failures($rows));
    foreach (['php.version', 'php.extensions', 'data.dir', 'data.webroot', 'data.modes', 'installed', 'config', 'db.open', 'db.sqlite', 'db.journal', 'wake.shard', 'keys.relay', 'keys.admin', 'keys.admission', 'web'] as $n) {
        doc_level($rows, $n);
    }
    contains('second process', doc_msg($rows, 'wake.shard'));
    contains('ms', doc_msg($rows, 'wake.shard'));
    // Whether it beat 50 ms depends on how busy this machine is; the verdict itself is tested with synthetic timings.
    ok(in_array(doc_level($rows, 'wake.shard'), ['ok', 'warn'], true), doc_msg($rows, 'wake.shard'));
    // Without --url the web checks are a warning that says how to run them.
    $rows = Doctor::run(['dataDir' => $data]);
    eq('warn', doc_level($rows, 'web'));
    contains('--url', doc_msg($rows, 'web'));
});

test('4.18.7 doctor: a data folder that is missing, an install that was never finished, damaged secrets, a broken config and a newer schema each fail', function () {
    // No data folder at all.
    $rows = Doctor::run(['dataDir' => Tmp::dir('nodata') . '/data', 'web' => false]);
    eq('fail', doc_level($rows, 'data.dir'));
    // Not installed (no lock).
    $data = doc_data();
    unlink($data . '/installed.lock');
    eq('fail', doc_level(Doctor::run(['dataDir' => $data, 'web' => false]), 'installed'));
    // Damaged and missing secrets. (Run as a command: with a debugger extension loaded, an exception keeps the frames it
    // unwound through alive, and on Windows an open SQLite file cannot be deleted by the test's clean-up.)
    [$root, $data] = inst_installed();
    file_put_contents($data . '/secrets/relay.key', "not a key\n");
    unlink($data . '/secrets/admin.json');
    unlink($data . '/secrets/admission.hmac');
    [$code, $out] = inst_cli($root, 'doctor.php', ['--skip-web', '--json']);
    eq(1, $code);
    $rows = json_decode($out, true)['checks'];
    eq('fail', doc_level($rows, 'keys.relay'));
    eq('fail', doc_level($rows, 'keys.admin'));
    eq('fail', doc_level($rows, 'keys.admission'));
    // A config that is not valid: later checks that need it say nothing instead of crashing.
    $data = doc_data();
    file_put_contents($data . '/config.json', '{"public_url":"ftp://x"}');
    $rows = Doctor::run(['dataDir' => $data, 'web' => false]);
    eq('fail', doc_level($rows, 'config'));
    contains('public_url', doc_msg($rows, 'config'));
    // A database from a newer relay. (Run as a command, so that the connection an exception unwinds through is released when
    // the process ends: on Windows an open SQLite file cannot be deleted by the test's clean-up.)
    [$root, $data] = inst_installed();
    $pdo = new PDO('sqlite:' . $data . '/relay.sqlite');
    $pdo->exec("UPDATE meta SET v = 99 WHERE k = 'schema_version'");
    $pdo = null;
    [$code, $out] = inst_cli($root, 'doctor.php', ['--skip-web', '--json']);
    eq(1, $code);
    $rows = json_decode($out, true)['checks'];
    eq('fail', doc_level($rows, 'db.open'));
    contains('99', doc_msg($rows, 'db.open'));
});

test('4.18.8 doctor: a data folder inside the served folder fails', function () {
    $data = doc_data();
    $rows = Doctor::run(['dataDir' => $data, 'web' => false, 'publicDir' => dirname($data)]);
    eq('fail', doc_level($rows, 'data.webroot'));
    eq('ok', doc_level(Doctor::run(['dataDir' => $data, 'web' => false, 'publicDir' => dirname($data) . '/public']), 'data.webroot'));
});

test('4.18.4 doctor: the wake shard check reports a second process it could not start instead of guessing, and knows wake.mode=db', function () {
    $data = doc_data();
    $rows = Doctor::wakeVisibility($data, $data . '/no-such-php-binary');
    eq('warn', doc_level($rows, 'wake.shard'));
    $rows = Doctor::wakeVisibility($data, PHP_BINARY, 'db');
    eq('ok', doc_level($rows, 'wake.shard'));
    contains('db', doc_msg($rows, 'wake.shard'));
    // The real thing, through the same second process the doctor starts: it must have SEEN the write (a warning about
    // being slow is allowed on a busy machine, one about never seeing it is not).
    $rows = Doctor::wakeVisibility($data, PHP_BINARY);
    ok(in_array(doc_level($rows, 'wake.shard'), ['ok', 'warn'], true));
    not_contains('NOT visible', doc_msg($rows, 'wake.shard'));
    contains(' ms', doc_msg($rows, 'wake.shard'));
});

test('4.18.4 doctor: a wake shard visible within 50 ms passes, a slower one or none at all warns and names wake.mode "db"', function () {
    foreach ([0.0, 0.4, 7.7, 49.9, 50.0] as $ms) {
        eq('ok', doc_level(Doctor::wakeResult($ms), 'wake.shard'), (string)$ms);
    }
    foreach ([50.1, 51.0, 200.0, 1400.0] as $ms) {
        $rows = Doctor::wakeResult($ms);
        eq('warn', doc_level($rows, 'wake.shard'), (string)$ms);
        contains('wake.mode', doc_msg($rows, 'wake.shard'));
    }
    $rows = Doctor::wakeResult(null);
    eq('warn', doc_level($rows, 'wake.shard'));
    contains('NOT visible', doc_msg($rows, 'wake.shard'));
    contains('"db"', doc_msg($rows, 'wake.shard'));
});

// ------------------------------------------------------------------------------------------------ the web

test('4.18.8 doctor web: with public/ as the document root no exposure probe succeeds and the dummy bearer arrives', function () {
    [$root, $data] = inst_installed();
    $srv = Server::start($root . '/public', ['prepend' => false, 'name' => 'doc-pub']);
    $rows = Doctor::run(['dataDir' => $data, 'url' => $srv->base(), 'adminTokenFile' => $data . '/admin-token.txt', 'slowBody' => false, 'publicDir' => $root . '/public']);
    eq([], doc_failures($rows));
    foreach (Doctor::EXPOSURE as $p) {
        eq('ok', doc_level($rows, 'web.exposure ' . $p), $p . ': ' . doc_msg($rows, 'web.exposure ' . $p));
    }
    eq('ok', doc_level($rows, 'web.authorization'), doc_msg($rows, 'web.authorization'));
    // php -S runs with its own ini defaults, so the comparison may legitimately report a difference; it must have run.
    ok(in_array(doc_level($rows, 'web.ini-difference'), ['ok', 'warn'], true));
    if (doc_level($rows, 'web.ini-difference') === 'warn') {
        contains('CLI "', doc_msg($rows, 'web.ini-difference'));
    }
    eq('ok', doc_level($rows, 'web.remote-addr'), doc_msg($rows, 'web.remote-addr'));
    // The web values were judged too (the limits that count are the web's).
    eq('ok', doc_level($rows, 'ini.memory_limit (web)'), doc_msg($rows, 'ini.memory_limit (web)'));
    doc_level($rows, 'ini.post_max_size (web)');
    doc_level($rows, 'ini.disable_functions (web)');
});

test('4.18.8 doctor web: with the relay folder as the document root data/, src/ and bin/ are reachable and each is a failure; nothing is written outside data/', function () {
    [$root, $data] = inst_installed();
    $srv = Server::start($root, ['prepend' => false, 'name' => 'doc-root']);
    $snapshot = static function () use ($root): array {
        $out = [];
        $it = new RecursiveIteratorIterator(new RecursiveDirectoryIterator($root, FilesystemIterator::SKIP_DOTS));
        foreach ($it as $f) {
            $p = str_replace('\\', '/', $f->getPathname());
            if (strpos($p, $root . '/data/') === 0) {
                continue;
            }
            $out[$p] = $f->getSize() . ':' . $f->getMTime();
        }
        ksort($out);
        return $out;
    };
    $before = $snapshot();
    $rows = Doctor::run(['dataDir' => $data, 'url' => $srv->base(), 'slowBody' => false, 'publicDir' => $root . '/public']);
    eq($before, $snapshot(), 'nothing outside data/ was created or changed');
    foreach (['/data/relay.sqlite', '/data/secrets/admission.hmac', '/data/secrets/relay.key', '/data/../data/relay.sqlite', '/bin/doctor.php', '/src/Db.php',
        '/data/first-key.txt', '/data/admin-token.txt', '/data/config.json', '/data/secrets/admin.json'] as $p) {
        eq('fail', doc_level($rows, 'web.exposure ' . $p), $p . ': ' . doc_msg($rows, 'web.exposure ' . $p));
    }
    eq('ok', doc_level($rows, 'web.exposure /.env'));
    eq('ok', doc_level($rows, 'web.exposure /INSTALL_ENABLED'), 'there is no such file after a command line install');
    eq('fail', Doctor::worst($rows));
    // The probe itself was only GET requests: the secret files are still intact.
    ok(is_file($data . '/secrets/relay.key') && is_file($data . '/relay.sqlite'), 'data/ intact');
});

test('4.18.8 doctor web: the web installer\'s INSTALL_ENABLED (it holds the owner\'s installer token) is an exposure failure when it can be fetched', function () {
    [$root, $data] = inst_installed();
    file_put_contents($root . '/INSTALL_ENABLED', "owner-chosen-installer-token-2026\n");
    $srv = Server::start($root, ['prepend' => false, 'name' => 'doc-root']);
    $rows = Doctor::run(['dataDir' => $data, 'url' => $srv->base(), 'slowBody' => false, 'publicDir' => $root . '/public']);
    eq('fail', doc_level($rows, 'web.exposure /INSTALL_ENABLED'));
    $pub = Server::start($root . '/public', ['prepend' => false, 'name' => 'doc-pub']);
    $rows = Doctor::run(['dataDir' => $data, 'url' => $pub->base(), 'slowBody' => false, 'publicDir' => $root . '/public']);
    eq('ok', doc_level($rows, 'web.exposure /INSTALL_ENABLED'));
});

test('4.18.8 doctor web: a relay unpacked into a folder of an existing site is probed at the URL\'s own path, so a served /relay/data/ is a failure', function () {
    [$root, $data] = inst_installed();
    // The site's document root is the folder that holds the relay folder: https://site/<relay>/data/... is served.
    $site = dirname($root);
    $name = basename($root);
    $srv = Server::start($site, ['prepend' => false, 'name' => 'doc-site']);
    $rows = Doctor::run(['dataDir' => $data, 'url' => $srv->base() . '/' . $name, 'slowBody' => false, 'publicDir' => $root . '/public']);
    foreach (['/data/relay.sqlite', '/data/secrets/relay.key', '/data/first-key.txt', '/data/admin-token.txt', '/data/config.json', '/bin/doctor.php', '/src/Db.php'] as $p) {
        eq('fail', doc_level($rows, 'web.exposure /' . $name . $p), $p . ' under the path: ' . doc_msg($rows, 'web.exposure /' . $name . $p));
    }
    // The site's root itself is asked too, and does not have these files.
    eq('ok', doc_level($rows, 'web.exposure /data/relay.sqlite'));
    eq('ok', doc_level($rows, 'web.exposure /.env'));
    eq('fail', Doctor::worst($rows));
    // A path with characters the doctor will not put in a request is refused, not probed.
    $rows = Doctor::run(['dataDir' => $data, 'url' => $srv->base() . '/a%20b', 'slowBody' => false, 'publicDir' => $root . '/public']);
    eq('fail', doc_level($rows, 'web'));
    contains('path', doc_msg($rows, 'web'));
});

test('4.9.1 doctor web: a host that strips the Authorization header fails the dummy bearer probe loudly (a stub that reports it stripped)', function () {
    $dir = Tmp::dir('strip');
    // A stub that answers /v1/health as the relay does, from a stack that dropped the Authorization header.
    file_put_contents($dir . '/index.php', '<?php header("Content-Type: application/json"); echo json_encode(["ok" => true, "time" => time(), "authHeaderSeen" => false]);');
    $srv = Server::start($dir, ['prepend' => false, 'name' => 'stub']);
    $r = HttpProbe::get($srv->base() . '/v1/health', ['Authorization' => 'Bearer probe']);
    $rows = Doctor::authorizationSeen($r);
    eq('fail', doc_level($rows, 'web.authorization'));
    contains('does NOT reach PHP', doc_msg($rows, 'web.authorization'));
    // And a stack that passes it, the real relay, is judged ok (checked above); here the stub that reports it seen.
    file_put_contents($dir . '/index.php', '<?php header("Content-Type: application/json"); echo json_encode(["ok" => true, "time" => time(), "authHeaderSeen" => isset($_SERVER["HTTP_AUTHORIZATION"])]);');
    $r = HttpProbe::get($srv->base() . '/v1/health', ['Authorization' => 'Bearer probe']);
    eq('ok', doc_level(Doctor::authorizationSeen($r), 'web.authorization'));
    $r = HttpProbe::get($srv->base() . '/v1/health');
    eq('fail', doc_level(Doctor::authorizationSeen($r), 'web.authorization'), 'no header sent: not seen');
});

test('4.7.2 doctor web: the slow-body probe sees a host that closes a stalled body and one that keeps waiting', function () {
    $dir = Tmp::dir('slow');
    file_put_contents($dir . '/stub.php', '<?php
$srv = stream_socket_server("tcp://127.0.0.1:" . $argv[1]);
$c = stream_socket_accept($srv, 15);
fread($c, 4096);
if ($argv[2] === "close") { usleep(400000); fclose($c); }
else { $t = microtime(true); while (microtime(true) - $t < 6 && !feof($c)) { usleep(50000); } }
');
    $run = static function (string $mode) use ($dir): array {
        $port = Server::freePort();
        $p = proc_open(array_merge([PHP_BINARY], Server::phpFlags(), [$dir . '/stub.php', (string)$port, $mode]), [1 => ['pipe', 'w'], 2 => ['pipe', 'w']], $pipes);
        $up = false;
        for ($i = 0; $i < 100 && !$up; $i++) {
            $c = @stream_socket_client('tcp://127.0.0.1:' . $port, $e1, $e2, 0.1);
            $up = $c !== false;
            if ($up) {
                fclose($c);
            } else {
                usleep(50000);
            }
        }
        // The probe above consumed the stub's one accept: start a second stub for the real probe.
        proc_terminate($p);
        proc_close($p);
        $port = Server::freePort();
        $p = proc_open(array_merge([PHP_BINARY], Server::phpFlags(), [$dir . '/stub.php', (string)$port, $mode]), [1 => ['pipe', 'w'], 2 => ['pipe', 'w']], $pipes);
        usleep(400000);
        $r = HttpProbe::slowBody('http://127.0.0.1:' . $port . '/v1/items', 100, '{"items":', 2.0);
        proc_terminate($p);
        proc_close($p);
        return $r;
    };
    $closed = $run('close');
    eq(true, $closed['closed'], 'the closing host');
    ok($closed['seconds'] < 1.9, 'noticed before the watch ran out');
    eq('ok', doc_level(Doctor::slowBody($closed, 2.0), 'web.slow-body'));
    $open = $run('hold');
    eq(false, $open['closed'], 'the waiting host');
    ok($open['seconds'] >= 1.9, 'watched for the full time');
    eq('warn', doc_level(Doctor::slowBody($open, 2.0), 'web.slow-body'));
});

test('4.18.7 doctor web: a URL that gives no answer is a failure, a bad --url is a failure, --skip-web skips', function () {
    $data = doc_data();
    // A server that accepts every connection and hangs up at once (a closed port would make Windows retry for seconds).
    $dir = Tmp::dir('hangup');
    file_put_contents($dir . '/stub.php', '<?php $s = stream_socket_server("tcp://127.0.0.1:" . $argv[1]); for ($i = 0; $i < 40; $i++) { $c = @stream_socket_accept($s, 20); if ($c) { fclose($c); } }');
    $port = Server::freePort();
    $p = proc_open(array_merge([PHP_BINARY], Server::phpFlags(), [$dir . '/stub.php', (string)$port]), [1 => ['pipe', 'w'], 2 => ['pipe', 'w']], $pipes);
    Tmp::after(static function () use ($p): void {
        proc_terminate($p);
        proc_close($p);
    });
    usleep(400000);
    $rows = Doctor::run(['dataDir' => $data, 'url' => 'http://127.0.0.1:' . $port, 'slowBody' => false]);
    ok(count(doc_failures($rows)) >= 9, 'every probe failed: ' . count(doc_failures($rows)));
    eq('fail', doc_level($rows, 'web.authorization'));
    eq('fail', doc_level(Doctor::run(['dataDir' => $data, 'url' => 'ftp://relay.example.com']), 'web'));
    eq('ok', doc_level(Doctor::run(['dataDir' => $data, 'url' => 'http://127.0.0.1:1', 'web' => false]), 'web'));
});

// ------------------------------------------------------------------------------------------------ the command line

test('4.18.7 doctor CLI: exit 0 with warnings only, 1 on a failure, 2 on usage; --json is JSON', function () {
    [$root, $data] = inst_installed();
    [$code, $out, $err] = inst_cli($root, 'doctor.php', ['--skip-web']);
    eq(0, $code, "stdout: $out stderr: $err");
    contains('0 failures', $out);
    [$code, $out] = inst_cli($root, 'doctor.php', ['--skip-web', '--json']);
    eq(0, $code);
    $j = json_decode($out, true);
    ok(is_array($j) && isset($j['checks'], $j['summary'], $j['result']), 'JSON: ' . $out);
    eq(0, $j['summary']['fail']);
    unlink($data . '/installed.lock');
    [$code, $out] = inst_cli($root, 'doctor.php', ['--skip-web']);
    eq(1, $code, $out);
    contains('[fail] installed', $out);
    foreach ([['--nonsense'], ['--slow-body-watch=0'], ['--slow-body-watch=abc']] as $args) {
        [$code] = inst_cli($root, 'doctor.php', $args);
        eq(2, $code, json_encode($args));
    }
});

test('4.18.7 doctor CLI: the admin token from the file is used and never printed', function () {
    [$root, $data] = inst_installed();
    $srv = Server::start($root . '/public', ['prepend' => false, 'name' => 'doc-cli']);
    [$code, $out, $err] = inst_cli($root, 'doctor.php', ['--url=' . $srv->base(), '--admin-token-file=' . $data . '/admin-token.txt', '--no-slow-body']);
    eq(0, $code, "stdout: $out stderr: $err");
    contains('web.ini-difference', $out);
    contains('web.authorization', $out);
    $secrets = inst_secrets($data);
    inst_no_secrets($out . $err, $secrets, 'the doctor output');
    inst_no_secrets($srv->log(), $secrets, 'the server log');
    // With no token file the comparison is skipped with an explanation; with a wrong one the refusal is reported.
    [, $out2] = inst_cli($root, 'doctor.php', ['--url=' . $srv->base(), '--no-slow-body']);
    contains('--admin-token-file', $out2);
    // A well-formed token that is not this relay's: the refusal is reported and the token is still not echoed.
    $wrong = Installer::newAdminToken()['token'];
    $bad = $root . '/bad-token.txt';
    file_put_contents($bad, $wrong . "\n");
    [, $out3, $err3] = inst_cli($root, 'doctor.php', ['--url=' . $srv->base(), '--admin-token-file=' . $bad, '--no-slow-body']);
    contains('refused the admin token', $out3);
    not_contains($wrong, $out3 . $err3, 'even a wrong token is not echoed');
    not_contains(substr($wrong, 21), $out3 . $err3, 'nor its secret part');
    // A file that is not a token at all is skipped, and its content is not echoed.
    file_put_contents($bad, "hunter2-not-a-token\n");
    [, $out4, $err4] = inst_cli($root, 'doctor.php', ['--url=' . $srv->base(), '--admin-token-file=' . $bad, '--no-slow-body']);
    contains('does not hold an admin token', $out4);
    not_contains('hunter2', $out4 . $err4);
});

test('4.18.6 gc CLI: runs a pass and prints counts, says "not due" until a minute has passed, --force runs it, --vacuum vacuums, bad arguments exit 2', function () {
    [$root] = inst_installed();
    [$code, $out, $err] = inst_cli($root, 'gc.php');
    eq(0, $code, $err);
    contains('gc: pass done', $out);
    contains('retired=0', $out);
    [$code, $out] = inst_cli($root, 'gc.php');
    eq(0, $code);
    contains('not due', $out);
    [$code, $out] = inst_cli($root, 'gc.php', ['--force']);
    eq(0, $code);
    contains('gc: pass done', $out);
    [$code, $out, $err] = inst_cli($root, 'gc.php', ['--vacuum']);
    eq(0, $code, $err);
    contains('vacuum done', $out);
    [$code] = inst_cli($root, 'gc.php', ['--bogus']);
    eq(2, $code);
    // An uninstalled relay is a failure, not a crash.
    [$code, , $err] = inst_cli(inst_scratch(), 'gc.php');
    eq(1, $code);
    contains('gc failed', $err);
});

test('4.18.2 bin scripts: a request to bin/*.php through the web runs nothing and installs nothing', function () {
    $root = inst_scratch();
    $srv = Server::start($root, ['prepend' => false, 'name' => 'bin-root']);
    foreach (['/bin/install.php', '/bin/doctor.php', '/bin/gc.php'] as $p) {
        foreach (['GET', 'POST'] as $m) {
            $r = $srv->request($m, $p . '?url=https://relay.example.com', ['Content-Type' => 'application/x-www-form-urlencoded'], $m === 'POST' ? '--url=https://relay.example.com' : null);
            eq('', $r['body'], "$m $p prints nothing");
        }
    }
    ok(!file_exists($root . '/data'), 'no data folder was created by a web request to bin/install.php');
    // Nor does src/*.php run anything.
    foreach (['/src/Installer.php', '/src/Doctor.php', '/src/Kernel.php'] as $p) {
        eq('', $srv->request('GET', $p)['body'], $p);
    }
    ok(!file_exists($root . '/data'), 'and nothing was created by src/');
});
