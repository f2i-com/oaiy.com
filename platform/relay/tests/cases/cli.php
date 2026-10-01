<?php
declare(strict_types=1);

use Oaiy\Relay\Auth;
use Oaiy\Relay\B64;
use Oaiy\Relay\Cli;
use Oaiy\Relay\Crypto;
use Oaiy\Relay\Devices;
use Oaiy\Relay\Enrolment;
use OaiyTest\Relay;
use OaiyTest\Server;
use OaiyTest\Tmp;

/** Run a bin/relay.php command in process. @return array{0:int,1:string,2:string} exit code, stdout, stderr */
function cli_run(Relay $r, string ...$args): array
{
    $out = '';
    $err = '';
    $cli = new Cli($r->data, function (string $s) use (&$out): void {
        $out .= $s;
    }, function (string $s) use (&$err): void {
        $err .= $s;
    });
    $code = $cli->run($args);
    return [$code, $out, $err];
}

// ------------------------------------------------------------------------------------------------ key

test('4.18.8 bin/relay key: an enrolment key goes to a 0600 file and only its path is printed; the key redeems', function () {
    $r = Relay::make();
    [$code, $out, $err] = cli_run($r, 'key', 'provider', '--ttl=600', '--name=FormLogic');
    eq([0, ''], [$code, $err]);
    ok(preg_match('#\n  (\S+/keys/[A-Za-z0-9_-]{11}\.txt)\n#', $out, $m) === 1, $out);
    $file = $m[1];
    $uri = trim((string)file_get_contents($file));
    ok(strncmp($uri, 'oaiy://enroll?', 14) === 0);
    parse_str(substr($uri, 14), $q);
    not_contains($q['s'], $out, 'the secret is not printed');
    not_contains('oaiy://', $out);
    eq('provider', $q['r']);
    eq((string)(Relay::T0 + 600), $q['x']);
    if (stripos(PHP_OS, 'WIN') !== 0) {
        eq('0600', substr(sprintf('%o', fileperms($file)), -4));
    }
    // It redeems.
    $d = Enrolment::derive(B64::decN($q['s'], 16));
    $body = json_encode(['kid' => $d['kid'], 'role' => 'provider', 'name' => 'FormLogic', 'n' => B64::enc(random_bytes(16)), 'keys' => ['ed25519' => B64::enc(Crypto::signKeypairFromSeed(random_bytes(32))[0]), 'x25519' => B64::enc(sodium_crypto_box_publickey(sodium_crypto_box_keypair()))]]);
    $res = $r->call(null, 'POST', '/v1/enroll', $body, [], ['X-OAIY-Proof' => B64::enc(Crypto::sign($d['sk'], Enrolment::DOMAIN . $body))]);
    eq(201, $res['status'], $res['body']);
    eq('FormLogic', $r->ctx()->db->val('SELECT name FROM devices WHERE id = ?', [$res['json']['deviceId']]));
});

test('4.18.8 bin/relay key --print writes the key to the terminal, on request only; the desktop limit and bad arguments are refused without a key', function () {
    $r = Relay::make();
    [$code, $out] = cli_run($r, 'key', 'provider', '--print');
    eq(0, $code);
    ok(strncmp($out, 'oaiy://enroll?', 14) === 0);
    foreach ([['key'], ['key', 'phone'], ['key', 'web'], ['key', 'desktop', 'extra'], ['key', 'provider', '--ttl=0'], ['key', 'provider', '--ttl=86401'], ['key', 'provider', '--ttl=abc'], ['key', 'provider', '--ttl=1.5']] as $args) {
        [$c, $o, $e] = cli_run($r, ...$args);
        eq(2, $c, implode(' ', $args));
        not_contains('oaiy://enroll', $o . $e);
    }
    // The installer's first key is pending: one more desktop key fits, a third does not.
    eq(0, cli_run($r, 'key', 'desktop')[0]);
    [$c, $o, $e] = cli_run($r, 'key', 'desktop');
    eq(1, $c);
    contains('already has its 2 desktops', $e);
    not_contains('oaiy://', $o . $e);
});

// ------------------------------------------------------------------------------------------------ devices, revoke

test('4.18.8 bin/relay devices: every device with its state and no token or key material', function () {
    $r = Relay::make();
    $d = $r->desktop('Front desk');
    $ph = $r->phone($d, 'Kitchen phone');
    $v = $r->provider();
    $r->call($ph, 'GET', '/v1/poll');
    Devices::revoke($r->ctx(), $v->id);
    [$code, $out] = cli_run($r, 'devices');
    eq(0, $code);
    foreach ([$d->id, $ph->id, $v->id, 'Front desk', 'Kitchen phone', 'online', 'revoked', 'offline'] as $s) {
        contains($s, $out);
    }
    contains('3 devices', $out);
    foreach ([$d, $ph, $v] as $a) {
        not_contains($a->token, $out);
        not_contains(B64::enc($a->edPk), $out);
    }
});

test('4.18.8 bin/relay revoke: a phone, a provider, and a desktop with its phones; twice is fine; unknown and malformed ids are refused', function () {
    $r = Relay::make();
    $d = $r->desktop();
    $d2 = $r->desktop('Two');
    $ph = $r->phone($d);
    $other = $r->phone($d2);
    $v = $r->provider();
    [$c, $o] = cli_run($r, 'revoke', $ph->id);
    eq(0, $c);
    contains('revoked: ' . $ph->id, $o);
    eq('revoked', $r->call($ph, 'GET', '/v1/poll')['json']['error']['code']);
    [$c, $o] = cli_run($r, 'revoke', $ph->id);
    eq(0, $c);
    contains('already revoked', $o);
    [$c, $o] = cli_run($r, 'revoke', $v->id);
    eq(0, $c);
    // A desktop, from the host: the way a desktop is removed. Its phones go with it; another desktop's do not.
    $mine = $r->phone($d);
    [$c, $o] = cli_run($r, 'revoke', $d->id);
    eq(0, $c);
    contains($d->id, $o);
    contains($mine->id, $o);
    eq('revoked', $r->call($d, 'GET', '/v1/poll')['json']['error']['code']);
    eq('revoked', $r->call($mine, 'GET', '/v1/poll')['json']['error']['code']);
    eq(200, $r->call($other, 'GET', '/v1/poll')['status']);
    eq(200, $r->call($d2, 'GET', '/v1/poll')['status']);
    [$c, , $e] = cli_run($r, 'revoke', 'dev-' . str_repeat('Z', 22));
    eq([1, "no such device\n"], [$c, $e]);
    foreach ([['revoke'], ['revoke', 'nonsense'], ['revoke', $d->id, 'x']] as $args) {
        eq(2, cli_run($r, ...$args)[0], implode(' ', $args));
    }
});

// ------------------------------------------------------------------------------------------------ roster-reset, reset

test('4.18.8 bin/relay roster-reset: forgets the roster row (one app or all) and revokes no one', function () {
    $r = Relay::make();
    $d = $r->desktop();
    $d2 = $r->desktop('Two');
    $ph = $r->phone($d);
    foreach ([[$d, 'aokie'], [$d, 'other'], [$d2, 'aokie']] as [$who, $app]) {
        eq(200, $r->call($who, 'POST', '/v1/roster', ['appId' => $app, 'revision' => 4, 'thumbprints' => $who === $d && $app === 'aokie' ? [Crypto::thumbprint($ph->edPk)] : []])['status']);
    }
    eq(3, (int)$r->ctx()->db->val('SELECT COUNT(*) FROM roster'));
    [$c, $o] = cli_run($r, 'roster-reset', $d->id, 'other');
    eq(0, $c);
    contains('removed 1 roster row', $o);
    eq(2, (int)$r->ctx()->db->val('SELECT COUNT(*) FROM roster'));
    [$c, $o] = cli_run($r, 'roster-reset', $d->id);
    contains('removed 1 roster row', $o);
    eq([$d2->id], array_column($r->ctx()->db->all('SELECT desktop_dev FROM roster'), 'desktop_dev'));
    eq(200, $r->call($ph, 'GET', '/v1/poll')['status'], 'the phone was not revoked');
    foreach ([['roster-reset'], ['roster-reset', 'nonsense'], ['roster-reset', $d->id, 'bad app'], ['roster-reset', $d->id, 'a', 'b']] as $args) {
        eq(2, cli_run($r, ...$args)[0], implode(' ', $args));
    }
});

test('4.18.8 bin/relay reset limits clears locks and buckets (not the status counters); reset epoch makes every client reset', function () {
    $r = Relay::make();
    $d = $r->desktop();
    $bad = 'oaiyrt1.' . explode('.', $d->token)[1] . '.' . B64::enc(random_bytes(32));
    for ($i = 0; $i < 20; $i++) {
        $r->call($bad, 'GET', '/v1/poll');
        Tmp::setClock(Relay::T0 + 65 * ($i + 1));
    }
    eq(401, $r->call($d, 'GET', '/v1/poll')['status'], 'the pair (token id, address) is locked');
    [$c, $o] = cli_run($r, 'reset', 'limits');
    eq(0, $c);
    contains('cleared', $o);
    eq(200, $r->call($d, 'GET', '/v1/poll')['status'], 'the lock is gone');
    eq(0, (int)$r->ctx()->db->val('SELECT COUNT(*) FROM tokid_fail'));
    ok((int)$r->ctx()->db->val("SELECT COUNT(*) FROM rl WHERE k LIKE 's:%'") >= 1, 'status counters stay');
    $epoch = $r->ctx()->db->metaStr('epoch');
    eq(0, cli_run($r, 'reset', 'epoch')[0]);
    neq($epoch, $r->ctx()->db->metaStr('epoch'));
    eq(true, $r->call($d, 'GET', '/v1/poll', null, ['epoch' => $epoch])['json']['reset']);
    eq(2, cli_run($r, 'reset')[0]);
    eq(2, cli_run($r, 'reset', 'everything')[0]);
});

test('4.18.8 bin/relay status: the facts and no secret', function () {
    Relay::sqliteOnly();
    $r = Relay::make();
    $d = $r->desktop();
    $ph = $r->phone($d);
    $r->call($d, 'GET', '/v1/poll');
    [$c, $o] = cli_run($r, 'status');
    eq(0, $c);
    foreach (['relay id      rly-', 'public url    ' . $r->publicUrl, 'version       0.1.0', 'database      sqlite (wal', 'devices       desktop=1 phone=1', 'call features off', 'calibrated    no', 'epoch         '] as $s) {
        contains($s, $o);
    }
    foreach ([$d->token, $ph->token, $r->adminToken(), trim(file_get_contents($r->data . '/secrets/relay.key')), trim(file_get_contents($r->data . '/secrets/admission.hmac'))] as $secret) {
        not_contains($secret, $o);
    }
});

// ------------------------------------------------------------------------------------------------ backup, restore

test('4.18.5 bin/relay backup: a consistent SQLite copy made with VACUUM INTO; an existing file is not overwritten', function () {
    Relay::sqliteOnly();
    $r = Relay::make();
    $d = $r->desktop();
    $v = $r->provider();
    $r->call($v, 'POST', '/v1/items', ['items' => [['to' => $d->inbox(), 'lane' => 'cmd', 'id' => 'b1', 'body' => 'x']]]);
    $file = $r->dir . '/copy.sqlite';
    [$c, $o] = cli_run($r, 'backup', '--out=' . $file);
    eq(0, $c, $o);
    ok(is_file($file));
    $p = new PDO('sqlite:' . $file);
    eq('ok', $p->query('PRAGMA integrity_check')->fetchColumn());
    eq(1, (int)$p->query('SELECT COUNT(*) FROM items')->fetchColumn());
    eq(2, (int)$p->query('SELECT COUNT(*) FROM devices')->fetchColumn());
    $p = null;
    [$c, , $e] = cli_run($r, 'backup', '--out=' . $file);
    eq(1, $c);
    contains('already exists', $e);
    [$c, $o] = cli_run($r, 'backup');
    eq(0, $c);
    contains('/backups/relay-', $o);
    eq(1, count(glob($r->data . '/backups/relay-*.sqlite')));
});

test('4.3 bin/relay restore: the backup goes back, a new epoch makes every client reset, tokens made after it are unknown, and the old database is kept', function () {
    Relay::sqliteOnly();
    $r = Relay::make();
    $d = $r->desktop();
    $v = $r->provider();
    $post = fn(string $id) => $r->call($v, 'POST', '/v1/items', ['items' => [['to' => $d->inbox(), 'lane' => 'cmd', 'id' => $id, 'body' => 'x']]])['json']['results'][0]['status'];
    eq('queued', $post('before'));
    $file = $r->dir . '/b.sqlite';
    eq(0, cli_run($r, 'backup', '--out=' . $file)[0]);
    // Life goes on after the backup.
    eq('queued', $post('after'));
    $late = $r->desktop('Late');
    $epoch = $r->ctx()->db->metaStr('epoch');
    $polled = $r->call($d, 'GET', '/v1/poll', null, ['epoch' => $epoch]);
    eq([1, 2], array_column($polled['json']['items'], 'seq'));
    // Refused without --yes.
    eq(2, cli_run($r, 'restore', $file)[0]);
    [$c, $o, $e] = cli_run($r, 'restore', $file, '--yes');
    eq(0, $c, $o . $e);
    contains('new epoch', $o);
    $db = $r->ctx()->db;
    eq(['before'], array_column($db->all('SELECT id FROM items ORDER BY seq'), 'id'));
    eq(0, (int)$db->val('SELECT COUNT(*) FROM devices WHERE id = ?', [$late->id]));
    neq($epoch, $db->metaStr('epoch'));
    $res = $r->call($d, 'GET', '/v1/poll', null, ['epoch' => $epoch, 'since' => '2']);
    eq(true, $res['json']['reset'], 'the next poll of every client answers reset');
    eq(1, $res['json']['cursor'], 'with the restored mailbox\'s highest seq');
    eq(401, $r->call($late, 'GET', '/v1/poll')['status'], 'a token made after the backup is unknown');
    eq(200, $r->call($d, 'GET', '/v1/poll')['status'], 'one that existed is fine');
    eq(1, count(glob($r->data . '/backups/pre-restore-*.sqlite')), 'the database before the restore is kept');
    $kept = new PDO('sqlite:' . glob($r->data . '/backups/pre-restore-*.sqlite')[0]);
    eq(2, (int)$kept->query('SELECT COUNT(*) FROM items')->fetchColumn());
});

test('4.3 bin/relay restore: refuses another relay\'s backup, a newer schema, a damaged file and a missing one, and changes nothing', function () {
    Relay::sqliteOnly();
    $r = Relay::make();
    $d = $r->desktop();
    $other = Relay::make();
    $otherFile = $other->dir . '/other.sqlite';
    cli_run($other, 'backup', '--out=' . $otherFile);
    $before = json_encode($r->ctx()->db->all('SELECT id FROM devices'));
    [$c, , $e] = cli_run($r, 'restore', $otherFile, '--yes');
    eq(1, $c);
    contains('another relay id', $e);
    // A newer schema.
    $newer = $r->dir . '/newer.sqlite';
    cli_run($r, 'backup', '--out=' . $newer);
    $p = new PDO('sqlite:' . $newer);
    $p->exec("UPDATE meta SET v = 99 WHERE k = 'schema_version'");
    $p = null;
    [$c, , $e] = cli_run($r, 'restore', $newer, '--yes');
    eq(1, $c);
    contains('schema 99', $e);
    // Not a database.
    file_put_contents($r->dir . '/junk.sqlite', str_repeat('not a database ', 500));
    [$c] = cli_run($r, 'restore', $r->dir . '/junk.sqlite', '--yes');
    eq(1, $c);
    [$c, , $e] = cli_run($r, 'restore', $r->dir . '/missing.sqlite', '--yes');
    eq(1, $c);
    contains('no such backup', $e);
    eq($before, json_encode($r->ctx()->db->all('SELECT id FROM devices')), 'nothing changed');
    eq(200, $r->call($d, 'GET', '/v1/poll')['status']);
    eq([], glob($r->data . '/backups/pre-restore-*') ?: [], 'no restore started, so no safety copy either');
});

// ------------------------------------------------------------------------------------------------ export, import

test('4.18.8 bin/relay export and import: the identity, the tokens and the mailboxes move together to a fresh host', function () {
    Relay::sqliteOnly();
    $r = Relay::make();
    $d = $r->desktop();
    $v = $r->provider();
    $r->call($v, 'POST', '/v1/items', ['items' => [['to' => $d->inbox(), 'lane' => 'cmd', 'id' => 'moved', 'body' => 'still here']]]);
    $info = $r->call(null, 'GET', '/v1/info')['json'];
    $dir = $r->dir . '/export';
    [$c, $o, $e] = cli_run($r, 'export', $dir);
    eq(0, $c, $o . $e);
    foreach (['relay.sqlite', 'config.json', 'MANIFEST.txt', 'secrets/relay.key', 'secrets/admin.json', 'secrets/admission.hmac'] as $f) {
        ok(is_file($dir . '/' . $f), $f);
    }
    contains('secrets', $o);
    // A brand new, empty data folder on "another host" (the same config: the hostname is kept).
    $fresh = Tmp::dir('newhost') . '/data';
    $new = new Cli($fresh, function (string $s) use (&$out2): void {
        $out2 = ($out2 ?? '') . $s;
    }, function (string $s) use (&$err2): void {
        $err2 = ($err2 ?? '') . $s;
    });
    eq(0, $new->run(['import', $dir, '--yes']), (string)($out2 ?? '') . (string)($err2 ?? ''));
    contains('imported relay ' . $info['relayId'], (string)$out2);
    $moved = new Relay();
    $moved->dir = dirname($fresh);
    $moved->data = $fresh;
    $moved->publicUrl = $r->publicUrl;
    $ni = $moved->call(null, 'GET', '/v1/info')['json'];
    eq($info['relayId'], $ni['relayId'], 'the same identity');
    eq($info['relayKey'], $ni['relayKey']);
    $poll = $moved->call($d, 'GET', '/v1/poll');
    eq(200, $poll['status'], 'the old token works on the new host: ' . $poll['body']);
    eq(['still here'], array_column($poll['json']['items'], 'body'));
    eq($r->ctx()->db->metaStr('epoch'), $poll['json']['epoch'], 'the epoch moved too: no client needs to reset');
    eq(200, $moved->call($r->adminToken(), 'GET', '/v1/admin/status')['status'], 'and so does the admin token, whose hash moved in secrets/');
});

test('4.18.8 bin/relay import: refuses an installed relay, a damaged or changed export, a missing manifest and a data folder in the web root', function () {
    Relay::sqliteOnly();
    $r = Relay::make();
    $dir = $r->dir . '/export';
    eq(0, cli_run($r, 'export', $dir)[0]);
    // Into an installed data folder.
    [$c, , $e] = cli_run($r, 'import', $dir, '--yes');
    eq(1, $c);
    contains('already installed', $e);
    $fresh = function (): Cli {
        $d = Tmp::dir('imp') . '/data';
        return new Cli($d, function (string $s): void {
        }, function (string $s): void {
        });
    };
    eq(2, $fresh()->run(['import', $dir]), '--yes is required');
    // A changed file: the checksum no longer matches.
    file_put_contents($dir . '/config.json', file_get_contents($dir . '/config.json') . ' ');
    eq(1, $fresh()->run(['import', $dir, '--yes']));
    // A missing manifest.
    unlink($dir . '/MANIFEST.txt');
    eq(1, $fresh()->run(['import', $dir, '--yes']));
    eq(1, $fresh()->run(['import', $r->dir . '/nothing-here', '--yes']));
});

test('4.18.8 bin/relay export: refuses an existing folder and a folder inside the web root', function () {
    Relay::sqliteOnly();
    $r = Relay::make();
    $dir = $r->dir . '/exists';
    mkdir($dir);
    [$c, , $e] = cli_run($r, 'export', $dir);
    eq(1, $c);
    contains('already exists', $e);
    [$c, , $e] = cli_run($r, 'export', dirname(__DIR__, 2) . '/public/exported-secrets');
    eq(1, $c);
    contains('web root', $e);
    ok(!file_exists(dirname(__DIR__, 2) . '/public/exported-secrets'));
    eq(2, cli_run($r, 'export')[0]);
});

test('4.18.6 bin/relay gc: runs a pass when due, says so when not, --force runs anyway', function () {
    $r = Relay::make();
    [$c, $o] = cli_run($r, 'gc');
    eq(0, $c);
    contains('gc: {', $o);
    [, $o] = cli_run($r, 'gc');
    contains('not due', $o);
    [, $o] = cli_run($r, 'gc', '--force', '--vacuum');
    contains('gc: {', $o);
    contains('vacuumed', $o);
});

// ------------------------------------------------------------------------------------------------ the script

test('4.18.8 bin/relay.php as a process: usage, exit codes, OAIY_RELAY_DATA, and no output of any secret', function () {
    $r = Relay::make();
    $run = function (array $args) use ($r): array {
        $p = proc_open(array_merge([PHP_BINARY], Server::phpFlags(), [dirname(__DIR__, 2) . '/bin/relay.php'], $args), [1 => ['pipe', 'w'], 2 => ['pipe', 'w']], $pipes, null, array_merge(getenv(), ['OAIY_RELAY_DATA' => $r->data]));
        $out = stream_get_contents($pipes[1]);
        $err = stream_get_contents($pipes[2]);
        return [proc_close($p), $out, $err];
    };
    [$c, $o] = $run([]);
    eq(2, $c);
    contains('usage: php bin/relay.php', $o);
    [$c, , $e] = $run(['frobnicate']);
    eq(2, $c);
    contains('unknown command', $e);
    [$c, $o, $e] = $run(['status']);
    eq([0, ''], [$c, $e]);
    contains('relay id', $o);
    [$c, $o] = $run(['help']);
    eq(0, $c);
    [$c, $o, $e] = $run(['key', 'provider']);
    eq(0, $c, $e);
    not_contains('oaiy://', $o . $e);
    [$c, , $e] = $run(['revoke', 'dev-' . str_repeat('Q', 22)]);
    eq(1, $c);
});

test('4.18.2 bin/relay.php refuses to run under a web server: nothing is printed and nothing is done', function () {
    $r = Relay::make();
    $srv = Server::start(inst_scratch(), ['prepend' => false]); // a copy of the package: never the working tree and its data/ folder
    $res = $srv->request('GET', '/bin/relay.php');
    eq('', $res['body']);
});
