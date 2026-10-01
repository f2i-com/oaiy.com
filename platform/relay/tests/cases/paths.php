<?php
declare(strict_types=1);

use Oaiy\Relay\AddressHolds;
use Oaiy\Relay\ApiError;
use Oaiy\Relay\Cli;
use Oaiy\Relay\Doctor;
use Oaiy\Relay\Fs;
use Oaiy\Relay\Holds;
use OaiyTest\Relay;
use OaiyTest\Tmp;

/**
 * The relay's own code reads folders of marker files (the hold registry, the address holds, the signal files, the secrets of an export), and
 * it did it with glob(), which takes `[`, `]`, `{`, `}`, `?` and `*` in the PATH it is given for pattern syntax. A data folder that is, or sits
 * under, a folder named `a[b]` matched nothing: every count of held requests read zero and every limit on how many workers a credential may
 * pin let everything through (a fourth poll was accepted, on Windows and on Linux), and an export copied no secrets. Folders are read with a
 * plain listing now (Fs::entries). These tests put the data folder in a path made of every character that is syntax to a glob, a shell or a
 * URL, and check each place that lists a folder.
 */

/** Folder names a data path may hold, with what the filesystem of the host will not take left out. @return array<string,string> label => folder */
function paths_names(): array
{
    $names = [
        'brackets' => 'a[b]', 'empty brackets' => 'a[]b', 'a lone [' => 'x[y', 'a lone ]' => 'x]y', 'braces' => '{x,y}', 'a lone {' => 'x{y', 'a lone }' => 'x}y',
        'a space' => 'my relay', 'quotes' => "it's", 'hash' => 'a#b', 'percent' => 'a%20b', 'ampersand' => 'a&b', 'bang' => 'a!b', 'parens' => 'a(b)', 'commas' => 'a,b;c',
        'equals and plus' => 'a=b+c', 'at and dollar' => 'a@b$c', 'caret and tilde' => 'a^b~c', 'backtick' => 'a`b', 'an accent' => "caf\u{00E9}", 'a dot folder' => 'a.b.c',
        'a mix' => '[{a b}] (x)',
    ];
    if (DIRECTORY_SEPARATOR === '/') {
        $names += ['question mark' => 'a?b', 'asterisk' => 'a*b', 'every glob character' => 'a?[*]{b}'];
    }
    return $names;
}

foreach (paths_names() as $label => $folder) {
    test("4.7.2 a data folder whose path holds $label ($folder): the hold counts see their markers and bound a credential's polls, the address holds bound an address, the signal files are collected, the doctor is satisfied", function () use ($folder, $label) {
        $r = Relay::make(['wait' => ['max' => 8], 'capacity' => ['workers' => 20]], [], $folder);
        ok(is_dir($r->data), 'the relay was provisioned in ' . $r->data);
        $d = $r->desktop();
        $phone = $r->phone($d);
        $holds = $r->ctx()->holds;
        // The registry: three polls of one credential are three, the fourth is refused, and the count of everything agrees.
        $made = [];
        for ($i = 1; $i <= 3; $i++) {
            $made[] = $holds->acquire('poll', $phone->id, 'edge', 8, 0, Holds::POLL_INFLIGHT_MAX);
            eq($i, $holds->inFlight('poll', $phone->id), "in flight after $i ($label)");
        }
        eq(3, $holds->liveCount(), "live markers ($label)");
        eq(['poll' => 3] + array_fill_keys(['lookup', 'rbx', 'pair', 'stream', 'admin'], 0), $holds->byKind(), 'by kind');
        $e = null;
        try {
            $holds->acquire('poll', $phone->id, 'edge', 8, 0, Holds::POLL_INFLIGHT_MAX);
        } catch (ApiError $x) {
            $e = $x;
        }
        ok($e !== null && $e->status === 429, "the fourth poll of one credential is refused ($label)");
        // The same through a whole request: with the three markers in place a fourth poll that waits is 429 at once.
        $res = $r->call($phone, 'GET', '/v1/poll', null, ['wait' => '4']);
        eq(429, $res['status'], "a request for a fourth poll: " . $res['body']);
        foreach ($made as $h) {
            $h->release();
        }
        eq(0, $holds->liveCount());
        // The address holds: four waits of one address, the fifth refused.
        $held = [];
        for ($i = 1; $i <= 4; $i++) {
            $held[] = AddressHolds::acquire($r->data, 'pair', '203.0.113.9', 20, 4);
        }
        eq(4, AddressHolds::count($r->data, 'pair', '203.0.113.9'), "four address holds counted ($label)");
        $e = null;
        try {
            AddressHolds::acquire($r->data, 'pair', '203.0.113.9', 20, 4);
        } catch (ApiError $x) {
            $e = $x;
        }
        ok($e !== null && $e->status === 429, "the fifth wait of one address is refused ($label)");
        foreach ($held as $h) {
            $h->release();
        }
        // A stale marker is collected, a fresh one is not (the listing finds both).
        $stale = $r->data . '/holds/addr-pair/' . Oaiy\Relay\Signals::hash('198.51.100.1');
        mkdir($stale, 0700, true);
        file_put_contents($stale . '/20.' . bin2hex(random_bytes(6)), '');
        touch(glob_free_first($stale), time() - 600);
        eq(1, AddressHolds::collect($r->data), "a stale address hold is collected ($label)");
        // The signal files: an old generation file and an old wake temporary go.
        $signals = $r->ctx()->signals;
        @mkdir($r->data . '/holds/gen', 0700, true);
        @mkdir($r->data . '/wake', 0700, true);
        file_put_contents($r->data . '/holds/gen/' . str_repeat('a', 64), 'x');
        touch($r->data . '/holds/gen/' . str_repeat('a', 64), time() - 7200);
        file_put_contents($r->data . '/wake/shard.tmp', 'x');
        touch($r->data . '/wake/shard.tmp', time() - 600);
        ok($signals->collect() >= 2, "an old generation file and an old wake temporary are collected ($label)");
        ok(!is_file($r->data . '/holds/gen/' . str_repeat('a', 64)) && !is_file($r->data . '/wake/shard.tmp'));
        // The doctor makes a marker, counts it and takes it away: and says so.
        $rows = Doctor::holdAccounting($r->data);
        eq('ok', $rows[0]['level'], $rows[0]['message']);
    });
}

/** The one marker file in a folder (a test needs its path to age it). */
function glob_free_first(string $dir): string
{
    $all = Fs::entries($dir);
    return $all[0];
}

test('4.7.2 Fs::entries is a listing and not a pattern: names are matched as plain strings, hidden names are left out, a prefix, a suffix and folders-only filters work, and a folder that is not there is an empty list', function () {
    $dir = Tmp::dir('entries') . '/we[ir]d {dir}';
    mkdir($dir . '/sub', 0700, true);
    foreach (['b.tmp', 'a.tmp', 'shard.1.tmp', 'shard..tmp', 'shard.tmp', '.hidden', 'plain', 'pre[fix]one', 'pre[fix]two'] as $f) {
        file_put_contents($dir . '/' . $f, '');
    }
    $names = fn(array $l): array => array_map('basename', $l);
    eq(['a.tmp', 'b.tmp', 'plain', 'pre[fix]one', 'pre[fix]two', 'shard..tmp', 'shard.1.tmp', 'shard.tmp', 'sub'], $names(Fs::entries($dir)), 'all but the hidden one, sorted');
    eq(['sub'], $names(Fs::entries($dir, '', '', true)), 'folders only');
    eq(['a.tmp', 'b.tmp', 'plain', 'pre[fix]one', 'pre[fix]two', 'shard..tmp', 'shard.1.tmp', 'shard.tmp'], $names(Fs::entries($dir, '', '', false)), 'files only');
    eq(['pre[fix]one', 'pre[fix]two'], $names(Fs::entries($dir, 'pre[fix]')), 'a prefix with brackets is a plain string');
    eq(['shard..tmp', 'shard.1.tmp'], $names(Fs::entries($dir, 'shard.', '.tmp')), 'prefix and suffix must not overlap: shard.tmp is not "shard." and ".tmp"');
    eq(['a.tmp', 'b.tmp', 'shard..tmp', 'shard.1.tmp', 'shard.tmp'], $names(Fs::entries($dir, '', '.tmp')));
    eq(['.hidden'], $names(Fs::entries($dir, '.hidden')), 'a hidden name is found when asked for by name');
    eq([], Fs::entries($dir . '/not there'));
    eq([], Fs::entries($dir . '/plain'), 'a file is not a folder');
});

test('4.18.8 bin/relay export copies the secrets of a data folder whose path holds brackets, and an import of it works: the listing of secrets/ was a glob, and an export from such a folder had no keys in it', function () {
    Relay::sqliteOnly();
    $r = Relay::make([], [], 'a[b] {c}');
    $d = $r->desktop();
    $dir = $r->dir . '/export';
    $out = '';
    $err = '';
    $cli = new Cli($r->data, function (string $s) use (&$out): void {
        $out .= $s;
    }, function (string $s) use (&$err): void {
        $err .= $s;
    });
    eq(0, $cli->run(['export', $dir]), $out . $err);
    foreach (['relay.sqlite', 'config.json', 'MANIFEST.txt', 'secrets/relay.key', 'secrets/admin.json', 'secrets/admission.hmac'] as $f) {
        ok(is_file($dir . '/' . $f), "$f is in the export");
    }
    contains('secrets/relay.key', (string)file_get_contents($dir . '/MANIFEST.txt'));
    $fresh = Tmp::dir('newhost') . '/n[ew]/data';
    $out2 = '';
    $err2 = '';
    $new = new Cli($fresh, function (string $s) use (&$out2): void {
        $out2 .= $s;
    }, function (string $s) use (&$err2): void {
        $err2 .= $s;
    });
    eq(0, $new->run(['import', $dir, '--yes']), $out2 . $err2);
    ok(is_file($fresh . '/secrets/relay.key') && is_file($fresh . '/secrets/admission.hmac'), 'the keys arrived');
});

test('4.7.2 the doctor fails a data folder whose hold counts do not see their own markers: a count that reads zero would let every limit through, and nothing else says so', function () {
    $r = Relay::make();
    eq('ok', Doctor::holdAccounting($r->data)[0]['level']);
    // A data folder that cannot hold a marker (holds/ is a file, so no folder can be made in it): a FAIL that says why.
    $broken = Tmp::dir('broken') . '/data';
    mkdir($broken, 0700, true);
    file_put_contents($broken . '/holds', 'not a folder');
    $rows = Doctor::holdAccounting($broken);
    eq('fail', $rows[0]['level'], json_encode($rows));
    contains('hold marker could not be made', $rows[0]['message']);
});
