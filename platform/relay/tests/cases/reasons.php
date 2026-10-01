<?php
declare(strict_types=1);

use Oaiy\Relay\ApiError;
use Oaiy\Relay\Auth;
use Oaiy\Relay\B64;
use Oaiy\Relay\Holds;
use Oaiy\Relay\Installer;
use OaiyTest\Ceremony;
use OaiyTest\Relay;

/**
 * Every reason code that debug.error_sites can put in the log, driven by a real request (or, where the answer is a 403, which the log does not
 * record, by the error the code throws): a 401 or a 404 that the client is told one thing about must be told apart in the log, and a reason
 * that nothing asserts can be renamed or merged with another without a test noticing (the review found eighteen of them could).
 * The ones asserted elsewhere: the four causes of a native 401 that cannot be authenticated and the locked token (db_errors.php), the causes
 * of a 404 on a pid (db_errors.php), the database's (db_errors.php: db_open, db_busy, db_error_<code>, db_error_not_transient and
 * exception_<class>), and the ones thrown inside a post's transaction (revoke_race.php).
 */

/** The error_site lines of the log. @return list<array<string,mixed>> */
function rs_sites(Relay $r): array
{
    $lines = @file($r->data . '/logs/relay.log', FILE_IGNORE_NEW_LINES | FILE_SKIP_EMPTY_LINES) ?: [];
    return array_values(array_filter(array_map(fn($l) => json_decode($l, true), $lines), fn($j) => is_array($j) && ($j['event'] ?? '') === 'error_site'));
}

/** Run $fire, and require that exactly one more error_site line came of it, with this status and reason. */
function rs_expect(Relay $r, callable $fire, int $status, string $reason, string $what): void
{
    $before = count(rs_sites($r));
    $res = $fire();
    eq($status, $res['status'], "$what: " . $res['body'] . $r->errorSites());
    $after = rs_sites($r);
    eq($before + 1, count($after), "$what: the request left one more line in the log");
    eq($reason, $after[count($after) - 1]['reason'] ?? null, "$what: the reason in the log: " . json_encode($after[count($after) - 1] ?? null));
}

/** The reason of the ApiError that $fn throws (a 403, which the log does not record). */
function rs_thrown(callable $fn, int $status, string $what): ?string
{
    try {
        $fn();
    } catch (ApiError $e) {
        eq($status, $e->status, "$what: the status of what was thrown");
        return $e->reason;
    }
    fail("$what: nothing was thrown");
}

test('4.18.3 the causes of a native 401 that come after the secret was right each have their reason: a revoked token, a revoked device, an expired token', function () {
    $r = Relay::make();
    $d = $r->desktop();
    $db = $r->ctx()->db;
    [$id] = Auth::parseToken($d->token);
    $call = fn() => $r->call($d, 'GET', '/v1/poll');
    $db->exec('UPDATE tokens SET revoked_at = ? WHERE id = ?', [Relay::T0, $id]);
    rs_expect($r, $call, 401, 'token_revoked', 'a token that was revoked');
    $db->exec('UPDATE tokens SET revoked_at = NULL WHERE id = ?', [$id]);
    $db->exec('UPDATE devices SET revoked_at = ? WHERE id = ?', [Relay::T0, $d->id]);
    rs_expect($r, $call, 401, 'device_revoked', 'a device that was revoked, its token as it was');
    $db->exec('UPDATE devices SET revoked_at = NULL WHERE id = ?', [$d->id]);
    $db->exec('UPDATE tokens SET not_after = ? WHERE id = ?', [Relay::T0 - 5, $id]);
    rs_expect($r, $call, 401, 'token_expired', 'a token that is past its end');
    $db->exec('UPDATE tokens SET not_after = NULL WHERE id = ?', [$id]);
    eq(200, $call()['status'], 'and all three put back, the token works');
});

test('4.18.3 the causes of a 401 on an admin route each have their reason: no bearer, a token that is not an admin token\'s shape, an id that is not the relay\'s, the right id with a wrong secret, a relay with no admin record; and a refusal that has no cause to name is "unspecified"', function () {
    $r = Relay::make();
    $go = fn(?string $bearer) => fn() => $r->call($bearer, 'GET', '/v1/admin/status');
    rs_expect($r, $go(null), 401, 'no_bearer', 'no bearer on an admin route');
    rs_expect($r, $go('oaiyadm1.this-is-not-an-admin-token'), 401, 'malformed_admin_token', 'a token that is not an admin token');
    rs_expect($r, $go(Installer::newAdminToken()['token']), 401, 'unknown_admin_id', 'a well formed admin token of another relay');
    $real = $r->adminToken();
    [$prefix, $id] = explode('.', $real);
    rs_expect($r, $go($prefix . '.' . $id . '.' . B64::enc(random_bytes(32))), 401, 'wrong_admin_secret', 'this relay\'s admin id with a wrong secret');
    eq(200, $go($real)()['status'], 'the real token is served');
    $file = $r->data . '/secrets/admin.json';
    rename($file, $file . '.away');
    try {
        rs_expect($r, $go($real), 401, 'admin_record_missing', 'a relay whose admin record is gone');
    } finally {
        rename($file . '.away', $file);
    }
    // A 404 that nothing in the code gave a reason for (a route that does not exist) says so, and does not borrow another's.
    rs_expect($r, fn() => $r->call($r->desktop(), 'GET', '/v1/there-is-no-such-route'), 404, 'unspecified', 'a route that is not there');
});
test('4.18.3 the causes of a 404 on a pairing that the log can tell apart: a pid of another desktop, an unknown pid on a desktop route, an app that config.apps no longer lists (read and answered), and a pid that went while the phone waited', function () {
    [$r, $d, $c] = pair_setup();
    $c->open();
    $other = $r->desktop('Other');
    rs_expect($r, fn() => $r->call($other, 'POST', '/v1/pair/' . $c->pid . '/burn'), 404, 'pid_of_another_desktop', 'another desktop acts on this desktop\'s pid');
    $nobody = Ceremony::random($r, $d);
    rs_expect($r, fn() => $r->call($d, 'POST', '/v1/pair/' . $nobody->pid . '/burn'), 404, 'pid_unknown', 'a desktop acts on a pid that is not there');
    // A phone that waits for a pid, which is burned in the instant after its wait began.
    $w = Ceremony::random($r, $d);
    $w->open();
    $burn = static function (string $kind, string $principal, string $file) use ($r, $w): void {
        if ($kind === 'pair') {
            $r->ctx()->db->exec("UPDATE pairings SET state = 'expired' WHERE pid = ?", [$w->pid]);
            Holds::$afterMarker = null;
        }
    };
    try {
        Holds::$afterMarker = $burn;
        rs_expect($r, fn() => $w->get(['wait' => '2']), 404, 'pid_gone_while_waiting', 'a pid burned while the phone waited');
    } finally {
        Holds::$afterMarker = null;
    }
    // config.apps narrowed after the rendezvous was made: the phone's read and the phone's response are the 404 of an unknown pid, with their own reason.
    $a = Ceremony::random($r, $d);
    $a->open();
    $r->configure(['apps' => ['an-app-that-is-not-aokie']]);
    rs_expect($r, fn() => $c->get(), 404, 'app_not_allowed', 'a read of a rendezvous of an app that is no longer listed');
    rs_expect($r, fn() => $a->answer(), 404, 'app_not_allowed', 'a response to a rendezvous of an app that is no longer listed');
});

test('4.18.3 every cause of a 401 on the compatibility routes has its reason in the log: no bearer, an admission that is invalid, a desktop or a phone that is not there or does not match, a device revoked, a desktop revoked', function () {
    [$k, $a, $b, $plug, $ta, $tb] = aok_pair();
    $db = $k->r->ctx()->db;
    $go = fn(?string $bearer) => fn() => aok_call($k, $bearer, 'GET', 'challenge');
    rs_expect($k->r, $go(null), 401, 'no_bearer', 'no bearer');
    rs_expect($k->r, $go('not-an-admission'), 401, 'admission_invalid_or_expired', 'a bearer that is no admission');
    $desk = $k->desk->id;
    $phone = $a->id;
    $db->exec("UPDATE devices SET role = 'provider' WHERE id = ?", [$desk]);
    rs_expect($k->r, $go($plug), 401, 'desktop_not_found', 'a plugin admission whose desktop is gone');
    $db->exec("UPDATE devices SET role = 'desktop' WHERE id = ?", [$desk]);
    $db->exec("UPDATE devices SET role = 'provider' WHERE id = ?", [$phone]);
    rs_expect($k->r, $go($ta), 401, 'phone_not_found', 'a phone admission whose phone is gone');
    $db->exec("UPDATE devices SET role = 'phone' WHERE id = ?", [$phone]);
    $thumb = (string)$db->val('SELECT thumbprint FROM devices WHERE id = ?', [$phone]);
    $db->exec("UPDATE devices SET thumbprint = 'another' WHERE id = ?", [$phone]);
    rs_expect($k->r, $go($ta), 401, 'phone_does_not_match_admission', 'a phone whose key is not the admission\'s');
    $db->exec('UPDATE devices SET thumbprint = ? WHERE id = ?', [$thumb, $phone]);
    $db->exec('UPDATE devices SET revoked_at = ? WHERE id = ?', [Relay::T0, $phone]);
    rs_expect($k->r, $go($ta), 401, 'device_revoked', 'a phone that was revoked');
    $db->exec('UPDATE devices SET revoked_at = NULL WHERE id = ?', [$phone]);
    $db->exec('UPDATE devices SET revoked_at = ? WHERE id = ?', [Relay::T0, $desk]);
    rs_expect($k->r, $go($ta), 401, 'desktop_revoked', 'a phone whose desktop was revoked');
    $db->exec('UPDATE devices SET revoked_at = NULL WHERE id = ?', [$desk]);
    eq(200, aok_call($k, $ta, 'GET', 'challenge')['status'], 'all of it put back, the phone is served');
});

test('4.18.3 the causes of a 403 on the compatibility routes carry their reasons (the log records no 403, so what is thrown is what is asserted): a phone the roster no longer lists, an app config.apps no longer lists', function () {
    [$k, $a, $b, $plug, $ta, $tb] = aok_pair();
    // (a roster push that drops a phone revokes it, 401; a roster that does not list a phone that was not revoked is what the 403 is, so the row is
    // written as a desktop that listed only another phone would have written it)
    $k->r->ctx()->db->exec('UPDATE roster SET thumbprints = ? WHERE desktop_dev = ? AND app_id = ?', [json_encode(['a-thumbprint-that-is-not-a-phone-of-this-desktop']), $k->desk->id, $k->app]);
    eq('phone_not_in_roster', rs_thrown(fn() => $k->facade($ta), 403, 'a phone that the roster does not list'));
    $k->r->configure(['apps' => ['an-app-that-is-not-aokie']]);
    eq('app_not_allowed', rs_thrown(fn() => $k->facade($tb), 403, 'an app that is no longer listed'));
});

test('4.18.3 the 410 of a pairing on a desktop route (not recorded by the log, so what is thrown is asserted) says whether the rendezvous was burned or ran out', function () {
    [$r, $d, $c] = pair_setup();
    $c->open([], 5);
    $p = new Oaiy\Relay\Principal($d->id, 'desktop', 'a-token-id', []);
    $owned = fn() => $r->ctx()->db->write(fn(Oaiy\Relay\Db $db) => Oaiy\Relay\Pairing::owned($db, $p, $c->pid, Oaiy\Relay\Clock::now()));
    ok(is_array($owned()), 'its own, open and in its life: the row');
    eq(200, $c->burn()['status']);
    eq('pid_ended', rs_thrown($owned, 410, 'a rendezvous the desktop burned'));
    $e = OaiyTest\Ceremony::random($r, $d);
    $e->open([], 5);
    $ownedE = fn() => $r->ctx()->db->write(fn(Oaiy\Relay\Db $db) => Oaiy\Relay\Pairing::owned($db, $p, $e->pid, Oaiy\Relay\Clock::now()));
    ok(is_array($ownedE()));
    OaiyTest\Tmp::setClock(Relay::T0 + 6);
    eq('pid_expired', rs_thrown($ownedE, 410, 'a rendezvous that ran out of its five seconds'));
});
test('4.18.3 an exception that is neither the relay\'s error nor the database\'s is a 500 whose reason is its class, on the ordinary routes and on the compatibility ones (a fault is made in the registry\'s test hook, which production never sets)', function () {
    [$k, $a, $b, $plug, $ta, $tb] = aok_pair();
    $r = $k->r;
    $d = $r->desktop('Poller');
    $boom = static function (string $kind, string $principal, string $file): void {
        throw new \LogicException('a fault the relay did not expect');
    };
    try {
        Holds::$afterMarker = $boom;
        rs_expect($r, fn() => $r->call($d, 'GET', '/v1/poll', null, ['wait' => '1']), 500, 'exception_LogicException', 'a poll that meets an unexpected exception');
        $before = count(rs_sites($r));
        $res = aok_call($k, $ta, 'GET', 'stream', null, ['since' => '0'], ['Accept' => 'text/event-stream']);
        eq(500, $res['status'], $res['body']);
        eq(['error' => true, 'code' => 'internal', 'message' => 'The relay hit an internal error.'], $res['json'], 'the Aokie shape, with nothing of the exception in it');
        $sites = rs_sites($r);
        eq($before + 1, count($sites));
        eq('exception_LogicException', $sites[count($sites) - 1]['reason'] ?? null, 'and its reason in the log');
        not_contains('a fault the relay did not expect', $res['body'], 'the message of the exception is not shown');
    } finally {
        Holds::$afterMarker = null;
    }
});

slow_test('4.18.5 MySQL and MariaDB: a migration that cannot get the migration lock another connection holds is a 503 unavailable with Retry-After 5 and the reason db_migrate_lock, after the 20 seconds the lock is waited for (the request that meets an old schema while another migrates)', function () {
    Relay::mysqlOnly();
    $r = Relay::make();
    $ctx = $r->ctx();
    $c = $ctx->db->config()->db();
    $other = new PDO((string)$c['dsn'], $c['user'], $c['pass'], [PDO::ATTR_ERRMODE => PDO::ERRMODE_EXCEPTION]);
    $other->query("SELECT GET_LOCK('oaiy-relay-migrate', 1)")->fetchAll();
    try {
        $t = microtime(true);
        $e = throws(fn() => $ctx->db->migrate(), ApiError::class);
        $el = microtime(true) - $t;
    } finally {
        $other->query("SELECT RELEASE_LOCK('oaiy-relay-migrate')")->fetchAll();
    }
    eq([503, 'unavailable', 5, 'db_migrate_lock'], [$e->status, $e->errorCode, $e->retryAfter, $e->reason]);
    ok($e->database, 'and it is marked as the database\'s state, which the compatibility routes answer as a 500');
    ok($el > 15.0, 'the migration lock was waited for: ' . round($el, 1) . ' s');
});