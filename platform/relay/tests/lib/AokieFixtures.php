<?php
declare(strict_types=1);

namespace OaiyTest;

use Oaiy\Relay\Admission;
use Oaiy\Relay\B64;
use Oaiy\Relay\Config;
use Oaiy\Relay\Crypto;
use Oaiy\Relay\Devices;
use Oaiy\Relay\Ice;
use Oaiy\Relay\Signals;
use Oaiy\Relay\Stream;

/**
 * The recorded fixtures for the Aokie compatibility routes (platform/protocol/relay/v1/fixtures/aokie/): what the real relay
 * answers to the admission requests and the compatibility routes, with the keys and ids of the protocol package's vectors, for
 * the contract tests that run the shipped plugin's and phone's decoders against them. `php tests/fixtures.php --write` records
 * them; `--check` verifies the committed ones without regenerating.
 *
 * Randomness in a recording: the admission id (jti), the connection id and nonce of a challenge. The bearer is signed with a
 * fixed test secret (the one of Appendix A4) so a reader can verify it. ice.json is fully deterministic. Frames, pages and
 * streams are recorded from the same relay run in order, so a page's frames are the ones just posted.
 */
final class AokieFixtures
{
    public const CLOCK = 1790000000;
    public const ADMISSION_SECRET_HEX = '0101010101010101010101010101010101010101010101010101010101010101';
    public const TURN_SECRET = '0123456789abcdef0123456789abcdef';
    public const BASE = '/v1/aokie-companion/relay/';

    /** JSON with floats that are whole numbers kept as 1.0 (a frame's number must come back as it went in). */
    public static function encode($v): string
    {
        $text = json_encode($v, JSON_UNESCAPED_SLASHES | JSON_UNESCAPED_UNICODE | JSON_PRETTY_PRINT | JSON_PRESERVE_ZERO_FRACTION | JSON_THROW_ON_ERROR);
        return preg_replace_callback('/^( +)/m', static fn(array $m): string => str_repeat(' ', intdiv(strlen($m[1]), 2)), $text) . "\n";
    }

    /** A relay with the vector's desktop, endpoint key and two phones, call features on, TURN and STUN configured. */
    private static function rig(): AokieRig
    {
        $v = Vectors::get('keys');
        $k = new AokieRig();
        $k->r = Relay::make([], ['public_url' => 'https://relay.example.com']);
        $k->r->configure([
            'call' => ['enabled' => true],
            'turn' => ['urls' => ['turn:turn.example.com:3478?transport=udp', 'turns:turn.example.com:5349?transport=tcp'], 'secret' => self::TURN_SECRET, 'ttl' => 600],
            'stun' => ['urls' => ['stun:stun.example.com:3478']],
        ]);
        file_put_contents($k->r->data . '/secrets/admission.hmac', B64::enc(hex2bin(self::ADMISSION_SECRET_HEX)) . "\n");
        $k->desk = PairKit::actorWithId($k->r, 'desktop', $v['ids']['desktopDevice'], hex2bin($v['ed25519Seeds']['host']), hex2bin($v['x25519Secrets']['host']));
        $k->epSeed = hex2bin($v['ed25519Seeds']['desktopEndpoint']);
        [$k->epPk] = Crypto::signKeypairFromSeed($k->epSeed);
        $k->rev = 7;
        $k->streamOk();
        $a = PairKit::actorWithId($k->r, 'phone', $v['ids']['phoneDevice'], hex2bin($v['ed25519Seeds']['phone']), hex2bin($v['x25519Secrets']['phone']),
            ['owner_desktop' => $k->desk->id, 'app_id' => 'aokie', 'peer_thumbprint' => $k->epThumb(), 'grants' => ['state_read', 'caller_read', 'captions_read', 'rtc_signal']]);
        $b = PairKit::actorWithId($k->r, 'phone', 'dev-' . B64::enc(str_repeat("\x08", 16)), str_repeat("\x08", 32), str_repeat("\x09", 32),
            ['owner_desktop' => $k->desk->id, 'app_id' => 'aokie', 'peer_thumbprint' => $k->epThumb(), 'grants' => ['state_read']]);
        $k->phones = [$a, $b];
        $k->pushRoster();
        return $k;
    }

    /** What a recording keeps of an answer: the status, the headers a client acts on, and the body with objects kept as objects. @return array<string,mixed> */
    private static function answer(array $res): array
    {
        $keep = [];
        foreach (['retry-after', 'www-authenticate', 'x-oaiy-hold', 'cache-control', 'pragma'] as $h) {
            if (isset($res['headers'][$h])) {
                $keep[$h] = $res['headers'][$h];
            }
        }
        $out = ['status' => $res['status']];
        if ($keep) {
            $out['headers'] = $keep;
        }
        $out['body'] = json_decode($res['body'], false, 512, JSON_THROW_ON_ERROR);
        return $out;
    }

    /** @return array<string,mixed> */
    private static function request(string $method, string $path, string $auth, $body = null, array $query = [], array $headers = []): array
    {
        $q = ['method' => $method, 'path' => $path];
        if ($query) {
            $q['query'] = $query;
        }
        $q['auth'] = $auth;
        if ($headers) {
            $q['headers'] = $headers;
        }
        if ($body !== null) {
            $q['body'] = is_string($body) ? json_decode($body, false, 512, JSON_THROW_ON_ERROR) : $body;
            if (is_string($body)) {
                $q['bodyText'] = $body; // exactly what was sent: the carriers embed their frames verbatim
            }
        }
        return $q;
    }

    /** @return array<string,mixed> */
    private static function head(string $title, string $about): array
    {
        return ['protocol' => 'oaiy-relay/1', 'title' => $title, 'about' => $about];
    }

    /** @return array<string,mixed> the relay every file was recorded against */
    private static function relayInfo(): array
    {
        return ['url' => 'https://relay.example.com', 'clock' => self::CLOCK, 'stunUrls' => ['stun:stun.example.com:3478'],
            'turnUrls' => ['turn:turn.example.com:3478?transport=udp', 'turns:turn.example.com:5349?transport=tcp'], 'turnSecret' => self::TURN_SECRET, 'turnTtl' => 600,
            'admissionSecretHex' => self::ADMISSION_SECRET_HEX];
    }

    /**
     * Record everything. @return array<string,array<string,mixed>> file name => document
     */
    public static function record(): array
    {
        $k = self::rig();
        [$a, $b] = $k->phones;
        $v = Vectors::get('keys');
        $files = [];

        // ---- admissions
        $cases = [];
        $plugin = function (string $name, array $req, string $note, string $path = '/v1/aokie-companion/admission') use ($k, &$cases): array {
            $res = $k->r->call($k->desk, 'POST', $path, $req);
            $cases[] = ['name' => $name, 'role' => 'plugin', 'note' => $note, 'request' => self::request('POST', $path, 'the desktop\'s device token', $req), 'response' => self::answer($res)];
            return $res;
        };
        $mobile = function (string $name, Actor $ph, array $req, string $note, string $path = '/v1/aokie-companion/admission') use ($k, &$cases): array {
            $res = $k->r->call($ph, 'POST', $path, $req);
            $cases[] = ['name' => $name, 'role' => 'mobile', 'note' => $note, 'request' => self::request('POST', $path, 'the phone\'s device token', $req), 'response' => self::answer($res)];
            return $res;
        };
        $pluginReq = $k->pluginRequest();
        $pa = $plugin('plugin, stream, STUN and TURN', $pluginReq, 'What the desktop\'s broker sends (upstream.rs) plus supportedTransports. The two phones are the roster, revision 7.');
        $plugin('plugin, poll mode', array_merge($pluginReq, ['supportedTransports' => ['relay-poll']]), 'A carrier that only polls: relay.mode is "poll".', '/v1/admission');
        $plugin('plugin, a carrier that offers both', array_merge($pluginReq, ['supportedTransports' => ['relay', 'relay-poll']]), 'The stream wins when the host flushes.');
        $ma = $mobile('phone A, stream, STUN and TURN', $a, $k->mobileRequest($a), 'What the phone sends (managed_auth.rs) plus supportedTransports.');
        $mb = $mobile('phone B, poll mode', $b, $k->mobileRequest($b, ['supportedTransports' => ['relay-poll']]), 'The second phone has the narrower grants (state_read only).', '/v1/admission');
        if ($pa['status'] !== 200 || $ma['status'] !== 200 || $mb['status'] !== 200) {
            throw new \RuntimeException('an admission failed: ' . $pa['body'] . $ma['body'] . $mb['body']);
        }
        $ptok = $pa['json']['accessToken'];
        $atok = $ma['json']['accessToken'];
        $btok = $mb['json']['accessToken'];
        // ICE variants: the same admission with the settings changed
        $k->r->configure(['turn' => ['urls' => [], 'secret' => null]]);
        $mobile('phone A, STUN only (no TURN configured)', $a, $k->mobileRequest($a), 'turnCredentialExpiresAt is null and iceServers holds the STUN entry alone.');
        $k->r->configure(['stun' => ['urls' => []]]);
        $plugin('plugin, no ICE server at all', $pluginReq, 'iceServers is an empty list.');
        $k->r->configure(['stun' => ['urls' => ['stun:stun.example.com:3478']], 'turn' => ['urls' => ['turn:turn.example.com:3478?transport=udp', 'turns:turn.example.com:5349?transport=tcp'], 'secret' => self::TURN_SECRET, 'ttl' => 600, 'relay_only' => true]]);
        $mobile('phone A, relay only', $a, $k->mobileRequest($a), 'relayOnly true: every route goes through TURN.');
        $k->r->configure(['turn' => ['relay_only' => false, 'ttl' => 3600]]);
        $mobile('phone A, the longest credential (turn.ttl 3600)', $a, $k->mobileRequest($a), 'expiresAt is an hour ahead; the decoders allow 24.');
        $k->r->configure(['turn' => ['ttl' => 600]]);
        $files['admission.json'] = self::head('Admissions recorded from the relay, for the plugin\'s and the phone\'s decoders',
            'The requests the desktop\'s broker and a phone send to POST /v1/aokie-companion/admission (or /v1/admission) and the relay\'s answers, with the vectors\' keys and ids. The accessToken is signed with the admission secret named in relay.admissionSecretHex (Appendix A4\'s test secret), so a reader can verify the claims against the answer. The jti and the times are the recording\'s. A Rust contract test feeds each response to AdmissionResponse (plugin) or AdmissionResponse (phone) and checks what the README says.')
            + ['relay' => self::relayInfo(), 'identities' => [
                'desktopDevice' => $k->desk->id, 'desktopEndpointThumbprint' => $k->epThumb(), 'phoneA' => $a->id, 'phoneAThumbprint' => $k->thumb($a), 'phoneB' => $b->id, 'phoneBThumbprint' => $k->thumb($b),
            ], 'cases' => $cases];

        // ---- challenges
        $ch = [];
        foreach ([['the plugin\'s challenge', $ptok, 'plugin'], ['phone A\'s challenge', $atok, 'phone A'], ['phone B\'s challenge (the admission of the poll-mode case)', $btok, 'phone B']] as [$name, $tok, $who]) {
            $res = $k->call($tok, 'GET', 'challenge');
            $ch[] = ['name' => $name, 'bearer' => $tok, 'request' => self::request('GET', self::BASE . 'challenge', 'Bearer of ' . $who), 'response' => self::answer($res)];
        }
        $files['challenge.json'] = self::head('Endpoint challenges recorded from the relay',
            'GET /v1/aokie-companion/relay/challenge with each admission bearer of admission.json. A plugin\'s challenge carries its roster and never expectedPeerKeyThumbprint; a phone\'s carries expectedPeerKeyThumbprint and no roster member at all (EndpointChallengeFrame refuses both). connectionId and challengeNonce are random; expiresAt is 25 seconds after the relay clock.')
            + ['relay' => self::relayInfo(), 'cases' => $ch];

        // ---- frames
        $steps = [];
        $step = function (string $label, string $who, array $req, array $res, ?string $note = null) use (&$steps): void {
            $s = ['step' => $label, 'from' => $who, 'request' => $req, 'response' => self::answer($res)];
            if ($note !== null) {
                $s['note'] = $note;
            }
            $steps[] = $s;
        };
        $thumbA = $k->thumb($a);
        $frames = ['{}', '{"type":"hello","n":9007199254740993,"f":1.0,"u":"héllo 😀/","e":{},"a":[],"z":[{}]}', '{"kind":"snapshot","grants":["state_read"],"seq":999}'];
        $post = '{"to":"plugin","frames":[' . implode(',', $frames) . ']}';
        $step('phone A posts three frames to the plugin', 'phone A', self::request('POST', self::BASE . 'frames', 'Bearer of phone A', $post), $k->call($atok, 'POST', 'frames', $post),
            'The frames are stored as decoded and encoded again: {} stays an object, 9007199254740993 an integer, 1.0 a float, slashes and Unicode as they are. The relay records who posted them; nothing in a frame can say so.');
        $step('the plugin reads them', 'plugin', self::request('GET', self::BASE . 'frames', 'Bearer of the plugin', null, ['since' => '0', 'wait' => '0']), $k->read($ptok, 0, 0),
            'from, subjectId and grants are the relay\'s record of phone A\'s admission, not the frame\'s own members (the third frame claims seq 999).');
        $step('the plugin finds the tail', 'plugin', self::request('GET', self::BASE . 'frames', 'Bearer of the plugin', null, ['since' => '3', 'wait' => '0']), $k->read($ptok, 3, 0),
            'How the plugin primes its cursor (tail_cursor reads lastSeq): with nothing after since, lastSeq is since.');
        $toA = '{"to":"mobile:' . $thumbA . '","frames":[{"kind":"plugin_idle"},{"kind":"lease_granted","deviceId":"' . $a->id . '"}]}';
        $step('the plugin posts two frames to phone A', 'plugin', self::request('POST', self::BASE . 'frames', 'Bearer of the plugin', $toA), $k->call($ptok, 'POST', 'frames', $toA));
        $step('phone A reads them', 'phone A', self::request('GET', self::BASE . 'frames', 'Bearer of phone A', null, ['since' => '0', 'wait' => '0']), $k->read($atok, 0, 0),
            'Reading does not acknowledge: the same read answers the same until the frames\' 120 seconds are up.');
        $step('phone A reads them again, from where it left off', 'phone A', self::request('GET', self::BASE . 'frames', 'Bearer of phone A', null, ['since' => '1', 'wait' => '0']), $k->read($atok, 1, 0));
        $step('phone B has nothing', 'phone B', self::request('GET', self::BASE . 'frames', 'Bearer of phone B', null, ['since' => '0', 'wait' => '0']), $k->read($btok, 0, 0), 'A mailbox is the party\'s own.');
        $gone = '{"to":"mobile:' . B64::enc(hash('sha256', 'a phone this desktop no longer has', true)) . '","frames":[{"kind":"plugin_idle"}]}';
        $step('the plugin posts to a phone the desktop no longer has', 'plugin', self::request('POST', self::BASE . 'frames', 'Bearer of the plugin', $gone), $k->call($ptok, 'POST', 'frames', $gone),
            'Answered like a delivered post (200, the same three members) and stored nowhere: the shipped plugin ends the session of every phone on a 403, so the removal of one phone must not look like an error. The same answer covers a revoked phone, one the roster dropped, one the plugin\'s admission does not list, and another desktop\'s.');
        $files['frames.json'] = self::head('Frames posted and read through the relay',
            'A conversation through POST and GET /v1/aokie-companion/relay/frames, in order, with the bearers of admission.json. Request bodies are what the shipped carriers send (relay_post_body embeds the frames as they are).')
            + ['relay' => self::relayInfo(), 'bearers' => ['plugin' => $ptok, 'phone A' => $atok, 'phone B' => $btok], 'steps' => $steps];

        // ---- streams
        $streams = [];
        $run = function (string $name, string $note, string $bearer, int $since, float $seconds, string $who) use ($k, &$streams): void {
            $f = $k->facade($bearer);
            $body = '';
            Stream::run($k->r->ctx(), $f, $since, $seconds, static function (string $bytes) use (&$body): bool {
                $body .= $bytes;
                return true;
            }, static fn(): bool => false);
            $streams[] = ['name' => $name, 'note' => $note, 'request' => self::request('GET', self::BASE . 'stream', 'Bearer of ' . $who, null, $since > 0 ? ['since' => (string)$since] : [], ['Accept' => 'text/event-stream']), 'body' => $body];
        };
        $run('the plugin\'s stream with three frames waiting', 'The preamble, one frame event per frame with id = seq, and end with the cursor. The stream lasts min(20, wait.max) seconds; this recording is cut at 0.6.', $ptok, 0, 0.6, 'the plugin');
        $run('resuming after the second', 'A carrier that saw event 2 opens with since=2 (the plugin does; Last-Event-ID does the same).', $ptok, 2, 0.6, 'the plugin');
        $run('an idle stream', 'Nothing to say: the preamble, a keepalive comment 2 seconds in (both carriers\' 45 second freshness timer is fed by it), and end.', $btok, 0, 2.3, 'phone B');
        $files['stream.json'] = self::head('The framed stream, recorded byte for byte',
            'The body of GET /v1/aokie-companion/relay/stream as the relay writes it (status 200, Content-Type text/event-stream; charset=utf-8, Cache-Control no-store, X-Accel-Buffering no). Feed each body to the carriers\' SseParser, whole and split at every offset.')
            + ['relay' => self::relayInfo(), 'cases' => $streams];

        // ---- errors
        $errs = [];
        $err = function (string $name, array $req, array $res, string $note = '') use (&$errs): void {
            $e = ['name' => $name, 'request' => $req, 'response' => self::answer($res)];
            if ($note !== '') {
                $e['note'] = $note;
            }
            $errs[] = $e;
        };
        $err('a bearer this relay never issued', self::request('GET', self::BASE . 'challenge', 'Bearer nonsense'), $k->call('nonsense', 'GET', 'challenge'), 'One 401 whatever was wrong (forged, expired, malformed, another kind of token).');
        $bad = '{"to":"mobile:' . $k->thumb($b) . '","frames":[{}]}';
        $err('a phone addressing a phone', self::request('POST', self::BASE . 'frames', 'Bearer of phone A', $bad), $k->call($atok, 'POST', 'frames', $bad));
        $big = '{"to":"plugin","frames":[{"p":"' . str_repeat('x', 196610) . '"}]}';
        $bigRes = $k->call($atok, 'POST', 'frames', $big);
        $bigReq = self::request('POST', self::BASE . 'frames', 'Bearer of phone A', null);
        $bigReq['bodyNote'] = 'one frame of 196,618 bytes once encoded (the cap is 196,608): {"to":"plugin","frames":[{"p":"' . str_repeat('x', 3) . '... 196,610 x characters ..."}]}';
        $err('a frame over the cap', $bigReq, $bigRes);
        $err('no frames', self::request('POST', self::BASE . 'frames', 'Bearer of phone A', '{"to":"plugin","frames":[]}'), $k->call($atok, 'POST', 'frames', '{"to":"plugin","frames":[]}'));
        $err('an admission for a transport this relay cannot serve', self::request('POST', '/v1/aokie-companion/admission', 'the phone\'s device token', $k->mobileRequest($a, ['supportedTransports' => ['websocket']])),
            $k->r->call($a, 'POST', '/v1/aokie-companion/admission', $k->mobileRequest($a, ['supportedTransports' => ['websocket']])));
        $wrong = $k->mobileRequest($a, ['deviceId' => $b->id]);
        $err('an admission naming another phone', self::request('POST', '/v1/aokie-companion/admission', 'the phone\'s device token', $wrong), $k->r->call($a, 'POST', '/v1/aokie-companion/admission', $wrong));
        $err('a wrong method', self::request('POST', self::BASE . 'challenge', 'Bearer of the plugin', '{}'), $k->call($ptok, 'POST', 'challenge', '{}'), 'Refused before any handler runs, and still in the Aokie shape.');
        $err('an unknown route under the prefix', self::request('GET', self::BASE . 'nothing', 'Bearer of the plugin'), $k->call($ptok, 'GET', 'nothing'));
        // 429: a mailbox that would overflow
        $k->r->configure(['limits' => ['sigItems' => 2]]);
        $two = '{"to":"plugin","frames":[{},{},{}]}';
        $err('a batch that would overflow the mailbox', self::request('POST', self::BASE . 'frames', 'Bearer of phone A', $two), $k->call($atok, 'POST', 'frames', $two), 'All or none: nothing was stored. Retry-After is a header only; the body has three members.');
        $k->r->configure(['limits' => ['sigItems' => 1024]]);
        // 503: a stream at the hard limit
        $eff = $k->r->ctx()->eff;
        for ($i = 0; $i < $eff->heldHard; $i++) {
            $dir = $k->r->data . '/holds/poll/' . Signals::hash('fixture-' . $i);
            @mkdir($dir, 0700, true);
            file_put_contents($dir . '/20.' . bin2hex(random_bytes(6)), '');
        }
        $err('a stream when the host has no worker to spare', self::request('GET', self::BASE . 'stream', 'Bearer of phone A'), $k->call($atok, 'GET', 'stream'), 'The pool is at its hard limit. The plugin maps 503 to "unavailable for this app" (a re-bootstrap), so this is a last resort.');
        $refused = $k->call($atok, 'GET', 'frames', null, ['since' => '0', 'wait' => '2']);
        $err('a frames wait when the host has no worker to spare (not an error)', self::request('GET', self::BASE . 'frames', 'Bearer of phone A', null, ['since' => '0', 'wait' => '2']), $refused, 'A refused wait is 200 with hold.refused: the carrier degrades to a short poll.');
        // 401 revoked and 403 roster
        $k->r->ctx()->db->exec('UPDATE roster SET thumbprints = ? WHERE desktop_dev = ?', [json_encode([$k->thumb($a)]), $k->desk->id]);
        $err('a phone the desktop\'s roster no longer lists', self::request('GET', self::BASE . 'challenge', 'Bearer of phone B'), $k->call($btok, 'GET', 'challenge'));
        Devices::revoke($k->r->ctx(), $b->id);
        $err('a phone that was removed', self::request('GET', self::BASE . 'challenge', 'Bearer of phone B'), $k->call($btok, 'GET', 'challenge'), 'Held streams and waits of that phone end the same way.');
        $k->r->configure(['call' => ['enabled' => false]]);
        $err('call features off', self::request('GET', self::BASE . 'challenge', 'Bearer of the plugin'), $k->call($ptok, 'GET', 'challenge'), 'The default: the whole family answers 403 feature_disabled.');
        $files['errors.json'] = self::head('Errors of the compatibility routes',
            'Every error has exactly three members, error (true), code and message: the phone\'s error decoder is deny_unknown_fields. A wait is the Retry-After header. The shape covers what fails before a handler runs as well.')
            + ['relay' => self::relayInfo(), 'cases' => $errs];

        // ---- ICE
        $files['ice.json'] = self::ice($v);
        return $files;
    }

    /** ICE and TURN minting for fixed inputs: fully deterministic, and including FormLogic's own known answers. @return array<string,mixed> */
    private static function ice(array $keys): array
    {
        $flSecret = 'turn-rest-test-secret-not-a-real-key-0123456789';
        $flTurn = ['turn:turn.example.com:3478?transport=udp', 'turns:turn.example.com:5349?transport=tcp'];
        $defs = [
            ['name' => 'FormLogic\'s phone case', 'note' => 'Known answers computed by FormLogic\'s own unit tests (Python hmac and openssl): AokieCompanionIceConfigurationTest::testAMobileAdmissionCarriesItsOwnMintedCredential.',
                'stun' => ['stun:turn.example.com:3478'], 'turn' => $flTurn, 'secret' => $flSecret, 'ttl' => 600, 'relayOnly' => false, 'role' => 'mobile', 'appId' => 'app_test', 'subjectId' => 'device_test', 'now' => 1784160000],
            ['name' => 'FormLogic\'s plugin case', 'note' => 'The same secret for the plugin: another id, another credential (testAPluginAdmissionGetsAnotherCredentialFromTheSameSecret).',
                'stun' => ['stun:turn.example.com:3478'], 'turn' => $flTurn, 'secret' => $flSecret, 'ttl' => 600, 'relayOnly' => false, 'role' => 'plugin', 'appId' => 'app_test', 'subjectId' => 'aokie', 'now' => 1784160000],
            ['name' => 'the relay\'s test phone', 'note' => 'Appendix A5\'s secret with the vectors\' phone.',
                'stun' => ['stun:stun.example.com:3478'], 'turn' => $flTurn, 'secret' => self::TURN_SECRET, 'ttl' => 600, 'relayOnly' => false, 'role' => 'mobile', 'appId' => 'aokie', 'subjectId' => $keys['ids']['phoneDevice'], 'now' => self::CLOCK],
            ['name' => 'the shortest credential', 'note' => 'turn.ttl 60: the decoders need more than 30 seconds.',
                'stun' => [], 'turn' => ['turn:turn.example.com:3478'], 'secret' => self::TURN_SECRET, 'ttl' => 60, 'relayOnly' => false, 'role' => 'mobile', 'appId' => 'aokie', 'subjectId' => $keys['ids']['phoneDevice'], 'now' => self::CLOCK],
            ['name' => 'the longest credential', 'note' => 'turn.ttl 3600: an hour, below the decoders\' 24 hours.',
                'stun' => ['stun:stun.example.com:3478'], 'turn' => ['turns:turn.example.com:5349?transport=tcp'], 'secret' => self::TURN_SECRET, 'ttl' => 3600, 'relayOnly' => false, 'role' => 'plugin', 'appId' => 'aokie', 'subjectId' => 'aokie', 'now' => self::CLOCK],
            ['name' => 'relay only', 'note' => 'relayOnly needs a TURN entry.',
                'stun' => ['stun:stun.example.com:3478'], 'turn' => $flTurn, 'secret' => self::TURN_SECRET, 'ttl' => 600, 'relayOnly' => true, 'role' => 'mobile', 'appId' => 'aokie', 'subjectId' => $keys['ids']['phoneDevice'], 'now' => self::CLOCK],
            ['name' => 'STUN only', 'note' => 'No TURN configured: turnCredentialExpiresAt is null.',
                'stun' => ['stun:stun.example.com:3478', 'stuns:stun.example.com:5349'], 'turn' => [], 'secret' => null, 'ttl' => 600, 'relayOnly' => false, 'role' => 'mobile', 'appId' => 'aokie', 'subjectId' => $keys['ids']['phoneDevice'], 'now' => self::CLOCK],
            ['name' => 'nothing configured', 'note' => 'iceServers is an empty list.',
                'stun' => [], 'turn' => [], 'secret' => null, 'ttl' => 600, 'relayOnly' => false, 'role' => 'plugin', 'appId' => 'aokie', 'subjectId' => 'aokie', 'now' => self::CLOCK],
        ];
        $cases = [];
        foreach ($defs as $d) {
            $cfg = Config::fromArray(['public_url' => 'https://relay.example.com', 'stun' => ['urls' => $d['stun']], 'turn' => ['urls' => $d['turn'], 'secret' => $d['secret'], 'ttl' => $d['ttl'], 'relay_only' => $d['relayOnly']]], '/tmp/x');
            $r = Ice::forAdmission($cfg, $d['role'], $d['appId'], $d['subjectId'], $d['now']);
            $cases[] = [
                'name' => $d['name'], 'note' => $d['note'],
                'config' => ['stunUrls' => $d['stun'], 'turnUrls' => $d['turn'], 'turnSecret' => $d['secret'], 'turnTtl' => $d['ttl'], 'relayOnly' => $d['relayOnly']],
                'endpoint' => ['role' => $d['role'], 'appId' => $d['appId'], 'subjectId' => $d['subjectId']], 'now' => $d['now'],
                'expected' => ['iceServers' => $r['servers'], 'relayOnly' => $r['relayOnly'], 'turnCredentialExpiresAt' => $r['expiresAt']],
            ];
        }
        return self::head('ICE and TURN credentials for fixed inputs',
            'What Ice::forAdmission returns for a configuration, an endpoint and a time, including FormLogic\'s own known answers (its unit tests computed them with Python and openssl): username = <expiry>:<first 32 hex characters of HMAC-SHA256(secret, "aokie-turn-id" NUL role NUL appId NUL subjectId)>, credential = base64(HMAC-SHA1(secret, username)). The verifier recomputes every value with hmac and hashlib alone.')
            + ['cases' => $cases];
    }

    /** The names of the recorded files. @return list<string> */
    public const FILES = ['admission.json', 'challenge.json', 'frames.json', 'stream.json', 'errors.json', 'ice.json'];

    public static function dir(): string
    {
        return Fixtures::dir() . '/aokie';
    }

    /** A recording as the check reads it: encoded the way it is written, and read back as arrays. @param array<string,mixed> $files @return array<string,mixed> */
    public static function roundTrip(array $files): array
    {
        $out = [];
        foreach ($files as $name => $doc) {
            $out[$name] = json_decode(self::encode($doc), true, 512, JSON_THROW_ON_ERROR);
        }
        return $out;
    }

    /** The committed files. @return array<string,mixed> */
    public static function load(): array
    {
        $out = [];
        foreach (self::FILES as $name) {
            $raw = @file_get_contents(self::dir() . '/' . $name);
            $out[$name] = is_string($raw) ? json_decode($raw, true) : null;
        }
        return $out;
    }

    /** Rules of a recording that do not depend on its random members. @param array<string,mixed> $files as arrays @return list<string> problems */
    public static function check(array $files): array
    {
        $bad = [];
        $secret = hex2bin(self::ADMISSION_SECRET_HEX);
        foreach (self::FILES as $f) {
            if (!isset($files[$f]) || !is_array($files[$f])) {
                $bad[] = "$f is missing";
            }
        }
        if ($bad) {
            return $bad;
        }
        foreach ($files['admission.json']['cases'] as $i => $c) {
            $res = $c['response'];
            if ($res['status'] === 200) {
                $claims = Admission::verify($secret, (string)$res['body']['accessToken'], self::CLOCK);
                if ($claims === null) {
                    $bad[] = "admission case $i: the bearer does not verify";
                } elseif ($claims['role'] !== ($c['role'] === 'plugin' ? 'plugin' : 'mobile') || $claims['appId'] !== $res['body']['appId'] || $claims['subjectId'] !== $res['body']['subjectId']) {
                    $bad[] = "admission case $i: the bearer's claims are not the answer's";
                }
            }
        }
        foreach ($files['challenge.json']['cases'] as $i => $c) {
            $claims = Admission::verify($secret, (string)$c['bearer'], self::CLOCK);
            if ($claims === null || ($c['response']['body']['admissionJti'] ?? null) !== $claims['jti']) {
                $bad[] = "challenge case $i: it is not built from its bearer";
            }
        }
        foreach ($files['stream.json']['cases'] as $i => $c) {
            $body = (string)$c['body'];
            if (strncmp($body, Stream::PREAMBLE, strlen(Stream::PREAMBLE)) !== 0 || preg_match('/id: \d+\nevent: end\ndata: \{\}\n\n$/D', $body) !== 1) {
                $bad[] = "stream case $i does not open with the preamble and close with end";
            }
        }
        foreach ($files['errors.json']['cases'] as $i => $c) {
            if ($c['response']['status'] >= 400 && array_keys($c['response']['body']) !== ['error', 'code', 'message']) {
                $bad[] = "error case $i does not have exactly error, code, message";
            }
        }
        // ice.json is deterministic: what the relay computes now is what the file says.
        foreach ($files['ice.json']['cases'] as $i => $c) {
            $cfg = Config::fromArray(['public_url' => 'https://relay.example.com', 'stun' => ['urls' => $c['config']['stunUrls']],
                'turn' => ['urls' => $c['config']['turnUrls'], 'secret' => $c['config']['turnSecret'], 'ttl' => $c['config']['turnTtl'], 'relay_only' => $c['config']['relayOnly']]], '/tmp/x');
            $r = Ice::forAdmission($cfg, $c['endpoint']['role'], $c['endpoint']['appId'], $c['endpoint']['subjectId'], $c['now']);
            if (json_encode($c['expected']['iceServers']) !== json_encode($r['servers']) || $c['expected']['turnCredentialExpiresAt'] !== $r['expiresAt']) {
                $bad[] = "ice case $i is not what the relay computes";
            }
        }
        return $bad;
    }
}
