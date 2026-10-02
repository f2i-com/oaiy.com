<?php
declare(strict_types=1);

namespace OaiyTest;

use Oaiy\Relay\B64;
use Oaiy\Relay\Clock;
use Oaiy\Relay\Crypto;
use Oaiy\Relay\Db;
use Oaiy\Relay\Grants;
use Oaiy\Relay\Info;
use Oaiy\Relay\Json;

/**
 * Both ends of a pairing v3 ceremony, written from the design's formulas (sections 4.10.2 and 4.10.3) and checked against
 * Appendix A3 by tests/cases/pairing.php: the desktop's offer and receipt, the phone's response, the secret's derivations,
 * the short authentication string and the typed code. The relay never does any of this (it never sees the secret); the tests
 * need it to drive the relay the way a desktop and a phone will.
 */
final class PairKit
{
    private const CROCKFORD = '0123456789ABCDEFGHJKMNPQRSTVWXYZ';
    public const SALT = 'oaiy/pairing/3';

    /** The canonical form of the pairing family: keys sorted bytewise, no whitespace, integers only. */
    public static function canonical($v): string
    {
        if (is_array($v)) {
            if (Json::isList($v)) {
                return '[' . implode(',', array_map([self::class, 'canonical'], $v)) . ']';
            }
            $keys = array_map('strval', array_keys($v));
            sort($keys, SORT_STRING);
            $parts = [];
            foreach ($keys as $k) {
                $parts[] = Json::encode($k) . ':' . self::canonical($v[$k]);
            }
            return '{' . implode(',', $parts) . '}';
        }
        if (is_float($v)) {
            throw new \InvalidArgumentException('a float is not canonical');
        }
        return Json::encode($v);
    }

    /** The bits of a byte string as '0' and '1' characters. */
    public static function bits(string $bin): string
    {
        $out = '';
        foreach (str_split($bin) as $c) {
            $out .= str_pad(decbin(ord($c)), 8, '0', STR_PAD_LEFT);
        }
        return $out;
    }

    /** Crockford base32 of a bit string, padded on the right with zero bits to a multiple of five. */
    public static function crockford(string $bits): string
    {
        $out = '';
        foreach (str_split($bits . str_repeat('0', (5 - strlen($bits) % 5) % 5), 5) as $five) {
            $out .= self::CROCKFORD[bindec($five)];
        }
        return $out;
    }

    /** @return array{pid:string,pidBin:string,macKey:string} */
    public static function derive(string $s): array
    {
        $pid = Crypto::hkdf($s, self::SALT, 'rendezvous', 16);
        return ['pid' => B64::enc($pid), 'pidBin' => $pid, 'macKey' => Crypto::hkdf($s, self::SALT, 'mac', 32)];
    }

    public static function sasCheck(string $sas12): string
    {
        return self::CROCKFORD[ord(hash('sha256', "oaiy/pairing/3/sas-check\0" . $sas12, true)[0]) >> 3];
    }

    /** The 13 character code the phone shows, as XXXX-XXXX-XXXX-C (4.10.2). */
    public static function sas(string $desktopEdPk, string $phoneEdPk, string $nonce, string $pidBin): string
    {
        $raw = Crypto::hkdf($desktopEdPk . $phoneEdPk, $nonce, "oaiy/pairing/3/sas\0" . $pidBin, 8);
        $twelve = self::crockford(substr(self::bits($raw), 0, 60)); // the top 60 bits of the 8 bytes
        return substr($twelve, 0, 4) . '-' . substr($twelve, 4, 4) . '-' . substr($twelve, 8, 4) . '-' . self::sasCheck($twelve);
    }

    /** The 28 character typed form of a secret: 26 characters of it (128 bits and two zero bits) and 2 of check, in seven groups. */
    public static function typed(string $s): string
    {
        $out = self::crockford(self::bits($s)); // 128 bits, padded by two zero bits to 26 characters
        $check = hash('sha256', "oaiy/pairing/3/typed\0" . $s, true);
        $out .= self::crockford(substr(self::bits(substr($check, 0, 2)), 0, 10)); // the first 10 bits of the hash
        return implode('-', str_split($out, 4));
    }

    /**
     * A device with a chosen id and keys, inserted directly (the id of Appendix A3 cannot come out of the relay's own id
     * generator). Same rows as Devices::create.
     * @param array<string,mixed> $opt owner_desktop, app_id, peer_thumbprint, grants
     */
    public static function actorWithId(Relay $r, string $role, string $id, string $edSeed, string $xSecret, array $opt = []): Actor
    {
        $ctx = $r->ctx();
        [$pk, $sk] = Crypto::signKeypairFromSeed($edSeed);
        $x = sodium_crypto_scalarmult_base($xSecret);
        [$token, $tokenId, $hash] = $ctx->auth->mint();
        $now = Clock::now();
        $ctx->db->write(function (Db $db) use ($role, $id, $pk, $x, $opt, $tokenId, $hash, $now): void {
            $db->insert('devices', [
                'id' => $id, 'role' => $role, 'name' => ucfirst($role), 'ver' => '', 'caps' => '[]', 'ed25519' => B64::enc($pk), 'x25519' => B64::enc($x),
                'thumbprint' => Crypto::thumbprint($pk), 'owner_desktop' => $opt['owner_desktop'] ?? null, 'app_id' => $opt['app_id'] ?? null,
                'peer_thumbprint' => $opt['peer_thumbprint'] ?? null, 'grants' => json_encode(array_values($opt['grants'] ?? [])), 'flags' => '{"canCmd":false}',
                'origins' => null, 'created_at' => $now, 'keys_changed_at' => $now, 'last_poll_at' => null, 'last_seen_at' => null, 'revoked_at' => null,
                'presence_changed_at' => null, 'push_kind' => null, 'push_token' => null,
            ]);
            $db->insert('tokens', ['id' => $tokenId, 'device_id' => $id, 'secret_hash' => $hash, 'created_at' => $now, 'not_after' => null, 'revoked_at' => null, 'last_used_at' => null, 'grace_until' => null]);
        });
        return new Actor($id, $role, $token, $pk, $sk, $x);
    }
}

/**
 * One ceremony: a desktop that opens a rendezvous and a phone that answers it, with every key and secret in one place so a
 * test can change exactly one thing.
 */
final class Ceremony
{
    public Relay $r;
    public Actor $desk;
    public string $app = 'aokie';
    public string $s;
    public string $pid;
    public string $pidBin;
    public string $macKey;
    public string $deskSeed;
    public string $deskPk;
    public string $phoneSeed;
    public string $phonePk;
    public string $phoneXSecret;
    public string $phoneXPk;
    public string $nonce;
    public string $jti;
    public int $issuedAt;
    /** @var array<string,mixed> */
    public array $offer;
    public string $offerText;
    public string $offerMac;
    public string $phoneName = 'Test phone';
    private bool $fromVector = false;

    /** A ceremony with fresh random secrets and keys. */
    public static function random(Relay $r, Actor $desk, string $app = 'aokie'): self
    {
        $c = new self();
        $c->r = $r;
        $c->desk = $desk;
        $c->app = $app;
        $c->s = random_bytes(16);
        $c->deskSeed = random_bytes(32);
        $c->phoneSeed = random_bytes(32);
        $c->phoneXSecret = random_bytes(32);
        $c->nonce = random_bytes(32);
        $c->jti = 'pair-' . bin2hex(random_bytes(8));
        $c->build();
        return $c;
    }

    /** The ceremony of Appendix A3, byte for byte; the desktop is given the device id the vector names. */
    public static function a3(Relay $r): self
    {
        $v = Vectors::get('A3');
        $k = Vectors::get('keys');
        $c = new self();
        $c->r = $r;
        $c->desk = PairKit::actorWithId($r, 'desktop', $k['ids']['desktopDevice'], hex2bin($k['ed25519Seeds']['host']), hex2bin($k['x25519Secrets']['host']));
        $c->app = $v['inputs']['offer']['appId'];
        $c->s = hex2bin($v['inputs']['secretHex']);
        $c->deskSeed = hex2bin($k['ed25519Seeds']['desktopEndpoint']);
        $c->phoneSeed = hex2bin($k['ed25519Seeds']['phone']);
        $c->phoneXSecret = hex2bin($k['x25519Secrets']['phone']);
        $c->nonce = hex2bin($v['inputs']['nonceHex']);
        $c->jti = $v['inputs']['offer']['jti'];
        $c->issuedAt = $v['inputs']['offer']['issuedAt'];
        $c->fromVector = true;
        $c->build(true);
        return $c;
    }

    private function build(bool $fromVector = false): void
    {
        $d = PairKit::derive($this->s);
        $this->pid = $d['pid'];
        $this->pidBin = $d['pidBin'];
        $this->macKey = $d['macKey'];
        [$this->deskPk] = Crypto::signKeypairFromSeed($this->deskSeed);
        [$this->phonePk] = Crypto::signKeypairFromSeed($this->phoneSeed);
        $this->phoneXPk = sodium_crypto_scalarmult_base($this->phoneXSecret);
        if ($fromVector) {
            $this->offer = Vectors::get('A3.inputs.offer');
        } else {
            $this->issuedAt = Clock::now();
            $hostPk = $this->desk->edPk;
            $this->offer = [
                'kind' => 'aokie_mobile_pairing', 'schemaVersion' => 3, 'appId' => $this->app, 'desktopConnectionId' => $this->desk->id, 'desktopName' => 'Front desk PC',
                'desktopEndpointKey' => ['algorithm' => 'ed25519', 'publicKey' => B64::enc($this->deskPk), 'thumbprint' => Crypto::thumbprint($this->deskPk)],
                'desktopX25519' => B64::enc(sodium_crypto_scalarmult_base(random_bytes(32))),
                'hostIdentity' => ['ed25519' => B64::enc($hostPk), 'thumbprint' => Crypto::thumbprint($hostPk), 'x25519' => B64::enc($this->desk->xPk)],
                'nonce' => B64::enc($this->nonce), 'jti' => $this->jti, 'issuedAt' => $this->issuedAt, 'expiresAt' => $this->issuedAt + 600,
                'relay' => ['url' => $this->r->publicUrl, 'fingerprint' => Crypto::thumbprint(Info::loadKeys($this->r->data)[0])],
            ];
        }
        $this->offerText = PairKit::canonical($this->offer);
        $this->offerMac = B64::enc(hash_hmac('sha256', "oaiy/pairing/3/offer-mac\0" . $this->offerText, $this->macKey, true));
    }

    public function deskThumb(): string
    {
        return Crypto::thumbprint($this->deskPk);
    }

    public function phoneThumb(): string
    {
        return Crypto::thumbprint($this->phonePk);
    }

    /** @return array<string,mixed> */
    public function createDoc(int $ttl = 600): array
    {
        return ['pid' => $this->pid, 'offer' => $this->offerText, 'mac' => $this->offerMac, 'ttl' => $ttl, 'appId' => $this->app, 'desktopThumbprint' => $this->deskThumb()];
    }

    /** POST /v1/pair as the desktop. @param array<string,mixed> $over @return array<string,mixed> */
    public function open(array $over = [], int $ttl = 600): array
    {
        return $this->r->call($this->desk, 'POST', '/v1/pair', array_merge($this->createDoc($ttl), $over));
    }

    /** @return array<string,mixed> the claims of the phone's response */
    public function claims(): array
    {
        if ($this->fromVector) {
            return Vectors::get('A3.inputs.claims');
        }
        $v = $this->offer;
        return [
            'appId' => $this->app, 'desktopConnectionId' => $v['desktopConnectionId'], 'desktopKeyThumbprint' => $this->deskThumb(),
            'deviceId' => 'dev-' . B64::enc(substr(hash('sha256', $this->phonePk, true), 0, 16)), 'displayName' => $this->phoneName,
            'mobileEndpointKey' => ['algorithm' => 'ed25519', 'publicKey' => B64::enc($this->phonePk), 'thumbprint' => $this->phoneThumb()],
            'mobileX25519' => B64::enc($this->phoneXPk), 'pairingNonce' => $v['nonce'], 'jti' => $v['jti'],
            'issuedAt' => $this->issuedAt + 30, 'expiresAt' => $this->issuedAt + 150,
        ];
    }

    /** The response text the phone posts: the claims, the phone's signature over them and the MAC under the pairing secret. */
    public function responseText(?array $claims = null): string
    {
        $claims ??= $this->claims();
        $canon = PairKit::canonical($claims);
        $sig = B64::enc(Crypto::sign(Crypto::signKeypairFromSeed($this->phoneSeed)[1], "oaiy/pairing/3/response\0" . $canon));
        $mac = B64::enc(hash_hmac('sha256', "oaiy/pairing/3/response-mac\0" . $canon, $this->macKey, true));
        return '{"kind":"aokie_mobile_pairing_response","schemaVersion":3,"claims":' . $canon . ',"signature":"' . $sig . '","mac":"' . $mac . '"}';
    }

    /**
     * A client address of its own for each phone request, so that a test that makes many of them is not stopped by the
     * per-address bucket by accident (the tests of that bucket pass their own address).
     * @return array<string,string>
     */
    private static function addr(): array
    {
        static $n = 0;
        $n++;
        return ['REMOTE_ADDR' => '198.18.' . (intdiv($n, 250) % 250) . '.' . ($n % 250 + 1)];
    }

    /** POST /v1/pair/{pid}/response as the phone. @return array<string,mixed> */
    public function answer(?string $text = null, array $server = []): array
    {
        return $this->r->call(null, 'POST', '/v1/pair/' . $this->pid . '/response', ['response' => $text ?? $this->responseText()], [], [], $server ?: self::addr());
    }

    /** GET /v1/pair/{pid} as the phone. @param array<string,mixed> $query @return array<string,mixed> */
    public function get(array $query = [], array $server = []): array
    {
        return $this->r->call(null, 'GET', '/v1/pair/' . $this->pid, null, $query, [], $server ?: self::addr());
    }

    /** The desktop's receipt: its endpoint key's signature over the approval (4.10.2). */
    public function receipt(array $grants, ?int $issuedAt = null): array
    {
        $issuedAt ??= $this->issuedAt + 40;
        $text = \Oaiy\Relay\Pairing::receiptText($this->app, $grants, $issuedAt, $this->phoneThumb(), $this->pid);
        return ['issuedAt' => $issuedAt, 'signature' => B64::enc(Crypto::sign(Crypto::signKeypairFromSeed($this->deskSeed)[1], "oaiy/pairing/3/approval\0" . $text))];
    }

    /**
     * What the phone does with the receipt it reads from GET /v1/pair/{pid} (section 4.10.2, Interpretation 60), from what it can see: its own
     * app, the pid, its own thumbprint, and the desktop key it pinned from the offer. It never sees the decision, so the grants are the receipt's
     * own, and must be sorted, without repeats, all of the fourteen names it knows, and the ones the signature covers.
     * Returns why it refuses the receipt, or null when it accepts it.
     * @param array<string,mixed> $receipt
     */
    public function phoneRefusesReceipt(array $receipt): ?string
    {
        $g = $receipt['grants'] ?? null;
        if (!is_array($g) || !Json::isList($g)) {
            return 'no grants';
        }
        foreach ($g as $name) {
            if (!Grants::isKnown($name)) {
                return 'a grant it does not know';
            }
        }
        $sorted = $g;
        sort($sorted, SORT_STRING);
        if ($sorted !== $g) {
            return 'grants that are not sorted';
        }
        if (count(array_unique($g)) !== count($g)) {
            return 'a grant twice';
        }
        $sig = is_string($receipt['signature'] ?? null) ? B64::decN($receipt['signature'], 64) : null;
        if (!is_int($receipt['issuedAt'] ?? null) || $sig === null) {
            return 'a receipt that is not shaped';
        }
        $text = \Oaiy\Relay\Pairing::receiptText($this->app, $g, $receipt['issuedAt'], $this->phoneThumb(), $this->pid);
        return Crypto::verify($this->deskPk, "oaiy/pairing/3/approval\0" . $text, $sig) ? null : 'a signature that does not verify over these grants';
    }

    /** @return array<string,mixed> */
    public function decisionDoc(array $grants = Grants::DEFAULT, ?int $issuedAt = null): array
    {
        return [
            'approve' => true,
            'phone' => ['ed25519' => B64::enc($this->phonePk), 'x25519' => B64::enc($this->phoneXPk), 'thumbprint' => $this->phoneThumb()],
            'name' => $this->phoneName, 'appId' => $this->app, 'grants' => $grants, 'receipt' => $this->receipt($grants, $issuedAt),
        ];
    }

    /** POST /v1/pair/{pid}/decision as the desktop. @param array<string,mixed>|null $doc @return array<string,mixed> */
    public function decide(?array $doc = null): array
    {
        return $this->r->call($this->desk, 'POST', '/v1/pair/' . $this->pid . '/decision', $doc ?? $this->decisionDoc());
    }

    public function reject(?string $reason = 'mac mismatch'): array
    {
        return $this->r->call($this->desk, 'POST', '/v1/pair/' . $this->pid . '/reject', $reason === null ? null : ['reason' => $reason]);
    }

    public function burn(): array
    {
        return $this->r->call($this->desk, 'POST', '/v1/pair/' . $this->pid . '/burn');
    }

    /** The 13 character code the phone shows. */
    public function sas(): string
    {
        return PairKit::sas($this->deskPk, $this->phonePk, $this->nonce, $this->pidBin);
    }

    /** Open the sealed token as the phone would: with its X25519 key pair. Null when it does not open. */
    public function openToken(string $sealedB64u): ?string
    {
        $bin = B64::dec($sealedB64u);
        if ($bin === null) {
            return null;
        }
        $kp = sodium_crypto_box_keypair_from_secretkey_and_publickey($this->phoneXSecret, $this->phoneXPk);
        $open = sodium_crypto_box_seal_open($bin, $kp);
        return $open === false ? null : $open;
    }

    /** Run the ceremony up to the phone reading "approved". @return array<string,mixed> the phone's GET answer */
    public function complete(array $grants = Grants::DEFAULT): array
    {
        $o = $this->open();
        if ($o['status'] !== 201) {
            throw new \RuntimeException('open: ' . $o['body']);
        }
        $a = $this->answer();
        if ($a['status'] !== 202) {
            throw new \RuntimeException('answer: ' . $a['body']);
        }
        $d = $this->decide($this->decisionDoc($grants));
        if ($d['status'] !== 200) {
            throw new \RuntimeException('decide: ' . $d['body']);
        }
        return $this->get();
    }
}
