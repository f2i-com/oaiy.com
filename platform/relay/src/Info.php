<?php
declare(strict_types=1);

namespace Oaiy\Relay;

defined('OAIY_RELAY') or exit;

/**
 * GET /v1/info (section 4.8): the static signed capability document and the interactive identity proof.
 *
 * The body has no time member, so it is static and its signature can be cached; the signature proves only that the
 * bytes were once signed by the relay key, and anyone can copy body and signature, so it says nothing about who is
 * answering now. Identity is the interactive proof: a request that carries X-OAIY-Nonce is answered with a signature
 * over the nonce, the body's hash and the time header, made by the relay key. A copied body and static signature
 * served with a new nonce cannot produce it.
 */
final class Info
{
    public const VERSION_FILE = 'VERSION';
    public const DOMAIN_SIG = "oaiy/relay/1/info\0";
    public const DOMAIN_PROOF = "oaiy/relay/1/info-proof\0";

    private Config $cfg;
    private Effective $eff;
    private string $relayId;
    private string $pk;
    private string $sk;

    public function __construct(Config $cfg, Effective $eff, string $relayId, string $publicKey, string $secretKey)
    {
        $this->cfg = $cfg;
        $this->eff = $eff;
        $this->relayId = $relayId;
        $this->pk = $publicKey;
        $this->sk = $secretKey;
    }

    /** @return array{0:string,1:string} [public key, secret key] from data/secrets/relay.key (the b64u of the 32 byte seed) */
    public static function loadKeys(string $dataDir): array
    {
        $raw = @file_get_contents(Paths::secretsDir($dataDir) . '/relay.key');
        $seed = is_string($raw) ? B64::decN(trim($raw), 32) : null;
        if ($seed === null) {
            throw new \RuntimeException('relay.key is missing or damaged');
        }
        $keys = Crypto::signKeypairFromSeed($seed);
        sodium_memzero($seed); // the seed is only the secret key in another form
        return $keys;
    }

    public static function softwareVersion(): string
    {
        $v = @file_get_contents(Paths::root() . '/' . self::VERSION_FILE);
        $v = is_string($v) ? trim($v) : '';
        return preg_match('/^[0-9A-Za-z.+-]{1,32}$/D', $v) === 1 ? $v : '0.0.0';
    }

    /** @return list<string> the features this build implements and this relay has enabled */
    public function features(): array
    {
        $f = ['poll', 'items', 'presence', 'methods.post-forms'];
        return $f;
    }

    /** @return array<string,mixed> */
    public function document(): array
    {
        $lanes = [];
        foreach (Lanes::advertised($this->cfg->callEnabled()) as $lane) {
            [$def, $min, $max] = $this->eff->ttl($lane);
            $lanes[$lane] = ['body' => $this->eff->body($lane), 'ttl' => ['default' => $def, 'min' => $min, 'max' => $max]];
        }
        $c = $this->cfg->toArray();
        return [
            'protocol' => 'oaiy-relay/1',
            'minClient' => 1,
            'relayId' => $this->relayId,
            'relayKey' => ['algorithm' => 'ed25519', 'publicKey' => B64::enc($this->pk), 'thumbprint' => Crypto::thumbprint($this->pk)],
            'software' => ['name' => 'oaiy-relay', 'version' => self::softwareVersion()],
            'features' => $this->features(),
            'wait' => ['default' => min(20, $this->eff->waitMax), 'max' => $this->eff->waitMax, 'pollGapMs' => $this->cfg->gapMs(), 'fallbackS' => $this->cfg->fallbackS()],
            'presenceWindow' => $this->cfg->presenceWindow(),
            'limits' => [
                'batchItems' => $this->cfg->limit('batchItems'),
                'batchBytes' => $this->cfg->limit('batchBytes'),
                'mailboxItems' => $this->cfg->limit('mailboxItems'),
                'mailboxBytes' => $this->cfg->limit('mailboxBytes'),
                'bulkShare' => $this->cfg->limit('bulkShare'),
                'sigItems' => $this->cfg->limit('sigItems'),
                'sigSenderShare' => $this->cfg->limit('sigSenderShare'),
                'lookupWait' => $this->cfg->limit('lookupWait'),
                'lookupHeld' => $this->cfg->limit('lookupHeld'),
                'hdrBytes' => $this->cfg->limit('hdrBytes'),
                'slotBytes' => $this->cfg->limit('slotBytes'),
                'rosterMax' => $this->cfg->limit('rosterMax'),
                'desktops' => $this->cfg->limit('desktops'),
                'held' => ['soft' => $this->eff->heldSoft, 'hard' => $this->eff->heldHard, 'measured' => $this->eff->measured],
                'lanes' => $lanes,
            ],
            'turn' => $c['turn']['urls'] !== [],
            'cors' => $this->cfg->corsExtraOrigins(),
        ];
    }

    public function body(): string
    {
        return Json::encode($this->document());
    }

    /** X-OAIY-Sig: Ed25519("oaiy/relay/1/info" || 0x00 || body). */
    public static function staticSignature(string $secretKey, string $body): string
    {
        return B64::enc(Crypto::sign($secretKey, self::DOMAIN_SIG . $body));
    }

    /** X-OAIY-Proof: Ed25519("oaiy/relay/1/info-proof" || 0x00 || nonce || SHA-256(body) || decimal time). */
    public static function proof(string $secretKey, string $nonce, string $body, int $time): string
    {
        return B64::enc(Crypto::sign($secretKey, self::DOMAIN_PROOF . $nonce . hash('sha256', $body, true) . (string)$time));
    }

    public function sign(string $body): string
    {
        return self::staticSignature($this->sk, $body);
    }

    public function proveNonce(string $nonce, string $body, int $time): string
    {
        return self::proof($this->sk, $nonce, $body, $time);
    }

    /** A strong ETag over the body. */
    public static function etag(string $body): string
    {
        return '"' . B64::enc(substr(hash('sha256', $body, true), 0, 16)) . '"';
    }

    /** The nonce of a request: 16 to 32 bytes of canonical b64u, else null. */
    public static function parseNonce(string $h): ?string
    {
        $n = B64::dec($h);
        return $n !== null && strlen($n) >= 16 && strlen($n) <= 32 ? $n : null;
    }
}
