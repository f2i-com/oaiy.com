<?php
declare(strict_types=1);

namespace Oaiy\Relay;

defined('OAIY_RELAY') or exit;

/**
 * The numbers actually in force: the configured ones narrowed by what the calibration measured on this host
 * (worker pool, largest body, longest hold). A measurement can only lower a limit, never raise it.
 */
final class Effective
{
    public int $workers;
    public bool $measured;
    public int $heldSoft;
    public int $heldHard;
    public int $waitMax;
    public bool $streamOk;
    public ?int $maxBody;
    public ?int $maxHold;
    public ?int $calibratedAt;
    /** @var array<string,int> */
    public array $laneBody = [];
    private Config $cfg;

    public function __construct(Config $cfg, ?int $workers, ?int $maxBody, ?int $maxHold, bool $streamOk, ?int $calibratedAt)
    {
        $this->cfg = $cfg;
        $this->measured = $workers !== null || $cfg->configuredWorkers() !== null;
        $this->workers = $workers ?? $cfg->configuredWorkers() ?? 5;
        $this->heldSoft = max(2, (int)floor(0.6 * $this->workers));
        $this->heldHard = max(3, $this->workers - 1);
        $this->maxBody = $maxBody;
        $this->maxHold = $maxHold;
        $this->streamOk = $streamOk;
        $this->calibratedAt = $calibratedAt;
        $this->waitMax = $cfg->waitMax();
        if ($maxHold !== null) {
            $this->waitMax = max(0, min($this->waitMax, $maxHold - 5));
        }
        foreach (array_keys(Lanes::TABLE) as $lane) {
            $b = $cfg->laneBody($lane);
            $this->laneBody[$lane] = $maxBody !== null ? min($b, $maxBody) : $b;
        }
    }

    public static function load(Config $cfg, Db $db): self
    {
        $rows = $db->all("SELECT k, v FROM meta WHERE k IN ('cal_workers', 'cal_max_body', 'cal_max_hold', 'cal_stream_ok', 'calibrated_at')");
        $m = [];
        foreach ($rows as $r) {
            $m[(string)$r['k']] = $r['v'] === null ? null : (int)$r['v'];
        }
        return new self(
            $cfg,
            $m['cal_workers'] ?? null,
            $m['cal_max_body'] ?? null,
            $m['cal_max_hold'] ?? null,
            ($m['cal_stream_ok'] ?? 0) === 1,
            $m['calibrated_at'] ?? null
        );
    }

    /** @return array{0:int,1:int,2:int} default, min, max ttl for a lane */
    public function ttl(string $lane): array
    {
        return $this->cfg->laneTtl($lane);
    }

    public function body(string $lane): int
    {
        return $this->laneBody[$lane];
    }
}
