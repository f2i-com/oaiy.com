<?php
declare(strict_types=1);

namespace Oaiy\Relay;

defined('OAIY_RELAY') or exit;

/** Everything one request needs, built once from the data directory. */
final class Context
{
    public string $dataDir;
    public Config $cfg;
    public Db $db;
    public Limiter $limiter;
    public Auth $auth;
    public Effective $eff;
    public Signals $signals;
    public Holds $holds;
    public Mailbox $mb;
    public Poll $poll;
    public Gc $gc;
    private ?Info $info = null;

    public static function open(string $dataDir): self
    {
        $c = new self();
        $c->dataDir = rtrim(str_replace('\\', '/', $dataDir), '/');
        $c->cfg = Config::load($c->dataDir);
        $c->db = Db::open($c->cfg);
        $c->db->assertSchema();
        $c->wire();
        return $c;
    }

    /** Build the parts that depend on the calibration; call again after the calibration changes. */
    public function wire(): void
    {
        $this->eff = Effective::load($this->cfg, $this->db);
        $this->limiter = new Limiter($this->db);
        $this->auth = new Auth($this->db, $this->cfg, $this->limiter);
        $this->signals = new Signals($this->dataDir);
        $this->holds = new Holds($this->dataDir, $this->eff);
        $this->mb = new Mailbox($this->db, $this->cfg, $this->signals);
        $this->poll = new Poll($this->db, $this->cfg, $this->eff, $this->mb, $this->signals, $this->holds, $this->limiter);
        $this->gc = new Gc($this->db, $this->cfg, $this->mb, $this->signals);
        $this->info = null;
    }

    public function relayId(): string
    {
        $id = $this->db->metaStr('relay_id');
        if ($id === null) {
            throw new \RuntimeException('relay id missing');
        }
        return $id;
    }

    public function info(): Info
    {
        if ($this->info === null) {
            [$pk, $sk] = Info::loadKeys($this->dataDir);
            $this->info = new Info($this->cfg, $this->eff, $this->relayId(), $pk, $sk);
        }
        return $this->info;
    }
}
