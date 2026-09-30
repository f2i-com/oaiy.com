<?php
declare(strict_types=1);

namespace Oaiy\Relay;

defined('OAIY_RELAY') or exit;

/** Who a verified credential belongs to: a device (its row) or the admin token. */
final class Principal
{
    public const ADMIN = 'admin';

    public string $id;
    public string $role;
    public string $tokenId;
    /** @var array<string,mixed> the devices row (empty for the admin) */
    public array $device;

    /** @param array<string,mixed> $device */
    public function __construct(string $id, string $role, string $tokenId, array $device)
    {
        $this->id = $id;
        $this->role = $role;
        $this->tokenId = $tokenId;
        $this->device = $device;
    }

    public static function admin(string $tokenId): self
    {
        return new self('admin', self::ADMIN, $tokenId, []);
    }

    public function isAdmin(): bool
    {
        return $this->role === self::ADMIN;
    }

    public function isDesktop(): bool
    {
        return $this->role === 'desktop';
    }

    public function name(): string
    {
        return (string)($this->device['name'] ?? '');
    }

    /** @return list<string> */
    public function grants(): array
    {
        $g = json_decode((string)($this->device['grants'] ?? '[]'), true);
        return is_array($g) ? array_values(array_filter($g, 'is_string')) : [];
    }

    /** @return array<string,mixed> */
    public function flags(): array
    {
        $f = json_decode((string)($this->device['flags'] ?? '{}'), true);
        return is_array($f) ? $f : [];
    }

    public function canCmd(): bool
    {
        return ($this->flags()['canCmd'] ?? false) === true;
    }

    public function ownerDesktop(): ?string
    {
        $o = $this->device['owner_desktop'] ?? null;
        return is_string($o) && $o !== '' ? $o : null;
    }
}
