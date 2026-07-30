<?php
declare(strict_types=1);

namespace Oaiy\Api\Controllers;

use Psr\Http\Message\ResponseInterface;
use Psr\Http\Message\ServerRequestInterface;

/**
 * Service Library — serves a folder of example CustomService JSON files so the
 * landing/desktop page can list them and offer them for download.
 *
 * Drop a `*.json` file into the library folder and it appears automatically —
 * no rebuild, no manifest to edit. Each file is a single CustomService object
 * (or an array of them); the listing surfaces a little metadata (name,
 * description, icon, category) parsed from the JSON itself.
 *
 * Folder: `api/service-library/` by default; override with the
 * `SERVICE_LIBRARY_DIR` env var (absolute path).
 */
final class ServiceLibraryController
{
    private static function dir(): string
    {
        $dir = $_ENV['SERVICE_LIBRARY_DIR'] ?? (dirname(__DIR__, 2) . '/service-library');
        return rtrim((string) $dir, "/\\");
    }

    /** GET /api/service-library — list the available example files. */
    public function list(ServerRequestInterface $req, ResponseInterface $res): ResponseInterface
    {
        $dir = self::dir();
        $items = [];
        if (is_dir($dir)) {
            foreach (glob($dir . '/*.json') ?: [] as $path) {
                if (!is_file($path)) {
                    continue;
                }
                $file = basename($path);
                $raw = @file_get_contents($path);
                $meta = self::extractMeta($raw === false ? null : $raw);
                $items[] = [
                    'file'        => $file,
                    'name'        => $meta['name'],
                    'description' => $meta['description'],
                    'icon'        => $meta['icon'],
                    'category'    => $meta['category'],
                    'count'       => $meta['count'],
                    'size'        => $raw === false ? 0 : strlen($raw),
                    'downloadUrl' => '/api/service-library/' . rawurlencode($file),
                ];
            }
        }
        // Stable, predictable order (by display name).
        usort($items, static fn ($a, $b) => strcasecmp((string) $a['name'], (string) $b['name']));
        return self::json($res, ['services' => $items, 'count' => count($items)]);
    }

    /** GET /api/service-library/{file} — download one example file. */
    public function download(ServerRequestInterface $req, ResponseInterface $res, array $args): ResponseInterface
    {
        $file = (string) ($args['file'] ?? '');
        // Hard allow-list: a bare `name.json`, no separators, no traversal.
        if (str_contains($file, '..') || !preg_match('/^[A-Za-z0-9._-]+\.json$/', $file)) {
            return self::json($res, ['error' => 'invalid file name'], 400);
        }
        $base = realpath(self::dir());
        $real = realpath(self::dir() . '/' . $file);
        // Confine the resolved path inside the library dir (defence in depth
        // on top of the regex above), and require it to be a real file.
        if ($base === false || $real === false || !str_starts_with($real, $base . DIRECTORY_SEPARATOR) || !is_file($real)) {
            return self::json($res, ['error' => 'not found'], 404);
        }
        $body = (string) file_get_contents($real);
        $res->getBody()->write($body);
        return $res
            ->withHeader('Content-Type', 'application/json; charset=utf-8')
            ->withHeader('Content-Disposition', 'attachment; filename="' . $file . '"')
            ->withHeader('Cache-Control', 'no-store');
    }

    /**
     * Pull display metadata out of a service file. Accepts either a single
     * service object `{ name, description, icon, … }` or an array of them
     * (uses the first for display, reports the count). Never throws — a
     * malformed file just falls back to sensible defaults.
     */
    private static function extractMeta(?string $raw): array
    {
        $meta = ['name' => 'Service', 'description' => '', 'icon' => '', 'category' => '', 'count' => 1];
        if ($raw === null) {
            return $meta;
        }
        $data = json_decode($raw, true);
        if (!is_array($data)) {
            return $meta;
        }
        $isArray = array_is_list($data) && isset($data[0]) && is_array($data[0]);
        $svc = $isArray ? $data[0] : $data;
        if ($isArray) {
            $meta['count'] = count($data);
        }
        $meta['name']        = isset($svc['name']) ? (string) $svc['name'] : $meta['name'];
        $meta['description'] = isset($svc['description']) ? (string) $svc['description'] : $meta['description'];
        $meta['icon']        = isset($svc['icon']) ? (string) $svc['icon'] : $meta['icon'];
        // `category` isn't part of CustomService; allow an optional top-level
        // hint so the library can group examples (e.g. "Chat", "Image").
        $meta['category']    = isset($svc['category']) ? (string) $svc['category'] : $meta['category'];
        return $meta;
    }

    private static function json(ResponseInterface $res, array $data, int $status = 200): ResponseInterface
    {
        $res->getBody()->write(json_encode($data, JSON_UNESCAPED_SLASHES) ?: '{}');
        return $res->withHeader('Content-Type', 'application/json')->withStatus($status);
    }
}
