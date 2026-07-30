/**
 * Media Server Utilities
 *
 * Provides helper functions for constructing media server URLs with dynamic port support.
 *
 * IMPORTANT: every URL must include the per-launch capability token in the
 * path component (`/media/<token>/<alias>/<rest>`). Without it, the Rust
 * media server returns 400/403 and the `<video>`/`<audio>` element shows
 * an empty box. The token + port are cached on first init and reused for
 * the rest of the session.
 */

import { createLogger } from './logger.js';

const logger = createLogger('MediaUtils');

const DEFAULT_PORT = 31338;

// Cache the media server port + token. Both are populated by
// `initMediaServerPort()` / `initMediaServerToken()` (called once from
// App.tsx on app startup, before any node renders). Sync getters return
// the cached values so `pathToMediaUrl` can stay sync.
let cachedPort: number = DEFAULT_PORT;
let cachedToken: string = '';

/**
 * Initialize the media server port. Call this on app startup.
 * This should be called from the frontend after getting the port from Tauri.
 */
export async function initMediaServerPort(): Promise<number> {
    if (cachedPort !== DEFAULT_PORT) {
        return cachedPort;
    }

    try {
        // Try to get port from Tauri via window invoke
        if (typeof window !== 'undefined' && (window as any).__TAURI_INTERNALS__) {
            const port = await (window as any).__TAURI_INTERNALS__.invoke('get_media_server_port');
            if (port && port > 0) {
                cachedPort = port;
                logger.debug(`Media server port: ${cachedPort}`);
            }
        }
    } catch (e) {
        logger.warn('Failed to get media server port', { error: e });
    }

    return cachedPort;
}

/**
 * Initialize the per-launch media-server capability token. Call this on
 * app startup, alongside `initMediaServerPort()`. The token is required
 * in every `/media/...` URL; without it the server rejects the request
 * and `<video>` elements render empty.
 *
 * Returns the token (or empty string if Tauri isn't available or the
 * media server isn't running).
 */
export async function initMediaServerToken(): Promise<string> {
    if (cachedToken !== '') {
        return cachedToken;
    }

    try {
        if (typeof window !== 'undefined' && (window as any).__TAURI_INTERNALS__) {
            const token = await (window as any).__TAURI_INTERNALS__.invoke('get_media_token');
            if (typeof token === 'string' && token.length > 0) {
                cachedToken = token;
                // Don't log the token value — it's a per-launch capability.
                // Length-only is fine and useful for diagnostics.
                logger.debug(`Media server token cached (len=${cachedToken.length})`);
            }
        }
    } catch (e) {
        logger.warn('Failed to get media server token', { error: e });
    }

    return cachedToken;
}

/**
 * Get the cached media server port. Returns 0 if not initialized.
 */
export function getMediaServerPort(): number {
    return cachedPort || DEFAULT_PORT; // Fallback to default if not initialized
}

/**
 * Get the cached media-server token. Returns empty string if init hasn't
 * run yet — callers should treat that as "not ready" and skip URL minting
 * (the resulting URL would be rejected by the server anyway).
 */
export function getMediaServerToken(): string {
    return cachedToken;
}

/**
 * Set the media server port directly (useful for initialization)
 */
export function setMediaServerPort(port: number): void {
    cachedPort = port;
}

/**
 * Set the media-server token directly. Used by tests; production calls
 * `initMediaServerToken()` instead.
 */
export function setMediaServerToken(token: string): void {
    cachedToken = token;
}

/**
 * Convert a local file path to a media server URL.
 *
 * Returns the original path if:
 *   - the path doesn't live under one of the allowed alias roots
 *     (oaiy-output / Pictures / Videos / Downloads), or
 *   - the token hasn't been initialised yet — without it the server
 *     would reject the URL with 400 anyway, and returning the raw path
 *     at least lets the caller fall back to other channels (e.g. a
 *     Tauri `convertFileSrc` they might apply on top).
 *
 * URL format (server side: `axum` route `/media/*tail` parses
 * `<token>/<alias>/<rest>` with `splitn(3, '/')`):
 *
 *   http://127.0.0.1:<port>/media/<token>/<alias>/<file>
 *
 * The `media_server::serve_media_file` handler validates the token in
 * constant time, looks up the alias in the allowed-paths table, then
 * canonicalises the resolved path and refuses anything that escapes the
 * base via symlinks or `..` traversal.
 */
export function pathToMediaUrl(filePath: string): string {
    if (!filePath) return filePath;

    const port = getMediaServerPort();
    const token = getMediaServerToken();
    if (!token) {
        // Init hasn't completed (or Tauri unavailable). Returning the raw
        // path lets the caller decide what to do — slightly less broken
        // than emitting a URL the server will reject with 400.
        return filePath;
    }
    const normalizedPath = filePath.replace(/\\/g, '/');

    // Check for oaiy output directory (AppData/Roaming/oaiy/output/)
    const oaiyOutputMatch = normalizedPath.match(/\/oaiy\/output\/(.+)$/i);
    if (oaiyOutputMatch) {
        return `http://127.0.0.1:${port}/media/${token}/oaiy-output/${oaiyOutputMatch[1]}`;
    }

    // Check for Pictures directory
    const picturesMatch = normalizedPath.match(/\/Pictures\/(.+)$/i);
    if (picturesMatch) {
        return `http://127.0.0.1:${port}/media/${token}/pictures/${picturesMatch[1]}`;
    }

    // Check for Videos directory
    const videosMatch = normalizedPath.match(/\/Videos\/(.+)$/i);
    if (videosMatch) {
        return `http://127.0.0.1:${port}/media/${token}/videos/${videosMatch[1]}`;
    }

    // Check for Downloads directory
    const downloadsMatch = normalizedPath.match(/\/Downloads\/(.+)$/i);
    if (downloadsMatch) {
        return `http://127.0.0.1:${port}/media/${token}/downloads/${downloadsMatch[1]}`;
    }

    // Return original if no match — caller may decide to fall back to a
    // different URL scheme (e.g. Tauri convertFileSrc for an allowed path
    // outside our base list).
    return filePath;
}

/**
 * Check if a string looks like a local file path (not a URL)
 */
export function isLocalPath(value: string): boolean {
    if (!value || typeof value !== 'string') return false;

    // Check if it's already a URL
    if (value.startsWith('http://') || value.startsWith('https://') || value.startsWith('data:')) {
        return false;
    }

    // Check for common path patterns
    return (
        value.includes('/') ||
        value.includes('\\') ||
        /^[A-Za-z]:/.test(value) // Windows drive letter
    );
}
