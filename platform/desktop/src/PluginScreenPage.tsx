import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { TriangleAlert } from 'lucide-react';
import { API_BASE, bridge, plugins, type PluginRecord } from './api';
import { useToast } from './Toasts';

/**
 * Host for a plugin-contributed screen.
 *
 * A plugin ships its screen as a BODY FRAGMENT plus css/js files (declared in
 * `manifest.ui.screens[].files`); the host composes them into one document and
 * runs it in a sandboxed, opaque-origin iframe under a strict CSP. The plugin's
 * scripts therefore have no network of their own — everything they need reaches
 * the host through `window.PluginHost`, a postMessage RPC bridge this component
 * injects before any plugin script executes.
 *
 * That indirection is the point: the screen is third-party code, so it gets a
 * named capability surface (invoke a connector command, subscribe to declared
 * events, read a snapshot) instead of a network and an origin.
 */

interface Props {
  pluginId: string;
  /** The `ui.nav[].id` that was clicked. */
  navId: string;
}

interface UiNav {
  id?: string;
  label?: string;
  screen?: string;
}
interface UiScreen {
  id?: string;
  title?: string;
  entry?: string;
  files?: string[];
}

/** The bootstrap injected ahead of the plugin's own scripts. Plain ES5-ish so it
 *  runs before any transform, and self-contained: the iframe has no imports. */
const HOST_BOOTSTRAP = `
(function () {
  var seq = 0;
  var pending = {};
  var subs = [];
  function call(method, args) {
    return new Promise(function (resolve, reject) {
      var id = 'r' + ++seq;
      pending[id] = { resolve: resolve, reject: reject };
      parent.postMessage({ __pluginHost: 1, id: id, method: method, args: args || [] }, '*');
    });
  }
  window.addEventListener('message', function (e) {
    var m = e.data;
    if (!m || !m.__pluginHost) return;
    if (m.event) { subs.forEach(function (s) { try { s(m.event); } catch (_) {} }); return; }
    var p = pending[m.id];
    if (!p) return;
    delete pending[m.id];
    if (m.ok) p.resolve(m.data); else p.reject(new Error(m.error || 'host call failed'));
  });
  window.PluginHost = {
    command: function (name, payload) { return call('command', [name, payload]); },
    toast: function (kind, message) { return call('toast', [kind, message]); },
    snapshot: function () { return call('snapshot', []); },
    aiSources: function () { return call('aiSources', []); },
    restartPlugin: function () { return call('restartPlugin', []); },
    events: {
      subscribe: function (names, cb) {
        subs.push(cb);
        call('subscribe', [names || []]);
        return function () { subs = subs.filter(function (s) { return s !== cb; }); };
      }
    },
    // Surfaces this host does not implement. Rejecting by name is deliberate —
    // a plugin tab then shows its own honest error instead of hanging on a
    // promise that never settles or calling undefined.
    consent: function () { return Promise.reject(new Error('consent grants are not supported by OAIY Desktop')); },
    companionPairing: {
      status: function () { return Promise.reject(new Error('companion pairing is not supported by OAIY Desktop')); },
      createOffer: function () { return Promise.reject(new Error('companion pairing is not supported by OAIY Desktop')); },
      receiveResponse: function () { return Promise.reject(new Error('companion pairing is not supported by OAIY Desktop')); },
      approve: function () { return Promise.reject(new Error('companion pairing is not supported by OAIY Desktop')); }
    }
  };
})();
`;

export default function PluginScreenPage({ pluginId, navId }: Props) {
  const toast = useToast();
  const [record, setRecord] = useState<PluginRecord | null | undefined>(undefined);
  const [doc, setDoc] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const frameRef = useRef<HTMLIFrameElement>(null);
  /** Event names the screen subscribed to, and the poll cursor. */
  const subscribed = useRef<string[]>([]);
  const cursor = useRef<number>(0);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const snap = await plugins.list();
        if (!cancelled) setRecord(snap.plugins.find((p) => p.id === pluginId) ?? null);
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [pluginId]);

  const screen = useMemo(() => {
    const ui = (record?.manifest as unknown as { ui?: { nav?: UiNav[]; screens?: UiScreen[] } } | undefined)?.ui;
    if (!ui) return null;
    const nav = (ui.nav ?? []).find((n) => n.id === navId);
    return (ui.screens ?? []).find((s) => s.id === nav?.screen) ?? null;
  }, [record, navId]);

  // Compose the document: fragment + inlined css + the bootstrap + the plugin's
  // scripts in manifest order. Assembled here (not server-side) so the iframe can
  // use srcdoc, which is what gives it an opaque origin.
  useEffect(() => {
    if (!screen?.id || !screen.entry) return;
    let cancelled = false;
    (async () => {
      const base = `${API_BASE}/api/plugins/${encodeURIComponent(pluginId)}/ui/${encodeURIComponent(screen.id!)}`;
      const fetchText = async (rel: string) => {
        const resp = await fetch(`${base}/${rel.split('/').map(encodeURIComponent).join('/')}`, {
          cache: 'no-store',
        });
        if (!resp.ok) throw new Error(`${rel} → HTTP ${resp.status}`);
        return resp.text();
      };
      try {
        const files = screen.files ?? [screen.entry!];
        const body = await fetchText(screen.entry!);
        const css = (
          await Promise.all(files.filter((f) => f.endsWith('.css')).map(fetchText))
        ).join('\n');
        // Manifest order matters: app.js defines the registry the tab modules
        // register into, and it is listed first.
        const js = (
          await Promise.all(files.filter((f) => f.endsWith('.js')).map(fetchText))
        ).join('\n;\n');
        if (cancelled) return;
        setDoc(
          `<!doctype html><html><head><meta charset="utf-8">` +
            // No external anything: the plugin ships inline SVG and its own CSS.
            `<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; img-src data: blob:; font-src data:; connect-src 'none'">` +
            `<style>${css}</style></head><body>${body}` +
            `<script>${HOST_BOOTSTRAP}</script><script>${js}</script></body></html>`,
        );
        setError(null);
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [screen, pluginId]);

  /** Service one RPC from the screen. */
  const handleCall = useCallback(
    async (method: string, args: unknown[]): Promise<unknown> => {
      switch (method) {
        case 'command': {
          const [name, payload] = args as [string, unknown];
          const res = await bridge.connectorRequest(pluginId, name, payload);
          // Plugins answer with the SDK envelope { ok, data }; hand the screen
          // the inner data, which is what its call sites expect.
          const r = res.result as { ok?: boolean; data?: unknown } | undefined;
          return r && typeof r === 'object' && 'data' in r ? r.data : r;
        }
        case 'toast': {
          const [kind, message] = args as [string, string];
          toast.push({
            kind: kind === 'error' ? 'error' : kind === 'success' ? 'success' : 'info',
            title: String(message ?? ''),
          });
          return true;
        }
        case 'snapshot': {
          const snap = await plugins.list();
          return snap.plugins.find((p) => p.id === pluginId) ?? null;
        }
        case 'aiSources':
          return (await bridge.aiSources()).sources;
        case 'restartPlugin':
          await plugins.stop(pluginId).catch(() => {});
          await plugins.start(pluginId);
          return true;
        case 'subscribe': {
          const [names] = args as [string[]];
          subscribed.current = Array.isArray(names) ? names : [];
          // Start from the tail so a screen doesn't replay history on open.
          cursor.current = (await bridge.events(0, 1)).next;
          return true;
        }
        default:
          throw new Error(`unsupported host call "${method}"`);
      }
    },
    [pluginId, toast],
  );

  // RPC pump: only messages from OUR iframe are serviced.
  useEffect(() => {
    const onMessage = (e: MessageEvent) => {
      const m = e.data as { __pluginHost?: number; id?: string; method?: string; args?: unknown[] };
      if (!m || !m.__pluginHost || !m.id || !m.method) return;
      const frame = frameRef.current;
      if (!frame || e.source !== frame.contentWindow) return;
      handleCall(m.method, m.args ?? [])
        .then((data) => frame.contentWindow?.postMessage({ __pluginHost: 1, id: m.id, ok: true, data }, '*'))
        .catch((err) =>
          frame.contentWindow?.postMessage(
            { __pluginHost: 1, id: m.id, ok: false, error: err instanceof Error ? err.message : String(err) },
            '*',
          ),
        );
    };
    window.addEventListener('message', onMessage);
    return () => window.removeEventListener('message', onMessage);
  }, [handleCall]);

  // Forward declared plugin events to the screen.
  useEffect(() => {
    const id = window.setInterval(async () => {
      if (subscribed.current.length === 0) return;
      try {
        const res = await bridge.events(cursor.current, 100);
        cursor.current = res.next;
        for (const e of res.events) {
          const env = e.envelope as { name?: string };
          if (env?.name && subscribed.current.includes(env.name)) {
            frameRef.current?.contentWindow?.postMessage({ __pluginHost: 1, event: env }, '*');
          }
        }
      } catch {
        /* transient — the next tick retries */
      }
    }, 2000);
    return () => window.clearInterval(id);
  }, []);

  if (error) {
    return (
      <div className="panel">
        <div className="banner banner-err" role="alert">
          <span>Couldn't load the plugin screen: {error}</span>
        </div>
      </div>
    );
  }
  if (record === undefined) {
    return (
      <div className="panel">
        <div className="empty-state">Loading…</div>
      </div>
    );
  }
  if (record === null) {
    return (
      <div className="panel">
        <div className="empty-state">
          <TriangleAlert size={22} style={{ opacity: 0.5 }} />
          <p>That plugin is no longer installed.</p>
        </div>
      </div>
    );
  }
  if (!screen) {
    return (
      <div className="panel">
        <div className="empty-state">
          <TriangleAlert size={22} style={{ opacity: 0.5 }} />
          <p>This plugin doesn't ship that screen.</p>
          <p style={{ fontSize: 13, opacity: 0.7 }}>
            Its manifest declares no <code>ui.screens</code> entry for “{navId}”.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="panel">
      {record.state !== 'running' && (
        <div className="banner banner-err" role="alert">
          <span>
            <TriangleAlert size={13} /> “{record.manifest?.name ?? record.id}” is {record.state} — the
            screen loads, but anything it asks the plugin to do will fail until you start it.
          </span>
        </div>
      )}
      {doc === null ? (
        <div className="empty-state">Loading screen…</div>
      ) : (
        <iframe
          ref={frameRef}
          className="plugin-screen"
          title={screen.title ?? navId}
          srcDoc={doc}
          /* allow-scripts WITHOUT allow-same-origin: the screen runs at an opaque
             origin, so it has no access to the host's storage or the local API
             except through the PluginHost bridge above. */
          sandbox="allow-scripts allow-forms allow-modals"
        />
      )}
    </div>
  );
}
