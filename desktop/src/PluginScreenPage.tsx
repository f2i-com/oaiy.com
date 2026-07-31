import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { TriangleAlert } from 'lucide-react';
import {
  API_BASE,
  bridge,
  companion,
  plugins,
  type CompanionOfferRequest,
  type PluginRecord,
} from './api';
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
      // Resolves to a HANDLE (with .unsubscribe()), not the function itself —
      // callers do subscribe(...).then(h => handle = h).
      subscribe: function (names, cb) {
        subs.push(cb);
        return call('subscribe', [names || []]).then(function () {
          return {
            unsubscribe: function () {
              subs = subs.filter(function (s) { return s !== cb; });
            }
          };
        });
      }
    },
    // Consent is a plain connector surface on the plugin (consent.get/set/revoke
    // are declared commands), so it routes through the same gateway as any other
    // command — the plugin owns the grant, the host just carries it.
    consent: {
      get: function () { return call('command', ['consent.get']); },
      issue: function (grant) { return call('command', ['consent.set', grant]); },
      revoke: function () { return call('command', ['consent.revoke']); }
    },
    // Companion device trust. Served by the host, not by the plugin: the
    // desktop's endpoint key is the thing a phone authenticates AGAINST, so it
    // must not live in the process the phone is talking to.
    //
    // All SEVEN methods are defined, including the three the screen only calls
    // from a confirm step. A missing one throws synchronously inside a handler
    // that has already set busy = true and cannot clear it, which leaves every
    // button on the tab disabled until reload.
    companionPairing: {
      status: function () { return call('companion.status', []); },
      createOffer: function (body) { return call('companion.createOffer', [body]); },
      receiveResponse: function (response) { return call('companion.receiveResponse', [response]); },
      approve: function (id) { return call('companion.approve', [id]); },
      deny: function (id) { return call('companion.deny', [id]); },
      revoke: function (thumbprint) { return call('companion.revoke', [thumbprint]); },
      rotateDesktopKey: function () { return call('companion.rotate', []); }
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
        //
        // One <script> PER FILE, not one concatenated blob: concatenating puts
        // every module in a single script, so a throw anywhere aborts all the
        // modules after it — which silently cost us every tab that registers
        // after the first failure. Separate tags give identical global semantics
        // and identical ordering, but isolate a bad module to itself.
        const jsFiles = files.filter((f) => f.endsWith('.js'));
        const sources = await Promise.all(jsFiles.map(fetchText));
        if (cancelled) return;
        const scripts = sources.map((s) => `<script>${s}</script>`).join('');
        setDoc(
          `<!doctype html><html><head><meta charset="utf-8">` +
            // No external anything: the plugin ships inline SVG and its own CSS.
            `<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; img-src data: blob:; font-src data:; connect-src 'none'">` +
            `<style>${css}</style></head><body>${body}` +
            `<script>${HOST_BOOTSTRAP}</script>${scripts}</body></html>`,
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
          // Always send an idempotency key: the gateway REQUIRES one for the
          // plugin's journalled (physically side-effecting) commands, and a
          // fresh key per user action is the correct semantics for the rest.
          const key = `ui-${name}-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
          const res = await bridge.connectorRequest(pluginId, name, payload, key);
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
        // Companion trust. The screen never names a plugin: it gets its OWN id
        // from the host, so a screen cannot administer another plugin's phones
        // by asking nicely.
        case 'companion.status':
          return await companion.status(pluginId);
        case 'companion.createOffer':
          return await companion.createOffer(pluginId, (args[0] ?? {}) as CompanionOfferRequest);
        case 'companion.receiveResponse':
          return await companion.receiveResponse(pluginId, args[0]);
        case 'companion.approve':
          return await companion.approve(pluginId, String(args[0] ?? ''));
        case 'companion.deny':
          // The DELETE/deny routes answer 204, so there is no body to hand
          // back; the screen only awaits completion and then refreshes.
          await companion.deny(pluginId, String(args[0] ?? ''));
          return true;
        case 'companion.revoke':
          await companion.revoke(pluginId, String(args[0] ?? ''));
          return true;
        case 'companion.rotate':
          return await companion.rotate(pluginId);
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
            <TriangleAlert size={13} /> “{record.manifest?.name ?? record.id}” is {record.state}
            {record.reason ? ` — ${record.reason}` : ''}
            {/* An unhealthy plugin is RUNNING and still answers commands (health
                is a coarse signal), so telling the user to start it would be
                wrong — and often the screen itself is where the fix lives. */}
            {record.state === 'unhealthy'
              ? '. The screen still works — you may be able to resolve this here.'
              : '. Start it from Plugins before using this screen.'}
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
