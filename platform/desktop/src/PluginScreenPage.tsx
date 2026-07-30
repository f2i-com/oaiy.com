import { useEffect, useMemo, useState } from 'react';
import { TriangleAlert } from 'lucide-react';
import { API_BASE, plugins, type PluginRecord } from './api';

/**
 * Host for a plugin-contributed screen.
 *
 * A plugin may ship its own interface (`manifest.ui.screens[]`) — static
 * HTML/CSS/JS inside the plugin folder — and declare a nav entry that opens it.
 * The host serves those files from `GET /api/plugins/:id/ui/:screen/*path`,
 * restricted to the files the screen declares, and renders the entry document in
 * a sandboxed iframe.
 *
 * Sandboxed deliberately: plugin UI is third-party code. It gets scripts and
 * same-origin (it must call the local API to be useful), but no top-level
 * navigation and no popups, so a screen cannot take over the app window.
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
}

export default function PluginScreenPage({ pluginId, navId }: Props) {
  const [record, setRecord] = useState<PluginRecord | null | undefined>(undefined);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const snap = await plugins.list();
        if (cancelled) return;
        setRecord(snap.plugins.find((p) => p.id === pluginId) ?? null);
        setError(null);
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [pluginId]);

  const resolved = useMemo(() => {
    const ui = (record?.manifest as unknown as { ui?: { nav?: UiNav[]; screens?: UiScreen[] } } | undefined)?.ui;
    if (!ui) return null;
    const nav = (ui.nav ?? []).find((n) => n.id === navId);
    const screenId = nav?.screen;
    const screen = (ui.screens ?? []).find((s) => s.id === screenId);
    if (!screen?.entry || !screen.id) return null;
    return {
      title: screen.title ?? nav?.label ?? pluginId,
      src: `${API_BASE}/api/plugins/${encodeURIComponent(pluginId)}/ui/${encodeURIComponent(
        screen.id,
      )}/${screen.entry.split('/').map(encodeURIComponent).join('/')}`,
    };
  }, [record, navId, pluginId]);

  if (error) {
    return (
      <div className="panel">
        <div className="banner banner-err" role="alert">
          <span>Couldn't load the plugin: {error}</span>
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
  if (!resolved) {
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
            <TriangleAlert size={13} /> “{record.manifest?.name ?? record.id}” is {record.state} — its
            screen is shown, but anything it asks the plugin to do will fail until you start it.
          </span>
        </div>
      )}
      <iframe
        className="plugin-screen"
        title={resolved.title}
        src={resolved.src}
        /* Plugin UI is third-party: it gets scripts, same-origin (it must call the
           local API to be useful), its own forms and modals — but NOT
           allow-top-navigation or allow-popups, so a screen cannot navigate the
           app window away or spawn windows. */
        sandbox="allow-scripts allow-same-origin allow-forms allow-modals"
      />
    </div>
  );
}
