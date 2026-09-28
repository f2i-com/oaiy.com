/**
 * Host-side bridge for a custom-node UI rendered in an isolated sandbox iframe.
 *
 * The untrusted package UI code runs inside an iframe with `sandbox="allow-scripts"` and
 * NO `allow-same-origin` → a unique opaque origin that cannot read the app's
 * localStorage/cookies/secrets or touch the parent DOM / __TAURI_INTERNALS__. The iframe
 * content is an inline `srcdoc` carrying the bundled sandbox runner (virtual:sandbox-runtime)
 * plus a CSP that blocks fetch-type network egress (fetch/XHR/WS/img/form) and nested realms.
 * (Not airtight: a frame can always navigate ITSELF — location.href — which CSP fetch-directives
 * don't govern; so the frame's OWN node.data isn't fully contained. We keep app-wide secrets out
 * of node.data for exactly this reason — see sandbox-main.tsx.) We inline rather than load an
 * external `/sandbox.html` because an opaque-origin iframe cannot fetch a cross-origin ES
 * module (CORS forbids it). This component relays node data in / data-change + height out
 * over a narrow postMessage protocol (see src/sandbox/sandbox-main.tsx).
 *
 * This replaces the previous model where the package UI ran via `new Function` in the
 * MAIN realm with full app privileges.
 */
import React, { useCallback, useEffect, useRef, useState } from 'react';
import sandboxRuntime from 'virtual:sandbox-runtime';
import type { CustomNodeUIProps } from '../services/customNodeUILoader';

const CSP = [
  "default-src 'none'",
  "script-src 'unsafe-inline' 'unsafe-eval'",
  "style-src 'unsafe-inline'",
  'img-src data: blob:',
  'font-src data:',
  "connect-src 'none'",
  "object-src 'none'",
  "base-uri 'none'",
  "worker-src 'none'",
  "frame-src 'none'",
].join('; ');

// `</script` inside the bundled JS would prematurely close the inline <script> tag;
// the backslash is inert in JS string/regex contexts so this is semantically a no-op.
const RUNTIME = sandboxRuntime.replace(/<\/script/gi, '<\\/script');

const SRCDOC =
  `<!doctype html><html><head><meta charset="utf-8">` +
  `<meta http-equiv="Content-Security-Policy" content="${CSP}">` +
  `<style>html,body,#root{margin:0;padding:0}body{font-family:system-ui,-apple-system,sans-serif;` +
  `font-size:13px;color:#e5e7eb;background:transparent}</style></head>` +
  `<body><div id="root"></div><script>${RUNTIME}</script></body></html>`;

interface Props extends CustomNodeUIProps {
  /** Compiled UI bundle for this node — UNTRUSTED package code, runs only in the iframe. */
  code: string;
  /** Stable node-type id used to derive the compiler's sentinel global name. */
  nodeTypeId: string;
}

export const SandboxedCustomNodeUI: React.FC<Props> = ({
  code,
  nodeTypeId,
  id,
  data,
  selected,
  onDataChange,
}) => {
  const iframeRef = useRef<HTMLIFrameElement | null>(null);
  const [height, setHeight] = useState(40);

  // Latest values read by the (stable) message handler without re-subscribing.
  const dataRef = useRef(data);
  dataRef.current = data;
  const selectedRef = useRef(selected);
  selectedRef.current = selected;
  const onDataChangeRef = useRef(onDataChange);
  onDataChangeRef.current = onDataChange;
  const readyRef = useRef(false);

  const postToFrame = useCallback((msg: Record<string, unknown>) => {
    iframeRef.current?.contentWindow?.postMessage(msg, '*');
  }, []);

  useEffect(() => {
    const onMsg = (ev: MessageEvent) => {
      // Only accept messages from OUR iframe.
      if (ev.source !== iframeRef.current?.contentWindow) return;
      const m = ev.data as { __oaiybox?: string; updates?: Record<string, unknown>; px?: number };
      if (!m || typeof m.__oaiybox !== 'string') return;
      if (m.__oaiybox === 'boot') {
        readyRef.current = true;
        postToFrame({ __oaiybox: 'init', code, nodeId: nodeTypeId });
        postToFrame({ __oaiybox: 'props', data: dataRef.current, selected: selectedRef.current });
      } else if (m.__oaiybox === 'dataChange') {
        onDataChangeRef.current?.(m.updates ?? {});
      } else if (m.__oaiybox === 'height' && typeof m.px === 'number' && Number.isFinite(m.px)) {
        setHeight(Math.max(24, Math.min(m.px, 2000)));
      }
    };
    window.addEventListener('message', onMsg);
    return () => window.removeEventListener('message', onMsg);
  }, [code, nodeTypeId, postToFrame]);

  // Push prop changes to the frame once it's booted.
  useEffect(() => {
    if (readyRef.current) postToFrame({ __oaiybox: 'props', data, selected });
  }, [data, selected, postToFrame]);

  return (
    <iframe
      // Remount (→ fresh boot → init with the new code) when the node type or compiled UI
      // changes; srcDoc is constant so without this the iframe would keep stale code.
      key={`${nodeTypeId}:${code.length}`}
      ref={iframeRef}
      srcDoc={SRCDOC}
      sandbox="allow-scripts"
      title={`custom-node-ui-${id}`}
      style={{ width: '100%', height, border: 'none', display: 'block' }}
    />
  );
};

export default SandboxedCustomNodeUI;
