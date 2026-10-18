/**
 * Service Call — custom node component.
 *
 * Reads `data.service` (the picked service preset id), looks the service
 * up in the registry (BUILT_IN_SERVICES + localStorage `oaiy.customServices`),
 * and renders ONE input handle per declared `inputs[]` so multi-input
 * services like `{prompt, image}` get one wireable pin per var.
 *
 * Without a picked service (or when the picked service has no declared
 * inputs[]) the node falls back to a single generic `input` handle —
 * matches the legacy shape, won't break existing saved flows.
 *
 * The component subscribes to `subscribeToDynamicOptionsInvalidation` so
 * a service edited in the Settings → Services panel refreshes the canvas
 * node's handles live (the Services panel calls `invalidateDynamicOptions`
 * on every save; the hook fans the broadcast out to every subscriber).
 *
 * The dropped node's title + collapsed preview reflect the picked service
 * so a "Ollama Chat" service-as-palette entry shows up as "Ollama Chat"
 * on the canvas instead of the generic "Service Call".
 */

import { memo, useEffect, useMemo, useState } from 'react';
import { Position, useStore } from '@xyflow/react';
import { CollapsibleNodeWrapper, portTypeClass, type HandleConfig } from 'oaiy-ui-components';
import { subscribeToDynamicOptionsInvalidation, useNodeNotice } from 'oaiy-ui-components';
import type { ServiceInputDecl } from '../examples';
import { describeService, resolveService as lookupService } from '../contract';

interface ServiceCallNodeData {
  service?: string;
  endpoint?: string;
  _collapsed?: boolean;
  _status?: 'running' | 'completed' | 'error';
  onCollapsedChange?: (collapsed: boolean) => void;
}

interface ServiceCallNodeProps {
  id?: string;
  data: ServiceCallNodeData;
}

// The same lookup the compiler uses (core-service/contract.ts): the user's
// services, OAIY Desktop's (its services, OAIY's engine models) and the
// built-in examples — so a dragged or picked one resolves on the canvas too.
function resolveService(id?: string) {
  return id ? lookupService(id) : null;
}

const DefaultServiceIcon = (
  <svg className="w-3 h-3 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path
      strokeLinecap="round"
      strokeLinejoin="round"
      strokeWidth={2}
      d="M5 12h14M5 12a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v4a2 2 0 01-2 2M5 12a2 2 0 00-2 2v4a2 2 0 002 2h14a2 2 0 002-2v-4a2 2 0 00-2-2m-2-4h.01M17 16h.01"
    />
  </svg>
);

/** Pick the visual icon for the title bar: the service's user-set
 * emoji/text if any (e.g. "🤖" / "🎨"), else a generic server glyph. */
function renderIcon(emoji: string | undefined) {
  if (emoji && emoji.length > 0) {
    return <span className="text-xs leading-none">{emoji}</span>;
  }
  return DefaultServiceIcon;
}

/** An address with a place to break after each slash: a long one wraps between its parts, not inside a word. */
function breakable(address: string) {
  return address.split('/').flatMap((part, i, all) => (i < all.length - 1 ? [part, '/', <wbr key={i} />] : [part]));
}

function ServiceCallNode({ id: nodeId, data }: ServiceCallNodeProps) {
  // Bump on every Services-registry change so the resolved service +
  // derived handles refresh without forcing a workflow reload.
  const [refreshTick, setRefreshTick] = useState(0);
  useEffect(() => {
    return subscribeToDynamicOptionsInvalidation(() => setRefreshTick((t) => t + 1));
  }, []);

  const service = useMemo(() => resolveService(data.service), [data.service, refreshTick]);
  // Picked, but not installed here (a model removed from Engines, a service
  // uninstalled): the node says so, and keeps the pins its edges use.
  const notice = useNodeNotice('service_call', data as Record<string, unknown>);
  const missing = !!data.service && !service;
  const wiredInputs = useStore((st) =>
    missing && nodeId
      ? st.edges.filter((e) => e.target === nodeId && e.targetHandle).map((e) => e.targetHandle as string).sort().join('|')
      : '',
  );

  const declaredInputs: ServiceInputDecl[] = useMemo(() => {
    if (service?.inputs && service.inputs.length > 0) return service.inputs;
    if (missing && wiredInputs) {
      return wiredInputs.split('|').filter((v, i, a) => a.indexOf(v) === i).map((pin) => ({ id: pin, name: pin, type: 'any' as const }));
    }
    return [
      {
        id: 'input',
        name: 'Input',
        type: 'any',
        description: 'Substituted into the body template as {{input}} / {{inputRaw}}.',
      },
    ];
  }, [service, missing, wiredInputs]);

  const inputHandles = useMemo<HandleConfig[]>(
    () =>
      declaredInputs.map((inp) => ({
        id: inp.id,
        type: 'target' as const,
        position: Position.Left,
        // (its pin's kind as every card colours it, and its name as the service gives it)
        color: portTypeClass(inp.type),
        label: inp.name || inp.id,
        size: 'md' as const,
      })),
    [declaredInputs],
  );

  const outputHandles = useMemo<HandleConfig[]>(() => {
    const outs = service?.outputs && service.outputs.length > 0
      ? service.outputs
      : [{ id: 'response', name: 'Response', type: 'any' as const }];
    return outs.map((o) => ({
      id: o.id,
      type: 'source' as const,
      position: Position.Right,
      color: portTypeClass(o.type),
      label: o.name || o.id,
      size: 'md' as const,
    }));
  }, [service]);

  // (in words, as the pickers say it: not the raw id of a worker's model)
  const title = service?.name || (missing ? `${describeService(String(data.service)).text} (not available)` : 'Service Call');
  const endpointPreview = data.endpoint || service?.endpoint || '';
  // The service's name is the card's title (or its kind, over a name of the node's own): the line under the ports
  // says where it calls. (The name and the address side by side broke the name over two lines.)
  // (one of OAIY's own models: what it is, not the desktop's internal address it is reached at)
  const own = service?.group === 'engine';
  const collapsedPreview = service ? (
    own ? (
      service.description ? <div>{service.description}</div> : null
    ) : endpointPreview ? (
      <div className="font-mono text-[10px] text-slate-500 dark:text-slate-400" title={endpointPreview}>
        {breakable(endpointPreview.replace(/^https?:\/\//, ''))}
      </div>
    ) : null
  ) : (
    <div className="italic text-slate-600 dark:text-slate-400">Pick a service preset</div>
  );

  return (
    <CollapsibleNodeWrapper
      title={title}
      color="cyan"
      icon={renderIcon(service?.icon)}
      ownIcon={!!service?.icon}
      width={260}
      collapsedWidth={160}
      status={data._status}
      isCollapsed={data._collapsed}
      onCollapsedChange={(c) => data.onCollapsedChange?.(c)}
      collapsedPreview={collapsedPreview}
      inputHandles={inputHandles}
      outputHandles={outputHandles}
    >
      <div className="space-y-1 text-xs">
        {service ? (
          <>
            <div className="text-slate-500 dark:text-slate-400">
              {service.description || 'Custom HTTP service.'}
            </div>
            {endpointPreview && !own && (
              <div className="font-mono text-[10px] text-slate-600 dark:text-slate-400 truncate" title={endpointPreview}>
                {endpointPreview}
              </div>
            )}
            {declaredInputs.length > 1 && (
              <div className="text-[10px] text-slate-500 dark:text-slate-400">
                {declaredInputs.length} inputs · use{' '}
                <code className="text-cyan-500">{`{{${declaredInputs[0].id}}}`}</code>{' '}
                etc. in the body template.
              </div>
            )}
          </>
        ) : missing ? (
          <div
            role="status"
            className="rounded border border-amber-500/60 bg-amber-50 dark:bg-amber-900/30 px-2 py-1.5 text-[11px] leading-snug text-amber-900 dark:text-amber-100"
          >
            {notice || `The service “${data.service}” is not available here. Pick another in the Properties panel.`}
          </div>
        ) : (
          <div className="text-slate-600 dark:text-slate-400 italic">
            No service selected — pick one in the Properties panel or drop a
            service from the palette.
          </div>
        )}
      </div>
    </CollapsibleNodeWrapper>
  );
}

export default memo(ServiceCallNode);
