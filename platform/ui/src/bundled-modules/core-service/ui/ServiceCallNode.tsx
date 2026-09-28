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
import { Position } from '@xyflow/react';
import { CollapsibleNodeWrapper, type HandleConfig } from 'oaiy-ui-components';
import { subscribeToDynamicOptionsInvalidation } from 'oaiy-ui-components';
import {
  BUILT_IN_SERVICES,
  type CustomService,
  type ServiceInputDecl,
} from '../examples';

interface ServiceCallNodeData {
  service?: string;
  endpoint?: string;
  _collapsed?: boolean;
  _status?: 'running' | 'completed' | 'error';
  onCollapsedChange?: (collapsed: boolean) => void;
}

interface ServiceCallNodeProps {
  data: ServiceCallNodeData;
}

const STORAGE_KEY = 'oaiy.customServices';
// Services the OAIY Desktop is exposing — published here by
// lib/desktopServices.ts (ids prefixed `companion:`). Read so a dragged or
// picked companion service resolves on the CANVAS too, not just in the
// Properties panel (whose getService() already falls through to these).
const DESKTOP_STORAGE_KEY = 'oaiy.desktopServices';

function readKey(key: string): CustomService[] {
  try {
    if (typeof localStorage === 'undefined') return [];
    const raw = localStorage.getItem(key);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? (parsed as CustomService[]) : [];
  } catch {
    return [];
  }
}

function readCustomServices(): CustomService[] {
  return readKey(STORAGE_KEY);
}

function resolveService(id?: string): CustomService | null {
  if (!id) return null;
  // Custom + companion services first so user edits shadow same-id built-ins.
  const all = [...readCustomServices(), ...readKey(DESKTOP_STORAGE_KEY), ...BUILT_IN_SERVICES];
  return all.find((s) => s.id === id) ?? null;
}

// Pin-type → handle color. Mirrors the convention image_gen / video_gen
// use so an `image` output from a generator visually matches an `image`
// input on a service.
function colorFor(type: ServiceInputDecl['type']): string {
  switch (type) {
    case 'string':
      return '!bg-blue-500';
    case 'image':
      return '!bg-purple-500';
    case 'audio':
      return '!bg-teal-500';
    case 'video':
      return '!bg-orange-500';
    case 'any':
    default:
      return '!bg-cyan-500';
  }
}

function labelColorFor(type: ServiceInputDecl['type']): string {
  switch (type) {
    case 'string':
      return 'text-blue-700 dark:text-blue-400';
    case 'image':
      return 'text-purple-700 dark:text-purple-400';
    case 'audio':
      return 'text-teal-700 dark:text-teal-400';
    case 'video':
      return 'text-orange-700 dark:text-orange-400';
    case 'any':
    default:
      return 'text-cyan-700 dark:text-cyan-400';
  }
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

function ServiceCallNode({ data }: ServiceCallNodeProps) {
  // Bump on every Services-registry change so the resolved service +
  // derived handles refresh without forcing a workflow reload.
  const [refreshTick, setRefreshTick] = useState(0);
  useEffect(() => {
    return subscribeToDynamicOptionsInvalidation(() => setRefreshTick((t) => t + 1));
  }, []);

  const service = useMemo(() => resolveService(data.service), [data.service, refreshTick]);

  const declaredInputs: ServiceInputDecl[] = useMemo(() => {
    if (service?.inputs && service.inputs.length > 0) return service.inputs;
    return [
      {
        id: 'input',
        name: 'Input',
        type: 'any',
        description: 'Substituted into the body template as {{input}} / {{inputRaw}}.',
      },
    ];
  }, [service]);

  const inputHandles = useMemo<HandleConfig[]>(
    () =>
      declaredInputs.map((inp) => ({
        id: inp.id,
        type: 'target' as const,
        position: Position.Left,
        color: colorFor(inp.type),
        label: inp.name?.toLowerCase() || inp.id,
        labelColor: labelColorFor(inp.type),
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
      color: colorFor(o.type),
      label: o.name?.toLowerCase() || o.id,
      labelColor: labelColorFor(o.type),
      size: 'md' as const,
    }));
  }, [service]);

  const title = service?.name || 'Service Call';
  const endpointPreview = data.endpoint || service?.endpoint || '';
  const collapsedPreview = (
    <div className="text-slate-500 dark:text-slate-400 text-xs">
      {service ? (
        <>
          <span className="text-cyan-400 font-medium">{service.name}</span>
          {endpointPreview && (
            <span className="text-slate-500 ml-1 truncate">
              {endpointPreview.replace(/^https?:\/\//, '')}
            </span>
          )}
        </>
      ) : (
        <span className="italic text-slate-600 dark:text-slate-400">Pick a service preset</span>
      )}
    </div>
  );

  return (
    <CollapsibleNodeWrapper
      title={title}
      color="cyan"
      icon={renderIcon(service?.icon)}
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
            {endpointPreview && (
              <div className="font-mono text-[10px] text-slate-600 dark:text-slate-500 truncate">
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
