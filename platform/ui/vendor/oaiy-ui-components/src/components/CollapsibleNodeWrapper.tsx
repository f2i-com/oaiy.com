import { memo, useState, useCallback, useEffect, type ReactNode } from 'react';
import { Handle, Position, useNodeId, useStore } from '@xyflow/react';
import type { HandleDefinition } from 'oaiy-core';
import { useNodeDefinition } from '../hooks/useNodeDefinition';
import { useNodeNoticeDetail } from '../hooks/useNodeNotice';
import { resolveNodeIcon } from '../registry/nodeIcons';
import { portTypeClass, typeWords } from './portTypes';

export interface ValidationIssue {
  field: string;
  message: string;
}

export interface HandleConfig {
  id: string;
  type: 'source' | 'target';
  position: Position;
  /** Its dot's colour (a Tailwind class like '!bg-blue-500'), for a port the node's definition does not have: one it
   * has is coloured by its kind of value (`portTypeClass`). */
  color: string;
  label?: string;
  /** The label is the port's own name, given by the person (a macro's input "prompt"): shown as it is, not the
   * definition's word for the port. */
  ownLabel?: boolean;
  /** No longer shown: every port's name is in the card's own ink. */
  labelColor?: string;
  size?: 'sm' | 'md' | 'lg'; // default md
}

export interface CollapsibleNodeWrapperProps {
  /** Node title displayed in header */
  title: string;
  /** Color theme for the node (tailwind color name without prefix) */
  color: 'purple' | 'green' | 'blue' | 'amber' | 'cyan' | 'pink' | 'emerald' | 'orange' | 'red' | 'slate' | 'indigo' | 'teal' | 'violet';
  /** Icon SVG element (the palette's icon for the node's kind wins, where the host gives one) */
  icon: ReactNode;
  /** The icon is the card's own, not its kind's (a service's emoji on a Service Call): kept. */
  ownIcon?: boolean;
  /** Width of the node when expanded */
  width: number;
  /** Width when collapsed (at least 200: room for its ports' names) */
  collapsedWidth?: number;
  /** Status for visual styling (else the node's own `_status`, as the run sets it; `stopped`: where Stop found it) */
  status?: 'running' | 'completed' | 'error' | 'stopped';
  /** Validation issues to display warning indicator */
  validationIssues?: ValidationIssue[];
  /** Whether the node starts collapsed (it does: compact until expanded) */
  defaultCollapsed?: boolean;
  /** Controlled collapsed state */
  isCollapsed?: boolean;
  /** Callback when collapsed state changes */
  onCollapsedChange?: (collapsed: boolean) => void;
  /** Content shown when expanded (the form fields) */
  children: ReactNode;
  /** Content always shown even when collapsed (like key info) */
  collapsedPreview?: ReactNode;
  /** Resize handles component (only shown when expanded) */
  resizeHandles?: ReactNode;
  /** Handle configurations for left side (inputs) */
  inputHandles?: HandleConfig[];
  /** Handle configurations for right side (outputs) */
  outputHandles?: HandleConfig[];
  /** Handle configurations for top */
  topHandles?: HandleConfig[];
  /** Handle configurations for bottom */
  bottomHandles?: HandleConfig[];
  /** Additional className for the container */
  className?: string;
  /** Extra content to show in title row (badges, counts, etc) */
  titleExtra?: ReactNode;
}

// (`text`: the kind's word over the name, 9px: the 700 ink on the light header, about 4.5:1; the 600 was about 3:1)
const colorMap = {
  purple: {
    border: 'border-purple-400 dark:border-purple-600',
    headerBg: 'bg-purple-100 dark:bg-purple-900/50',
    iconBg: 'bg-purple-500 dark:bg-purple-600',
    text: 'text-purple-700 dark:text-purple-400',
    hoverBg: 'hover:bg-purple-200/50 dark:hover:bg-purple-500/30',
  },
  violet: {
    border: 'border-violet-400 dark:border-violet-600',
    headerBg: 'bg-violet-100 dark:bg-violet-900/50',
    iconBg: 'bg-violet-500 dark:bg-violet-600',
    text: 'text-violet-700 dark:text-violet-400',
    hoverBg: 'hover:bg-violet-200/50 dark:hover:bg-violet-500/30',
  },
  green: {
    border: 'border-green-400 dark:border-green-600',
    headerBg: 'bg-green-100 dark:bg-green-900/50',
    iconBg: 'bg-green-500 dark:bg-green-600',
    text: 'text-green-700 dark:text-green-400',
    hoverBg: 'hover:bg-green-200/50 dark:hover:bg-green-500/30',
  },
  blue: {
    border: 'border-blue-400 dark:border-blue-600',
    headerBg: 'bg-blue-100 dark:bg-blue-900/50',
    iconBg: 'bg-blue-500 dark:bg-blue-600',
    text: 'text-blue-700 dark:text-blue-400',
    hoverBg: 'hover:bg-blue-200/50 dark:hover:bg-blue-500/30',
  },
  amber: {
    border: 'border-amber-400 dark:border-amber-500',
    headerBg: 'bg-amber-100 dark:bg-amber-900/50',
    iconBg: 'bg-amber-500 dark:bg-amber-600',
    text: 'text-amber-700 dark:text-amber-400',
    hoverBg: 'hover:bg-amber-200/50 dark:hover:bg-amber-500/30',
  },
  cyan: {
    border: 'border-cyan-400 dark:border-cyan-600',
    headerBg: 'bg-cyan-100 dark:bg-cyan-900/50',
    iconBg: 'bg-cyan-500 dark:bg-cyan-600',
    text: 'text-cyan-700 dark:text-cyan-400',
    hoverBg: 'hover:bg-cyan-200/50 dark:hover:bg-cyan-500/30',
  },
  pink: {
    border: 'border-pink-400 dark:border-pink-600',
    headerBg: 'bg-pink-100 dark:bg-pink-900/50',
    iconBg: 'bg-pink-500 dark:bg-pink-600',
    text: 'text-pink-700 dark:text-pink-400',
    hoverBg: 'hover:bg-pink-200/50 dark:hover:bg-pink-500/30',
  },
  emerald: {
    border: 'border-emerald-400 dark:border-emerald-600',
    headerBg: 'bg-emerald-100 dark:bg-emerald-900/50',
    iconBg: 'bg-emerald-500 dark:bg-emerald-600',
    text: 'text-emerald-700 dark:text-emerald-400',
    hoverBg: 'hover:bg-emerald-200/50 dark:hover:bg-emerald-500/30',
  },
  orange: {
    border: 'border-orange-400 dark:border-orange-600',
    headerBg: 'bg-orange-100 dark:bg-orange-900/50',
    iconBg: 'bg-orange-500 dark:bg-orange-600',
    text: 'text-orange-700 dark:text-orange-400',
    hoverBg: 'hover:bg-orange-200/50 dark:hover:bg-orange-500/30',
  },
  red: {
    border: 'border-red-400 dark:border-red-600',
    headerBg: 'bg-red-100 dark:bg-red-900/50',
    iconBg: 'bg-red-500 dark:bg-red-600',
    text: 'text-red-700 dark:text-red-400',
    hoverBg: 'hover:bg-red-200/50 dark:hover:bg-red-500/30',
  },
  slate: {
    border: 'border-slate-400 dark:border-slate-600',
    headerBg: 'bg-slate-200 dark:bg-slate-700/50',
    iconBg: 'bg-slate-500 dark:bg-slate-600',
    text: 'text-slate-700 dark:text-slate-400',
    hoverBg: 'hover:bg-slate-300/50 dark:hover:bg-slate-500/30',
  },
  indigo: {
    border: 'border-indigo-400 dark:border-indigo-600',
    headerBg: 'bg-indigo-100 dark:bg-indigo-900/50',
    iconBg: 'bg-indigo-500 dark:bg-indigo-600',
    text: 'text-indigo-700 dark:text-indigo-400',
    hoverBg: 'hover:bg-indigo-200/50 dark:hover:bg-indigo-500/30',
  },
  teal: {
    border: 'border-teal-400 dark:border-teal-600',
    headerBg: 'bg-teal-100 dark:bg-teal-900/50',
    iconBg: 'bg-teal-500 dark:bg-teal-600',
    text: 'text-teal-700 dark:text-teal-400',
    hoverBg: 'hover:bg-teal-200/50 dark:hover:bg-teal-500/30',
  },
};

// A port's row: its handle sits in it, so the card can be as tall as its name and summary need.
const ROW = 22;
// (a collapsed card at least this wide: room for its ports' names side by side)
const MIN_COLLAPSED = 232;

// Run status → visible pill + screen-reader text. Semantic status hues (not the
// customizable accent): amber=running, green=completed, red=error, and a neutral
// grey for stopped (Stop found it running: nothing of its own went wrong), each
// with a dark variant so it reads on both the cream and dark canvas.
const STATUS_META = {
  running: { label: 'Running', pill: 'bg-amber-100 text-amber-700 dark:bg-amber-900/40 dark:text-amber-300', dot: 'bg-amber-500' },
  completed: { label: 'Completed', pill: 'bg-green-100 text-green-700 dark:bg-green-900/40 dark:text-green-300', dot: 'bg-green-500' },
  error: { label: 'Error', pill: 'bg-red-100 text-red-700 dark:bg-red-900/40 dark:text-red-300', dot: 'bg-red-500' },
  stopped: { label: 'Stopped', pill: 'bg-slate-200 text-slate-700 dark:bg-slate-700/60 dark:text-slate-200', dot: 'bg-slate-400' },
} as const;

/** A port's name as its row shows it: its label, else its id (a node's one output, `output`, as "out"). */
function portName(h: HandleConfig): string {
  if (h.label) return h.label;
  return h.id === 'default' || h.id === 'output' || h.id === 'out' ? 'out' : h.id === 'input' ? 'in' : h.id.replace(/_/g, ' ');
}

/**
 * The node's definition's port for a handle: by its id, else (a card naming its one input or output its own way,
 * Logic Block's `output`) the definition's only one of that side.
 */
function definedPort(h: HandleConfig, given: HandleConfig[], defined: HandleDefinition[] | undefined): HandleDefinition | undefined {
  if (!defined?.length) return undefined;
  return defined.find((d) => d.id === h.id) ?? (given.length === 1 && defined.length === 1 ? defined[0] : undefined);
}

/**
 * A node's card on the canvas. Compact until expanded (the default): its name (the node's own label where it has one,
 * its kind above it), its ports a row each with their names (inputs left, outputs right), and a line of what it is
 * set to; so a flow reads at a glance, how its steps feed each other, before any one is opened. Expanded, its form
 * below the ports. The handles sit on their ports' rows.
 */
function CollapsibleNodeWrapper({
  title,
  color,
  icon: givenIcon,
  ownIcon = false,
  width,
  collapsedWidth = MIN_COLLAPSED,
  status: givenStatus,
  validationIssues = [],
  defaultCollapsed = true,
  isCollapsed: controlledCollapsed,
  onCollapsedChange,
  children,
  collapsedPreview,
  resizeHandles,
  inputHandles: givenInputs = [],
  outputHandles: givenOutputs = [],
  topHandles: givenTop = [],
  bottomHandles: givenBottom = [],
  className = '',
  titleExtra,
}: CollapsibleNodeWrapperProps) {
  const [internalCollapsed, setInternalCollapsed] = useState(defaultCollapsed);
  // The node's own name (a flow's "Make the picture" over its kind's "Image Gen"), where it has one.
  const nodeId = useNodeId();
  const label = useStore((st) => {
    const raw = nodeId ? (st.nodeLookup.get(nodeId)?.data as { label?: unknown } | undefined)?.label : undefined;
    return typeof raw === 'string' ? raw.trim() : '';
  });
  const name = label && label !== title ? label : title;
  const kind = label && label !== title ? title : '';
  // Its run state as the run sets it (`_status`), where the card does not say it itself: a card made from its
  // definition alone, and a few others, never showed which step was running or which one failed.
  const dataStatus = useStore((st) => {
    const raw = nodeId ? (st.nodeLookup.get(nodeId)?.data as { _status?: unknown } | undefined)?._status : undefined;
    return raw === 'running' || raw === 'completed' || raw === 'error' || raw === 'stopped' ? raw : undefined;
  });
  const status = givenStatus ?? dataStatus;
  // What the host says of the node (the model it uses is offline or not installed): a warning mark on the card,
  // folded or not, a short line on the folded card and the whole notice on the expanded one. It was on the expanded
  // card alone, so a flow of folded cards showed nothing wrong until it ran.
  const nodeType = useStore((st) => (nodeId ? st.nodeLookup.get(nodeId)?.type : undefined));
  const nodeData = useStore((st) => (nodeId ? (st.nodeLookup.get(nodeId)?.data as Record<string, unknown> | undefined) : undefined));
  const notice = useNodeNoticeDetail(nodeType, nodeData);
  // Its definition: the palette's icon for it, and its ports' names and kinds of value.
  const definition = useNodeDefinition();
  const icon = (!ownIcon && resolveNodeIcon(definition?.icon)) || givenIcon;

  // Support both controlled and uncontrolled modes
  const isCollapsed = controlledCollapsed !== undefined ? controlledCollapsed : internalCollapsed;

  const colors = colorMap[color] ?? colorMap.slate;
  const hasIssues = validationIssues.length > 0;
  const statusClass = status ? `node-${status}` : '';
  const currentWidth = isCollapsed ? Math.max(collapsedWidth, MIN_COLLAPSED) : Math.max(width, MIN_COLLAPSED);
  // A port on a side is a row of the card; one on its top or bottom edge (a loop's link from its start to its end,
  // given among the inputs or outputs) sits on that edge, not in a row.
  const side = (h: HandleConfig) => h.position === Position.Left || h.position === Position.Right;
  const inputHandles = givenInputs.filter(side);
  const outputHandles = givenOutputs.filter(side);
  const topHandles = [...givenTop, ...givenInputs.filter((h) => h.position === Position.Top), ...givenOutputs.filter((h) => h.position === Position.Top)];
  const bottomHandles = [...givenBottom, ...givenInputs.filter((h) => h.position === Position.Bottom), ...givenOutputs.filter((h) => h.position === Position.Bottom)];
  const rows = Math.max(inputHandles.length, outputHandles.length);

  // Sync with controlled state
  useEffect(() => {
    if (controlledCollapsed !== undefined) {
      setInternalCollapsed(controlledCollapsed);
    }
  }, [controlledCollapsed]);

  const toggleCollapse = useCallback(() => {
    const newState = !isCollapsed;
    setInternalCollapsed(newState);
    onCollapsedChange?.(newState);
  }, [isCollapsed, onCollapsedChange]);

  // A port as its definition has it (where the node's module defines it): its name ("Prompt", the palette's and
  // Properties' word for it, not a card's own "prompt" or "desc") and its kind of value, as its dot's colour. A port
  // the definition does not have (a template's variables, a macro's ports) keeps what its card gives it.
  const portOf = (h: HandleConfig): { name: string; colour: string; tip: string } => {
    const given = h.type === 'target' ? givenInputs : givenOutputs;
    const def = definedPort(h, given, h.type === 'target' ? definition?.inputs : definition?.outputs);
    const name = (h.ownLabel && h.label) || def?.name || portName(h);
    return def
      ? { name, colour: portTypeClass(def.type), tip: `${name}: ${typeWords(def.type)}` }
      : { name, colour: h.color, tip: name };
  };

  // A side's handle sits in its port's row (centred on it); a top or bottom one is spread along that edge.
  const renderHandle = (handle: HandleConfig, index: number, total: number) => {
    const port = portOf(handle);
    return (
      <Handle
        // (by kind as well: a node may take in and give out a port of the same name, Image View's `image`)
        key={`${handle.type}:${handle.id}`}
        type={handle.type}
        position={handle.position}
        id={handle.id}
        className={`oaiy-port ${port.colour}`}
        style={handle.position === Position.Left || handle.position === Position.Right
          ? { top: '50%', zIndex: 20 }
          : { left: `${(currentWidth / (total + 1)) * (index + 1)}px`, zIndex: 20 }}
        isConnectable={true}
        title={port.tip}
      />
    );
  };

  const issuesText = validationIssues.map((i) => `${i.field}: ${i.message}`).join('\n');

  return (
    <div
      className={`oaiy-node ${colors.border} ${statusClass} ${isCollapsed ? 'collapsed' : 'expanded'} ${className}`}
      style={{ width: currentWidth }}
    >
      <div className="oaiy-node-card">
        {/* Header: the kind, the name, its state; double-click (or the chevron) opens it */}
        <div className={`oaiy-node-head ${colors.headerBg}`} onDoubleClick={toggleCollapse}>
          <div className={`oaiy-node-icon ${colors.iconBg}`}>{icon}</div>
          <div className="oaiy-node-names">
            {kind && <span className={`oaiy-node-kind ${colors.text}`}>{kind}</span>}
            <span className="oaiy-node-name" title={name}>{name}</span>
          </div>
          {titleExtra && !isCollapsed && <div className="flex items-center gap-1">{titleExtra}</div>}
          {status && (
            <div
              role="status"
              aria-live="polite"
              title={STATUS_META[status].label}
              className={`flex items-center gap-1 flex-shrink-0 ${isCollapsed ? '' : 'px-1.5 py-0.5 rounded-full ' + STATUS_META[status].pill}`}
            >
              <span aria-hidden="true" className={`w-1.5 h-1.5 rounded-full ${STATUS_META[status].dot} ${status === 'running' ? 'animate-pulse' : ''}`} />
              {!isCollapsed && <span className="text-[10px] font-medium leading-none">{STATUS_META[status].label}</span>}
              <span className="sr-only">{`Status: ${STATUS_META[status].label}`}</span>
            </div>
          )}
          {notice && (
            <div role="img" aria-label={`Warning: ${notice.text}`} className="oaiy-node-warn" title={notice.text}>
              <svg aria-hidden="true" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
              </svg>
            </div>
          )}
          {hasIssues && (
            <div
              role="img"
              aria-label={`Validation ${validationIssues.length === 1 ? 'issue' : 'issues'}: ${issuesText.replace(/\n/g, '; ')}`}
              className="w-4 h-4 bg-amber-500 rounded-full flex items-center justify-center flex-shrink-0"
              title={issuesText}
            >
              <svg aria-hidden="true" className="w-2.5 h-2.5 text-slate-900" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
              </svg>
            </div>
          )}
          <button
            type="button"
            className={`nodrag oaiy-node-toggle ${colors.hoverBg}`}
            onClick={toggleCollapse}
            title={isCollapsed ? 'Show its settings' : 'Make it compact'}
            aria-label={isCollapsed ? `Expand ${name} node` : `Collapse ${name} node`}
            aria-expanded={!isCollapsed}
          >
            <svg aria-hidden="true" className={`w-3 h-3 transition-transform duration-200 ${isCollapsed ? '' : 'rotate-180'}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
          </button>
        </div>

        {/* Its ports, a row each with its handle: what comes in on the left, what goes out on the right */}
        {rows > 0 && (
          <div className="oaiy-node-ports">
            {Array.from({ length: rows }).map((_, i) => (
              <div key={i} className="oaiy-node-port" style={{ minHeight: ROW }}>
                {inputHandles[i] && renderHandle(inputHandles[i], i, inputHandles.length)}
                {/* (its name in the card's own ink: the dot beside it says what kind of value it is) */}
                <span>{inputHandles[i] ? portOf(inputHandles[i]).name : ''}</span>
                <span>{outputHandles[i] ? portOf(outputHandles[i]).name : ''}</span>
                {outputHandles[i] && renderHandle(outputHandles[i], i, outputHandles.length)}
              </div>
            ))}
          </div>
        )}

        {isCollapsed ? (
          <>
            {notice && <div className="oaiy-node-notice" title={notice.text}>{notice.short}</div>}
            {collapsedPreview && <div className="oaiy-node-preview">{collapsedPreview}</div>}
          </>
        ) : (
          <div className="oaiy-node-body">
            {notice && <div role="status" className="oaiy-node-notice expanded">{notice.text}</div>}
            {children}
          </div>
        )}
      </div>

      {topHandles.map((h, i) => renderHandle(h, i, topHandles.length))}
      {bottomHandles.map((h, i) => renderHandle(h, i, bottomHandles.length))}

      {/* Resize Handles - only when expanded */}
      {!isCollapsed && resizeHandles}
    </div>
  );
}

export default memo(CollapsibleNodeWrapper);
