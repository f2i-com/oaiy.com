import { memo, useState, useCallback, useEffect, type ReactNode } from 'react';
import { Handle, Position } from '@xyflow/react';

export interface ValidationIssue {
  field: string;
  message: string;
}

export interface HandleConfig {
  id: string;
  type: 'source' | 'target';
  position: Position;
  color: string; // tailwind color class like 'bg-blue-500'
  label?: string;
  labelColor?: string; // tailwind text color class
  size?: 'sm' | 'md' | 'lg'; // default md
}

export interface CollapsibleNodeWrapperProps {
  /** Node title displayed in header */
  title: string;
  /** Color theme for the node (tailwind color name without prefix) */
  color: 'purple' | 'green' | 'blue' | 'amber' | 'cyan' | 'pink' | 'emerald' | 'orange' | 'red' | 'slate' | 'indigo' | 'teal' | 'violet';
  /** Icon SVG element */
  icon: ReactNode;
  /** Width of the node when expanded */
  width: number;
  /** Width when collapsed (default: 160) */
  collapsedWidth?: number;
  /** Status for visual styling */
  status?: 'running' | 'completed' | 'error';
  /** Validation issues to display warning indicator */
  validationIssues?: ValidationIssue[];
  /** Whether the node starts collapsed */
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

const colorMap = {
  purple: {
    border: 'border-purple-400 dark:border-purple-600',
    headerBg: 'bg-purple-100 dark:bg-purple-900/50',
    iconBg: 'bg-purple-500 dark:bg-purple-600',
    text: 'text-purple-600 dark:text-purple-400',
    hoverBg: 'hover:bg-purple-200/50 dark:hover:bg-purple-500/30',
  },
  violet: {
    border: 'border-violet-400 dark:border-violet-600',
    headerBg: 'bg-violet-100 dark:bg-violet-900/50',
    iconBg: 'bg-violet-500 dark:bg-violet-600',
    text: 'text-violet-600 dark:text-violet-400',
    hoverBg: 'hover:bg-violet-200/50 dark:hover:bg-violet-500/30',
  },
  green: {
    border: 'border-green-400 dark:border-green-600',
    headerBg: 'bg-green-100 dark:bg-green-900/50',
    iconBg: 'bg-green-500 dark:bg-green-600',
    text: 'text-green-600 dark:text-green-400',
    hoverBg: 'hover:bg-green-200/50 dark:hover:bg-green-500/30',
  },
  blue: {
    border: 'border-blue-400 dark:border-blue-600',
    headerBg: 'bg-blue-100 dark:bg-blue-900/50',
    iconBg: 'bg-blue-500 dark:bg-blue-600',
    text: 'text-blue-600 dark:text-blue-400',
    hoverBg: 'hover:bg-blue-200/50 dark:hover:bg-blue-500/30',
  },
  amber: {
    border: 'border-amber-400 dark:border-amber-500',
    headerBg: 'bg-amber-100 dark:bg-amber-900/50',
    iconBg: 'bg-amber-500 dark:bg-amber-600',
    text: 'text-amber-600 dark:text-amber-400',
    hoverBg: 'hover:bg-amber-200/50 dark:hover:bg-amber-500/30',
  },
  cyan: {
    border: 'border-cyan-400 dark:border-cyan-600',
    headerBg: 'bg-cyan-100 dark:bg-cyan-900/50',
    iconBg: 'bg-cyan-500 dark:bg-cyan-600',
    text: 'text-cyan-600 dark:text-cyan-400',
    hoverBg: 'hover:bg-cyan-200/50 dark:hover:bg-cyan-500/30',
  },
  pink: {
    border: 'border-pink-400 dark:border-pink-600',
    headerBg: 'bg-pink-100 dark:bg-pink-900/50',
    iconBg: 'bg-pink-500 dark:bg-pink-600',
    text: 'text-pink-600 dark:text-pink-400',
    hoverBg: 'hover:bg-pink-200/50 dark:hover:bg-pink-500/30',
  },
  emerald: {
    border: 'border-emerald-400 dark:border-emerald-600',
    headerBg: 'bg-emerald-100 dark:bg-emerald-900/50',
    iconBg: 'bg-emerald-500 dark:bg-emerald-600',
    text: 'text-emerald-600 dark:text-emerald-400',
    hoverBg: 'hover:bg-emerald-200/50 dark:hover:bg-emerald-500/30',
  },
  orange: {
    border: 'border-orange-400 dark:border-orange-600',
    headerBg: 'bg-orange-100 dark:bg-orange-900/50',
    iconBg: 'bg-orange-500 dark:bg-orange-600',
    text: 'text-orange-600 dark:text-orange-400',
    hoverBg: 'hover:bg-orange-200/50 dark:hover:bg-orange-500/30',
  },
  red: {
    border: 'border-red-400 dark:border-red-600',
    headerBg: 'bg-red-100 dark:bg-red-900/50',
    iconBg: 'bg-red-500 dark:bg-red-600',
    text: 'text-red-600 dark:text-red-400',
    hoverBg: 'hover:bg-red-200/50 dark:hover:bg-red-500/30',
  },
  slate: {
    border: 'border-slate-400 dark:border-slate-600',
    headerBg: 'bg-slate-200 dark:bg-slate-700/50',
    iconBg: 'bg-slate-500 dark:bg-slate-600',
    text: 'text-slate-600 dark:text-slate-400',
    hoverBg: 'hover:bg-slate-300/50 dark:hover:bg-slate-500/30',
  },
  indigo: {
    border: 'border-indigo-400 dark:border-indigo-600',
    headerBg: 'bg-indigo-100 dark:bg-indigo-900/50',
    iconBg: 'bg-indigo-500 dark:bg-indigo-600',
    text: 'text-indigo-600 dark:text-indigo-400',
    hoverBg: 'hover:bg-indigo-200/50 dark:hover:bg-indigo-500/30',
  },
  teal: {
    border: 'border-teal-400 dark:border-teal-600',
    headerBg: 'bg-teal-100 dark:bg-teal-900/50',
    iconBg: 'bg-teal-500 dark:bg-teal-600',
    text: 'text-teal-600 dark:text-teal-400',
    hoverBg: 'hover:bg-teal-200/50 dark:hover:bg-teal-500/30',
  },
};

const sizeMap = {
  sm: '!w-4 !h-4',
  md: '!w-5 !h-5',
  lg: '!w-6 !h-6',
};

// Run status → visible pill + screen-reader text. Semantic status hues (not the
// customizable accent): amber=running, green=completed, red=error, each with a
// dark variant so it reads on both the cream and dark canvas.
const STATUS_META = {
  running: { label: 'Running', pill: 'bg-amber-100 text-amber-700 dark:bg-amber-900/40 dark:text-amber-300', dot: 'bg-amber-500' },
  completed: { label: 'Completed', pill: 'bg-green-100 text-green-700 dark:bg-green-900/40 dark:text-green-300', dot: 'bg-green-500' },
  error: { label: 'Error', pill: 'bg-red-100 text-red-700 dark:bg-red-900/40 dark:text-red-300', dot: 'bg-red-500' },
} as const;

function CollapsibleNodeWrapper({
  title,
  color,
  icon,
  width,
  collapsedWidth = 160,
  status,
  validationIssues = [],
  defaultCollapsed = false,
  isCollapsed: controlledCollapsed,
  onCollapsedChange,
  children,
  collapsedPreview,
  resizeHandles,
  inputHandles = [],
  outputHandles = [],
  topHandles = [],
  bottomHandles = [],
  className = '',
  titleExtra,
}: CollapsibleNodeWrapperProps) {
  const [internalCollapsed, setInternalCollapsed] = useState(defaultCollapsed);

  // Support both controlled and uncontrolled modes
  const isCollapsed = controlledCollapsed !== undefined ? controlledCollapsed : internalCollapsed;

  const colors = colorMap[color];
  const hasIssues = validationIssues.length > 0;
  const statusClass = status ? `node-${status}` : '';
  const currentWidth = isCollapsed ? collapsedWidth : width;

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

  // Calculate handle positions - evenly distribute on edge
  const getHandleStyle = (index: number, total: number, position: Position): React.CSSProperties => {
    const headerHeight = 40; // header is ~40px

    if (position === Position.Left || position === Position.Right) {
      // Vertical distribution. When collapsed, distribute handles compactly
      // (smaller gap/offset) rather than clamping every index to a single 50px
      // point — that stacked all of a multi-input/output node's handles on top of
      // one another, leaving them visually indistinguishable + untargetable.
      const startOffset = isCollapsed ? headerHeight + 6 : headerHeight + 15;
      const gap = isCollapsed ? 12 : 25;
      const top = startOffset + (index * gap);
      return { top: `${top}px` };
    } else {
      // Horizontal distribution (top/bottom)
      const totalWidth = currentWidth;
      const gap = totalWidth / (total + 1);
      const left = gap * (index + 1);
      return { left: `${left}px` };
    }
  };

  const renderHandle = (handle: HandleConfig, index: number, total: number) => {
    const sizeClass = sizeMap[handle.size || 'md'];
    const baseStyle = getHandleStyle(index, total, handle.position);
    // Add z-index to ensure handles render above labels
    const style = { ...baseStyle, zIndex: 20 };

    return (
      <Handle
        key={handle.id}
        type={handle.type}
        position={handle.position}
        id={handle.id}
        className={`${handle.color} ${sizeClass}`}
        style={style}
        isConnectable={true}
      />
    );
  };

  // Render handle labels only when expanded
  const renderHandleLabels = (handles: HandleConfig[], position: Position) => {
    if (isCollapsed) return null;

    const headerHeight = 40;
    const startOffset = headerHeight + 15;
    const gap = 25;

    return handles.map((handle, index) => {
      if (!handle.label) return null;
      const top = startOffset + (index * gap) - 4;
      // Position labels with pointer-events: none so they don't block handle interaction
      // Left labels: position to left of node, right-aligned (text ends near handle)
      // Right labels: position to right of node, left-aligned (text starts near handle)
      const labelStyle: React.CSSProperties = position === Position.Left
        ? { right: `calc(100% + 8px)`, top: `${top}px`, pointerEvents: 'none', zIndex: 10 }
        : { left: `calc(100% + 8px)`, top: `${top}px`, pointerEvents: 'none', zIndex: 10 };

      return (
        <div
          key={`label-${handle.id}`}
          className={`absolute text-[9px] ${handle.labelColor || 'text-slate-600 dark:text-slate-400'} bg-white/90 dark:bg-slate-900/80 px-1 rounded whitespace-nowrap`}
          style={labelStyle}
        >
          {handle.label}
        </div>
      );
    });
  };

  return (
    <div
      className={`border-2 ${colors.border} rounded-lg shadow-xl overflow-visible relative group transition-all duration-200 ${statusClass} ${className}`}
      style={{ width: currentWidth }}
    >
      {/* Inner wrapper - clips header and content to match rounded corners */}
      <div className="bg-white dark:bg-slate-800 rounded-md overflow-hidden">
        {/* Header - draggable, collapse button on right */}
        <div
          className={`${colors.headerBg} px-2 py-1.5 border-b border-slate-300 dark:border-slate-700 flex items-center gap-2 select-none`}
        >
          {/* Icon */}
          <div className={`w-5 h-5 rounded-full ${colors.iconBg} flex items-center justify-center flex-shrink-0`}>
            {icon}
          </div>

          {/* Title - truncate when collapsed */}
          <span className={`${colors.text} font-semibold text-xs ${titleExtra ? '' : 'flex-grow'} truncate`}>
            {title}
          </span>

          {/* Title extra content (badges, counts, etc) */}
          {titleExtra && !isCollapsed && (
            <div className="flex items-center gap-1 flex-grow">
              {titleExtra}
            </div>
          )}

          {/* Run-status pill — visible label when expanded, always announced to
              screen readers (role=status + aria-live) so status changes are heard. */}
          {status && (
            <div
              role="status"
              aria-live="polite"
              title={STATUS_META[status].label}
              className={`flex items-center gap-1 flex-shrink-0 ${isCollapsed ? '' : 'px-1.5 py-0.5 rounded-full ' + STATUS_META[status].pill}`}
            >
              <span
                aria-hidden="true"
                className={`w-1.5 h-1.5 rounded-full ${STATUS_META[status].dot} ${status === 'running' ? 'animate-pulse' : ''}`}
              />
              {!isCollapsed && (
                <span className="text-[10px] font-medium leading-none">{STATUS_META[status].label}</span>
              )}
              <span className="sr-only">{`Status: ${STATUS_META[status].label}`}</span>
            </div>
          )}

          {/* Validation indicator in header when collapsed */}
          {isCollapsed && hasIssues && (
            <div
              role="img"
              aria-label={`Validation ${validationIssues.length === 1 ? 'issue' : 'issues'}: ${validationIssues.map(i => `${i.field}: ${i.message}`).join('; ')}`}
              className="w-4 h-4 bg-amber-500 rounded-full flex items-center justify-center flex-shrink-0"
              title={validationIssues.map(i => `${i.field}: ${i.message}`).join('\n')}
            >
              <svg aria-hidden="true" className="w-2.5 h-2.5 text-slate-900" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
              </svg>
            </div>
          )}

          {/* Collapse toggle button - nodrag to prevent drag when clicking */}
          <button
            type="button"
            className={`nodrag p-0.5 rounded ${colors.hoverBg} transition-colors flex-shrink-0`}
            onClick={toggleCollapse}
            title={isCollapsed ? 'Expand' : 'Collapse'}
            aria-label={isCollapsed ? `Expand ${title} node` : `Collapse ${title} node`}
            aria-expanded={!isCollapsed}
          >
            <svg
              aria-hidden="true"
              className={`w-3 h-3 ${colors.text} transition-transform duration-200 ${isCollapsed ? '' : 'rotate-180'}`}
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
          </button>
        </div>

        {/* Content */}
        {isCollapsed ? (
          // Collapsed preview (minimal info)
          collapsedPreview && (
            <div className="px-2 py-1.5 text-[10px]">
              {collapsedPreview}
            </div>
          )
        ) : (
          // Full content
          <div className="p-3 space-y-3">
            {children}
          </div>
        )}
      </div>

      {/* Handles - rendered by wrapper with calculated positions */}
      {inputHandles.map((h, i) => renderHandle(h, i, inputHandles.length))}
      {outputHandles.map((h, i) => renderHandle(h, i, outputHandles.length))}
      {topHandles.map((h, i) => renderHandle(h, i, topHandles.length))}
      {bottomHandles.map((h, i) => renderHandle(h, i, bottomHandles.length))}

      {/* Handle labels - only when expanded */}
      {renderHandleLabels(inputHandles, Position.Left)}
      {renderHandleLabels(outputHandles, Position.Right)}

      {/* Validation Warning Indicator - only when expanded */}
      {!isCollapsed && hasIssues && (
        <div
          role="img"
          aria-label={`Validation ${validationIssues.length === 1 ? 'issue' : 'issues'}: ${validationIssues.map(i => `${i.field}: ${i.message}`).join('; ')}`}
          className="absolute -bottom-1 -right-1 w-5 h-5 bg-amber-500 rounded-full flex items-center justify-center cursor-help shadow-lg border-2 border-slate-800 z-10"
          title={validationIssues.map(i => `${i.field}: ${i.message}`).join('\n')}
        >
          <svg aria-hidden="true" className="w-3 h-3 text-slate-900" fill="currentColor" viewBox="0 0 20 20">
            <path
              fillRule="evenodd"
              d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z"
              clipRule="evenodd"
            />
          </svg>
        </div>
      )}

      {/* Resize Handles - only when expanded */}
      {!isCollapsed && resizeHandles}
    </div>
  );
}

export default memo(CollapsibleNodeWrapper);
