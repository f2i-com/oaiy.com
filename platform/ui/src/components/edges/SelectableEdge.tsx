import { memo } from 'react';
import {
  BaseEdge,
  EdgeLabelRenderer,
  getBezierPath,
  useReactFlow,
  useStore,
  type EdgeProps,
} from '@xyflow/react';

function SelectableEdge({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  selected,
  source,
  target,
}: EdgeProps) {
  const { setEdges } = useReactFlow();
  // (a selected node's wires stand out with it: what feeds it and what it feeds, at a glance)
  const nodeSelected = useStore((st) => !!(st.nodeLookup.get(source)?.selected || st.nodeLookup.get(target)?.selected));

  // A soft curve between the two ports (right angles made long links a grid of crossing lines).
  const [edgePath, labelX, labelY] = getBezierPath({
    sourceX,
    sourceY,
    sourcePosition,
    targetX,
    targetY,
    targetPosition,
  });

  const handleDelete = (event: React.MouseEvent) => {
    event.stopPropagation();
    setEdges((edges) => edges.filter((edge) => edge.id !== id));
  };

  // Colors based on selection state — driven by the theme tokens so edges
  // track light/dark and the user's customizable accent. These are CSS custom
  // properties, so they must be applied via `style` (SVG attributes don't
  // resolve var()); selected uses the brand accent, idle uses muted text tones.
  const lit = selected || nodeSelected;
  const strokeColor = lit
    ? 'rgb(var(--accent-primary))'
    : 'rgb(var(--color-text-tertiary) / 0.75)';
  const strokeWidth = selected ? 3 : lit ? 2.25 : 1.75;

  return (
    <>
      {/* Arrow marker definition */}
      <defs>
        <marker
          id={`arrow-${id}`}
          markerWidth="9"
          markerHeight="9"
          refX="8"
          refY="4.5"
          orient="auto"
          markerUnits="userSpaceOnUse"
        >
          <path
            d="M1,1 L8,4.5 L1,8 L2.5,4.5 Z"
            style={{ fill: strokeColor }}
          />
        </marker>
      </defs>
      {/* Invisible wider path for easier clicking */}
      <path
        id={`${id}-interaction`}
        d={edgePath}
        fill="none"
        stroke="transparent"
        strokeWidth={25}
        className="react-flow__edge-interaction"
        style={{ pointerEvents: 'stroke', cursor: 'pointer' }}
      />
      {/* Visible edge path with arrow */}
      <BaseEdge
        id={id}
        path={edgePath}
        markerEnd={`url(#arrow-${id})`}
        style={{
          stroke: strokeColor,
          strokeWidth: strokeWidth,
        }}
      />
      {/* Delete button when selected */}
      {selected && (
        <EdgeLabelRenderer>
          <div
            style={{
              position: 'absolute',
              transform: `translate(-50%, -50%) translate(${labelX}px,${labelY}px)`,
              pointerEvents: 'all',
            }}
            className="nodrag nopan"
          >
            <button
              onClick={handleDelete}
              className="bg-red-600 hover:bg-red-500 text-white rounded-full w-5 h-5 flex items-center justify-center text-xs cursor-pointer shadow-lg border border-red-400 transition-colors"
            >
              ×
            </button>
          </div>
        </EdgeLabelRenderer>
      )}
    </>
  );
}

export default memo(SelectableEdge);
