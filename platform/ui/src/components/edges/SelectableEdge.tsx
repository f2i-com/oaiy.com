import { memo } from 'react';
import {
  BaseEdge,
  EdgeLabelRenderer,
  getSmoothStepPath,
  useReactFlow,
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
}: EdgeProps) {
  const { setEdges } = useReactFlow();

  const [edgePath, labelX, labelY] = getSmoothStepPath({
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
  const strokeColor = selected
    ? 'rgb(var(--accent-primary))'
    : 'rgb(var(--color-text-tertiary))';
  const strokeWidth = selected ? 4 : 3;
  const startCircleColor = selected
    ? 'rgb(var(--accent-primary))'
    : 'rgb(var(--color-text-secondary))';

  return (
    <>
      {/* Arrow marker definition */}
      <defs>
        <marker
          id={`arrow-${id}`}
          markerWidth="12"
          markerHeight="12"
          refX="10"
          refY="6"
          orient="auto"
          markerUnits="userSpaceOnUse"
        >
          <path
            d="M2,2 L10,6 L2,10 L4,6 Z"
            style={{ fill: strokeColor }}
          />
        </marker>
      </defs>
      {/* Start circle marker at source */}
      <circle
        cx={sourceX}
        cy={sourceY}
        r={selected ? 7 : 6}
        strokeWidth={1.5}
        style={{ fill: startCircleColor, stroke: strokeColor }}
      />
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
