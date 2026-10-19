import { memo } from 'react';
import { X } from 'lucide-react';
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

  // Its colour by state, in index.css (`.react-flow__edge-path.lit` / `.picked`, `.oaiy-edge-arrow`): the theme's
  // tokens and the accent, and a hover that shows (an inline stroke outranked the stylesheet's hover).
  const lit = selected || nodeSelected;
  const state = selected ? 'picked' : lit ? 'lit' : '';

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
          <path d="M1,1 L8,4.5 L1,8 L2.5,4.5 Z" className={`oaiy-edge-arrow ${state}`} />
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
        className={state}
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
            {/* (named: its one glyph was its name to a screen reader) */}
            <button type="button" onClick={handleDelete} className="oaiy-edge-delete" aria-label="Delete this wire" title="Delete this wire">
              <X size={12} strokeWidth={2.5} aria-hidden="true" />
            </button>
          </div>
        </EdgeLabelRenderer>
      )}
    </>
  );
}

export default memo(SelectableEdge);
