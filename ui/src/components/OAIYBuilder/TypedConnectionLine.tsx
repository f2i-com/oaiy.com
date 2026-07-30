/**
 * Connection line drawn while the user drags from a handle.
 *
 * React Flow gives us `connectionStatus` derived from the
 * `isValidConnection` callback on the parent ReactFlow component:
 *  - `'valid'`   — hover target is a compatible handle (green line).
 *  - `'invalid'` — hover target is INCOMPATIBLE (red line + dashed).
 *  - `null`      — not over any target (default blue, current look).
 *
 * Only renders the SVG <path>; the wrapping <g class="react-flow__connectionline">
 * comes from React Flow's internal wrapper, so we stay focused on the
 * line's geometry + colour.
 */

import type { ConnectionLineComponentProps } from '@xyflow/react';
import { getBezierPath } from '@xyflow/react';

function strokeFor(status: 'valid' | 'invalid' | null): string {
  switch (status) {
    case 'valid':
      // Slightly darker green than tailwind-500 so it pops on both themes.
      return '#10b981';
    case 'invalid':
      return '#ef4444';
    default:
      return 'rgb(var(--accent-primary))'; // default connection chrome uses the accent token
  }
}

export default function TypedConnectionLine({
  fromX,
  fromY,
  toX,
  toY,
  fromPosition,
  toPosition,
  connectionStatus,
}: ConnectionLineComponentProps) {
  const [path] = getBezierPath({
    sourceX: fromX,
    sourceY: fromY,
    sourcePosition: fromPosition,
    targetX: toX,
    targetY: toY,
    targetPosition: toPosition,
  });

  const stroke = strokeFor(connectionStatus);
  const dashed = connectionStatus === 'invalid';

  return (
    <g>
      <path
        d={path}
        fill="none"
        strokeWidth={3}
        strokeDasharray={dashed ? '6 4' : undefined}
        style={{ stroke, pointerEvents: 'none' }}
      />
      {/* Small dot at the cursor end — mirrors React Flow's stock line look. */}
      <circle
        cx={toX}
        cy={toY}
        r={4}
        stroke="white"
        strokeWidth={1.5}
        style={{ fill: stroke, pointerEvents: 'none' }}
      />
    </g>
  );
}
