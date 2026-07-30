/**
 * useEdgeScrolling Hook
 *
 * Provides automatic panning when the mouse moves near the edges of a React Flow canvas.
 * Extracted from OAIYBuilder.tsx for maintainability.
 */

import { useEffect, useRef } from 'react';
import type { Viewport } from '@xyflow/react';

interface UseEdgeScrollingOptions {
  /** Ref to the wrapper element containing the React Flow canvas */
  wrapperRef: React.RefObject<HTMLDivElement | null>;
  /** Ref to the React Flow instance with viewport methods */
  reactFlowInstance: React.RefObject<{
    getViewport: () => Viewport;
    setViewport: (viewport: Viewport, options?: { duration?: number }) => void;
  } | null>;
  /** Translation extent bounds [[minX, minY], [maxX, maxY]] */
  translateExtent: [[number, number], [number, number]];
  /** Distance from edge to trigger scrolling (px). Default: 60 */
  edgeZone?: number;
  /** Pan speed (px per frame). Default: 10 */
  panSpeed?: number;
  /** Whether edge scrolling is enabled. Default: true */
  enabled?: boolean;
}

/**
 * Hook that provides auto-panning when mouse moves near canvas edges.
 * Useful for better UX when dragging nodes or connections near viewport boundaries.
 */
export function useEdgeScrolling({
  wrapperRef,
  reactFlowInstance,
  translateExtent,
  edgeZone = 60,
  panSpeed = 10,
  enabled = true,
}: UseEdgeScrollingOptions): void {
  const edgeScrollRef = useRef<number | null>(null);
  // Keep the latest translateExtent in a ref so the animate closure reads current
  // bounds without translateExtent being an effect dependency (which would tear down
  // and re-add listeners every drag frame, cancelling the in-flight RAF).
  const translateExtentRef = useRef(translateExtent);
  translateExtentRef.current = translateExtent;

  useEffect(() => {
    if (!enabled) return;

    const wrapper = wrapperRef.current;
    if (!wrapper) return;

    const handleMouseMove = (e: MouseEvent) => {
      // Skip edge scrolling if mouse is over UI overlay elements
      const target = e.target as HTMLElement;
      if (target.closest('[role="menu"], [role="toolbar"], .react-flow__controls, .react-flow__minimap, button, a, input, select, textarea')) {
        // Stop any existing animation when entering UI elements
        if (edgeScrollRef.current) {
          cancelAnimationFrame(edgeScrollRef.current);
          edgeScrollRef.current = null;
        }
        return;
      }

      const rect = wrapper.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;

      // Calculate pan direction based on mouse position near edges
      let panX = 0;
      let panY = 0;

      if (x < edgeZone) {
        panX = panSpeed * ((edgeZone - x) / edgeZone); // Faster as you get closer to edge
      } else if (x > rect.width - edgeZone) {
        panX = -panSpeed * ((x - (rect.width - edgeZone)) / edgeZone);
      }

      if (y < edgeZone) {
        panY = panSpeed * ((edgeZone - y) / edgeZone);
      } else if (y > rect.height - edgeZone) {
        panY = -panSpeed * ((y - (rect.height - edgeZone)) / edgeZone);
      }

      // Stop any existing animation
      if (edgeScrollRef.current) {
        cancelAnimationFrame(edgeScrollRef.current);
        edgeScrollRef.current = null;
      }

      // Start panning if in edge zone
      if ((panX !== 0 || panY !== 0) && reactFlowInstance.current) {
        const animate = () => {
          if (!reactFlowInstance.current) return;

          const viewport = reactFlowInstance.current.getViewport();
          const newX = viewport.x + panX;
          const newY = viewport.y + panY;

          // Apply translate extent bounds (read latest value from ref)
          const [[minX, minY], [maxX, maxY]] = translateExtentRef.current;
          const zoom = viewport.zoom;

          // Clamp to translateExtent (converted to viewport coordinates)
          const clampedX = Math.min(Math.max(newX, -maxX * zoom + rect.width), -minX * zoom);
          const clampedY = Math.min(Math.max(newY, -maxY * zoom + rect.height), -minY * zoom);

          reactFlowInstance.current.setViewport(
            { x: clampedX, y: clampedY, zoom },
            { duration: 0 }
          );

          edgeScrollRef.current = requestAnimationFrame(animate);
        };
        edgeScrollRef.current = requestAnimationFrame(animate);
      }
    };

    const handleMouseLeave = () => {
      if (edgeScrollRef.current) {
        cancelAnimationFrame(edgeScrollRef.current);
        edgeScrollRef.current = null;
      }
    };

    wrapper.addEventListener('mousemove', handleMouseMove);
    wrapper.addEventListener('mouseleave', handleMouseLeave);

    return () => {
      wrapper.removeEventListener('mousemove', handleMouseMove);
      wrapper.removeEventListener('mouseleave', handleMouseLeave);
      if (edgeScrollRef.current) {
        cancelAnimationFrame(edgeScrollRef.current);
      }
    };
  }, [wrapperRef, reactFlowInstance, edgeZone, panSpeed, enabled]);
}
