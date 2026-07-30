import { useState, useCallback, useEffect, useRef, type MouseEvent } from 'react';

export interface ResizeConstraints {
  minWidth: number;
  maxWidth: number;
  minHeight: number;
  maxHeight: number;
}

export interface UseNodeResizeOptions {
  initialWidth: number;
  initialHeight: number;
  constraints?: Partial<ResizeConstraints>;
}

const defaultConstraints: ResizeConstraints = {
  minWidth: 200,
  maxWidth: 600,
  minHeight: 120,
  maxHeight: 800,
};

export function useNodeResize(options: UseNodeResizeOptions) {
  const { initialWidth, initialHeight, constraints = {} } = options;

  const resolvedConstraints = { ...defaultConstraints, ...constraints };

  const [size, setSize] = useState({ width: initialWidth, height: initialHeight });

  // Use refs to avoid dependency issues in callbacks
  const sizeRef = useRef(size);
  const constraintsRef = useRef(resolvedConstraints);

  // Keep size ref in sync with state
  useEffect(() => {
    sizeRef.current = size;
  }, [size]);

  // Sync constraints ref on every render to always have latest values
  // This runs on every render intentionally - no dependency array needed
  useEffect(() => {
    constraintsRef.current = resolvedConstraints;
  }); // eslint-disable-line react-hooks/exhaustive-deps

  // Store references to event handlers for cleanup
  const handlersRef = useRef<{
    mousemove: ((e: globalThis.MouseEvent) => void) | null;
    mouseup: (() => void) | null;
  }>({ mousemove: null, mouseup: null });

  // Cleanup function to remove any active event listeners
  const cleanupListeners = useCallback(() => {
    if (handlersRef.current.mousemove) {
      document.removeEventListener('mousemove', handlersRef.current.mousemove);
      handlersRef.current.mousemove = null;
    }
    if (handlersRef.current.mouseup) {
      document.removeEventListener('mouseup', handlersRef.current.mouseup);
      handlersRef.current.mouseup = null;
    }
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      cleanupListeners();
    };
  }, [cleanupListeners]);

  const handleResizeStart = useCallback(
    (e: MouseEvent, direction: 'e' | 's' | 'se') => {
      e.preventDefault();
      e.stopPropagation();

      // Clean up any existing listeners first
      cleanupListeners();

      const startX = e.clientX;
      const startY = e.clientY;
      // Capture current size at start of resize
      const startSize = { ...sizeRef.current };

      const handleMouseMove = (moveEvent: globalThis.MouseEvent) => {
        const deltaX = moveEvent.clientX - startX;
        const deltaY = moveEvent.clientY - startY;
        const constraints = constraintsRef.current;

        setSize({
          width:
            direction === 's'
              ? startSize.width
              : Math.max(
                  constraints.minWidth,
                  Math.min(constraints.maxWidth, startSize.width + deltaX)
                ),
          height:
            direction === 'e'
              ? startSize.height
              : Math.max(
                  constraints.minHeight,
                  Math.min(constraints.maxHeight, startSize.height + deltaY)
                ),
        });
      };

      const handleMouseUp = () => {
        cleanupListeners();
      };

      // Store references for cleanup
      handlersRef.current.mousemove = handleMouseMove;
      handlersRef.current.mouseup = handleMouseUp;

      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
    },
    [cleanupListeners]
  );

  return { size, handleResizeStart };
}
