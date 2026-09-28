import { memo, useState, useRef, useCallback, useEffect, type ReactNode } from 'react';

interface ResizableNodeWrapperProps {
  children: ReactNode;
  minWidth?: number;
  minHeight?: number;
  defaultWidth?: number;
  defaultHeight?: number;
  className?: string;
  autoHeight?: boolean;
}

function ResizableNodeWrapper({
  children,
  minWidth = 200,
  minHeight = 100,
  defaultWidth = 280,
  defaultHeight,
  className = '',
  autoHeight = true,
}: ResizableNodeWrapperProps) {
  const [size, setSize] = useState({
    width: defaultWidth,
    height: defaultHeight,
  });
  const [isResizing, setIsResizing] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  // Use refs to track resize state without causing re-renders
  const sizeRef = useRef(size);
  const startPosRef = useRef({ x: 0, y: 0 });
  const startSizeRef = useRef({ width: 0, height: 0 });

  // Store references to event handlers for cleanup
  const handlersRef = useRef<{
    mousemove: ((e: MouseEvent) => void) | null;
    mouseup: (() => void) | null;
  }>({ mousemove: null, mouseup: null });

  // Keep size ref in sync
  useEffect(() => {
    sizeRef.current = size;
  }, [size]);

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

  const handleMouseDown = useCallback((e: React.MouseEvent, direction: 'e' | 's' | 'se') => {
    e.preventDefault();
    e.stopPropagation();

    // Clean up any existing listeners first
    cleanupListeners();

    setIsResizing(true);
    startPosRef.current = { x: e.clientX, y: e.clientY };
    startSizeRef.current = {
      width: sizeRef.current.width,
      height: sizeRef.current.height || containerRef.current?.offsetHeight || minHeight
    };

    const handleMouseMove = (moveEvent: MouseEvent) => {
      const deltaX = moveEvent.clientX - startPosRef.current.x;
      const deltaY = moveEvent.clientY - startPosRef.current.y;

      setSize(prev => ({
        width: direction === 's' ? prev.width : Math.max(minWidth, startSizeRef.current.width + deltaX),
        height: direction === 'e' ? prev.height : Math.max(minHeight, startSizeRef.current.height + deltaY),
      }));
    };

    const handleMouseUp = () => {
      setIsResizing(false);
      cleanupListeners();
    };

    // Store references for cleanup
    handlersRef.current.mousemove = handleMouseMove;
    handlersRef.current.mouseup = handleMouseUp;

    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);
  }, [minWidth, minHeight, cleanupListeners]);

  return (
    <div
      ref={containerRef}
      className={`relative ${className}`}
      style={{
        width: size.width,
        height: autoHeight && !size.height ? 'auto' : size.height,
        minWidth,
        minHeight: autoHeight ? undefined : minHeight,
      }}
    >
      {children}

      {/* Right resize handle - nodrag prevents React Flow from starting node drag */}
      <div
        className={`nodrag absolute top-0 right-0 w-2 h-full cursor-ew-resize hover:bg-blue-500/30 transition-colors ${
          isResizing ? 'bg-blue-500/30' : ''
        }`}
        onMouseDown={(e) => handleMouseDown(e, 'e')}
        role="slider"
        aria-label="Resize width"
        aria-orientation="horizontal"
      />

      {/* Bottom resize handle - nodrag prevents React Flow from starting node drag */}
      <div
        className={`nodrag absolute bottom-0 left-0 w-full h-2 cursor-ns-resize hover:bg-blue-500/30 transition-colors ${
          isResizing ? 'bg-blue-500/30' : ''
        }`}
        onMouseDown={(e) => handleMouseDown(e, 's')}
        role="slider"
        aria-label="Resize height"
        aria-orientation="vertical"
      />

      {/* Corner resize handle - nodrag prevents React Flow from starting node drag */}
      <div
        className={`nodrag absolute bottom-0 right-0 w-3 h-3 cursor-nwse-resize hover:bg-blue-500/50 transition-colors rounded-tl ${
          isResizing ? 'bg-blue-500/50' : ''
        }`}
        onMouseDown={(e) => handleMouseDown(e, 'se')}
        role="slider"
        aria-label="Resize both dimensions"
      >
        <svg
          className="w-3 h-3 text-slate-500 dark:text-slate-400"
          viewBox="0 0 24 24"
          fill="currentColor"
          aria-hidden="true"
        >
          <path d="M22 22H20V20H22V22ZM22 18H20V16H22V18ZM18 22H16V20H18V22ZM22 14H20V12H22V14ZM18 18H16V16H18V18ZM14 22H12V20H14V22Z" />
        </svg>
      </div>
    </div>
  );
}

export default memo(ResizableNodeWrapper);
