import { memo, useCallback, useRef, useEffect, useMemo, useState } from 'react';
import { Position } from '@xyflow/react';
// Side-effect import: points the loader at the bundled monaco instead of the
// CDN it defaults to. Must precede the first <Editor> mount.
import '../../../services/monacoSetup';
import Editor from '@monaco-editor/react';
import { useNodeResize, CollapsibleNodeWrapper, type HandleConfig } from 'oaiy-ui-components';

// Hook to detect current theme from document class
function useDocumentTheme() {
  const [isDark, setIsDark] = useState(() =>
    typeof document !== 'undefined' && document.documentElement.classList.contains('dark')
  );

  useEffect(() => {
    const observer = new MutationObserver(() => {
      setIsDark(document.documentElement.classList.contains('dark'));
    });
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ['class'] });
    return () => observer.disconnect();
  }, []);

  return isDark;
}

interface LogicBlockNodeData {
  code?: string;
  inputCount?: number;
  inputNames?: string[];
  _status?: 'running' | 'completed' | 'error';
  _collapsed?: boolean;
  onCodeChange?: (value: string) => void;
  onInputCountChange?: (value: number) => void;
  onInputNamesChange?: (value: string[]) => void;
  onCollapsedChange?: (value: boolean) => void;
  showBodyProperties?: boolean;
}

interface LogicBlockNodeProps {
  data: LogicBlockNodeData;
}

const LogicBlockIcon = (
  <span className="text-white font-bold text-xs">ƒ</span>
);

function LogicBlockNode({ data }: LogicBlockNodeProps) {
  const isDarkTheme = useDocumentTheme();
  const { size, handleResizeStart } = useNodeResize({
    initialWidth: 300,
    initialHeight: 260,
    constraints: { minWidth: 240, maxWidth: 600, minHeight: 200, maxHeight: 600 },
  });

  const inputCount = data.inputCount || 1;
  const inputNames = data.inputNames || ['input'];

  const onCodeChangeRef = useRef(data.onCodeChange);
  const onInputCountChangeRef = useRef(data.onInputCountChange);
  const onInputNamesChangeRef = useRef(data.onInputNamesChange);
  const onCollapsedChangeRef = useRef(data.onCollapsedChange);

  useEffect(() => {
    onCodeChangeRef.current = data.onCodeChange;
    onInputCountChangeRef.current = data.onInputCountChange;
    onInputNamesChangeRef.current = data.onInputNamesChange;
    onCollapsedChangeRef.current = data.onCollapsedChange;
  });

  const handleEditorChange = useCallback((value: string | undefined) => {
    if (value !== undefined) {
      onCodeChangeRef.current?.(value);
    }
  }, []);

  const handleCollapsedChange = useCallback((collapsed: boolean) => {
    onCollapsedChangeRef.current?.(collapsed);
  }, []);

  const handleInputNameChange = useCallback((index: number, name: string) => {
    const newNames = [...inputNames];
    newNames[index] = name;
    onInputNamesChangeRef.current?.(newNames);
  }, [inputNames]);

  const handleAddInput = useCallback(() => {
    if (inputCount < 6) {
      const newCount = inputCount + 1;
      const newNames = [...inputNames, `var${newCount}`];
      onInputCountChangeRef.current?.(newCount);
      onInputNamesChangeRef.current?.(newNames);
    }
  }, [inputCount, inputNames]);

  const handleRemoveInput = useCallback(() => {
    if (inputCount > 1) {
      const newCount = inputCount - 1;
      const newNames = inputNames.slice(0, newCount);
      onInputCountChangeRef.current?.(newCount);
      onInputNamesChangeRef.current?.(newNames);
    }
  }, [inputCount, inputNames]);

  const editorHeight = Math.max(80, size.height - 180);

  const collapsedPreview = (
    <div className="text-slate-400">
      <span className="text-blue-400">{inputCount} input{inputCount > 1 ? 's' : ''}</span>
    </div>
  );

  const inputHandles = useMemo<HandleConfig[]>(() => {
    // Use standard handle IDs that match the node definition: input, input2, input3, etc.
    const standardIds = ['input', 'input2', 'input3', 'input4', 'input5'];
    return standardIds.slice(0, inputCount).map((id, index) => ({
      id,
      type: 'target' as const,
      position: Position.Left,
      color: '!bg-blue-500',
      label: inputNames[index] || id,
      labelColor: 'text-blue-400',
      size: 'md' as const,
    }));
  }, [inputNames, inputCount]);

  const outputHandles = useMemo<HandleConfig[]>(() => [
    { id: 'output', type: 'source', position: Position.Right, color: '!bg-green-500', size: 'lg' },
  ], []);

  const resizeHandles = (
    <>
      <div
        className="nodrag absolute top-0 right-0 w-2 h-full cursor-ew-resize opacity-0 group-hover:opacity-100 hover:bg-blue-500/30 transition-all"
        onMouseDown={(e) => handleResizeStart(e, 'e')}
      />
      <div
        className="nodrag absolute bottom-0 left-0 w-full h-2 cursor-ns-resize opacity-0 group-hover:opacity-100 hover:bg-blue-500/30 transition-all"
        onMouseDown={(e) => handleResizeStart(e, 's')}
      />
      <div
        className="nodrag absolute bottom-1 right-1 w-3 h-3 cursor-nwse-resize opacity-0 group-hover:opacity-100 transition-opacity"
        onMouseDown={(e) => handleResizeStart(e, 'se')}
      >
        <svg className="w-3 h-3 text-slate-500" viewBox="0 0 24 24" fill="currentColor">
          <path d="M22 22H20V20H22V22ZM22 18H20V16H22V18ZM18 22H16V20H18V22Z" />
        </svg>
      </div>
    </>
  );

  return (
    <CollapsibleNodeWrapper
      title="Logic Block"
      color="blue"
      icon={LogicBlockIcon}
      width={size.width}
      collapsedWidth={140}
      status={data._status}
      isCollapsed={data._collapsed}
      onCollapsedChange={handleCollapsedChange}
      collapsedPreview={collapsedPreview}
      inputHandles={inputHandles}
      outputHandles={outputHandles}
      resizeHandles={resizeHandles}
    >
      {data.showBodyProperties !== false && (
        <>
          {/* Input Variables */}
          <div>
            <div className="flex items-center justify-between mb-1">
              <label className="text-slate-600 dark:text-slate-400 text-xs">Input Variables</label>
              <div className="flex gap-1">
                <button
                  onClick={handleRemoveInput}
                  disabled={inputCount <= 1}
                  className="w-5 h-5 flex items-center justify-center bg-slate-200 dark:bg-slate-700 hover:bg-slate-300 dark:hover:bg-slate-600 disabled:opacity-50 disabled:cursor-not-allowed rounded text-slate-700 dark:text-slate-300 text-xs"
                >
                  −
                </button>
                <button
                  onClick={handleAddInput}
                  disabled={inputCount >= 6}
                  className="w-5 h-5 flex items-center justify-center bg-slate-200 dark:bg-slate-700 hover:bg-slate-300 dark:hover:bg-slate-600 disabled:opacity-50 disabled:cursor-not-allowed rounded text-slate-700 dark:text-slate-300 text-xs"
                >
                  +
                </button>
              </div>
            </div>
            <div className="flex flex-col gap-1">
              {Array.from({ length: inputCount }).map((_, index) => (
                <input
                  key={index}
                  type="text"
                  className="nodrag nowheel w-full bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded px-1.5 py-0.5 text-xs text-yellow-700 dark:text-yellow-400 focus:outline-none focus:border-blue-500 font-mono"
                  value={inputNames[index] || ''}
                  onChange={(e) => handleInputNameChange(index, e.target.value)}
                  onMouseDown={(e) => e.stopPropagation()}
                  placeholder={`var${index + 1}`}
                />
              ))}
            </div>
          </div>

          {/* Code Editor */}
          <div
            className="nodrag nowheel border border-slate-300 dark:border-slate-600 rounded overflow-hidden"
            style={{ height: editorHeight }}
            onMouseDown={(e) => e.stopPropagation()}
          >
            <Editor
              height="100%"
              defaultLanguage="javascript"
              value={data.code || '// Transform the inputs\nreturn input;'}
              theme={isDarkTheme ? 'vs-dark' : 'light'}
              options={{
                minimap: { enabled: false },
                lineNumbers: 'off',
                fontSize: 12,
                scrollBeyondLastLine: false,
                wordWrap: 'on',
                padding: { top: 8 },
              }}
              onChange={handleEditorChange}
            />
          </div>
        </>
      )}
    </CollapsibleNodeWrapper>
  );
}

export default memo(LogicBlockNode);
