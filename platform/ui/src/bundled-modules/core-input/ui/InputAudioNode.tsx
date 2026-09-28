import { memo, useState, useRef, useEffect, useCallback, useMemo } from 'react';
import { Position, useReactFlow, useNodeId } from '@xyflow/react';
import { CollapsibleNodeWrapper, type HandleConfig } from 'oaiy-ui-components';

interface AudioInputNodeData {
    filePath?: string;
    fileName?: string;
    _status?: 'running' | 'completed' | 'error';
    _collapsed?: boolean;
    showBodyProperties?: boolean;
    onCollapsedChange?: (value: boolean) => void;
}

interface AudioInputNodeProps {
    data: AudioInputNodeData;
}

const AudioInputIcon = (
    <svg className="w-3 h-3 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM9 10l12-3" />
    </svg>
);

function InputAudioNode({ data }: AudioInputNodeProps) {
    const nodeId = useNodeId();
    const { updateNodeData } = useReactFlow();
    const [isDragging, setIsDragging] = useState(false);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);
    const onCollapsedChangeRef = useRef(data.onCollapsedChange);

    useEffect(() => {
        onCollapsedChangeRef.current = data.onCollapsedChange;
    });

    const handleCollapsedChange = useCallback((collapsed: boolean) => {
        onCollapsedChangeRef.current?.(collapsed);
    }, []);

    const handleFile = useCallback((file: File) => {
        if (!nodeId) return;
        setError(null);

        const isAudio = file.type.startsWith('audio/') ||
            /\.(mp3|wav|ogg|flac|m4a|aac|wma)$/i.test(file.name);

        if (!isAudio) {
            setError('Please select an audio file');
            return;
        }

        // Get file path if available (Tauri adds .path property to File objects)
        // @ts-expect-error - path exists on File in Tauri context
        const filePath = file.path as string | undefined;

        if (filePath) {
            updateNodeData(nodeId, {
                filePath,
                fileName: file.name
            });
            return;
        }

        // Browser fallback: no .path on a dropped File. Stash the bytes as
        // a `blob:` URL and use that as the path — the oaiy-web read_file
        // shim already fetches blob URLs and returns the raw bytes, so
        // downstream nodes consume it unchanged. (Persistence across
        // reloads is handled by the dialog shim's blobStore for picker-
        // sourced URLs; drag-drop URLs are session-scoped, which matches
        // user expectation for ad-hoc drops.)
        try {
            // Append the original filename as a URL fragment so the
            // input_file runtime's extension-sniff classifies the dropped
            // file correctly (the picker shim does the same — see
            // tauri-shim/dialog.ts pickViaInput). Without it, a bare
            // blob:URL has no extension and read_file gets a
            // `readAs: 'text'` request for binary audio data.
            const baseUrl = URL.createObjectURL(file);
            const safeName = encodeURIComponent(file.name || '');
            const blobUrl = safeName ? `${baseUrl}#${safeName}` : baseUrl;
            updateNodeData(nodeId, { filePath: blobUrl, fileName: file.name });
        } catch {
            setError('Could not read dropped file');
        }
    }, [nodeId, updateNodeData]);

    const handleDrop = (e: React.DragEvent) => {
        e.preventDefault();
        e.stopPropagation();
        setIsDragging(false);
        const file = e.dataTransfer.files[0];
        if (file) handleFile(file);
    };

    const handleDragOver = (e: React.DragEvent) => {
        e.preventDefault();
        e.stopPropagation();
        setIsDragging(true);
    };

    const handleDragLeave = (e: React.DragEvent) => {
        e.preventDefault();
        e.stopPropagation();
        setIsDragging(false);
    };

    const handleClick = async () => {
        // Prefer native Tauri file picker for guaranteed path access.
        // Tauri 2 exposes invoke under `__TAURI_INTERNALS__`, not the
        // legacy `__TAURI__` global (see reference_plugin_ui_tauri_access).
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const internals = (window as any).__TAURI_INTERNALS__;
        if (internals && nodeId) {
            setIsLoading(true);
            try {
                const result = await internals.invoke('plugin:oaiy-filesystem|pick_file', {
                    filters: [{
                        name: 'Audio Files',
                        extensions: ['mp3', 'wav', 'ogg', 'flac', 'm4a', 'aac', 'wma']
                    }]
                }) as string | null;

                if (result) {
                    const fileName = result.split(/[/\\]/).pop() || 'audio';
                    updateNodeData(nodeId, { filePath: result, fileName });
                }
                setIsLoading(false);
            } catch (err) {
                setIsLoading(false);
                setError(`Failed to pick file: ${err}`);
            }
        } else {
            // Fallback to HTML file input for browser-only dev mode
            fileInputRef.current?.click();
        }
    };

    const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (file) handleFile(file);
        e.target.value = ''; // allow re-selecting the same file (no onChange otherwise)
    };

    // Extract filename from path
    const fileName = useMemo(() => {
        if (data.fileName) return data.fileName;
        if (!data.filePath) return null;
        const parts = data.filePath.replace(/\\/g, '/').split('/');
        return parts[parts.length - 1];
    }, [data.filePath, data.fileName]);

    const hasFile = !!data.filePath;

    const collapsedPreview = (
        <div className="text-slate-400">
            {hasFile ? (
                <span className="text-teal-400 truncate block">{fileName}</span>
            ) : (
                <span className="italic text-slate-500">No file</span>
            )}
        </div>
    );

    const outputHandles = useMemo<HandleConfig[]>(() => [
        { id: 'audio', type: 'source', position: Position.Right, color: '!bg-teal-500', size: 'lg' },
    ], []);

    return (
        <CollapsibleNodeWrapper
            title="Audio Input"
            color="teal"
            icon={AudioInputIcon}
            width={260}
            collapsedWidth={130}
            status={data._status}
            isCollapsed={data._collapsed}
            onCollapsedChange={handleCollapsedChange}
            collapsedPreview={collapsedPreview}
            inputHandles={[]}
            outputHandles={outputHandles}
        >
            {data.showBodyProperties !== false && (
                <>
                    <div
                        onClick={handleClick}
                        onDrop={handleDrop}
                        onDragOver={handleDragOver}
                        onDragLeave={handleDragLeave}
                        className={`
                            nodrag w-full h-24 border-2 border-dashed rounded-lg flex flex-col items-center justify-center cursor-pointer transition-all
                            ${isDragging
                                ? 'border-teal-500 bg-teal-900/30'
                                : hasFile
                                    ? 'border-teal-600 bg-slate-100 dark:bg-slate-900'
                                    : 'border-slate-300 dark:border-slate-600 bg-slate-100 dark:bg-slate-900 hover:border-teal-600'
                            }
                        `}
                    >
                        {isLoading ? (
                            <div className="text-center">
                                <svg className="w-6 h-6 mx-auto text-teal-500 animate-spin" fill="none" viewBox="0 0 24 24">
                                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                                </svg>
                            </div>
                        ) : hasFile ? (
                            <div className="text-center px-2">
                                <svg className="w-6 h-6 mx-auto text-teal-500 mb-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2z" />
                                </svg>
                                <p className="text-teal-400 text-xs truncate max-w-full">{fileName}</p>
                            </div>
                        ) : (
                            <>
                                <svg className="w-6 h-6 text-slate-500 mb-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM9 10l12-3" />
                                </svg>
                                <p className="text-slate-600 dark:text-slate-400 text-xs">Drop audio file or click</p>
                            </>
                        )}
                    </div>

                    <input
                        ref={fileInputRef}
                        type="file"
                        accept="audio/*,.mp3,.wav,.ogg,.flac,.m4a,.aac,.wma"
                        onChange={handleInputChange}
                        className="hidden"
                    />
                </>
            )}

            {error && (
                <div className="flex items-center gap-2 px-2 py-1.5 bg-red-900/30 border border-red-600/50 rounded text-red-400 text-xs mt-2">
                    <svg className="w-3 h-3 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                    <span>{error}</span>
                </div>
            )}
        </CollapsibleNodeWrapper>
    );
}

export default memo(InputAudioNode);
