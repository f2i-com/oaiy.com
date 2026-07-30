import { useEffect, useRef, useState, useCallback } from 'react';
import { createPortal } from 'react-dom';
import type { Node } from '@xyflow/react';
import { open } from '@tauri-apps/plugin-dialog';
import { uiLogger as logger } from '../../utils/logger';
import { useFocusTrap } from '../../hooks/useFocusTrap';

// Input node types that should appear in the run modal
const INPUT_NODE_TYPES = ['input_text', 'input_file', 'input_video', 'input_audio', 'input_folder'];

interface InputNodeInfo {
  id: string;
  type: string;
  label: string;
  value: string;
  nodeData: Record<string, unknown>;
}

interface RunWorkflowModalProps {
  isOpen: boolean;
  nodes: Node[];
  onRun: (updatedInputs: Map<string, Record<string, unknown>>) => void;
  onCancel: () => void;
}

export default function RunWorkflowModal({
  isOpen,
  nodes,
  onRun,
  onCancel,
}: RunWorkflowModalProps) {
  const dialogRef = useRef<HTMLDivElement>(null);
  const firstInputRef = useRef<HTMLTextAreaElement | HTMLInputElement>(null);
  // When there are no input nodes the modal is short-circuited to
  // null anyway, so the ref-fallback to first focusable inside the
  // dialog is all we need.
  useFocusTrap(dialogRef, isOpen, firstInputRef as React.RefObject<HTMLElement | null>);

  // Extract input nodes from the workflow
  const inputNodes: InputNodeInfo[] = nodes
    .filter(node => INPUT_NODE_TYPES.includes(node.type || ''))
    .map(node => {
      const data = node.data as Record<string, unknown>;
      const def = data.__definition as { name?: string } | undefined;

      // Get the label - use the label property, or fall back to node type name
      const label = (data.label as string) || def?.name || node.type || 'Input';

      // Get the value based on node type
      let value = '';
      if (node.type === 'input_text') {
        value = (data.value as string) || '';
      } else if (node.type === 'input_file' || node.type === 'input_video' || node.type === 'input_audio') {
        value = (data.filePath as string) || '';
      } else if (node.type === 'input_folder') {
        value = (data.path as string) || '';
      }

      return {
        id: node.id,
        type: node.type || '',
        label,
        value,
        nodeData: data,
      };
    });

  // Local state for form values
  const [formValues, setFormValues] = useState<Map<string, string>>(new Map());

  // Initialize form values when modal opens
  useEffect(() => {
    if (isOpen) {
      const initial = new Map<string, string>();
      inputNodes.forEach(node => {
        initial.set(node.id, node.value);
      });
      setFormValues(initial);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isOpen]);

  // Handle escape key
  useEffect(() => {
    if (!isOpen) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        onCancel();
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onCancel]);

  const handleValueChange = useCallback((nodeId: string, value: string) => {
    setFormValues(prev => {
      const next = new Map(prev);
      next.set(nodeId, value);
      return next;
    });
  }, []);

  const handleFilePick = useCallback(async (nodeId: string, nodeType: string) => {
    try {
      let filters: { name: string; extensions: string[] }[] = [];
      let directory = false;

      if (nodeType === 'input_file') {
        filters = [
          { name: 'Images', extensions: ['png', 'jpg', 'jpeg', 'gif', 'webp', 'bmp'] },
          { name: 'Text Files', extensions: ['txt', 'json', 'md', 'csv', 'xml', 'yaml', 'yml'] },
          { name: 'All Files', extensions: ['*'] },
        ];
      } else if (nodeType === 'input_video') {
        filters = [
          { name: 'Video Files', extensions: ['mp4', 'mov', 'avi', 'mkv', 'webm'] },
          { name: 'All Files', extensions: ['*'] },
        ];
      } else if (nodeType === 'input_audio') {
        filters = [
          { name: 'Audio Files', extensions: ['mp3', 'wav', 'ogg', 'flac', 'm4a', 'aac', 'wma'] },
          { name: 'All Files', extensions: ['*'] },
        ];
      } else if (nodeType === 'input_folder') {
        directory = true;
      }

      const result = await open({
        multiple: false,
        directory,
        filters: directory ? undefined : filters,
      });

      if (result) {
        handleValueChange(nodeId, result as string);
      }
    } catch (err) {
      logger.error('File picker error', { error: err });
    }
  }, [handleValueChange]);

  const handleRun = useCallback(() => {
    // Build updated node data
    const updates = new Map<string, Record<string, unknown>>();

    inputNodes.forEach(node => {
      const newValue = formValues.get(node.id) ?? node.value;

      if (node.type === 'input_text') {
        updates.set(node.id, { value: newValue });
      } else if (node.type === 'input_file' || node.type === 'input_video' || node.type === 'input_audio') {
        // Input_file's compiler checks fileContent FIRST and falls back
        // to filePath only when fileContent is empty. That means when a
        // workflow was saved with an embedded data URL (fileContent
        // populated, filePath empty), typing a new path here without
        // clearing fileContent would silently reuse the old embedded
        // image. Detect a real path change and clear the cached
        // content so the node re-reads from disk.
        const currentFilePath = (node.nodeData.filePath as string) || '';
        if (newValue !== currentFilePath) {
          updates.set(node.id, { filePath: newValue, fileContent: '' });
        } else {
          updates.set(node.id, { filePath: newValue });
        }
      } else if (node.type === 'input_folder') {
        updates.set(node.id, { path: newValue });
      }
    });

    onRun(updates);
  }, [inputNodes, formValues, onRun]);

  if (!isOpen) return null;

  // If no input nodes, don't show modal (caller should handle this case)
  if (inputNodes.length === 0) {
    return null;
  }

  const portalContainer = document.body;

  const getNodeIcon = (type: string) => {
    switch (type) {
      case 'input_text':
        return (
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h7" />
          </svg>
        );
      case 'input_file':
        return (
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 21h10a2 2 0 002-2V9.414a1 1 0 00-.293-.707l-5.414-5.414A1 1 0 0012.586 3H7a2 2 0 00-2 2v14a2 2 0 002 2z" />
          </svg>
        );
      case 'input_video':
        return (
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
          </svg>
        );
      case 'input_audio':
        return (
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM9 10l12-3" />
          </svg>
        );
      case 'input_folder':
        return (
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z" />
          </svg>
        );
      default:
        return null;
    }
  };

  const getNodeColor = (type: string) => {
    switch (type) {
      case 'input_text':
        return 'text-green-400 bg-green-500/20';
      case 'input_file':
        return 'text-green-400 bg-green-500/20';
      case 'input_video':
        return 'text-orange-400 bg-orange-500/20';
      case 'input_audio':
        return 'text-teal-400 bg-teal-500/20';
      case 'input_folder':
        return 'text-green-400 bg-green-500/20';
      default:
        return 'text-slate-400 bg-slate-500/20';
    }
  };

  return createPortal(
    <div className="fixed inset-0 z-[100] flex items-center justify-center">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/60 backdrop-blur-sm"
        onClick={onCancel}
      />

      {/* Dialog */}
      <div
        ref={dialogRef}
        className="relative bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-600 rounded-xl shadow-2xl w-full max-w-2xl mx-4 max-h-[80vh] short:max-h-[92vh] flex flex-col overflow-hidden animate-scaleIn"
        role="dialog"
        aria-modal="true"
        aria-labelledby="run-workflow-title"
        aria-describedby="run-workflow-description"
      >
        {/* Header */}
        <div className="flex items-center gap-3 p-5 short:p-3 border-b border-slate-200 dark:border-slate-700">
          <div className="flex-shrink-0 p-2 rounded-full bg-blue-500/20">
            <svg className="w-6 h-6 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </div>
          <div>
            <h2 id="run-workflow-title" className="text-lg font-semibold text-slate-900 dark:text-slate-100">
              Run Workflow
            </h2>
            <p id="run-workflow-description" className="text-sm text-slate-500 dark:text-slate-400">
              Review and modify inputs before running
            </p>
          </div>
        </div>

        {/* Form Content */}
        <div className="flex-1 overflow-y-auto p-5 short:p-3 space-y-4 short:space-y-3 custom-scrollbar">
          {inputNodes.map((node, index) => (
            <div key={node.id} className="space-y-2">
              <label className="flex items-center gap-2 text-sm font-medium text-slate-700 dark:text-slate-200">
                <span className={`p-1.5 rounded ${getNodeColor(node.type)}`}>
                  {getNodeIcon(node.type)}
                </span>
                {node.label}
              </label>

              {node.type === 'input_text' ? (
                <textarea
                  ref={index === 0 ? firstInputRef as React.RefObject<HTMLTextAreaElement> : undefined}
                  value={formValues.get(node.id) ?? node.value}
                  onChange={(e) => handleValueChange(node.id, e.target.value)}
                  placeholder="Enter text..."
                  rows={3}
                  className="w-full px-3 py-2 bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded-lg text-slate-900 dark:text-slate-100 placeholder-slate-400 dark:placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-y min-h-[80px]"
                />
              ) : (
                <>
                  {/* Image / video preview when the InputFileNode has
                      one. imagePreview is set by InputFileNode for image
                      types (data URL). For non-images this block is
                      skipped. Clicking it triggers the file picker so
                      the user can swap. */}
                  {(() => {
                    const preview =
                      typeof node.nodeData.imagePreview === 'string'
                        ? (node.nodeData.imagePreview as string)
                        : '';
                    const currentPath =
                      formValues.get(node.id) ?? node.value;
                    const isImage =
                      preview.startsWith('data:image/') ||
                      /\.(png|jpe?g|gif|webp|bmp)$/i.test(currentPath);
                    if (!isImage) return null;
                    const previewSrc = preview
                      ? preview
                      : currentPath && !currentPath.startsWith('data:')
                        ? `file://${currentPath.replace(/\\/g, '/')}`
                        : '';
                    if (!previewSrc) return null;
                    return (
                      <button
                        type="button"
                        onClick={() => handleFilePick(node.id, node.type)}
                        title="Click to change image"
                        className="group block w-full max-w-sm rounded-lg overflow-hidden border border-slate-300 dark:border-slate-600 bg-slate-100 dark:bg-slate-900 hover:border-blue-400 transition-colors"
                      >
                        <img
                          src={previewSrc}
                          alt={(node.nodeData.fileName as string) || 'Selected image'}
                          className="block w-full max-h-48 object-contain bg-checker"
                          onError={(e) => {
                            // Hide broken images quietly — text path still
                            // visible below.
                            (e.target as HTMLImageElement).style.display = 'none';
                          }}
                        />
                        {typeof node.nodeData.fileName === 'string' && (
                          <div className="px-2 py-1 text-xs text-slate-600 dark:text-slate-400 truncate text-left">
                            {node.nodeData.fileName as string}
                          </div>
                        )}
                      </button>
                    );
                  })()}
                <div className="flex gap-2">
                  <input
                    ref={index === 0 ? firstInputRef as React.RefObject<HTMLInputElement> : undefined}
                    type="text"
                    value={formValues.get(node.id) ?? node.value}
                    onChange={(e) => handleValueChange(node.id, e.target.value)}
                    placeholder={node.type === 'input_folder' ? 'Select or enter folder path...' : 'Select or enter file path...'}
                    className="flex-1 px-3 py-2 bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded-lg text-slate-900 dark:text-slate-100 placeholder-slate-400 dark:placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                  <button
                    type="button"
                    onClick={() => handleFilePick(node.id, node.type)}
                    className="px-3 py-2 bg-slate-100 dark:bg-slate-700 hover:bg-slate-200 dark:hover:bg-slate-600 border border-slate-300 dark:border-slate-600 rounded-lg text-slate-700 dark:text-slate-200 transition-colors"
                    title="Browse..."
                  >
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 19a2 2 0 01-2-2V7a2 2 0 012-2h4l2 2h4a2 2 0 012 2v1M5 19h14a2 2 0 002-2v-5a2 2 0 00-2-2H9a2 2 0 00-2 2v5a2 2 0 01-2 2z" />
                    </svg>
                  </button>
                </div>
                {/* When the saved workflow embedded the file as a data
                    URL (fileContent + fileName but no filePath), the
                    text input stays empty — the thumbnail above is the
                    only hint that anything is loaded. Show a small
                    caption with the embedded fileName so the user knows
                    what will be used if they don't change the path.
                    Empty input + this caption → existing embedded
                    content is used. Type a real path or click Browse to
                    override. */}
                {(() => {
                  const currentInput = formValues.get(node.id) ?? node.value;
                  const fileName =
                    typeof node.nodeData.fileName === 'string'
                      ? (node.nodeData.fileName as string)
                      : '';
                  if (currentInput || !fileName) return null;
                  return (
                    <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
                      Using embedded file:{' '}
                      <span className="font-mono text-slate-700 dark:text-slate-300">
                        {fileName}
                      </span>{' '}
                      — type a path above or click Browse to replace it.
                    </p>
                  );
                })()}
                </>
              )}
            </div>
          ))}
        </div>

        {/* Actions */}
        <div className="flex justify-end gap-3 px-5 py-4 short:px-4 short:py-2.5 bg-slate-50 dark:bg-slate-900/50 border-t border-slate-200 dark:border-slate-700">
          <button
            onClick={onCancel}
            className="btn btn-secondary btn-md"
          >
            Cancel
          </button>
          <button
            onClick={handleRun}
            className="btn btn-primary btn-md"
          >
            <svg className="w-4 h-4 mr-1.5" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM9.555 7.168A1 1 0 008 8v4a1 1 0 001.555.832l3-2a1 1 0 000-1.664l-3-2z" clipRule="evenodd" />
            </svg>
            Run
          </button>
        </div>
      </div>
    </div>,
    portalContainer
  );
}

// Helper to check if a workflow has input nodes
export function hasInputNodes(nodes: Node[]): boolean {
  return nodes.some(node => INPUT_NODE_TYPES.includes(node.type || ''));
}
