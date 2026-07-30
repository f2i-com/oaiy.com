import { useRef } from 'react';
import { useConfirmDialog } from '../hooks/useConfirmDialog';

interface ProjectImportButtonProps {
  /**
   * Replaces the current project with the one parsed from `file`. Resolves
   * with the re-registered embedded-service counts so we can surface them.
   */
  importProject: (
    file: File,
  ) => Promise<{ servicesAdded: number; servicesSkipped: number }>;
  /** Current project name — shown in the "this will replace…" confirm. */
  projectName: string;
  onShowToast: (
    message: string,
    type?: 'success' | 'error' | 'info' | 'warning',
  ) => void;
}

/**
 * Header button: load a previously-exported project JSON back into the app.
 *
 * Lives as its own component (rather than inline in OAIYApp) specifically so it
 * can call `useConfirmDialog()` — OAIYApp *renders* the ConfirmDialogProvider
 * and therefore sits ABOVE that context, where the hook would throw. As a
 * child placed inside the provider's subtree, this component is in-context.
 */
export default function ProjectImportButton({
  importProject,
  projectName,
  onShowToast,
}: ProjectImportButtonProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const confirm = useConfirmDialog();

  const handleFile = async (file: File) => {
    const ok = await confirm({
      title: 'Import project?',
      message: `This replaces your current project "${projectName}". The web build doesn't auto-save it to a file — export it first (Ctrl/⌘ + S) if you want to keep a copy.`,
      confirmLabel: 'Replace & import',
      cancelLabel: 'Cancel',
      variant: 'danger',
    });
    if (!ok) return;
    try {
      const { servicesAdded } = await importProject(file);
      onShowToast(
        servicesAdded > 0
          ? `Project imported · re-added ${servicesAdded} service${servicesAdded === 1 ? '' : 's'}`
          : 'Project imported',
        'success',
      );
    } catch (err) {
      onShowToast(
        `Import failed: ${err instanceof Error ? err.message : 'invalid project file'}`,
        'error',
      );
    }
  };

  return (
    <>
      <input
        ref={inputRef}
        type="file"
        accept="application/json,.json"
        className="hidden"
        onChange={(e) => {
          const file = e.target.files?.[0];
          // Clear the value so re-picking the SAME file still fires onChange.
          e.target.value = '';
          if (file) void handleFile(file);
        }}
      />
      <button
        type="button"
        onClick={() => inputRef.current?.click()}
        className="p-1.5 rounded-md transition-colors"
        style={{ color: 'rgb(var(--color-text-tertiary))' }}
        onMouseOver={(e) => {
          e.currentTarget.style.backgroundColor = 'rgb(var(--color-bg-tertiary) / 0.7)';
        }}
        onMouseOut={(e) => {
          e.currentTarget.style.backgroundColor = '';
        }}
        aria-label="Import project from JSON"
        title={'Import project (.json)\n\nReplaces the current project — export first (Ctrl/⌘ + S) to keep a copy.'}
      >
        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"
          />
        </svg>
      </button>
    </>
  );
}
