import { useRef, useState } from 'react';
import { Download, FilePlus2, FolderInput, Upload } from 'lucide-react';
import { useConfirmDialog } from '../hooks/useConfirmDialog';
import Menu, { MenuItem } from './ui/Menu';

interface ImportMenuProps {
  /**
   * Replaces the current project with the one parsed from `file`. Resolves
   * with the re-registered embedded-service counts so we can surface them.
   */
  importProject: (
    file: File,
  ) => Promise<{ servicesAdded: number; servicesSkipped: number }>;
  /** Adds the flows in a flow file (or loads a .oaiy package) beside the project's own. */
  onImportFlows: () => void;
  /** Downloads the project as a file (Ctrl/⌘+S). */
  onExportProject: () => void;
  /** Current project name — shown in the "this will replace…" confirm. */
  projectName: string;
  onShowToast: (
    message: string,
    type?: 'success' | 'error' | 'info' | 'warning',
  ) => void;
}

/**
 * The editor's one place to bring flows in, and to take the project out.
 *
 * Two kinds of file come in, and they do different things, so the menu says
 * which is which: a flow file adds its flows to this project; a project file
 * replaces the project (after a confirm). Export sits beside them because the
 * confirm tells you to export first.
 *
 * Lives as its own component (rather than inline in OAIYApp) so it can call
 * `useConfirmDialog()`: OAIYApp renders the provider and sits above it.
 */
export default function ImportMenu({
  importProject,
  onImportFlows,
  onExportProject,
  projectName,
  onShowToast,
}: ImportMenuProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const buttonRef = useRef<HTMLButtonElement>(null);
  const [open, setOpen] = useState(false);
  const confirm = useConfirmDialog();

  const handleFile = async (file: File) => {
    const ok = await confirm({
      title: 'Replace this project?',
      message: `The project in the file replaces "${projectName}" and its flows. Export this one first (Ctrl/⌘ + S) if you want to keep a copy.`,
      confirmLabel: 'Replace and import',
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
        ref={buttonRef}
        type="button"
        onClick={() => setOpen((o) => !o)}
        className={`oaiy-icon-btn${open ? ' on' : ''}`}
        aria-label="Import or export"
        aria-haspopup="menu"
        aria-expanded={open}
        title="Import a flow or a project, or export this project"
      >
        <FolderInput size={16} />
      </button>
      <Menu open={open} onClose={() => setOpen(false)} anchor={buttonRef.current} label="Import or export">
        <MenuItem
          icon={<FilePlus2 size={15} />}
          label="Import flows…"
          hint="Adds the flows in a flow file (.json) or a package (.oaiy) to this project"
          onSelect={onImportFlows}
        />
        <MenuItem
          icon={<Upload size={15} />}
          label="Import a project…"
          hint="Replaces this project with one exported before (.json)"
          onSelect={() => inputRef.current?.click()}
        />
        <hr />
        <MenuItem
          icon={<Download size={15} />}
          label="Export this project"
          hint="Downloads it as .json (Ctrl/⌘ + S)"
          onSelect={onExportProject}
        />
      </Menu>
    </>
  );
}
