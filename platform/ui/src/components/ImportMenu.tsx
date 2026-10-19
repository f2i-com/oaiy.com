import { useRef, useState } from 'react';
import { Download, FilePlus2, FolderInput, FolderPlus, Upload } from 'lucide-react';
import { useConfirmDialog } from '../hooks/useConfirmDialog';
import { parseImportedProject } from '../utils/ProjectIO';
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
  /** Downloads the project as a file. */
  onExportProject: () => void;
  /** Asks to replace the project with a new, empty one (its own confirm says what goes). */
  onNewProject?: () => void;
  /** Current project name — shown in the "this will replace…" confirm. */
  projectName: string;
  onShowToast: (
    message: string,
    type?: 'success' | 'error' | 'info' | 'warning',
  ) => void;
}

/**
 * The editor's Projects menu: one place to start a project, to bring flows in,
 * and to take the project out.
 *
 * Two kinds of file come in, and they do different things, so the menu says
 * which is which: a flow file adds its flows to this project; a project file
 * replaces the project (after a confirm). Export sits beside them because the
 * confirm tells you to export first.
 *
 * A project file is read before anything is asked: one that is not a project
 * (not JSON, a flow file) is said so at once, rather than after "Replace this
 * project?" was answered, in the parser's own words.
 *
 * Lives as its own component (rather than inline in OAIYApp) so it can call
 * `useConfirmDialog()`: OAIYApp renders the provider and sits above it.
 */
export default function ImportMenu({
  importProject,
  onImportFlows,
  onExportProject,
  onNewProject,
  projectName,
  onShowToast,
}: ImportMenuProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const buttonRef = useRef<HTMLButtonElement>(null);
  const [open, setOpen] = useState(false);
  const confirm = useConfirmDialog();

  const handleFile = async (file: File) => {
    try {
      parseImportedProject(await file.text());
    } catch (err) {
      onShowToast(err instanceof Error ? err.message : "This file isn't an OAIY project.", 'error');
      return;
    }
    const ok = await confirm({
      title: 'Replace this project?',
      message: `The project in the file replaces "${projectName}" and its flows. Export this one first (Projects → Export this project) if you want to keep a copy.`,
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
        `The project was not imported: ${err instanceof Error ? err.message : 'the file could not be read'}`,
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
        aria-label="Projects"
        aria-haspopup="menu"
        aria-expanded={open}
        title="Projects: start a new one, import flows or a project, or export this one"
      >
        <FolderInput size={16} />
      </button>
      <Menu open={open} onClose={() => setOpen(false)} anchor={buttonRef.current} label="Projects">
        {onNewProject && (
          <MenuItem
            icon={<FolderPlus size={15} />}
            label="New project…"
            hint="Replaces this project with an empty one (asks first)"
            onSelect={onNewProject}
          />
        )}
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
          hint="Downloads it as a .json file"
          onSelect={onExportProject}
        />
      </Menu>
    </>
  );
}
