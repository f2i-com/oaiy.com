/**
 * Interactive turn deliverables & workspace-changes card inspired by DeepSeek Harness
 * (`@deepseek-ai/dsh-ui-deliverables` + `@deepseek-ai/dsh-workspace-changes`).
 *
 * Shows net files created, modified, or deleted during a turn, exact `+added / -deleted` line
 * statistics, one-click Open in Editor, and expandable color-coded unified diff hunks.
 */
import type { DeliverableFile } from '../../agent/deliverables';
import { h } from '../dom';
import { fileIcon, icon } from '../icons';
import { duration } from './tools';

function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(2)} MB`;
}

const KIND_BADGE: Record<DeliverableFile['kind'], { label: string; title: string }> = {
  created: { label: 'NEW', title: 'Created in this turn' },
  modified: { label: 'MOD', title: 'Modified in this turn' },
  deleted: { label: 'DEL', title: 'Deleted in this turn' },
};

export function renderDeliverablesCard(
  files: DeliverableFile[],
  meta: { durationMs?: number; toolCount?: number; checkpointId?: string },
  onOpen: (path: string) => void,
  onRevert?: (checkpointId: string) => boolean | Promise<boolean>,
): HTMLElement {
  const totalAdded = files.reduce((s, f) => s + f.added, 0);
  const totalDeleted = files.reduce((s, f) => s + f.deleted, 0);
  const rows: HTMLDetailsElement[] = [];

  const list = h('div.deliverables-list');
  for (const file of files) {
    const badge = KIND_BADGE[file.kind];
    const name = file.path.split('/').pop() || file.path;
    const kind = fileIcon(name, false, false);
    const hasDiff = file.hunks.length > 0;
    const stats = h('span.deliverable-stats');
    if (file.binary) {
      stats.append(h('span.deliverable-bin', `binary · ${formatBytes(file.sizeBytes)}`));
    } else {
      if (file.added > 0) stats.append(h('span.diff-stat-add', `+${file.added}`));
      if (file.deleted > 0) stats.append(h('span.diff-stat-del', `-${file.deleted}`));
      if (file.added === 0 && file.deleted === 0) stats.append(h('span.deliverable-bin', formatBytes(file.sizeBytes)));
    }

    const actions = h('span.deliverable-actions');
    if (file.kind !== 'deleted') {
      actions.append(
        h(
          'button.deliverable-open',
          {
            type: 'button',
            title: `Open /${file.path} in the editor`,
            onclick: (e: Event) => {
              e.preventDefault();
              e.stopPropagation();
              onOpen(file.path);
            },
          },
          'Open',
        ),
      );
    }

    const diffBody = h('div.deliverable-diff');
    if (hasDiff) {
      const pre = h('div.diff-hunks');
      for (const hunk of file.hunks) {
        pre.append(h('div.diff-hunk-head', hunk.header));
        for (const line of hunk.lines) {
          const prefix = line.kind === 'add' ? '+' : line.kind === 'del' ? '−' : ' ';
          pre.append(
            h(
              `div.diff-line.diff-${line.kind}`,
              h('span.diff-gutter', prefix),
              h('span.diff-text', line.text || ' '),
            ),
          );
        }
      }
      if (file.truncatedDiff) {
        pre.append(h('div.diff-truncated', 'Diff preview truncated — open the file to view all changes.'));
      }
      diffBody.append(pre);
    } else {
      diffBody.append(
        h(
          'div.diff-empty',
          file.binary
            ? `Binary file (${formatBytes(file.sizeBytes)}) — click Open to inspect.`
            : 'No line diff available.',
        ),
      );
    }

    const row = h(
      'details.deliverable-file',
      { 'data-kind': file.kind },
      h(
        'summary.deliverable-row',
        icon('chevron-right', 'deliverable-caret'),
        h('span.deliverable-badge', { class: `kind-${file.kind}`, title: badge.title }, badge.label),
        h('span.deliverable-icon', { class: `tone-${kind.tone}` }, icon(kind.icon)),
        h('span.deliverable-path', { title: `/${file.path}` }, file.path),
        stats,
        actions,
      ),
      diffBody,
    ) as HTMLDetailsElement;
    // Auto-expand if only a single small text file was changed so the diff is immediately scannable.
    if (files.length === 1 && hasDiff && file.added + file.deleted <= 40) row.open = true;
    rows.push(row);
    list.append(row);
  }

  let allExpanded = rows.length > 0 && rows.every((r) => r.open);
  const toggleBtn = h(
    'button.deliverables-toggle',
    {
      type: 'button',
      onclick: () => {
        allExpanded = !allExpanded;
        for (const r of rows) r.open = allExpanded;
        toggleBtn.textContent = allExpanded ? 'Collapse diffs' : 'Expand diffs';
      },
    },
    allExpanded ? 'Collapse diffs' : 'Expand diffs',
  );
  const syncToggle = () => {
    allExpanded = rows.length > 0 && rows.every((r) => r.open);
    toggleBtn.textContent = allExpanded ? 'Collapse diffs' : 'Expand diffs';
  };
  for (const r of rows) r.addEventListener('toggle', syncToggle);

  const metaBits: string[] = [];
  if (meta.durationMs && meta.durationMs >= 200) metaBits.push(duration(meta.durationMs));
  if (meta.toolCount && meta.toolCount > 0) metaBits.push(`${meta.toolCount} tool${meta.toolCount === 1 ? '' : 's'}`);

  const totalStats = h('span.deliverables-totals');
  if (totalAdded > 0) totalStats.append(h('span.diff-stat-add', `+${totalAdded}`));
  if (totalDeleted > 0) totalStats.append(h('span.diff-stat-del', `-${totalDeleted}`));

  let revertEl: HTMLElement | null = null;
  if (meta.checkpointId && onRevert) {
    const cpId = meta.checkpointId;
    const btn = h(
      'button.deliverables-revert',
      {
        type: 'button',
        title: 'Revert the file changes made in this turn (/undo)',
        onclick: async () => {
          btn.disabled = true;
          try {
            const ok = await onRevert(cpId);
            if (ok) {
              btn.classList.add('reverted');
              btn.replaceChildren(icon('check'), h('span', 'Reverted'));
              btn.title = 'Turn changes reverted';
            } else {
              btn.disabled = false;
            }
          } catch {
            btn.disabled = false;
          }
        },
      },
      icon('undo'),
      h('span', 'Revert'),
    ) as HTMLButtonElement;
    revertEl = btn;
  }

  return h(
    'section.deliverables-card',
    { 'aria-label': `Files changed (${files.length})` },
    h(
      'div.deliverables-head',
      icon('layers', 'deliverables-head-icon'),
      h('span.deliverables-title', `${files.length} file${files.length === 1 ? '' : 's'} changed`),
      totalStats,
      ...(metaBits.length ? [h('span.deliverables-meta', metaBits.join(' · '))] : []),
      h('span.deliverables-head-actions', ...(revertEl ? [revertEl] : []), toggleBtn),
    ),
    list,
  );
}
