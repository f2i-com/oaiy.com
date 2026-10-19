import { memo, useState, useEffect, useRef, useCallback, useMemo, type ChangeEvent } from 'react';
import { Position } from '@xyflow/react';
import { useNodeResize, CollapsibleNodeWrapper, type HandleConfig } from 'oaiy-ui-components';
import { BLOCKED_MEDIA_WORDS, mediaUrlBlocked } from '../../../lib/localMedia';
import { desktopFileBytes } from '../../core-service/contractRuntime';
import { triggerDownloadFromBlob } from '../../../lib/download';
import { loadableSrc } from '../../../lib/mediaSrc';
import { cellText, isSafeUrl, outputItems, summaryOf, tableColumns, type OutputItem, type OutputKind } from './outputValue';

/**
 * The file a run made, as bytes to save: OAIY's own files with the desktop's credential (fetched bare they were a
 * refusal, saved as the picture), anything else as it is; an answer that is not the file is an error, not a download.
 */
async function outputBlob(url: string): Promise<Blob> {
  const own = await desktopFileBytes(url);
  if (own) return new Blob([own.bytes as BlobPart], { type: own.mime });
  const response = await fetch(url);
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.blob();
}

/** Items with their files made loadable, and a bare blob: URL's kind read from the blob (its address says nothing). */
function useLoadableItems(items: OutputItem[]): OutputItem[] {
  const [ready, setReady] = useState<OutputItem[]>(items);
  const key = JSON.stringify(items.map((i) => [i.kind, i.src.length > 300 ? i.src.slice(0, 300) + i.src.length : i.src, i.poster]));
  useEffect(() => {
    let live = true;
    setReady(items);
    void Promise.all(items.map(async (item) => {
      let next = item;
      if (item.kind === 'text' && item.src.startsWith('blob:')) {
        const type = await fetch(item.src).then((r) => r.blob()).then((b) => b.type).catch(() => '');
        const kind: OutputKind | null = type.startsWith('image/') ? 'image' : type.startsWith('video/') ? 'video'
          : type.startsWith('audio/') ? 'audio' : type.startsWith('model/') ? 'model3d' : null;
        if (kind) next = { ...item, kind };
      }
      if (next.kind !== 'text' && next.kind !== 'json' && next.kind !== 'table') {
        next = { ...next, src: await loadableSrc(next.src), poster: next.poster ? await loadableSrc(next.poster) : undefined };
      }
      return next;
    })).then((done) => { if (live) setReady(done); });
    return () => { live = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);
  return ready;
}

/** A file's ending for a download, from its name or its type. */
function extensionOf(item: OutputItem, blob: Blob): string {
  const fromName = /\.([A-Za-z0-9]{2,5})$/.exec(item.name ?? '')?.[1];
  if (fromName) return fromName.toLowerCase();
  const sub = blob.type.split('/')[1]?.split(/[;+]/)[0];
  const known: Record<string, string> = { jpeg: 'jpg', mpeg: 'mp3', 'x-wav': 'wav', wave: 'wav', 'gltf-binary': 'glb', quicktime: 'mov' };
  if (sub) return known[sub] ?? sub;
  return item.kind === 'image' ? 'png' : item.kind === 'video' ? 'mp4' : item.kind === 'audio' ? 'wav' : item.kind === 'model3d' ? 'glb' : 'bin';
}

interface OutputNodeData {
  label?: string;
  /** What reached it on the last run (outputValue.ts sorts it). */
  outputValue?: unknown;
  /** The last run did not finish (it failed, or was stopped): whatever it shows is not that run's result. */
  _stale?: boolean;
  _collapsed?: boolean;
  onLabelChange?: (value: string) => void;
  onCollapsedChange?: (value: boolean) => void;
  onShowToast?: (message: string, type?: 'success' | 'error' | 'info' | 'warning') => void;
  showBodyProperties?: boolean;
}

interface OutputNodeProps {
  data: OutputNodeData;
}

const OutputIcon = (
  <svg className="w-3 h-3 text-white" fill="currentColor" viewBox="0 0 20 20">
    <path fillRule="evenodd" d="M6.267 3.455a3.066 3.066 0 001.745-.723 3.066 3.066 0 013.976 0 3.066 3.066 0 001.745.723 3.066 3.066 0 012.812 2.812c.051.643.304 1.254.723 1.745a3.066 3.066 0 010 3.976 3.066 3.066 0 00-.723 1.745 3.066 3.066 0 01-2.812 2.812 3.066 3.066 0 00-1.745.723 3.066 3.066 0 01-3.976 0 3.066 3.066 0 00-1.745-.723 3.066 3.066 0 01-2.812-2.812 3.066 3.066 0 00-.723-1.745 3.066 3.066 0 010-3.976 3.066 3.066 0 00.723-1.745 3.066 3.066 0 012.812-2.812zm7.44 5.252a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
  </svg>
);

function OutputNode({ data }: OutputNodeProps) {
  const { size, handleResizeStart } = useNodeResize({
    initialWidth: 280,
    initialHeight: 260,
    constraints: { minWidth: 220, maxWidth: 600, minHeight: 200, maxHeight: 800 },
  });
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [failed, setFailed] = useState<Set<number>>(new Set());
  const [copyFeedback, setCopyFeedback] = useState<'success' | 'error' | null>(null);
  const copyTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const onCollapsedChangeRef = useRef(data.onCollapsedChange);

  useEffect(() => {
    onCollapsedChangeRef.current = data.onCollapsedChange;
  });

  useEffect(() => {
    return () => {
      if (copyTimeoutRef.current) clearTimeout(copyTimeoutRef.current);
    };
  }, []);

  const handleCollapsedChange = useCallback((collapsed: boolean) => {
    onCollapsedChangeRef.current?.(collapsed);
  }, []);

  // What reached it, sorted by what it is (outputValue.ts), with its files made loadable.
  const sorted = useMemo(() => outputItems(data.outputValue), [data.outputValue]);
  const items = useLoadableItems(sorted);
  const hasOutput = items.length > 0;
  const isMultiple = items.length > 1;
  const kind = items[0]?.kind;
  const safeIndex = Math.min(selectedIndex, Math.max(0, items.length - 1));
  const current = items[safeIndex];
  const blockedAny = items.some((i) => /^https?:\/\//i.test(i.src.trim()) && mediaUrlBlocked(i.src));

  useEffect(() => {
    setFailed(new Set());
    setSelectedIndex((prev) => (prev >= sorted.length ? 0 : prev));
  }, [sorted]);

  /** Its value as text: what Copy copies, and what Save saves when it is not a file. */
  const asText = (): string =>
    typeof data.outputValue === 'string' ? data.outputValue : JSON.stringify(data.outputValue, null, 2) ?? '';

  const handleCopy = async () => {
    if (!hasOutput) return;
    try {
      await navigator.clipboard.writeText(asText());
      setCopyFeedback('success');
    } catch {
      setCopyFeedback('error');
    }
    if (copyTimeoutRef.current) clearTimeout(copyTimeoutRef.current);
    copyTimeoutRef.current = setTimeout(() => setCopyFeedback(null), 2000);
  };

  // Save: each file it shows (pictures, video, sound, a 3D model) as a download; anything else as a text or JSON file.
  const handleSave = async () => {
    if (!hasOutput) return;
    const name = data.label || 'output';
    if (kind === 'text' || kind === 'json' || kind === 'table') {
      const json = kind !== 'text';
      triggerDownloadFromBlob(`${name}.${json ? 'json' : 'txt'}`, new Blob([asText()], { type: json ? 'application/json' : 'text/plain' }));
      data.onShowToast?.(`Saved ${name}.${json ? 'json' : 'txt'} to Downloads`, 'success');
      return;
    }
    let saved = 0;
    let missed = 0;
    for (let i = 0; i < items.length; i++) {
      const item = items[i];
      if (!isSafeUrl(item.src)) { missed++; continue; }
      try {
        const blob = await outputBlob(item.src);
        triggerDownloadFromBlob(`${name}${items.length > 1 ? `_${i + 1}` : ''}.${extensionOf(item, blob)}`, blob);
        saved++;
        // (a beat between downloads, or the browser blocks the rest)
        if (i < items.length - 1) await new Promise((resolve) => setTimeout(resolve, 200));
      } catch (err) {
        console.error(`Could not save ${name} ${i + 1}:`, err);
        missed++;
      }
    }
    const noun = kind === 'image' ? 'picture' : kind === 'video' ? 'video' : kind === 'audio' ? 'sound' : '3D model';
    if (missed === 0) data.onShowToast?.(saved === 1 ? `The ${noun} is in your Downloads` : `${saved} ${noun}s are in your Downloads`, 'success');
    else if (saved === 0) data.onShowToast?.(`The ${noun}${items.length > 1 ? 's' : ''} could not be saved`, 'error');
    else data.onShowToast?.(`${saved} saved, ${missed} could not be`, 'warning');
  };

  const contentHeight = Math.max(60, size.height - 160);

  /** A picture, with what to say when it does not load. */
  const picture = (item: OutputItem, index: number, className: string) =>
    failed.has(index) || !isSafeUrl(item.src) ? (
      <span className="text-[10px] italic text-slate-500 dark:text-slate-400">This picture cannot be shown</span>
    ) : (
      <img
        src={item.src}
        alt={`${data.label || 'Output'}${isMultiple ? ` ${index + 1}` : ''}`}
        className={className}
        onError={() => setFailed((prev) => new Set([...prev, index]))}
      />
    );

  /** A sound's player (the browser draws it in the theme's colours: the page's `color-scheme`; inverted as well it was
   * light on the dark theme). */
  const player = (item: OutputItem, compact: boolean) =>
    isSafeUrl(item.src) ? (
      <audio
        key={item.src}
        controls
        preload="metadata"
        src={item.src}
        className={`nodrag w-full ${compact ? 'h-7' : 'h-8'}`}
        aria-label={item.name ? `Play ${item.name}` : 'Play the sound'}
      />
    ) : (
      <span className="truncate text-[10px] text-slate-500">{item.name ?? item.src}</span>
    );

  // Compact: the result itself where it is small (pictures, a player, a video's first frame), else a line saying what it is.
  const collapsedPreview = !hasOutput ? (
    <span className="italic">{data._stale ? 'No result: the last run did not finish' : 'No result yet'}</span>
  ) : kind === 'image' ? (
    // (one row: several pictures side by side, the rest counted, so the card keeps its height)
    <div className="flex items-end gap-1 overflow-hidden">
      {items.slice(0, isMultiple ? 3 : 1).map((item, i) => (
        <span key={i} className="block flex-none">{picture(item, i, isMultiple ? 'block h-14 w-14 rounded object-cover' : 'block max-h-20 max-w-[110px] rounded object-contain')}</span>
      ))}
      {items.length > 3 && <span className="text-[10px]">+{items.length - 3}</span>}
    </div>
  ) : kind === 'audio' ? (
    <div className="flex flex-col gap-1">
      {player(items[0], true)}
      <span className="truncate">{summaryOf(items)}</span>
    </div>
  ) : kind === 'video' && isSafeUrl(items[0].src) ? (
    <div className="flex flex-col gap-1">
      <video src={items[0].src} muted playsInline preload="metadata" className="block max-h-24 max-w-full rounded" aria-label="The video's first frame" />
      <span className="truncate">{summaryOf(items)}</span>
    </div>
  ) : (
    <span className={kind === 'json' || kind === 'table' ? 'font-mono text-[10px]' : ''}>{summaryOf(items)}</span>
  );

  const inputHandles = useMemo<HandleConfig[]>(() => [
    { id: 'result', type: 'target', position: Position.Left, color: '!bg-blue-500', size: 'lg' },
  ], []);

  const resizeHandles = (
    <>
      <div
        className="nodrag absolute top-0 right-0 w-2 h-full cursor-ew-resize opacity-0 group-hover:opacity-100 hover:bg-emerald-500/30 transition-all"
        onMouseDown={(e) => handleResizeStart(e, 'e')}
      />
      <div
        className="nodrag absolute bottom-0 left-0 w-full h-2 cursor-ns-resize opacity-0 group-hover:opacity-100 hover:bg-emerald-500/30 transition-all"
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

  const frame = 'rounded border border-slate-300 dark:border-slate-700 bg-slate-100 dark:bg-slate-900';

  return (
    <CollapsibleNodeWrapper
      title="Output"
      color="emerald"
      icon={OutputIcon}
      width={size.width}
      collapsedWidth={140}
      isCollapsed={data._collapsed}
      onCollapsedChange={handleCollapsedChange}
      collapsedPreview={collapsedPreview}
      inputHandles={inputHandles}
      resizeHandles={resizeHandles}
    >
      {data.showBodyProperties !== false && (
        <>
          <div>
            <label className="text-slate-600 dark:text-slate-400 text-xs block mb-1">Label</label>
            <input
              type="text"
              aria-label="Label"
              className="nodrag nowheel w-full bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded px-2 py-1.5 text-sm text-slate-800 dark:text-slate-200 focus:outline-none focus:border-emerald-500"
              placeholder="Its name among the flow's results"
              value={data.label || ''}
              onChange={(e: ChangeEvent<HTMLInputElement>) => data.onLabelChange?.(e.target.value)}
              onMouseDown={(e) => e.stopPropagation()}
            />
          </div>

          <div>
            <div className="mb-1 flex items-center justify-between gap-2">
              <label className="text-slate-600 dark:text-slate-400 text-xs">Result</label>
              {/* (what it is, beside its label: not for text, which says it below) */}
              {hasOutput && kind !== 'text' && <span className="truncate text-[10px] text-slate-500 dark:text-slate-400">{summaryOf(items)}</span>}
            </div>
            {blockedAny && (
              <p className="text-amber-600 dark:text-amber-400 text-[10px] mb-1" data-blocked-media>{BLOCKED_MEDIA_WORDS}</p>
            )}
            {!hasOutput ? (
              <div className={`${frame} flex items-center justify-center p-3 text-center text-xs italic text-slate-500 dark:text-slate-400`} style={{ height: contentHeight }}>
                {data._stale ? 'No result: the last run did not finish' : 'Its result shows here after a run'}
              </div>
            ) : kind === 'image' ? (
              <div className={`${frame} overflow-hidden`} style={{ height: contentHeight }}>
                {isMultiple ? (
                  <div className="grid grid-cols-2 gap-1 p-1 overflow-auto h-full nowheel">
                    {items.map((item, index) => (
                      <button
                        key={`img-${index}`}
                        type="button"
                        onClick={() => setSelectedIndex(index)}
                        aria-pressed={index === safeIndex}
                        aria-label={`Picture ${index + 1}`}
                        className={`nodrag relative aspect-square rounded border overflow-hidden ${index === safeIndex ? 'border-emerald-500 ring-1 ring-emerald-500/50' : 'border-slate-300 dark:border-slate-600'}`}
                      >
                        {picture(item, index, 'w-full h-full object-cover')}
                      </button>
                    ))}
                  </div>
                ) : (
                  <div className="flex h-full w-full items-center justify-center">{picture(items[0], 0, 'w-full h-full object-contain')}</div>
                )}
              </div>
            ) : kind === 'video' ? (
              <div className={`${frame} overflow-hidden`} style={{ height: contentHeight }}>
                {current && isSafeUrl(current.src) ? (
                  <video key={current.src} src={current.src} controls playsInline className="nodrag w-full h-full object-contain" />
                ) : (
                  <div className="flex h-full items-center justify-center p-2 text-xs text-slate-500"><span className="truncate">{current?.name ?? current?.src}</span></div>
                )}
              </div>
            ) : kind === 'audio' ? (
              <div className={`${frame} flex flex-col gap-2 p-2 overflow-auto nowheel`} style={{ maxHeight: Math.max(contentHeight, 72) }}>
                {items.map((item, index) => (
                  <div key={`audio-${index}`} className="flex flex-col gap-1">
                    {player(item, false)}
                    {item.name && <span className="truncate text-[10px] text-slate-500 dark:text-slate-400">{item.name}</span>}
                  </div>
                ))}
              </div>
            ) : kind === 'model3d' ? (
              <div className={`${frame} flex flex-col gap-2 p-2`}>
                {items.map((item, index) => (
                  <div key={`model-${index}`} className="flex items-center gap-2">
                    {item.poster && isSafeUrl(item.poster) ? (
                      <img src={item.poster} alt="" className="h-14 w-14 flex-none rounded object-cover" />
                    ) : (
                      <span className="grid h-14 w-14 flex-none place-items-center rounded bg-slate-200 text-[10px] text-slate-500 dark:bg-slate-800 dark:text-slate-400">GLB</span>
                    )}
                    <span className="min-w-0 text-xs text-slate-700 dark:text-slate-300">
                      <span className="block truncate font-medium">{item.name ?? '3D model'}</span>
                      <span className="text-[10px] text-slate-500 dark:text-slate-400">A 3D model: Save keeps its file</span>
                    </span>
                  </div>
                ))}
              </div>
            ) : kind === 'table' ? (
              <TablePreview rows={items[0].value as Array<Record<string, unknown>>} height={contentHeight} />
            ) : (
              <textarea
                readOnly
                aria-label="Result"
                value={kind === 'json' ? items[0].src : asText()}
                style={{ height: contentHeight }}
                className={`nodrag nowheel w-full rounded px-2 py-2 resize-none ${kind === 'json' ? 'font-mono text-[11px]' : ''}`}
                onMouseDown={(e) => e.stopPropagation()}
              />
            )}
          </div>

          {hasOutput && (
            <div className="flex gap-1.5">
              <button
                type="button"
                onClick={handleCopy}
                title={kind === 'text' || kind === 'json' || kind === 'table' ? 'Copy the result' : 'Copy what reached it (its address or data)'}
                className={`nodrag flex-1 px-2 py-1.5 text-xs rounded flex items-center justify-center gap-1 ${copyFeedback === 'success' ? 'bg-green-700 text-white' :
                  copyFeedback === 'error' ? 'bg-red-700 text-white' :
                    'bg-slate-200 dark:bg-slate-700 hover:bg-slate-300 dark:hover:bg-slate-600 text-slate-700 dark:text-slate-300'
                  }`}
              >
                {copyFeedback === 'success' ? 'Copied' : copyFeedback === 'error' ? 'Could not copy' : 'Copy'}
              </button>
              <button
                type="button"
                onClick={handleSave}
                title="Save it to your Downloads"
                className="nodrag flex-1 px-2 py-1.5 bg-emerald-700 hover:bg-emerald-600 text-white text-xs rounded"
              >
                Save
              </button>
            </div>
          )}
        </>
      )}
    </CollapsibleNodeWrapper>
  );
}

/** A list of records as a table: a column per key (the first 12), a row each (the first 200), long cells wrapped. */
function TablePreview({ rows, height }: { rows: Array<Record<string, unknown>>; height: number }) {
  const cols = tableColumns(rows).slice(0, 12);
  const shown = rows.slice(0, 200);
  return (
    <div className="nodrag nowheel overflow-auto rounded border border-slate-300 dark:border-slate-700" style={{ maxHeight: Math.max(height, 96) }}>
      <table className="w-full border-collapse text-left text-[11px]">
        <thead className="sticky top-0 bg-slate-100 dark:bg-slate-800">
          <tr>
            {cols.map((c) => (
              <th key={c} scope="col" className="whitespace-nowrap border-b border-slate-300 px-2 py-1 font-semibold text-slate-700 dark:border-slate-700 dark:text-slate-300">{c}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {shown.map((r, i) => (
            <tr key={i} className="odd:bg-white even:bg-slate-50 dark:odd:bg-slate-900 dark:even:bg-slate-900/60">
              {cols.map((c) => {
                const v = r[c];
                return (
                  <td
                    key={c}
                    className={`max-w-[220px] border-b border-slate-200 px-2 py-1 align-top text-slate-700 dark:border-slate-800 dark:text-slate-300 ${typeof v === 'number' ? 'text-right tabular-nums' : ''}`}
                    // (a long word breaks only where it must: "Slovenia" stays whole, and the table scrolls sideways)
                    style={{ overflowWrap: 'break-word' }}
                  >
                    {cellText(v)}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
      {rows.length > shown.length && (
        <p className="m-0 px-2 py-1 text-[10px] text-slate-500 dark:text-slate-400">The first {shown.length} of {rows.length} rows (Copy or Save has them all)</p>
      )}
    </div>
  );
}

export default memo(OutputNode);
