/**
 * Give a flow to OAIY's agent: as a tool of its own, or in front of one of the
 * agent's tools (run before it, or instead of it). Shown in OAIY's window.
 */
import { useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import type { Flow } from 'oaiy-core';
import { useFocusTrap } from '../../hooks/useFocusTrap';
import { AGENT_TOOLS, agentUse, publishAsHook, publishAsTool, toolName, withdrawHook, withdrawTool } from '../../lib/oaiyAgentTools';

type Use = 'tool' | 'before' | 'instead';

const USES: Array<{ id: Use; title: string; says: string }> = [
  { id: 'tool', title: 'A tool of its own', says: "The agent can use the flow like any of its tools: its input nodes are the tool's parameters, by their labels." },
  {
    id: 'before',
    title: 'Before one of its tools',
    says: 'The flow runs first each time the agent uses the tool, given the call (inputs labelled like its parameters, or "tool" and "input"). It answers "ok" to let it go ahead, "STOP: why" to stop it, {"input": {…}} to change it; anything else is a note for the agent.',
  },
  { id: 'instead', title: 'Instead of one of its tools', says: "The flow runs in the tool's place, given the call the same way, and what it returns is the tool's result." },
];

interface Props {
  flow: Flow;
  onClose: () => void;
  /** Said when something was given or taken back. */
  onDone: (message: string) => void;
}

export default function AgentToolDialog({ flow, onClose, onDone }: Props) {
  const dialogRef = useRef<HTMLDivElement>(null);
  const [use, setUse] = useState<Use>('tool');
  const [name, setName] = useState(toolName(flow.name));
  const [description, setDescription] = useState(flow.description || `Runs the flow "${flow.name}".`);
  const [tool, setTool] = useState('write_file');
  const [now, setNow] = useState<Awaited<ReturnType<typeof agentUse>>>({});
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  useFocusTrap(dialogRef, true);

  // How the flow is given now: the dialog opens on that.
  useEffect(() => {
    let live = true;
    void agentUse(flow).then((u) => {
      if (!live) return;
      setNow(u);
      if (u.hook) {
        setUse(u.hook.mode);
        setTool(u.hook.tool);
      } else if (u.tool) {
        setName(u.tool.name);
        if (u.tool.description) setDescription(u.tool.description);
      }
    });
    return () => {
      live = false;
    };
  }, [flow]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && onClose();
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [onClose]);

  const give = async () => {
    setBusy(true);
    setError(null);
    try {
      if (use === 'tool') {
        await publishAsTool(flow, { name, description });
        onDone(`"${toolName(name)}" is one of the agent's tools now.`);
      } else {
        await publishAsHook(flow, { tool, mode: use });
        onDone(`"${flow.name}" runs ${use === 'before' ? 'before' : 'instead of'} the agent's ${tool} now (within a minute).`);
      }
      onClose();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const takeBack = async () => {
    setBusy(true);
    try {
      if (now.tool) await withdrawTool(flow);
      if (now.hook) await withdrawHook(flow);
      onDone(`"${flow.name}" is no longer given to the agent.`);
      onClose();
    } finally {
      setBusy(false);
    }
  };

  const field = 'w-full px-3 py-2 text-sm rounded-md bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 text-slate-800 dark:text-slate-100 focus:outline-none focus:ring-2 focus:ring-accent';
  const given = now.tool ? `a tool, "${now.tool.name}"` : now.hook ? `${now.hook.mode === 'before' ? 'before' : 'instead of'} ${now.hook.tool}` : null;

  return createPortal(
    <div className="fixed inset-0 z-[100] flex items-center justify-center">
      <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" onClick={onClose} />
      <div
        ref={dialogRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby="agent-tool-title"
        className="relative bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-600 rounded-xl shadow-2xl w-full max-w-lg mx-4 overflow-hidden animate-scaleIn"
      >
        <div className="p-5 space-y-4">
          <div>
            <h3 id="agent-tool-title" className="text-lg font-semibold text-slate-800 dark:text-slate-100">
              Give “{flow.name}” to the agent
            </h3>
            <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">{given ? `It is given now: ${given}.` : 'OAIY’s agent runs it on this computer.'}</p>
          </div>

          <div className="grid gap-2" role="radiogroup" aria-label="How the agent uses it">
            {USES.map((u) => (
              <label
                key={u.id}
                className={`flex gap-3 p-3 rounded-lg border cursor-pointer transition-colors ${use === u.id ? 'border-accent bg-accent/10' : 'border-slate-200 dark:border-slate-700 hover:border-slate-400 dark:hover:border-slate-500'}`}
              >
                <input type="radio" name="agent-use" className="mt-1 accent-[rgb(var(--accent-primary))]" checked={use === u.id} onChange={() => setUse(u.id)} />
                <span>
                  <span className="block text-sm font-medium text-slate-800 dark:text-slate-100">{u.title}</span>
                  <span className="block mt-0.5 text-xs leading-relaxed text-slate-500 dark:text-slate-400">{u.says}</span>
                </span>
              </label>
            ))}
          </div>

          {use === 'tool' ? (
            <div className="space-y-3">
              <label className="block text-xs font-medium text-slate-600 dark:text-slate-300">
                Its name, as the agent calls it
                <input className={`${field} mt-1 font-mono`} spellCheck={false} value={name} onChange={(e) => setName(e.target.value)} />
              </label>
              <label className="block text-xs font-medium text-slate-600 dark:text-slate-300">
                What it does, and when to use it
                <textarea className={`${field} mt-1`} rows={2} value={description} onChange={(e) => setDescription(e.target.value)} />
              </label>
            </div>
          ) : (
            <label className="block text-xs font-medium text-slate-600 dark:text-slate-300">
              The agent’s tool
              <input className={`${field} mt-1 font-mono`} spellCheck={false} list="agent-tool-names" value={tool} onChange={(e) => setTool(e.target.value.trim())} />
              <datalist id="agent-tool-names">
                {AGENT_TOOLS.map((t) => (
                  <option key={t.name} value={t.name}>
                    {t.does}
                  </option>
                ))}
              </datalist>
              <span className="block mt-1 font-normal text-slate-500 dark:text-slate-400">{AGENT_TOOLS.find((t) => t.name === tool)?.does ?? 'One of the agent’s tools, by name.'}</span>
            </label>
          )}

          {error && <p className="text-sm text-red-600 dark:text-red-400">{error}</p>}
        </div>

        <div className="flex items-center justify-between gap-3 px-5 py-4 border-t border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-900/40">
          <div>
            {given && (
              <button type="button" disabled={busy} onClick={() => void takeBack()} className="text-sm text-red-600 dark:text-red-400 hover:underline disabled:opacity-50">
                Take it back
              </button>
            )}
          </div>
          <div className="flex gap-2">
            <button type="button" onClick={onClose} className="px-4 py-2 text-sm rounded-md text-slate-700 dark:text-slate-200 bg-slate-200 dark:bg-slate-700 hover:bg-slate-300 dark:hover:bg-slate-600">
              Cancel
            </button>
            <button
              type="button"
              disabled={busy || (use === 'tool' ? !name.trim() : !tool)}
              onClick={() => void give()}
              className="px-4 py-2 text-sm font-medium rounded-md text-white disabled:opacity-50"
              style={{ backgroundColor: 'rgb(var(--accent-fill, var(--accent-primary)))' }}
            >
              {busy ? 'Giving…' : 'Give it to the agent'}
            </button>
          </div>
        </div>
      </div>
    </div>,
    document.body,
  );
}
