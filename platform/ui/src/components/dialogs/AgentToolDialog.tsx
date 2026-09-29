/**
 * Give a flow to OAIY's agent: as a tool of its own, or in front of one of the
 * agent's tools (run before it, or instead of it). Shown in OAIY's window.
 */
import { useEffect, useState } from 'react';
import { Wrench } from 'lucide-react';
import type { Flow } from 'oaiy-core';
import Dialog from '../ui/Dialog';
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
  const [use, setUse] = useState<Use>('tool');
  const [name, setName] = useState(toolName(flow.name));
  const [description, setDescription] = useState(flow.description || `Runs the flow "${flow.name}".`);
  const [tool, setTool] = useState('write_file');
  const [now, setNow] = useState<Awaited<ReturnType<typeof agentUse>>>({});
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

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

  const given = now.tool ? `a tool, "${now.tool.name}"` : now.hook ? `${now.hook.mode === 'before' ? 'before' : 'instead of'} ${now.hook.tool}` : null;

  return (
    <Dialog
      open
      onClose={onClose}
      title={<>Give “{flow.name}” to the agent</>}
      description={given ? `It is given now: ${given}.` : 'OAIY’s agent runs it on this computer.'}
      icon={<Wrench size={16} />}
      tone="accent"
      size="md"
      dismissible={!busy}
      footer={
        <>
          {given && (
            <button type="button" disabled={busy} onClick={() => void takeBack()} className="btn btn-danger">
              Take it back
            </button>
          )}
          <span className="spacer" />
          <button type="button" onClick={onClose} disabled={busy} className="btn btn-secondary">
            Cancel
          </button>
          <button
            type="button"
            disabled={busy || (use === 'tool' ? !name.trim() : !tool)}
            onClick={() => void give()}
            className="btn btn-primary"
          >
            {busy ? 'Giving…' : 'Give it to the agent'}
          </button>
        </>
      }
    >
      <div className="grid gap-2" role="radiogroup" aria-label="How the agent uses it">
        {USES.map((u) => (
          <label
            key={u.id}
            className={`flex cursor-pointer gap-3 rounded-[var(--r-ctl)] border p-3 transition-colors ${use === u.id ? 'border-accent bg-accent/10' : 'border-edge-primary hover:border-edge-strong'}`}
          >
            <input type="radio" name="agent-use" className="mt-1 accent-[rgb(var(--accent-primary))]" checked={use === u.id} onChange={() => setUse(u.id)} />
            <span>
              <span className="block text-[13px] font-semibold text-content-primary">{u.title}</span>
              <span className="mt-0.5 block text-[12px] leading-relaxed text-content-secondary">{u.says}</span>
            </span>
          </label>
        ))}
      </div>

      {use === 'tool' ? (
        <>
          <label className="oaiy-field">
            <span>Its name, as the agent calls it</span>
            <input className="oaiy-input mono" spellCheck={false} value={name} onChange={(e) => setName(e.target.value)} />
          </label>
          <label className="oaiy-field">
            <span>What it does, and when to use it</span>
            <textarea className="oaiy-textarea" rows={2} value={description} onChange={(e) => setDescription(e.target.value)} />
          </label>
        </>
      ) : (
        <label className="oaiy-field">
          <span>The agent’s tool</span>
          <input className="oaiy-input mono" spellCheck={false} list="agent-tool-names" value={tool} onChange={(e) => setTool(e.target.value.trim())} />
          <datalist id="agent-tool-names">
            {AGENT_TOOLS.map((t) => (
              <option key={t.name} value={t.name}>
                {t.does}
              </option>
            ))}
          </datalist>
          <p className="oaiy-help faint">{AGENT_TOOLS.find((t) => t.name === tool)?.does ?? 'One of the agent’s tools, by name.'}</p>
        </label>
      )}

      {error && <p className="oaiy-error-text">{error}</p>}
    </Dialog>
  );
}
