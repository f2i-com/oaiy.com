/**
 * "Add a service…" — the last entry of every Service dropdown.
 *
 * Says where services come from (models in OAIY → Engines; Python rigs and
 * local servers in OAIY → Services, which appear in the dropdown by
 * themselves once installed) and opens the editor's own service form, tagged
 * for the node it was opened from, so an HTTP service the user describes
 * here is offered on that node straight away.
 *
 * Opened from a dropdown resolver (outside any React tree), so it mounts its
 * own small root with the providers the form needs.
 */
import { createRoot, type Root } from 'react-dom/client';
import { v4 as uuidv4 } from 'uuid';
import type { CustomService, ServiceNodeTag } from 'oaiy-core/modules/core-service/examples';
import { NODE_OUTPUT } from 'oaiy-core/modules/core-service/contract';
import { ServiceForm } from '../components/panels/settings/ServicesTab';
import { ConfirmDialogProvider } from '../hooks/useConfirmDialog';
import { saveService } from '../utils/serviceRegistry';
import { refreshDesktopServices } from './desktopServices';
import { oaiyDesktop } from './oaiyAgentTools';

const NODE_LABEL: Record<string, string> = {
  ai_llm: 'AI LLM',
  image_gen: 'Image Gen',
  video_gen: 'Video Gen',
  text_to_speech: 'Text to Speech',
  music_gen: 'Music Gen',
  speech_to_text: 'Speech to Text',
  sound_effect: 'Sound Effect',
  model_3d: '3D Model',
  background_removal: 'Remove Background',
  image_upscale: 'Upscale Image',
};

let open: { root: Root; host: HTMLElement } | null = null;

function close(): void {
  if (!open) return;
  const { root, host } = open;
  open = null;
  root.unmount();
  host.remove();
}

/** A service described here, for `nodeType`: tagged for it, making what it makes. */
export function serviceForNode(draft: CustomService, nodeType: string): CustomService {
  const tags = new Set<ServiceNodeTag>([...(draft.nodeTypes ?? [])]);
  if (nodeType) tags.add(nodeType as ServiceNodeTag);
  tags.add('service_call');
  const out: CustomService = { ...draft, nodeTypes: [...tags] };
  const makes = NODE_OUTPUT[nodeType];
  if (makes && !out.output) out.output = makes;
  return out;
}

function Dialog({ nodeType }: { nodeType: string }) {
  const inOaiy = oaiyDesktop() !== null;
  const label = NODE_LABEL[nodeType] || 'this node';
  const initial: CustomService = {
    id: uuidv4(),
    name: '',
    description: '',
    endpoint: '',
    method: 'POST',
    headers: '{}',
    bodyTemplate: '{{inputRaw}}',
    responseType: 'json',
    responsePath: '',
    apiKeyConstant: '',
    installHint: '',
    nodeTypes: nodeType ? [nodeType as ServiceNodeTag, 'service_call'] : ['service_call'],
  };
  const onSave = (svc: CustomService) => {
    if (!svc.name.trim() || !svc.endpoint.trim()) {
      window.alert('Give the service a name and its endpoint URL.');
      return;
    }
    saveService(serviceForNode(svc, nodeType));
    close();
  };
  return (
    <div
      className="fixed inset-0 z-[1000] flex items-center justify-center bg-black/50 p-4"
      onMouseDown={(e) => {
        if (e.target === e.currentTarget) close();
      }}
    >
      <div className="max-h-[90vh] w-full max-w-xl overflow-y-auto rounded-lg border border-slate-300 bg-white p-4 text-slate-900 shadow-xl dark:border-slate-700 dark:bg-slate-900 dark:text-slate-100">
        <div className="mb-3 flex items-start justify-between gap-3">
          <h2 className="text-base font-semibold">Add a service for {label}</h2>
          <button type="button" onClick={close} className="text-slate-500 hover:text-slate-800 dark:hover:text-slate-200" aria-label="Close">
            ✕
          </button>
        </div>
        <ul className="mb-3 list-disc space-y-1 pl-5 text-xs text-slate-600 dark:text-slate-300">
          {inOaiy ? (
            <>
              <li>
                <strong>A model for OAIY's engine</strong> (pictures, video, speech, music, sound effects, 3D): add it in
                OAIY → <strong>Engines</strong>. Once its files are here it is listed under “OAIY engine”.
              </li>
              <li>
                <strong>Your own Python rig or local server</strong> (Ollama, llama.cpp, anything with an HTTP API): add it in
                OAIY → <strong>Services</strong>. Once installed it is listed under “Your services”; its template's{' '}
                <code>node</code> block says how a node calls it.
              </li>
            </>
          ) : (
            <li>
              In the OAIY app, its engine's models and your Python rigs are listed here by themselves.
            </li>
          )}
          <li>
            <strong>Any HTTP service</strong>: describe it below. It is saved in this editor (Settings → Services) and offered
            on {label} nodes; its body template gets the node's values as <code>{'{{prompt}}'}</code>,{' '}
            <code>{'{{image}}'}</code>, <code>{'{{input}}'}</code> and so on.
          </li>
        </ul>
        <ServiceForm initial={initial} isNew onSave={onSave} onCancel={close} />
        {inOaiy && (
          <button
            type="button"
            className="mt-3 text-xs text-blue-600 hover:underline dark:text-blue-400"
            onClick={() => void refreshDesktopServices()}
          >
            Added one in OAIY? Look again now
          </button>
        )}
      </div>
    </div>
  );
}

/** Open the dialog for a node of `nodeType` (only one at a time). */
export function openAddServiceDialog(nodeType: string): void {
  if (typeof document === 'undefined') return;
  close();
  const host = document.createElement('div');
  host.dataset.oaiyAddService = '';
  document.body.appendChild(host);
  const root = createRoot(host);
  open = { root, host };
  root.render(
    <ConfirmDialogProvider>
      <Dialog nodeType={nodeType} />
    </ConfirmDialogProvider>,
  );
}
