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
 * own small root with the providers the form needs. The form is the editor's
 * one service dialog (ServiceForm), with this node's notes at its top.
 */
import { useState } from 'react';
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

function AddServiceDialog({ nodeType }: { nodeType: string }) {
  const inOaiy = oaiyDesktop() !== null;
  const label = NODE_LABEL[nodeType] || 'this node';
  // Built once per opening: ServiceForm compares its draft with this to know
  // whether closing would throw an edit away.
  const [initial] = useState<CustomService>(() => ({
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
  }));
  const [error, setError] = useState<string | null>(null);
  const onSave = (svc: CustomService) => {
    if (!svc.name.trim() || !svc.endpoint.trim()) {
      setError('Give the service a name and its endpoint URL.');
      return;
    }
    saveService(serviceForNode(svc, nodeType));
    close();
  };
  return (
    <ServiceForm
      initial={initial}
      isNew
      onSave={onSave}
      onCancel={close}
      title={`Add a service for ${label}`}
      error={error}
      intro={
        <ul className="oaiy-note m-0 flex list-disc flex-col gap-1" style={{ paddingLeft: 28 }}>
          {inOaiy ? (
            <>
              <li>
                <strong>A model for OAIY's engine</strong> (pictures, video, speech, music, sound effects, 3D): add it in
                OAIY → <strong>Engines</strong>. Once its files are here it is listed under “OAIY engine”.
              </li>
              <li>
                <strong>Your own Python rig or local server</strong> (ComfyUI, anything with an HTTP API): add it in
                OAIY → <strong>Services</strong>. Once installed it is listed under “Your services”; its template's{' '}
                <code className="oaiy-code">node</code> block says how a node calls it.
              </li>
            </>
          ) : (
            <li>
              In the OAIY app, its engine's models and your Python rigs are listed here by themselves.
            </li>
          )}
          <li>
            <strong>Any HTTP service</strong>: describe it below. It is saved in this editor (Settings → Services) and offered
            on {label} nodes; its body template gets the node's values as <code className="oaiy-code">{'{{prompt}}'}</code>,{' '}
            <code className="oaiy-code">{'{{image}}'}</code>, <code className="oaiy-code">{'{{input}}'}</code> and so on.
          </li>
        </ul>
      }
      footerStart={
        inOaiy ? (
          <button type="button" className="btn btn-ghost btn-sm" onClick={() => void refreshDesktopServices()}>
            Added one in OAIY? Look again now
          </button>
        ) : undefined
      }
    />
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
      <AddServiceDialog nodeType={nodeType} />
    </ConfirmDialogProvider>,
  );
}
