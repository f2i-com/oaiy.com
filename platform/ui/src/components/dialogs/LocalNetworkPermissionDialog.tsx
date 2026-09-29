import { useRef, useState } from 'react';
import { AlertTriangle } from 'lucide-react';
import type { LocalNetworkPermissionRequest } from 'oaiy-core';
import Dialog from '../ui/Dialog';

interface LocalNetworkPermissionDialogProps {
  request: LocalNetworkPermissionRequest;
  onResponse: (allowed: boolean, remember: boolean) => void;
}

export default function LocalNetworkPermissionDialog({
  request,
  onResponse,
}: LocalNetworkPermissionDialogProps) {
  const [remember, setRemember] = useState(true);
  // Initial focus goes to Deny — this is a permission prompt, the
  // safe default is "no" even though "yes" is the more visually
  // prominent button. Matches the ConfirmDialog danger convention.
  const denyButtonRef = useRef<HTMLButtonElement>(null);

  // Escape and the overlay = Deny (without remembering). Nothing is ever
  // committed by dismissing; the user can re-run the workflow if they meant Allow.
  const deny = () => onResponse(false, false);

  return (
    <Dialog
      open
      onClose={deny}
      title="Let this flow reach your network?"
      description="A flow wants to connect to a service on this computer or your local network."
      icon={<AlertTriangle size={16} />}
      tone="warning"
      size="sm"
      role="alertdialog"
      initialFocusRef={denyButtonRef}
      footer={
        <>
          <button ref={denyButtonRef} type="button" onClick={deny} className="btn btn-secondary">
            Deny
          </button>
          <button type="button" onClick={() => onResponse(true, remember)} className="btn btn-primary">
            {remember ? 'Allow and remember' : 'Allow once'}
          </button>
        </>
      }
    >
      <div className="flex flex-col gap-1">
        <span className="oaiy-label">Address</span>
        <div className="font-mono text-[15px] text-content-primary">{request.hostPort}</div>
        <div className="truncate font-mono text-[11.5px] text-content-faint" title={request.url}>{request.url}</div>
      </div>

      {request.purpose && (
        <div className="flex flex-col gap-1">
          <span className="oaiy-label">Purpose</span>
          <div className="text-content-primary">{request.purpose}</div>
        </div>
      )}

      <div className="oaiy-note warn">
        Allow it only if you trust this flow and the service running at this address.
      </div>

      <label className="oaiy-check">
        <input
          type="checkbox"
          checked={remember}
          onChange={(e) => setRemember(e.target.checked)}
        />
        <div className="flex flex-col gap-0.5">
          <strong>Remember this address</strong>
          <span className="oaiy-help faint">It is added to the allowed addresses in Settings → Security.</span>
        </div>
      </label>
    </Dialog>
  );
}
