import { updateController, useUpdateState } from '../pwa';
import { updateMessage } from '../pwa/updateController';

/**
 * A new build of the editor is waiting (or another tab of it just took one).
 *
 * Nothing reloads by itself: a person may be part-way through wiring a flow. The message stays
 * until they reload or say later. See pwa/updateController.ts.
 */
export default function UpdateNotice() {
  const message = updateMessage(useUpdateState());
  if (!message) return null;
  return (
    <div className="oaiy-update-notice" role="status" aria-live="polite" data-testid="update-notice">
      <span>{message}</span>
      <button type="button" className="btn btn-primary btn-sm" onClick={() => updateController.apply()}>
        Reload
      </button>
      <button type="button" className="btn btn-ghost btn-sm" onClick={() => updateController.dismiss()}>
        Later
      </button>
    </div>
  );
}
