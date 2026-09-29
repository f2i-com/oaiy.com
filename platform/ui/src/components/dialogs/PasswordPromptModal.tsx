import { useState, useRef } from 'react';
import { Lock } from 'lucide-react';
import Dialog from '../ui/Dialog';

/**
 * Masked password prompt for opening an ENCRYPTED shared flow. Replaces the
 * unmasked `window.prompt` fallback in openSharedFlow.ts (which renders the
 * typed password in cleartext) with a proper themed `type=password` dialog.
 * The password is used only in-browser to decrypt the flow — never sent
 * anywhere (see flowCrypto.ts). The overlay and Escape both cancel, as in
 * every dialog (it is the editor's one Dialog, which needs no providers, so it
 * works during boot too).
 */
interface Props {
  /** The previous attempt's failure message, or null on the first prompt. */
  lastError: string | null;
  onSubmit: (password: string) => void;
  onCancel: () => void;
}

export default function PasswordPromptModal({ lastError, onSubmit, onCancel }: Props) {
  const [password, setPassword] = useState('');
  const inputRef = useRef<HTMLInputElement>(null);

  return (
    <Dialog
      open
      onClose={onCancel}
      title="This flow is encrypted"
      description="Enter its password to open it. It is used only in your browser to decrypt the flow, never sent anywhere."
      icon={<Lock size={16} />}
      tone="accent"
      size="sm"
      initialFocusRef={inputRef}
      footer={
        <>
          <button type="button" onClick={onCancel} className="btn btn-secondary">
            Cancel
          </button>
          <button type="submit" form="oaiy-flow-password" disabled={!password} className="btn btn-primary">
            Open flow
          </button>
        </>
      }
    >
      {lastError && <div className="oaiy-banner">{lastError}</div>}
      <form
        id="oaiy-flow-password"
        onSubmit={(e) => {
          e.preventDefault();
          if (password) onSubmit(password);
        }}
      >
        <label className="oaiy-field">
          <span>Password</span>
          <input
            ref={inputRef}
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            className="oaiy-input"
            placeholder="Password"
            autoComplete="off"
            aria-label="Flow decryption password"
          />
        </label>
      </form>
    </Dialog>
  );
}
