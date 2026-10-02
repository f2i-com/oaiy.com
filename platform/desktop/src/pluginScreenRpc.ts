/** A WindowProxy survives iframe navigation; a document's grant must not. */
export class PluginScreenDocument {
  private active = true;
  private loads = 0;
  private port: MessagePort | null = null;
  private readonly controller = new AbortController();
  readonly signal = this.controller.signal;

  constructor(readonly nonce: string, private readonly source: Window | null, private readonly onRevoke: () => void) {}

  accepts(event: MessageEvent): boolean {
    if (!this.active || !this.source || event.source !== this.source || !event.data || typeof event.data !== 'object') return false;
    const descriptor = Object.getOwnPropertyDescriptor(event.data, 'documentNonce');
    return !!descriptor && 'value' in descriptor && descriptor.value === this.nonce;
  }

  /** Transfer binds the port to the original document, rather than its proxy. */
  connect(event: MessageEvent, receive: (event: MessageEvent) => void): boolean {
    if (this.port || !this.accepts(event) || event.ports.length !== 1) return false;
    const port = event.ports[0];
    this.port = port;
    port.addEventListener('message', message => {
      if (!this.active || !message.data || typeof message.data !== 'object') return;
      const descriptor = Object.getOwnPropertyDescriptor(message.data, 'documentNonce');
      if (descriptor && 'value' in descriptor && descriptor.value === this.nonce) receive(message);
    });
    port.start();
    return true;
  }

  /** The expected srcdoc may load once. Any subsequent document is untrusted. */
  loaded(): void { if (++this.loads > 1) this.revoke(); }

  post(message: Record<string, unknown>): void {
    if (this.active) this.port?.postMessage({ ...message, __pluginHost: 1, documentNonce: this.nonce });
  }

  revoke(): void {
    if (!this.active) return;
    this.active = false;
    this.controller.abort();
    this.port?.close();
    this.onRevoke();
  }
}
