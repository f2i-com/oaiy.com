/** Synchronous MessagePorts for jsdom only; no native or browser permission. */
export class TestMessagePort extends EventTarget {
  peer!: TestMessagePort;
  closed = false;
  postMessage(data: unknown): void { if (!this.closed) this.peer.deliver(data); }
  deliver(data: unknown): void { if (!this.closed) this.dispatchEvent(new MessageEvent('message', { data })); }
  start(): void {}
  close(): void { this.closed = true; }
}
export function testPorts() {
  const port1 = new TestMessagePort(), port2 = new TestMessagePort();
  port1.peer = port2; port2.peer = port1;
  return { port1, port2 };
}
export function testMessageEvent(init: MessageEventInit): MessageEvent {
  // jsdom's ports IDL conversion otherwise replaces subclass test ports with
  // plain EventTarget wrappers, unlike transferred native browser MessagePorts.
  const event = new MessageEvent('message', { ...init, ports: [] });
  Object.defineProperty(event, 'ports', { value: init.ports ?? [] });
  return event;
}
export function installTestChannel(target: Window, onRequest?: (data: unknown) => void) {
  let latest!: ReturnType<typeof testPorts>;
  Object.defineProperty(target, 'MessageChannel', { configurable: true, value: class {
    port1: TestMessagePort; port2: TestMessagePort;
    constructor() {
      latest = testPorts(); this.port1 = latest.port1; this.port2 = latest.port2;
      if (onRequest) this.port2.addEventListener('message', event => onRequest((event as MessageEvent).data));
    }
  } });
  return () => latest;
}
