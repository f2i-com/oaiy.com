/**
 * Browser shim for `@tauri-apps/api/event`.
 *
 * Backs Tauri's event bus with an in-page EventTarget. Desktop events that
 * originated in Rust (download progress, service status, …) simply never fire
 * in the web build today — listeners attach harmlessly and the UI shows its
 * idle state. Front-end `emit`s round-trip locally, which is enough for the
 * renderer's own intra-app signalling.
 */

export type UnlistenFn = () => void;

export interface Event<T> {
  event: string;
  id: number;
  payload: T;
}

export type EventCallback<T> = (event: Event<T>) => void;

const bus = new EventTarget();
let idSeq = 1;

export async function listen<T>(
  event: string,
  handler: EventCallback<T>,
): Promise<UnlistenFn> {
  const id = idSeq++;
  const cb = (e: Event_) => handler({ event, id, payload: (e as CustomEvent).detail as T });
  bus.addEventListener(event, cb as EventListener);
  return () => bus.removeEventListener(event, cb as EventListener);
}

export async function once<T>(
  event: string,
  handler: EventCallback<T>,
): Promise<UnlistenFn> {
  const unlisten = await listen<T>(event, (e) => {
    unlisten();
    handler(e);
  });
  return unlisten;
}

export async function emit(event: string, payload?: unknown): Promise<void> {
  bus.dispatchEvent(new CustomEvent(event, { detail: payload }));
}

export async function emitTo(_target: unknown, event: string, payload?: unknown): Promise<void> {
  return emit(event, payload);
}

// Tauri ships a TauriEvent enum of built-in lifecycle events; provide the
// common members as harmless string constants so imports resolve.
export const TauriEvent = {
  WINDOW_RESIZED: 'tauri://resize',
  WINDOW_MOVED: 'tauri://move',
  WINDOW_CLOSE_REQUESTED: 'tauri://close-requested',
  WINDOW_FOCUS: 'tauri://focus',
  WINDOW_BLUR: 'tauri://blur',
  DRAG_DROP: 'tauri://drag-drop',
} as const;

// Local alias so the EventListener cast above reads cleanly.
type Event_ = globalThis.Event;
