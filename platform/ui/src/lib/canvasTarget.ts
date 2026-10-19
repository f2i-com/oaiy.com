/**
 * Whether a key pressed with focus on `target` is the canvas's: on the flow's canvas, or on the editor itself (the
 * builder, which takes focus when an empty part of the canvas is clicked). ⌘A and ⌘V did nothing after such a click:
 * the canvas's own element was not under the focus, the builder around it was.
 */
export function isCanvasTarget(target: Pick<Element, 'closest'> | null, builder: unknown): boolean {
  if (!target) return false;
  return target === builder || !!target.closest('.react-flow');
}
