/**
 * A link in a previewed SoftN app to another site, a mail address or a phone number, shown over the
 * app (preview.ts, from SoftN's `{ type: 'link', href }`). The app's frame can open nothing: it is
 * sandboxed. So the preview says where the link goes, the host as the browser reads it and the
 * whole address, and opens it only when the person presses Open: in the desktop, OAIY's window
 * hands it to the system's browser (embed.rs, `on_new_window`); in a browser, a tab of its own. The
 * app is anyone's code, and nothing it does on its own sends anyone elsewhere. Escape, the close
 * button or Stay here put it away; a newer link takes its place; a new frame takes it away.
 *
 * Open is a button, not a link: a link is followed by a middle click, a drag to the tab bar or its
 * context menu too, none of which wait. It waits OPEN_SETTLE_MS from when the address is shown
 * (OpenGate), so a run of presses meant for the app never reaches it. The keyboard comes to the bar
 * itself, never to Open, and only from the app's frame (where the link was clicked).
 */
import { h } from '../ui/dom';
import { icon } from '../ui/icons';
import { OpenGate, shownName, type OutsideLink } from './outsideLink';

export class LinkBar {
  readonly element: HTMLDivElement;
  private readonly lead = h('span.preview-link-lead-text', '');
  private readonly name = h('strong.preview-link-name', '');
  private readonly address = h('p.preview-link-address', '');
  private readonly kindIcon = h('span.preview-link-icon', { 'aria-hidden': 'true' });
  private readonly openButton: HTMLButtonElement;
  private readonly gate = new OpenGate();
  private link: OutsideLink | null = null;

  constructor() {
    this.openButton = h(
      'button.primary',
      {
        type: 'button',
        onpointerdown: () => this.gate.press(performance.now()),
        onclick: (event: MouseEvent) => this.open(event),
      },
      'Open link',
    ) as HTMLButtonElement;
    this.element = h(
      'div.preview-link-bar',
      { role: 'dialog', 'aria-label': 'A link out of the app', tabindex: '-1' },
      this.kindIcon,
      h(
        'div.preview-link-text',
        h('p.preview-link-lead', this.lead, ' ', this.name),
        // The whole address, whatever it is to: a mail's copies and words are in it. It scrolls, so Open and Stay here stay in view.
        this.address,
        h('div.preview-link-actions', this.openButton, h('button', { type: 'button', onclick: () => this.hide() }, 'Stay here')),
      ),
      h('button.preview-link-close', { type: 'button', 'aria-label': 'Close', title: 'Close', onclick: () => this.hide() }, icon('x')),
    ) as HTMLDivElement;
    this.element.hidden = true;
    // Escape puts the bar away, and only the bar (not the preview's full screen as well).
    window.addEventListener('keydown', (event) => {
      if (event.key !== 'Escape' || this.element.hidden) return;
      event.stopImmediatePropagation();
      this.hide();
    });
  }

  /** The link shown now, if any. */
  get shown(): OutsideLink | null {
    return this.element.hidden ? null : this.link;
  }

  /** Shows `link`, in place of any shown before; Open waits from now. */
  show(link: OutsideLink): void {
    this.link = link;
    this.lead.textContent = link.kind === 'mail' ? 'This app wants to write to' : link.kind === 'phone' ? 'This app wants to call' : 'This app links to';
    this.name.textContent = shownName(link.name);
    this.name.title = link.name;
    this.address.textContent = link.href;
    this.kindIcon.replaceChildren(icon(link.kind === 'mail' ? 'message' : link.kind === 'phone' ? 'phone' : 'globe'));
    this.openButton.textContent = link.kind === 'mail' ? 'Write email' : link.kind === 'phone' ? 'Call' : 'Open link';
    this.gate.arm(performance.now());
    this.element.hidden = false;
    // (the keyboard to the bar where it was in the app's frame, or nowhere: a page typing elsewhere is not taken from)
    const at = document.activeElement;
    if (at === null || at === document.body || at.tagName === 'IFRAME') this.element.focus({ preventScroll: true });
  }

  hide(): void {
    this.element.hidden = true;
    this.link = null;
  }

  private open(event: MouseEvent): void {
    const link = this.link;
    if (!link) return;
    // A pointer's click is timed from when its press began; one from the keyboard or a screen reader is its own press.
    if (!this.gate.click(performance.now(), event.detail === 0)) return;
    // A window of its own, which cannot reach back into this one: the system's browser, in the desktop.
    window.open(link.href, '_blank', 'noopener,noreferrer');
    this.hide();
  }
}
