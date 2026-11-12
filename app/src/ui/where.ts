/**
 * Where the Agent is and keeps things, in the person's words. In a browser tab that is "this browser"; in OAIY's own
 * window there is no browser to its owner, only a program on their computer, and "Keys are stored encrypted in this
 * browser" read as a mistake there. The storage is the same kind (the page's own, on the machine): the words differ.
 */
import { pageHost } from '@oaiy/shared/capabilities/host';

/** Whether this page is shown in OAIY Desktop's own window. */
export const IN_WINDOW = pageHost().kind === 'oaiy-window';

/** "this browser", or "this computer" in OAIY's window. */
export const HERE = IN_WINDOW ? 'this computer' : 'this browser';

/** "in this browser", or "on this computer" in OAIY's window. */
export const KEPT_HERE = IN_WINDOW ? 'on this computer' : 'in this browser';

/** "this tab", or "this window" in OAIY's window. */
export const THIS_PAGE = IN_WINDOW ? 'this window' : 'this tab';
