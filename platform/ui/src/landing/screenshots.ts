/**
 * The screenshots the landing page shows.
 *
 * They are the README's pictures of a demo setup (the business, the people and their numbers are
 * made up), made into WebP by scripts/make-site-images.py. The size is the file's own, so the page
 * reserves the room before the picture arrives; tests/site-assets.mjs checks each file is the size
 * written here, is under 200 KB, and is described.
 */
export interface Screenshot {
  /** From the site's root. */
  src: string;
  width: number;
  height: number;
  alt: string;
}

export const SCREENSHOTS = {
  agent: {
    src: '/images/agent.webp',
    width: 1440,
    height: 900,
    alt: "OAIY Desktop with the Agent open on a project called Weather report: its files, a Python file in the editor, the output in the terminal, and the agent's reply with a table of cities and temperatures.",
  },
  flows: {
    src: '/images/flows.webp',
    width: 1440,
    height: 900,
    alt: "The flow editor in OAIY Desktop: a flow that removes the background from a photo, upscales it and writes a caption, beside the palette of nodes, some of them OAIY's own models.",
  },
  call: {
    src: '/images/receptionist-call.webp',
    width: 1440,
    height: 900,
    alt: 'A phone call as the AI Receptionist handles it: it greets the caller, the caller asks to book a lawn mow, and the receptionist looks up free times.',
  },
  calendar: {
    src: '/images/calendar.webp',
    width: 1440,
    height: 900,
    alt: "The Calendar's week with two booking requests waiting for the owner to confirm or decline, and the week's appointments beneath them.",
  },
} as const satisfies Record<string, Screenshot>;

export const SCREENSHOT_LIST: Screenshot[] = Object.values(SCREENSHOTS);
