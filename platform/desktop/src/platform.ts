/** The kind of computer this window runs on, as far as the dashboard needs to know: for its examples and its hints. */
export type HostOS = 'macos' | 'windows' | 'linux' | 'other';

/**
 * The system, as the webview says it (Windows' WebView2, macOS's WebKit and Linux's WebKitGTK each name their own):
 * `nav` is the browser's `navigator`, given in tests.
 */
export function hostOS(nav: { userAgent?: string; platform?: string } | undefined = typeof navigator === 'undefined' ? undefined : navigator): HostOS {
  const said = `${nav?.platform ?? ''} ${nav?.userAgent ?? ''}`;
  if (/Windows|Win32|Win64/i.test(said)) return 'windows';
  if (/Macintosh|MacIntel|Mac OS X/i.test(said)) return 'macos';
  if (/Linux|X11/i.test(said)) return 'linux';
  return 'other';
}

/**
 * Example folders for a path box, as this system writes them: a big drive's data and models folders, and another
 * folder of models a person already has (in a person's words: "ckpts" was a word for those who train models).
 */
export function pathExamples(os: HostOS = hostOS()): { data: string; models: string; extra: string } {
  if (os === 'windows') return { data: 'D:\\OAIY', models: 'E:\\models', extra: 'E:\\more-models' };
  if (os === 'macos') return { data: '/Volumes/Data/OAIY', models: '/Volumes/Data/models', extra: '/Volumes/Data/more-models' };
  return { data: '/mnt/data/OAIY', models: '/mnt/data/models', extra: '/mnt/data/more-models' };
}
