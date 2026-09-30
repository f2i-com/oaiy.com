/**
 * Looking for a secret where it must not be: an origin's storage, and a page's memory (design 8, E3).
 *
 * A search that finds nothing proves nothing until the same search is shown to find something, so every scan here has a
 * POSITIVE CONTROL in the tests that use it: the same call on a page that was handed the secret on purpose.
 */

/** The places in `dump` (what `oaiyTest.dumpStorage()` returns) whose text holds the needle, or any long piece of it. */
export function findInStorage(dump, needles) {
  const found = [];
  const check = (where, text) => {
    if (typeof text !== 'string') return;
    for (const needle of needles) if (text.includes(needle)) found.push(`${where} holds ${needle.length > 8 ? `${needle.slice(0, 4)}…(${needle.length} chars)` : needle}`);
  };
  check('document.cookie', dump.cookies);
  for (const [k, v] of Object.entries(dump.localStorage)) check(`localStorage[${k}]`, `${k}=${v}`);
  for (const [k, v] of Object.entries(dump.sessionStorage)) check(`sessionStorage[${k}]`, `${k}=${v}`);
  for (const cache of dump.caches) for (const item of cache.items) check(`cache ${cache.name} ${item.url}`, `${item.url}\n${item.body}`);
  for (const db of dump.indexedDB) for (const [store, values] of Object.entries(db.stores)) values.forEach((v, i) => check(`indexedDB ${db.name}/${store}[${i}]`, v));
  return found;
}

/**
 * Whether the needles are in the JavaScript heap of the process that runs `target` (a page, or an out-of-process frame). It takes
 * a heap snapshot over the DevTools protocol and searches every string in it.
 */
export async function heapHolds(context, target, needles) {
  const client = await context.newCDPSession(target);
  try {
    const chunks = [];
    client.on('HeapProfiler.addHeapSnapshotChunk', ({ chunk }) => chunks.push(chunk));
    await client.send('HeapProfiler.enable');
    await client.send('HeapProfiler.collectGarbage');
    await client.send('HeapProfiler.takeHeapSnapshot', { reportProgress: false });
    const snapshot = chunks.join('');
    return needles.filter((needle) => snapshot.includes(needle));
  } finally {
    await client.detach().catch(() => {});
  }
}

/** A key's bytes as a page might hold them other than as text: base64, hex and percent-encoded forms of it. */
export function forms(secret) {
  return [secret, Buffer.from(secret).toString('base64'), Buffer.from(secret).toString('hex'), encodeURIComponent(secret)].filter((v, i, all) => all.indexOf(v) === i);
}
