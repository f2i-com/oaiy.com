/**
 * Looking for a secret where it must not be: an origin's storage, what a page was sent, and a page's memory (design 8, E3).
 *
 * WHAT THESE SCANS CAN AND CANNOT SHOW. A scan looks for the secret, and for a list of shapes a page might hold it in (see `needlesFor`:
 * as it is, reversed, in base64 and hex, percent-encoded, as a list of character codes, each also for a prefix, a suffix and a middle
 * piece), in text, in the bytes of typed arrays and ArrayBuffers (of what a page was sent, and of what is reachable from its `window`),
 * in IndexedDB, Cache Storage and the origin private file system, and in the JavaScript heap's strings. A secret in a form that is not on
 * that list is not found, and a scan that finds nothing is NOT a proof that there is nothing. The forms that are known not to be found:
 *   - SHIFTED: every character moved (a Caesar or ROT13 shift, each code plus one, XORed with a byte, any cipher);
 *   - SPLIT: in runs shorter than 20 characters, or over several messages, several storage entries or several objects that are never joined
 *     in one place a scan looks (a page that receives the key 14 characters at a time in 14 replies holds no 20 of them together);
 *   - in a BLOB (or a File): the bytes of a Blob are not in the heap snapshot, not reachable as text from `window`, and not read by the
 *     buffer scan, which reads ArrayBuffers and typed arrays; a Blob in a page's memory, in a `blob:` URL it made, or in a message that
 *     carried one is not searched (the storage dump reads a file system file and a cache body by its bytes, and an IndexedDB Blob as `{}`);
 *   - held only by a CLOSURE (a buffer no property of `window` reaches), or in a worker's heap, or in another process.
 * The proof is the
 * rule the scans check, not the scans: no operation of the port returns anything that depends on a key (design 3.2, and
 * web/providers/src/fixed.ts for why even an error's words are not passed on). The scans are what would show that rule broken in the
 * ordinary ways, and they are shown to find each of the shapes they claim to (leakscan.test.mjs, harness.test.mjs).
 *
 * A search that finds nothing proves nothing until the same search is shown to find something, so every scan has a POSITIVE CONTROL in
 * the tests that use it: the same call on a page that was handed the secret on purpose.
 *
 * Not covered: the heaps of workers a page starts (a worker is given only what its page holds, so a key there passed through a page
 * that received it, and the scan of what the page was sent is where that shows), and other people's processes.
 */

const unique = (list) => [...new Set(list)];
const reversed = (text) => [...text].reverse().join('');

/** The base64 of `bytes` at each of the three alignments it can have inside a longer run, without the edges that depend on neighbours. */
function base64Runs(bytes, alphabet) {
  const runs = [];
  for (let offset = 0; offset < 3; offset++) {
    const encoded = Buffer.concat([Buffer.alloc(offset), bytes]).toString(alphabet).replace(/=+$/, '');
    runs.push(encoded.slice([0, 2, 3][offset], encoded.length - 2));
  }
  return runs;
}

/** The width of a piece of a secret that is looked for, and how far apart they start: any run of `PIECE + STRIDE - 1` characters of it holds one. */
const PIECE = 14;
const STRIDE = 7;

/**
 * Every shape of `secret` (and of pieces of it) the scans search for. A piece is 14 characters and they start every 7, so any run of 20
 * or more characters of the key is found, not only the key whole; a run of fewer is not looked for (a channel that leaks a few characters
 * at a time is why the port passes nothing that depends on a key at all, see fixed.ts). A piece is long enough that a hit is not chance.
 * @param {string} secret
 * @returns {string[]}
 */
export function needlesFor(secret) {
  const windows = [];
  for (let at = 0; at + PIECE <= secret.length; at += STRIDE) windows.push(secret.slice(at, at + PIECE));
  if (secret.length >= PIECE) windows.push(secret.slice(-PIECE));
  const pieces = unique([secret, ...windows]).filter((p) => p.length >= 12);
  const needles = [];
  for (const piece of pieces) {
    for (const text of [piece, reversed(piece)]) {
      const bytes = Buffer.from(text);
      needles.push(text, bytes.toString('hex'), bytes.toString('hex').toUpperCase(), encodeURIComponent(text), bytes.join(','), bytes.join(', '));
      needles.push(bytes.toString('base64'), bytes.toString('base64url'), ...base64Runs(bytes, 'base64'), ...base64Runs(bytes, 'base64url'));
    }
  }
  return unique(needles).filter((n) => n.length >= 12);
}

/** The secret in its usual forms only, for a control that must not be confused by a piece of it: as it is, reversed, base64, hex, percent. */
export function forms(secret) {
  return unique([secret, reversed(secret), Buffer.from(secret).toString('base64'), Buffer.from(secret).toString('hex'), encodeURIComponent(secret)]);
}

const LATIN1 = (view) => Buffer.from(view.buffer, view.byteOffset, view.byteLength).toString('latin1');
const isBuffer = (v) => v instanceof ArrayBuffer || ArrayBuffer.isView(v);

/**
 * A value a page was sent (or holds), as text a scan can search: strings as they are; the bytes of every ArrayBuffer and typed array as
 * text and as a list of character codes; an array of characters joined; a Map and a Set as their entries. `JSON.stringify` alone turns a
 * Uint8Array into `{"0":115,"1":107,...}`, in which nothing is found: the review's leak was exactly that.
 * The same algorithm is in fixtures/shell/shell.js (the page cannot import this file).
 */
export function encodeForScan(value, seen = new WeakSet()) {
  if (value === null || typeof value !== 'object') return typeof value === 'bigint' ? String(value) : value;
  if (isBuffer(value)) {
    const bytes = value instanceof ArrayBuffer ? new Uint8Array(value) : new Uint8Array(value.buffer, value.byteOffset, value.byteLength);
    const encoded = { $bytes: LATIN1(bytes), $codes: Array.from(bytes).join(',') };
    // A 16- or 32-bit view holds characters as units, not as bytes: they are text and codes too.
    if (ArrayBuffer.isView(value) && value.BYTES_PER_ELEMENT > 1 && !(value instanceof DataView)) {
      const units = Array.from(value).map(Number);
      encoded.$units = units.join(',');
      encoded.$chars = units.map((u) => (Number.isInteger(u) && u >= 0 && u <= 0x10ffff ? String.fromCodePoint(u) : '')).join('');
    }
    return encoded;
  }
  if (seen.has(value)) return '[cycle]';
  seen.add(value);
  if (value instanceof Map) return { $map: [...value].map(([k, v]) => [encodeForScan(k, seen), encodeForScan(v, seen)]) };
  if (value instanceof Set) return { $set: [...value].map((v) => encodeForScan(v, seen)) };
  if (Array.isArray(value)) {
    const out = value.map((v) => encodeForScan(v, seen));
    if (value.length >= 8 && value.every((v) => typeof v === 'string' && v.length <= 2)) return { $items: out, $joined: value.join('') };
    return out;
  }
  const out = {};
  for (const [k, v] of Object.entries(value)) out[k] = encodeForScan(v, seen);
  return out;
}

/** `encodeForScan` as one string. */
export const dumpText = (value) => JSON.stringify(encodeForScan(value));

/** The needles (of `needlesFor`) found in `text`. */
export const findInText = (text, needles) => needles.filter((needle) => text.includes(needle));

/** The places in `dump` (what `oaiyTest.dumpStorage()` returns) whose text holds the needle, or any long piece of it. */
export function findInStorage(dump, needles) {
  const found = [];
  // One line for a place, with how many shapes were found in it (a key in one place is many needles: its whole and each of its pieces).
  const check = (where, text) => {
    if (typeof text !== 'string') return;
    const hits = needles.filter((needle) => text.includes(needle));
    if (hits.length > 0) found.push(`${where} holds the secret (${hits.length} of ${needles.length} shapes, first: ${hits[0].slice(0, 4)}…${hits[0].length} chars)`);
  };
  check('document.cookie', dump.cookies);
  for (const [k, v] of Object.entries(dump.localStorage)) check(`localStorage[${k}]`, `${k}=${v}`);
  for (const [k, v] of Object.entries(dump.sessionStorage)) check(`sessionStorage[${k}]`, `${k}=${v}`);
  for (const cache of dump.caches) {
    check(`cache ${cache.name}`, cache.name);
    for (const item of cache.items) check(`cache ${cache.name} ${item.url}`, `${item.url}\n${item.body}\n${item.bytes ?? ''}\n${item.codes ?? ''}`);
  }
  for (const db of dump.indexedDB) for (const [store, values] of Object.entries(db.stores)) values.forEach((v, i) => check(`indexedDB ${db.name}/${store}[${i}]`, v));
  for (const file of dump.opfs ?? []) check(`opfs ${file.path}`, `${file.path}\n${file.text}\n${file.codes}`);
  for (const script of dump.serviceWorkers ?? []) check(`service worker ${script}`, script);
  return found;
}

/** Whether the secret's shapes are in a page's received messages, given as the text the shell kept (`oaiyTest.received()`). */
export const findInReceived = (received, needles) => findInStorage({ indexedDB: [], localStorage: {}, sessionStorage: {}, caches: [], cookies: received }, needles).map((f) => f.replace('document.cookie', 'received messages'));

/**
 * Whether the needles are in the JavaScript heap of the process that runs `target` (a page, or an out-of-process frame). It takes
 * a heap snapshot over the DevTools protocol and searches every string in it. A heap snapshot holds strings, not the bytes of an
 * ArrayBuffer: `buffersHold` is the scan for those.
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

const MOST_OBJECTS = 300_000;
const LARGEST_BUFFER = 8 * 1024 * 1024;

/**
 * Whether the needles are in the bytes of a typed array or an ArrayBuffer a page holds. A heap snapshot does not contain those bytes
 * (only their sizes, and V8 lists the arrays themselves as native nodes that cannot be fetched back, and `Runtime.queryObjects` does not
 * report typed arrays), so this walks what is REACHABLE FROM `window` by own data properties, arrays, Maps and Sets (without calling a
 * getter), reads every buffer it meets, and searches the bytes HERE as text and, for 16- and 32-bit views, as characters. The needles
 * are not sent into the page: scanning must not put the key in the memory being scanned.
 *
 * What it does not see: a buffer that only a closure holds. That is a limit, stated here: what a page holds it received over the port,
 * and `findInReceived` reads everything the port delivered, buffers included, so a key that reached a page at all is found there.
 *
 * @param {{ read?: number, skipped?: number, objects?: number }} [report] receives what was walked
 */
export async function buffersHold(context, target, needles, report = {}) {
  const client = await context.newCDPSession(target);
  try {
    await client.send('Runtime.enable');
    const reply = await client.send('Runtime.evaluate', {
      returnByValue: true,
      expression: `((limit, largest) => {
        const latin1 = (units) => {
          let out = '';
          for (let i = 0; i < units.length; i += 8192) out += String.fromCharCode.apply(null, units.subarray(i, i + 8192));
          return out;
        };
        const seen = new WeakSet();
        const found = [];
        let skipped = 0;
        let objects = 0;
        const queue = [window];
        while (queue.length > 0 && objects < limit) {
          const value = queue.pop();
          if (value === null || (typeof value !== 'object' && typeof value !== 'function')) continue;
          if (seen.has(value)) continue;
          seen.add(value);
          objects++;
          if (value instanceof ArrayBuffer || ArrayBuffer.isView(value)) {
            const bytes = value instanceof ArrayBuffer ? new Uint8Array(value) : new Uint8Array(value.buffer, value.byteOffset, value.byteLength);
            if (bytes.length > largest) {
              skipped++;
              continue;
            }
            const wide = ArrayBuffer.isView(value) && value.BYTES_PER_ELEMENT > 1 && !(value instanceof DataView) && !(value instanceof BigInt64Array) && !(value instanceof BigUint64Array);
            found.push({ text: latin1(bytes), units: wide ? latin1(value) : '' });
            continue;
          }
          if (value !== window && typeof Node !== 'undefined' && value instanceof Node) continue;
          let keys;
          try {
            keys = Reflect.ownKeys(value);
          } catch {
            continue;
          }
          for (const key of keys) {
            try {
              const descriptor = Object.getOwnPropertyDescriptor(value, key);
              if (descriptor && 'value' in descriptor) queue.push(descriptor.value);
            } catch {
              // a property the page cannot read
            }
          }
          try {
            if (value instanceof Map) for (const [k, v] of value) queue.push(k, v);
            else if (value instanceof Set) for (const v of value) queue.push(v);
          } catch {
            // not iterable after all
          }
        }
        return { found, skipped, objects };
      })(${MOST_OBJECTS}, ${LARGEST_BUFFER})`,
    });
    if (reply.exceptionDetails) throw new Error(`scanning buffers: ${reply.exceptionDetails.text} ${reply.exceptionDetails.exception?.description ?? ''}`);
    const { found, skipped, objects } = reply.result.value;
    report.read = found.length;
    report.skipped = skipped;
    report.objects = objects;
    const hits = new Set();
    for (const buffer of found) for (const needle of needles) if (buffer.text.includes(needle) || (buffer.units && buffer.units.includes(needle))) hits.add(needle);
    return needles.filter((needle) => hits.has(needle));
  } finally {
    await client.detach().catch(() => {});
  }
}
