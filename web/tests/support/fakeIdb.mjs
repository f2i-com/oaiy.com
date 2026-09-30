/**
 * A stand-in IndexedDB for Node, with the behaviours the providers origin depends on.
 *
 *   - Values are structured clones (a CryptoKey and typed arrays survive), and keys are strings.
 *   - A transaction is ACTIVE from its creation until control returns to the event loop, and again while one of its request
 *     callbacks runs (and the promise continuations after it). A request made outside that window throws
 *     TransactionInactiveError, as a browser's does: so code that awaits a slow non-IndexedDB promise (WebCrypto) between two
 *     requests of one transaction fails here, as it fails there.
 *   - Transactions on one database run one at a time, in the order they were made; a transaction completes only when its
 *     requests are done and none was added.
 *   - A request that fails aborts its transaction. Failures, a hanging store, a refusal to open, "clear site data" and a
 *     connection closed under the caller can all be arranged by a test.
 */

const domError = (name, message) => new DOMException(message, name);

export function createFakeIdb() {
  const databases = new Map();
  const knobs = { openError: null, failures: [], holds: [], held: [], hangOpen: false, hungOpens: [] };
  const counters = { puts: {}, gets: {}, transactions: 0 };

  function injected(list, op, store) {
    for (const f of list) {
      if (f.op !== op || (f.store && f.store !== store)) continue;
      if (f.skip > 0) {
        f.skip--;
        continue;
      }
      if (f.times <= 0) continue;
      f.times--;
      return f;
    }
    return null;
  }

  class Request {
    constructor(exec, op, store) {
      this.exec = exec;
      this.op = op;
      this.store = store;
      this.result = undefined;
      this.error = null;
      this.onsuccess = null;
      this.onerror = null;
      this.readyState = 'pending';
    }
  }

  class Transaction {
    constructor(conn, names, mode) {
      this.conn = conn;
      this.db = conn.db;
      this.names = names;
      this.mode = mode;
      this.pending = [];
      this.state = 'active';
      this.finished = false;
      this.error = null;
      this.oncomplete = null;
      this.onerror = null;
      this.onabort = null;
      counters.transactions++;
      this.db.queue.push(this);
      setImmediate(() => this.deactivateAndRun());
    }

    objectStore(name) {
      if (!this.names.includes(name)) throw domError('NotFoundError', `store ${name} is not in this transaction`);
      const writable = this.mode === 'readwrite';
      const add = (op, exec, write = false) => {
        if (this.finished || this.state !== 'active') throw domError('TransactionInactiveError', 'The transaction is not active.');
        if (write && !writable) throw domError('ReadOnlyError', 'The transaction is read-only.');
        const req = new Request(exec, op, name);
        this.pending.push(req);
        return req;
      };
      const data = () => this.db.stores.get(name);
      const seen = (v) => (v === undefined ? undefined : structuredClone(v));
      return {
        get: (key) => add('get', () => { counters.gets[name] = (counters.gets[name] ?? 0) + 1; return seen(data().get(key)); }),
        getAll: () => add('getAll', () => [...data().keys()].sort().map((k) => seen(data().get(k)))),
        getAllKeys: () => add('getAllKeys', () => [...data().keys()].sort()),
        count: () => add('count', () => data().size),
        put: (value, key) => add('put', () => { counters.puts[name] = (counters.puts[name] ?? 0) + 1; data().set(key, structuredClone(value)); return key; }, true),
        add: (value, key) => add('add', () => {
          if (data().has(key)) throw domError('ConstraintError', 'The key already exists.');
          data().set(key, structuredClone(value));
          return key;
        }, true),
        delete: (key) => add('delete', () => { data().delete(key); }, true),
        clear: () => add('clear', () => { data().clear(); }, true),
      };
    }

    abort() {
      if (this.finished) throw domError('InvalidStateError', 'The transaction has finished.');
      this.fail(domError('AbortError', 'The transaction was aborted.'));
    }

    deactivateAndRun() {
      if (this.finished) return;
      this.state = 'inactive';
      this.waiting = true;
      this.db.pump();
    }

    /** Runs when this is the transaction at the front of its database. */
    begin() {
      this.waiting = false;
      this.step();
    }

    step() {
      if (this.finished) return;
      const req = this.pending.shift();
      if (!req) return this.complete();
      const failure = injected(knobs.failures, req.op, req.store);
      const perform = () => {
        if (this.finished) return;
        let error = failure ? failure.error : null;
        if (!error) {
          try {
            req.result = req.exec();
          } catch (e) {
            error = e;
          }
        }
        this.state = 'active';
        if (error) {
          req.error = error;
          req.readyState = 'done';
          try { req.onerror?.({ target: req }); } catch { /* the page's handler threw */ }
          this.fail(error);
          return;
        }
        req.readyState = 'done';
        try {
          req.onsuccess?.({ target: req });
        } catch (e) {
          this.fail(e);
          return;
        }
        // Continuations of a promise on this request run before this: they may add requests.
        setImmediate(() => {
          this.state = 'inactive';
          this.step();
        });
      };
      if (injected(knobs.holds, req.op, req.store)) knobs.held.push(perform);
      else setImmediate(perform);
    }

    complete() {
      if (this.finished) return;
      this.finished = true;
      this.state = 'finished';
      this.db.current = null;
      setImmediate(() => {
        try { this.oncomplete?.({ target: this }); } catch { /* ignore */ }
      });
      this.db.pump();
    }

    fail(error) {
      if (this.finished) return;
      this.error = error;
      this.finished = true;
      this.state = 'finished';
      this.db.current = null;
      setImmediate(() => {
        try { this.onerror?.({ target: this }); } catch { /* ignore */ }
        try { this.onabort?.({ target: this }); } catch { /* ignore */ }
      });
      this.db.pump();
    }
  }

  class Connection {
    constructor(db, version) {
      this.db = db;
      this.version = version;
      this.closed = false;
      this.onversionchange = null;
      this.onclose = null;
      this.objectStoreNames = { contains: (n) => db.stores.has(n) };
    }

    createObjectStore(name) {
      this.db.stores.set(name, new Map());
      return {};
    }

    transaction(names, mode = 'readonly') {
      if (this.closed) throw domError('InvalidStateError', 'The database connection is closing.');
      const list = Array.isArray(names) ? names : [names];
      for (const n of list) if (!this.db.stores.has(n)) throw domError('NotFoundError', `no store ${n}`);
      return new Transaction(this, list, mode);
    }

    close() {
      this.closed = true;
      this.db.conns.delete(this);
    }
  }

  function makeDb(name) {
    const db = { name, version: 0, stores: new Map(), conns: new Set(), queue: [], current: null };
    db.pump = () => {
      if (db.current) return;
      const next = db.queue.find((t) => t.waiting && !t.finished);
      if (!next) return;
      db.queue.splice(db.queue.indexOf(next), 1);
      db.current = next;
      next.begin();
    };
    return db;
  }

  const factory = {
    open(name, version = 1) {
      const req = { result: undefined, error: null, onsuccess: null, onerror: null, onupgradeneeded: null, onblocked: null };
      const go = () => {
        if (knobs.openError) {
          req.error = knobs.openError;
          return req.onerror?.({ target: req });
        }
        let db = databases.get(name);
        if (!db) {
          db = makeDb(name);
          databases.set(name, db);
        }
        const conn = new Connection(db, Math.max(version, db.version));
        db.conns.add(conn);
        req.result = conn;
        if (version > db.version) {
          const oldVersion = db.version;
          db.version = version;
          try { req.onupgradeneeded?.({ target: req, oldVersion, newVersion: version }); } catch (e) { req.error = e; return req.onerror?.({ target: req }); }
        }
        req.onsuccess?.({ target: req });
      };
      setImmediate(() => (knobs.hangOpen ? knobs.hungOpens.push(go) : go()));
      return req;
    },
  };

  return {
    factory,
    knobs,
    counters,
    /** The map behind one store of one database, by the name it was made with. */
    raw: (store, dbName) => (dbName ? databases.get(dbName) : [...databases.values()][0])?.stores.get(store),
    databaseNames: () => [...databases.keys()],
    /** "Clear site data": every database gone, every connection closed and told so. */
    wipe() {
      for (const db of databases.values()) for (const conn of [...db.conns]) { conn.closed = true; conn.onclose?.(); }
      databases.clear();
    },
    /** Connections closed without a word. */
    closeQuietly() {
      for (const db of databases.values()) for (const conn of db.conns) conn.closed = true;
    },
    /** Make `op` on `store` fail after `skip` of them, `times` times. */
    fail(op, error, { store, skip = 0, times = 1 } = {}) {
      knobs.failures.push({ op, store, skip, times, error });
    },
    /** Make `op` on `store` never answer until `release()`. */
    hold(op, { store, skip = 0, times = 1 } = {}) {
      knobs.holds.push({ op, store, skip, times });
    },
    release() {
      knobs.hangOpen = false;
      knobs.holds.length = 0;
      for (const go of knobs.hungOpens.splice(0)) go();
      for (const go of knobs.held.splice(0)) go();
    },
  };
}
