/**
 * VALUE REFERENCES — how a node reads data a provider's graph points it at.
 *
 * A graph names values it does not hold: "$inputs.callId" is the id the run was
 * started with, "$nodes.decide.hasCall" is what an earlier node answered, and
 * "{{ nodes.summarise.content }}" is the same idea inside free text. Nodes are
 * compiled with their `data` frozen as a literal, so without this the reference
 * ITSELF is what the node acts on: a call was rejected with the callId
 * `"$inputs.callId"`, which is not a call, and the plugin said so.
 *
 * Two spellings, one language:
 *   - a reference standing ALONE as the whole string yields the referenced
 *     value itself, object and all — an answers map is addressed this way;
 *   - inside larger text it stringifies, and the `$` is optional there because
 *     graphs are authored both ways.
 *
 * Shared rather than reimplemented per module: two copies of these rules drift,
 * and the failure when they do is a node quietly acting on different data than
 * its neighbour.
 */

/** Roots a reference may address, and where each reads from at run time. */
const ROOTS = ['nodes', 'inputs', 'input', 'event', 'upstream'] as const;

/**
 * A JS expression that resolves every reference inside `template` at run time.
 *
 * Emitted as a self-contained IIFE rather than a shared declaration so that two
 * nodes in one graph cannot collide on a name — the trap that made two logic
 * blocks in one flow fail to compile.
 *
 * @param template   the node's frozen data, embedded as a literal
 * @param ctxVar     expression for the per-node output map
 * @param inputsVar  expression for the run's inputs
 * @param upstreamVar expression for this node's incoming value
 */
export function resolveRefsExpr(
  template: unknown,
  ctxVar = 'workflow_context',
  inputsVar = '__inputs',
  upstreamVar = 'null',
): string {
  return `(function (t, ctx, ins, up) {
    var read = function (root, path) {
      var base = root === 'nodes' ? ctx
        : (root === 'inputs' || root === 'input') ? ins
        : root === 'upstream' ? up
        : (ins && typeof ins === 'object' && ins.event !== undefined) ? ins.event : ins;
      var cur = base;
      for (var i = 0; i < path.length; i++) {
        if (cur === null || cur === undefined) return undefined;
        cur = cur[path[i]];
      }
      return cur;
    };
    var parts = function (p) { return p ? p.slice(1).split('.') : []; };
    var SEL = /^\\$(${ROOTS.join('|')})((?:\\.[^.\\s]+)*)$/;
    var TPL = /\\{\\{\\s*\\$?(${ROOTS.join('|')})((?:\\.[^.\\s}]+)*)\\s*\\}\\}/g;
    var walk = function (v, depth) {
      if (depth > 32) return v;
      if (typeof v === 'string') {
        var m = SEL.exec(v);
        if (m) return read(m[1], parts(m[2]));
        return v.replace(TPL, function (_m, r, p) {
          var got = read(r, parts(p));
          if (got === null || got === undefined) return '';
          return typeof got === 'object' ? JSON.stringify(got) : String(got);
        });
      }
      if (Array.isArray(v)) return v.map(function (x) { return walk(x, depth + 1); });
      if (v && typeof v === 'object') {
        var o = {};
        for (var k in v) if (Object.prototype.hasOwnProperty.call(v, k)) o[k] = walk(v[k], depth + 1);
        return o;
      }
      return v;
    };
    return walk(t, 0);
  })(${JSON.stringify(template)}, ${ctxVar}, ${inputsVar}, ${upstreamVar})`;
}

/** Blank (absent / null / empty text) means "no declaration here". */
export function hasValue(value: unknown): boolean {
  return value !== undefined && value !== null && value !== '';
}
