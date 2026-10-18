/**
 * The generator trampoline appended to every compiled workflow.
 *
 * A leaf module on purpose: it is a plain string with no dependencies, so the
 * test suite can import the REAL trampoline under Node without dragging in the
 * whole runtime graph. Keeping a second copy in the tests is exactly how the
 * `__gen.throw` resume bug below stayed invisible — the test agreed with itself.
 */

/**
 * The driver that turns the generator produced by `rewriteAwaitToYield` into a
 * sequence of `host.call`s. Appended to every compiled workflow.
 *
 * Exported so tests can drive the REAL trampoline instead of a copy — an
 * earlier copy in the test suite is exactly how the `__gen.throw` bug below
 * survived.
 *
 * # Resuming after a host error
 *
 * A failed module call comes back as `{ __error__ }`, and the generator is
 * resumed by THROWING at its `await` site. Two things can happen next, and
 * only one of them used to be handled:
 *
 *   * user code does not catch — the error propagates out of the generator,
 *     `__gen.throw` itself throws, and the workflow fails. This always worked.
 *   * user code DOES catch (`try { await X() } catch { … }`) — the generator
 *     swallows it and keeps running, so `__gen.throw` RETURNS the next
 *     `{value, done}` just like `__gen.next` would. That return value used to
 *     be discarded, which left the generator live but unreachable: no further
 *     host call, no `__system.finish`, and a flow that hung until the browser's
 *     10-minute timeout. Any flow wrapping a fallible await in try/catch hit it.
 *
 * `__handle` is the shared resume path so both `__step` (normal) and
 * `__stepThrow` (error) drive the generator identically.
 *
 * An `await` of anything but a module call is the value awaited, as `await`
 * gives it: a promise's once it settles, any other value itself. Both resumed
 * with nothing before, so `await Promise.resolve(42)` in a logic block was
 * `undefined`.
 */
export const WORKFLOW_TRAMPOLINE: string = `let __gen = __run_workflow();
function __handle(item) {
   if (item.done) {
      // (an answer past what one call carries goes in parts, the last one with finish: in one call, a flow whose
      // steps' values passed 4 MiB together failed as it ended, its work done)
      var __text = JSON.stringify(item.value);
      var __at = 0;
      if (typeof __text === 'string') {
         while (__text.length - __at > 1048576) {
            host.call("__system.finish_part", [__text.slice(__at, __at + 1048576)], function(){});
            __at += 1048576;
         }
         __text = __text.slice(__at);
      }
      host.call("__system.finish", [__text], function(){});
      return;
   }
   if (item.value && item.value._kind) {
      host.call(item.value._kind, item.value._args, function(res) {
         if (res && typeof res === 'object' && res.__error__) {
             __stepThrow(res.__error__);
         } else {
             __step(res);
         }
      });
   } else if (item.value && typeof item.value.then === 'function') {
      item.value.then(function(v) { __step(v); }, function(e) { __stepThrow(e && e.message ? e.message : "" + e); });
   } else {
      __step(item.value);
   }
}
function __step(val) {
   try {
       __handle(val === undefined ? __gen.next() : __gen.next(val));
   } catch(e) {
       host.call("__system.finish_error", ["" + e], function(){});
   }
}
function __stepThrow(message) {
   try {
       __handle(__gen.throw(new Error(message)));
   } catch(e) {
       host.call("__system.finish_error", ["" + e], function(){});
   }
}
__step();
`;
