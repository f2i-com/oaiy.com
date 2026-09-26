# Proposal: an application bridge for zipp-wasm

bot.computer's sandbox gives guest programs synchronous file and network access (Node's `fs.readFileSync`, the shell's `cat`, `curl`) by blocking the Worker on a `SharedArrayBuffer` while the page answers. zipp-wasm's synchronous host channel only accepts a fixed set of operations (`db.*`, `ls.*`, `nav.clipboard*`, `accel.*`). So today bot.computer tunnels its calls through `ls.getItem` with a reserved key prefix. That works, and ordinary `localStorage` keys stay inert, but it hides an application protocol inside a storage API.

## The change

Add one more bridge to `crates/zipp-wasm/src/lib.rs`, following the existing ones:

- **`Engine.setAppBridge(adapter)`:** `adapter.call(kind: string, args: string[]) => string`, a synchronous, trusted host adapter like the others.
- **Capability names `app.<kind>`:** each one must match `^app\.[a-z][a-z0-9_.]{0,58}$`. Grant them with `setSyncHostCapabilities(["app.fs.read", …])`.
  - Unlike the fixed names, their arity is decided by the host adapter, but the existing envelope limits still apply: `MAX_SYNC_BRIDGE_ARGS`, `MAX_SYNC_BRIDGE_BYTES`, and the kind length.
  - An ungranted `app.*` kind is refused before the adapter is looked up, like every other operation.
- **Guest side:** `__zippHostCall("app.fs.read", path)`, with a small `app.call(kind, ...args)` wrapper in `preamble.js`.
- **Errors:** as now, the guest sees an opaque failure when the adapter throws. An adapter that wants to pass an error back returns it as data (`{"err": …}`), which bot.computer already does.

With that in place, bot.computer drops the `ls.getItem` tunnel. The same guest scripts (`prelude.js`, `shell.js`) would then run unchanged on zipp-wasm and on coder-cli's native runner, where `__zippHostCall` is already an open application channel.
