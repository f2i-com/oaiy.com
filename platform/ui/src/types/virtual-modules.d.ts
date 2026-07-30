declare module 'virtual:sandbox-runtime' {
  /**
   * Self-contained IIFE bundle of the custom-node UI sandbox runner
   * (src/sandbox/sandbox-main.tsx), produced by sandboxRuntimePlugin in vite.config.ts.
   * Inlined into a sandboxed iframe's srcdoc by SandboxedCustomNodeUI.
   */
  const runtime: string;
  export default runtime;
}
