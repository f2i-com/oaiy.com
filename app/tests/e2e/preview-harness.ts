// Exposes the live preview to the end-to-end runner (tests/e2e/webpage.mjs):
// a project in memory, the preview pane, and the agent's tools against both.
import '../../src/styles.css';
import { NetGate } from '../../src/gate/netgate';
import { Preview } from '../../src/preview/preview';
import { Vfs } from '../../src/vfs/vfs';
import { runTool } from '../../src/agent/tools';
import { SOFTN_STARTER } from '../../src/softn/softn';

const vfs = new Vfs();
const preview = new Preview(vfs, () => 'harness');
document.getElementById('host')!.append(preview.element);
preview.setVisible(true);

/** One of the agent's tools, against the harness's project and preview. */
async function tool(name: string, input: Record<string, unknown>) {
  const result = await runTool({ id: `t${Date.now()}`, name, input }, { vfs, gate: new NetGate(), reads: new Map(), shell: { cwd: '/', env: {} }, preview });
  return { content: result.content, isError: result.isError, images: result.images?.length ?? 0, files: result.files };
}

Object.assign(window, {
  harness: {
    write: (path: string, text: string) => vfs.writeFile(`/${path}`, text, { parents: true }),
    writeBytes: (path: string, bytes: number[]) => vfs.writeFile(`/${path}`, new Uint8Array(bytes), { parents: true }),
    read: (path: string) => vfs.readBytes(`/${path}`),
    starter: (root: string) => {
      for (const [path, text] of SOFTN_STARTER) vfs.writeFile(`/${root}/${path}`, text, { parents: true });
    },
    preview,
    tool,
    frame: () => document.querySelector<HTMLIFrameElement>('.preview iframe'),
  },
});
document.title = 'ready';
