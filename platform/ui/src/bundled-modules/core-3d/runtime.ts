/**
 * "3D Model": a GLB from a picture of an object, through a contract service
 * (OAIY's engine by default: a job that takes a few minutes on the GPU). With
 * no picture, one is made from the description first, with the picture
 * service, asking for the whole object on a plain background (what the 3D
 * model works best from).
 */
import type { RuntimeContext, RuntimeModule, RuntimeMethod } from 'oaiy-core/src/module-types';
import type { CustomService } from '../core-service/examples';
import { imageDataUrl, runContract } from '../core-service/contractRuntime';

/** How a picture made for a 3D model is asked for. */
export function pictureFor(description: string): string {
  return `${description.trim()}. One whole object, centered and fully in view, on a plain white background, soft studio light, no text.`;
}

function createModel3DMethods(ctx: RuntimeContext): Record<string, RuntimeMethod> {
  async function make(
    contract: CustomService | string,
    pictureContract: CustomService | string,
    vars: Record<string, unknown>,
    nodeId: string,
    filename: string,
  ): Promise<{ model: string; path: string; picture: string | null }> {
    ctx.onNodeStatus?.(nodeId, 'running');
    try {
      let picture = vars.image != null && vars.image !== '' ? await imageDataUrl(ctx, vars.image) : null;
      if (!picture) {
        const description = typeof vars.prompt === 'string' ? vars.prompt.trim() : '';
        if (!description) throw new Error('3D Model: connect a picture, or describe the object');
        ctx.log('info', '[3D Model] making a picture of it first');
        const made = await runContract(
          ctx,
          pictureContract,
          { prompt: pictureFor(description), model: vars.pictureModel || undefined },
          { nodeId, label: '3D Model (picture)' },
        );
        picture = made.image ?? null;
        if (!picture) throw new Error('3D Model: no picture was made from the description');
        ctx.onImage?.(nodeId, picture);
      }
      const r = await runContract(
        ctx,
        contract,
        { image: picture, resolution: vars.resolution, faces: vars.faces, seed: vars.seed, model: vars.model || undefined },
        { nodeId, filename, label: '3D Model' },
      );
      if (!r.path) throw new Error('3D Model: the service returned no model');
      ctx.onNodeStatus?.(nodeId, 'completed');
      ctx.log('success', `[3D Model] ${r.path}`);
      return { model: r.url || r.path, path: r.path, picture };
    } catch (e) {
      ctx.onNodeStatus?.(nodeId, 'error');
      throw e;
    }
  }

  return { make: make as RuntimeMethod };
}

const Core3DRuntime: RuntimeModule = {
  name: 'Model3D',
  createMethods: createModel3DMethods,
  methods: {},
  async cleanup(): Promise<void> {},
};

export default Core3DRuntime;
