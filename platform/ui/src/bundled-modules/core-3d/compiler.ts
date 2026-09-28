/**
 * Compiles "3D Model" into a call to its runtime: the picture (the Image
 * input) or the description (the Description input, else the property) to
 * make one from first, the service that makes the model (OAIY's engine by
 * default, `engine:model3d`) and the one that makes the picture.
 */
import type { ModuleCompiler, ModuleCompilerContext } from 'oaiy-core/src/module-types';
import { contractModel, nodeContract, resolveService } from '../core-service/contract';

const Core3DCompiler: ModuleCompiler = {
  name: 'Model3D',

  getNodeTypes() {
    return ['model_3d'];
  },

  compileNode(nodeType: string, ctx: ModuleCompilerContext): string | null {
    if (nodeType !== 'model_3d') return null;
    const { node, inputs, outputVar, skipVarDeclaration } = ctx;
    const data = node.data as Record<string, unknown>;
    const letOrAssign = skipVarDeclaration ? '' : 'let ';

    const serviceId = String(data.service || 'engine:model3d');
    const preset = resolveService(serviceId);
    const contract = nodeContract(serviceId, preset, 'model3d') ?? JSON.stringify(serviceId);
    const pictureId = String(data.pictureService || 'engine:image');
    const picturePreset = resolveService(pictureId);
    const pictureContract = nodeContract(pictureId, picturePreset, 'image') ?? JSON.stringify(pictureId);

    const imageVar = inputs.get('image') || 'null';
    const promptIn = inputs.get('prompt');
    const promptLit = JSON.stringify(String(data.prompt || ''));
    const resolution = Number(data.resolution) === 1536 ? 1536 : 1024;
    const faces = Math.min(2_000_000, Math.max(1000, Number(data.faces) || 200_000));
    const seed = Number(data.seed);

    return `
  // --- Node: ${node.id} (model_3d) ---
  ${letOrAssign}${outputVar} = await Model3D.make(
    ${contract},
    ${pictureContract},
    {
      image: ${imageVar},
      prompt: ${promptIn ? `${promptIn} || ${promptLit}` : promptLit},
      resolution: ${resolution},
      faces: ${faces},
      seed: ${Number.isFinite(seed) && seed >= 0 ? seed : 'null'},
      model: ${JSON.stringify(contractModel(preset, ''))},
      pictureModel: ${JSON.stringify(contractModel(picturePreset, ''))},
    },
    "${node.id}",
    ${JSON.stringify(String(data.filename || 'model'))}
  );
  let ${outputVar}_model = ${outputVar}.model;
  let ${outputVar}_path = ${outputVar}.path;
  let ${outputVar}_picture = ${outputVar}.picture;
  workflow_context["${node.id}"] = ${outputVar};`;
  },
};

export default Core3DCompiler;
