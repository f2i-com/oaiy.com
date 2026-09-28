# 3D models

NROB makes 3D models from a picture with Pixal3D, natively in Rust
(`nrob-diffusion`, `kind: "model3d"`). Give it a picture of one object (a
product, a character, a prop, a building) on a plain or transparent background,
and you get a textured GLB. The GLB is glTF 2.0, Y up, with the object's front
(the side the picture shows) facing +Z. It fits the cube from -0.5 to 0.5 and
has PBR materials (base colour, metallic, roughness).

The studio serves it as asynchronous jobs on `/v1/3d/models`, like music and
sound effects.

## The models

Pixal3D needs three downloads. Pick the Pixal3D folder on the **Models** page,
and the studio finds the other two beside it (or add each one).

| Part | Download | Licence |
|---|---|---|
| Pixal3D (the flow transformers and decoders) | [TencentARC/Pixal3D](https://huggingface.co/TencentARC/Pixal3D) | MIT |
| DINOv3 ViT-L/16, the image encoder | `facebook/dinov3-vitl16-pretrain-lvd1689m` (transformers layout: `config.json`, `model.safetensors`) | Meta's DINOv3 License |
| NAF, the feature upsampler | `naf_release.pth` from [valeoai/NAF](https://github.com/valeoai/NAF) | Apache-2.0 |

The studio recognises Pixal3D from its `pipeline.json`
(`Trellis2ImageTo3DPipeline`), DINOv3 from its `config.json`
(`model_type: dinov3_vit`, ViT-L/16 only), and NAF by its file name. It adds
`pixal3d` and enables it once all three parts are set.

`naf_release.pth` is the one pickled file the studio accepts. The worker reads it
with nrob's own tensor reader, which takes only the tensors and refuses anything
else in the pickle, so no Python and no code from the file ever runs.

## How it works

Pixal3D is TRELLIS.2's cascade with pixel-aligned conditioning: each voxel sees
the image features at the point where it projects into the picture.

1. **The picture.** It is cut out (from its alpha, or by flooding a plain
   background from the edges), squared around the object with a margin, and put
   on black. DINOv3 describes it at 512 and 1024 pixels, and NAF upsamples its
   features to each voxel's pixel.
2. **The structure.** A flow transformer makes a 16³ latent. A decoder turns it
   into 64³ occupancy, pooled to 32³ voxels.
3. **The shape.** A second transformer makes the shape latent on those voxels,
   at 512. Its decoder proposes the finer voxels, and a third transformer makes
   the shape again there, at 1024 (or 1536).
4. **The texture.** A fourth transformer makes the texture latent on the same
   voxels.
5. **The surface.** The shape decoder gives each voxel a vertex and marks the
   edges the surface crosses (a flexible dual grid, up to about 9 million
   triangles). The texture decoder gives each voxel its base colour, metallic,
   roughness and alpha.
6. **The mesh.** The dual grid has open and non-manifold edges, so it is rebuilt
   as a closed shell one voxel out from it. This is dual contouring of its
   distance in a narrow band, as TRELLIS.2's GLB export does. Parts that are
   closed at that scale are filled, so they have no inner sheet. The shell is
   simplified with meshoptimizer to the face budget (200,000 by default).
7. **The textures.** The mesh is cut into charts. Each chart grows across edges
   while its faces stay within 60° of its mean normal, and is laid flat on its
   plane. A face that would overlap another in the projection starts its own
   chart. The charts are packed into a 2048² atlas. Each texel samples the
   texture voxels at the nearest point of the decoded surface. The gaps are
   filled so that filtering and mipmaps never show the background.

   Only the faces that show get space in the atlas. The mesh is drawn from 96
   directions to find them. Hidden faces (mostly the shell's inner sheet) and
   tiny scraps get vertex colours instead, and point at a white patch of the
   atlas.

The steps, guidance and schedule come from Pixal3D's own `pipeline.json`. One
model is on the GPU at a time.

## Speed and memory

On an RTX 5090, a model takes 90 to 100 seconds at 1024:

- about 60 s for the four flow stages on the GPU;
- about 35 s for the mesh and textures on the CPU (remesh 25 s, simplify 10 s,
  unwrap and bake 1 s).

The official pipeline takes 105 s on the same GPU. Peak VRAM is about 15 GB.
A GLB with 200,000 faces and 2048² textures is 10 to 14 MB.

## API

### `POST /v1/3d/models`

This creates a job and returns it at once. The fields:

| Field | Meaning |
|---|---|
| `image` | The picture: a `data:` URL (PNG, JPEG or WebP). Also accepted as `input_reference`. |
| `resolution` | 1024 (default) or 1536: the finest grid the shape reaches. |
| `faces` | The simplified mesh's triangle budget, 1,000 to 2,000,000 (default 200,000). |
| `texture_size` | The baked textures' size: 512, 1024, 2048 (default) or 4096; 0 for vertex colours only. |
| `fov_degrees` | The camera the picture was taken with, 5 to 120 degrees (default 30). |
| `seed` | Repeats a model exactly. |
| `steps` | The flow stages' steps (default: `pipeline.json`'s). |
| `model` | A 3D model; left out, the default. |

| Request | Does |
|---|---|
| `GET /v1/3d/models/{id}` | Polls a job (`queued`, `in_progress`, `completed`, `failed`; `progress` 0-100, and `stage`). A finished job also has `faces`, `vertices`, `bytes` and `matte` (`alpha` or `background`: how the object was cut out). |
| `GET /v1/3d/models/{id}/content` | Downloads the GLB (`model/gltf-binary`). |
| `GET /v1/3d/models/{id}/input` | The picture as it was cut out and squared (PNG). |
| `GET /v1/3d/models` | Lists 3D model jobs. |
| `POST /v1/3d/models/{id}/cancel`, `DELETE /v1/3d/models/{id}` | Stops a job, or forgets it. |

```sh
curl http://127.0.0.1:8080/v1/3d/models -H "Content-Type: application/json" \
  -d "{\"image\": \"data:image/png;base64,$(base64 -w0 lamp.png)\", \"seed\": 3}"
```

The Playground has a **3D** tab, and the gallery shows finished models. Companion
apps find the endpoint and the models in `GET /v1/discovery` (`models.model3d`,
with each model's resolutions and default faces).

## Pictures that work

The model is only as good as the picture. Use one object, whole and centred,
on a plain white, grey or transparent background, in soft even light. A
three-quarter view (its front and one side visible) works best. Text, other
objects or a busy background end up in the model.

Pixal3D's metallic and roughness come out nearly the same over the whole
object, and change from seed to seed. Three seeds of one picture gave metallic
1.0, 0.74 and 0.0. A fully metallic model looks dark in a scene lit only by
lights. An environment map (three.js's `RoomEnvironment`, SoftN Scene3D's
`environment="studio"`) shows it as intended.

## Verification

Each stage was compared with the official pipeline from the same inputs and
starting noise:

| Stage | Result |
|---|---|
| DINOv3 features | within 0.24% RMS |
| NAF-upsampled features | within 0.23% RMS |
| One transformer step (dense) | within 0.97% RMS (correlation 0.99995) |
| One transformer step (sparse) | within 0.68% RMS |
| Structure occupancy | 99.56% of voxels agree |
| Whole-stage latents | correlation 0.994 to 0.997 |
| Decoders, on the reference's latents | 99.94% of voxels in common; dual vertices correlate 0.99977; texture correlates 0.999996 |

The textured GLB made from the reference's noise has the same roughness and
metallic as the reference's, and the same shape.
