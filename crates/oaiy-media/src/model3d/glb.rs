//! A mesh as a binary glTF 2.0 file (.glb): positions, normals, vertex colours
//! (and, when baked, UVs with base-colour and metallic-roughness textures) and one
//! PBR material. Loads in three.js's GLTFLoader, Blender, and model viewers.
use super::mesh::Mesh;
use oaiy_engine::json::Json;

/// Baked PBR textures, as PNG bytes, with each vertex's UV and normal (the normals
/// of the mesh before it was cut along its charts' seams, so the seams do not show).
pub struct Textures {
    pub uvs: Vec<[f32; 2]>,
    pub normals: Vec<[f32; 3]>,
    pub base_color_png: Vec<u8>,
    /// glTF's layout: roughness in G, metallic in B.
    pub metallic_roughness_png: Vec<u8>,
    pub transparent: bool,
}

fn pad4(buf: &mut Vec<u8>, byte: u8) {
    while buf.len() % 4 != 0 {
        buf.push(byte);
    }
}

/// sRGB (as the voxels hold colour) to linear (as glTF's vertex colours are).
fn linear(v: f32) -> f32 {
    if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

/// The mesh as a GLB. Its material is single-sided: the remeshed surface is closed
/// and faces out (and where it is open, it has both sides), as TRELLIS.2's export.
pub fn write(mesh: &Mesh, textures: Option<&Textures>) -> Vec<u8> {
    let normals = match textures {
        Some(t) => t.normals.clone(),
        None => mesh.normals(),
    };
    let n = mesh.positions.len();
    let mut bin: Vec<u8> = Vec::new();
    let mut views = Vec::new();
    let mut accessors = Vec::new();
    let mut add_view = |bin: &mut Vec<u8>, bytes: &[u8], target: Option<i64>| -> usize {
        pad4(bin, 0);
        let offset = bin.len();
        bin.extend_from_slice(bytes);
        let mut v = vec![("buffer", Json::Int(0)), ("byteOffset", Json::Int(offset as i64)), ("byteLength", Json::Int(bytes.len() as i64))];
        if let Some(t) = target {
            v.push(("target", Json::Int(t)));
        }
        views.push(Json::obj(v));
        views.len() - 1
    };
    let f32s = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
    // Positions, with their bounds (glTF requires them).
    let mut lo = [f32::INFINITY; 3];
    let mut hi = [f32::NEG_INFINITY; 3];
    for p in &mesh.positions {
        for k in 0..3 {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
        }
    }
    let flat: Vec<f32> = mesh.positions.iter().flatten().copied().collect();
    let view = add_view(&mut bin, &f32s(&flat), Some(34962));
    accessors.push(Json::obj([
        ("bufferView", Json::Int(view as i64)),
        ("componentType", Json::Int(5126)),
        ("count", Json::Int(n as i64)),
        ("type", Json::str("VEC3")),
        ("min", Json::Arr(lo.iter().map(|v| Json::Num(*v as f64)).collect())),
        ("max", Json::Arr(hi.iter().map(|v| Json::Num(*v as f64)).collect())),
    ]));
    let position = accessors.len() - 1;
    let flat: Vec<f32> = normals.iter().flatten().copied().collect();
    let view = add_view(&mut bin, &f32s(&flat), Some(34962));
    accessors.push(Json::obj([("bufferView", Json::Int(view as i64)), ("componentType", Json::Int(5126)), ("count", Json::Int(n as i64)), ("type", Json::str("VEC3"))]));
    let normal = accessors.len() - 1;
    let mut attributes = vec![("POSITION", Json::Int(position as i64)), ("NORMAL", Json::Int(normal as i64))];
    // Vertex colours: the whole colour, or (with textures) a factor on it.
    if let Some(colors) = &mesh.colors {
        let bytes: Vec<u8> = colors.iter().flat_map(|c| [linear(c[0]), linear(c[1]), linear(c[2]), c[3]].map(|v| (v * 255.).round().clamp(0., 255.) as u8)).collect();
        let view = add_view(&mut bin, &bytes, Some(34962));
        accessors.push(Json::obj([("bufferView", Json::Int(view as i64)), ("componentType", Json::Int(5121)), ("normalized", Json::Bool(true)), ("count", Json::Int(n as i64)), ("type", Json::str("VEC4"))]));
        attributes.push(("COLOR_0", Json::Int(accessors.len() as i64 - 1)));
    }
    if let Some(t) = textures {
        let flat: Vec<f32> = t.uvs.iter().flatten().copied().collect();
        let view = add_view(&mut bin, &f32s(&flat), Some(34962));
        accessors.push(Json::obj([("bufferView", Json::Int(view as i64)), ("componentType", Json::Int(5126)), ("count", Json::Int(n as i64)), ("type", Json::str("VEC2"))]));
        attributes.push(("TEXCOORD_0", Json::Int(accessors.len() as i64 - 1)));
    }
    let idx: Vec<u8> = mesh.triangles.iter().flatten().flat_map(|i| i.to_le_bytes()).collect();
    let view = add_view(&mut bin, &idx, Some(34963));
    accessors.push(Json::obj([("bufferView", Json::Int(view as i64)), ("componentType", Json::Int(5125)), ("count", Json::Int(mesh.triangles.len() as i64 * 3)), ("type", Json::str("SCALAR"))]));
    let indices = accessors.len() - 1;

    // The material: textured, or the vertex colours with the mesh's average metallic and roughness.
    let mut images = Vec::new();
    let mut textures_json = Vec::new();
    let (metallic, roughness) = match &mesh.metal_rough {
        Some(mr) if !mr.is_empty() => {
            let (m, r) = mr.iter().fold((0f64, 0f64), |(m, r), v| (m + v[0] as f64, r + v[1] as f64));
            (m / mr.len() as f64, r / mr.len() as f64)
        }
        _ => (0., 1.),
    };
    let mut pbr = vec![("baseColorFactor", Json::Arr(vec![Json::Num(1.); 4]))];
    let mut alpha_mode = "OPAQUE";
    if let Some(t) = textures {
        for png in [&t.base_color_png, &t.metallic_roughness_png] {
            let view = add_view(&mut bin, png, None);
            images.push(Json::obj([("bufferView", Json::Int(view as i64)), ("mimeType", Json::str("image/png"))]));
            textures_json.push(Json::obj([("source", Json::Int(images.len() as i64 - 1)), ("sampler", Json::Int(0))]));
        }
        pbr.push(("baseColorTexture", Json::obj([("index", Json::Int(0))])));
        pbr.push(("metallicRoughnessTexture", Json::obj([("index", Json::Int(1))])));
        pbr.push(("metallicFactor", Json::Num(1.)));
        pbr.push(("roughnessFactor", Json::Num(1.)));
        if t.transparent {
            alpha_mode = "BLEND";
        }
    } else {
        pbr.push(("metallicFactor", Json::Num((metallic * 1000.).round() / 1000.)));
        pbr.push(("roughnessFactor", Json::Num((roughness * 1000.).round() / 1000.)));
        // Opaque, unless much of the surface is see-through.
        if mesh.colors.as_ref().is_some_and(|c| c.iter().filter(|v| v[3] < 0.5).count() * 10 > c.len()) {
            alpha_mode = "BLEND";
        }
    }
    pad4(&mut bin, 0);
    let mut doc = vec![
        ("asset", Json::obj([("version", Json::str("2.0")), ("generator", Json::str("OAIY (Pixal3D)"))])),
        ("scene", Json::Int(0)),
        ("scenes", Json::Arr(vec![Json::obj([("nodes", Json::Arr(vec![Json::Int(0)]))])])),
        ("nodes", Json::Arr(vec![Json::obj([("mesh", Json::Int(0)), ("name", Json::str("model"))])])),
        (
            "meshes",
            Json::Arr(vec![Json::obj([("primitives", Json::Arr(vec![Json::obj([("attributes", Json::obj(attributes)), ("indices", Json::Int(indices as i64)), ("material", Json::Int(0)), ("mode", Json::Int(4))])]))])]),
        ),
        ("materials", Json::Arr(vec![Json::obj([("pbrMetallicRoughness", Json::obj(pbr)), ("doubleSided", Json::Bool(false)), ("alphaMode", Json::str(alpha_mode))])])),
        ("accessors", Json::Arr(accessors)),
        ("bufferViews", Json::Arr(views)),
        ("buffers", Json::Arr(vec![Json::obj([("byteLength", Json::Int(bin.len() as i64))])])),
    ];
    if !images.is_empty() {
        doc.push(("images", Json::Arr(images)));
        doc.push(("textures", Json::Arr(textures_json)));
        doc.push(("samplers", Json::Arr(vec![Json::obj([("magFilter", Json::Int(9729)), ("minFilter", Json::Int(9987)), ("wrapS", Json::Int(33071)), ("wrapT", Json::Int(33071))])])));
    }
    let mut json = Json::obj(doc).to_json().into_bytes();
    pad4(&mut json, b' ');
    let total = 12 + 8 + json.len() + 8 + bin.len();
    let mut out = Vec::with_capacity(total);
    out.extend_from_slice(b"glTF");
    out.extend_from_slice(&2u32.to_le_bytes());
    out.extend_from_slice(&(total as u32).to_le_bytes());
    out.extend_from_slice(&(json.len() as u32).to_le_bytes());
    out.extend_from_slice(b"JSON");
    out.extend_from_slice(&json);
    out.extend_from_slice(&(bin.len() as u32).to_le_bytes());
    out.extend_from_slice(b"BIN\0");
    out.extend_from_slice(&bin);
    out
}
