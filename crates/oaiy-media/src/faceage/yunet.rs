//! YuNet (Shiqi Yu and others; OpenCV Zoo's `face_detection_yunet_2023mar.onnx`, MIT): the faces in a picture, found as
//! OpenCV's `FaceDetectorYN` finds them. The network is run from the published ONNX file as it is (its eight operators:
//! convolutions, ReLU, 2x2 max pooling, nearest doubling, addition, and the transposes, reshapes and sigmoids of its
//! outputs); any other operator is refused. The picture goes in as BGR, 0 to 255, padded below and to the right to a
//! multiple of 32; each of the three strides (8, 16, 32) gives a score (the square root of class times objectness),
//! a box and five landmarks a cell; then OpenCV's non-maximum suppression on whole-pixel boxes.
use super::onnx::{self, Graph, Node};
use super::prep::Rgb;
use candle_core::{Device, Result, Tensor};
use std::collections::HashMap;
use std::path::Path;

/// OpenCV's defaults for this model.
pub const SCORE_THRESHOLD: f32 = 0.9;
pub const NMS_THRESHOLD: f32 = 0.3;
pub const TOP_K: usize = 5000;
const STRIDES: [usize; 3] = [8, 16, 32];
const OPS: [&str; 8] = ["Conv", "Relu", "MaxPool", "Resize", "Add", "Transpose", "Reshape", "Sigmoid"];

/// A face: its box (left, top, width, height), five landmarks (the eyes, the nose, the corners of the mouth) and score.
#[derive(Clone, Debug, PartialEq)]
pub struct Face {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
    pub landmarks: [f32; 10],
    pub score: f32,
}

pub struct YuNet {
    graph: Graph,
    weights: HashMap<String, Tensor>,
    ints: HashMap<String, Vec<i64>>,
    input: String,
    dev: Device,
}

fn bad(what: impl std::fmt::Display) -> candle_core::Error {
    candle_core::Error::Msg(format!("not YuNet's 2023 face detector: {what}"))
}

impl YuNet {
    pub fn load(path: &Path, dev: &Device) -> Result<Self> {
        let bytes = std::fs::read(path)?;
        let graph = onnx::read(&bytes).map_err(bad)?;
        if let Some(n) = graph.nodes.iter().find(|n| !OPS.contains(&n.op.as_str())) {
            return Err(bad(format!("it has a {} operator", n.op)));
        }
        for stride in STRIDES {
            for kind in ["cls", "obj", "bbox", "kps"] {
                if !graph.outputs.iter().any(|o| *o == format!("{kind}_{stride}")) {
                    return Err(bad(format!("it has no {kind}_{stride} output")));
                }
            }
        }
        let input = match graph.inputs.as_slice() {
            [one] => one.clone(),
            _ => return Err(bad("it takes more than one input")),
        };
        let mut weights = HashMap::new();
        let mut ints = HashMap::new();
        for (name, t) in &graph.initializers {
            // (an empty one, Resize's unused region of interest, is neither)
            if !t.ints.is_empty() {
                ints.insert(name.clone(), t.ints.clone());
            } else if !t.floats.is_empty() {
                weights.insert(name.clone(), Tensor::from_vec(t.floats.clone(), t.dims.clone(), dev)?);
            }
        }
        Ok(Self { graph, weights, ints, input, dev: dev.clone() })
    }

    /// The network's outputs by name, for an input [1, 3, H, W] (BGR, 0 to 255, H and W multiples of 32).
    pub fn raw(&self, input: &Tensor) -> Result<HashMap<String, Tensor>> {
        let mut values: HashMap<String, Tensor> = HashMap::new();
        values.insert(self.input.clone(), input.clone());
        for node in &self.graph.nodes {
            let arg = |i: usize| -> Result<Tensor> {
                let name = node.inputs.get(i).ok_or_else(|| bad(format!("{} has too few inputs", node.name)))?;
                values.get(name).or_else(|| self.weights.get(name)).cloned().ok_or_else(|| bad(format!("{name} is missing")))
            };
            let out = match node.op.as_str() {
                "Conv" => self.conv(node, &arg(0)?, &arg(1)?, node.inputs.get(2).map(|_| arg(2)).transpose()?)?,
                "Relu" => arg(0)?.relu()?,
                "MaxPool" => {
                    let (k, s) = (node.ints("kernel_shape").unwrap_or(&[]), node.ints("strides").unwrap_or(&[]));
                    let pads = node.ints("pads").unwrap_or(&[0, 0, 0, 0]);
                    if k != [2, 2] || s != [2, 2] || pads.iter().any(|&p| p != 0) || node.int("ceil_mode").unwrap_or(0) != 0 {
                        return Err(bad(format!("{} pools other than 2x2", node.name)));
                    }
                    arg(0)?.max_pool2d_with_stride((2, 2), (2, 2))?
                }
                "Resize" => {
                    if node.text("mode") != Some("nearest") || node.text("coordinate_transformation_mode") != Some("asymmetric") || node.text("nearest_mode") != Some("floor") {
                        return Err(bad(format!("{} resizes other than to the nearest", node.name)));
                    }
                    let scales = node.inputs.get(2).and_then(|n| self.graph.initializers.get(n)).map(|t| t.floats.clone()).unwrap_or_default();
                    if scales.len() != 4 || scales[0] != 1. || scales[1] != 1. || scales[2].fract() != 0. || scales[3].fract() != 0. {
                        return Err(bad(format!("{} scales by other than whole numbers", node.name)));
                    }
                    let x = arg(0)?;
                    let (_, _, h, w) = x.dims4()?;
                    x.upsample_nearest2d(h * scales[2] as usize, w * scales[3] as usize)?
                }
                "Add" => arg(0)?.broadcast_add(&arg(1)?)?,
                "Transpose" => {
                    let perm: Vec<usize> = node.ints("perm").ok_or_else(|| bad("a transpose without its order"))?.iter().map(|&p| p as usize).collect();
                    arg(0)?.permute(perm)?.contiguous()?
                }
                "Reshape" => {
                    let x = arg(0)?;
                    let shape = node.inputs.get(1).and_then(|n| self.ints.get(n)).ok_or_else(|| bad("a reshape to a shape it computes"))?;
                    let known: usize = shape.iter().enumerate().map(|(i, &d)| if d == 0 { x.dims()[i] } else if d < 0 { 1 } else { d as usize }).product();
                    let dims: Vec<usize> = shape.iter().enumerate().map(|(i, &d)| if d == 0 { x.dims()[i] } else if d < 0 { x.elem_count() / known.max(1) } else { d as usize }).collect();
                    x.reshape(dims)?
                }
                "Sigmoid" => candle_nn::ops::sigmoid(&arg(0)?)?,
                other => return Err(bad(format!("it has a {other} operator"))),
            };
            let name = node.outputs.first().ok_or_else(|| bad(format!("{} has no output", node.name)))?;
            values.insert(name.clone(), out);
        }
        self.graph.outputs.iter().map(|o| values.remove(o).map(|t| (o.clone(), t)).ok_or_else(|| bad(format!("{o} was not computed")))).collect()
    }

    fn conv(&self, node: &Node, x: &Tensor, w: &Tensor, b: Option<Tensor>) -> Result<Tensor> {
        let strides = node.ints("strides").unwrap_or(&[1, 1]);
        let pads = node.ints("pads").unwrap_or(&[0, 0, 0, 0]);
        let dilations = node.ints("dilations").unwrap_or(&[1, 1]);
        let group = node.int("group").unwrap_or(1) as usize;
        if strides.len() != 2 || strides[0] != strides[1] || pads.len() != 4 || pads.iter().any(|&p| p != pads[0]) || dilations != [1, 1] {
            return Err(bad(format!("{} is a convolution of a kind it does not have", node.name)));
        }
        let y = x.conv2d(w, pads[0] as usize, strides[0] as usize, 1, group)?;
        match b {
            Some(b) => y.broadcast_add(&b.reshape((1, (), 1, 1))?),
            None => Ok(y),
        }
    }

    /// The picture as YuNet takes it: [1, 3, H, W] BGR, 0 to 255, padded with black to multiples of 32.
    pub fn input(&self, img: &Rgb) -> Result<(Tensor, usize, usize)> {
        let (pw, ph) = (img.width.div_ceil(32) * 32, img.height.div_ceil(32) * 32);
        let plane = pw * ph;
        let mut data = vec![0f32; plane * 3];
        for y in 0..img.height {
            for x in 0..img.width {
                let px = &img.data[(y * img.width + x) * 3..(y * img.width + x) * 3 + 3];
                for (c, v) in [px[2], px[1], px[0]].into_iter().enumerate() {
                    data[c * plane + y * pw + x] = v as f32;
                }
            }
        }
        Ok((Tensor::from_vec(data, (1, 3, ph, pw), &self.dev)?, pw, ph))
    }

    /// The faces in a picture scoring `score_threshold` or more, after suppression, best first.
    pub fn detect(&self, img: &Rgb, score_threshold: f32, nms_threshold: f32, top_k: usize) -> Result<Vec<Face>> {
        let (input, pw, ph) = self.input(img)?;
        let raw = self.raw(&input)?;
        let flat = |name: String| -> Result<Vec<f32>> { raw.get(&name).ok_or_else(|| bad(format!("no {name}")))?.flatten_all()?.to_vec1() };
        let mut faces = Vec::new();
        for stride in STRIDES {
            let (cols, rows) = (pw / stride, ph / stride);
            let (cls, obj, bbox, kps) = (flat(format!("cls_{stride}"))?, flat(format!("obj_{stride}"))?, flat(format!("bbox_{stride}"))?, flat(format!("kps_{stride}"))?);
            if cls.len() != rows * cols || bbox.len() != rows * cols * 4 || kps.len() != rows * cols * 10 {
                return Err(bad("its outputs are not the picture's size"));
            }
            let s = stride as f32;
            for r in 0..rows {
                for c in 0..cols {
                    let i = r * cols + c;
                    let score = (cls[i].clamp(0., 1.) * obj[i].clamp(0., 1.)).sqrt();
                    if score < score_threshold {
                        continue;
                    }
                    let (cx, cy) = ((c as f32 + bbox[i * 4]) * s, (r as f32 + bbox[i * 4 + 1]) * s);
                    let (w, h) = (bbox[i * 4 + 2].exp() * s, bbox[i * 4 + 3].exp() * s);
                    let mut landmarks = [0f32; 10];
                    for n in 0..5 {
                        landmarks[2 * n] = (kps[i * 10 + 2 * n] + c as f32) * s;
                        landmarks[2 * n + 1] = (kps[i * 10 + 2 * n + 1] + r as f32) * s;
                    }
                    faces.push(Face { x: cx - w / 2., y: cy - h / 2., w, h, landmarks, score });
                }
            }
        }
        if faces.len() <= 1 {
            return Ok(faces);
        }
        Ok(suppress(&faces, score_threshold, nms_threshold, top_k).into_iter().map(|i| faces[i].clone()).collect())
    }
}

/// OpenCV's `NMSBoxes` on its whole-pixel boxes (each coordinate cut to an integer): candidates above the threshold,
/// best first (ties in their order), each kept unless it overlaps one already kept by more than `nms_threshold`.
fn suppress(faces: &[Face], score_threshold: f32, nms_threshold: f32, top_k: usize) -> Vec<usize> {
    let boxes: Vec<[i64; 4]> = faces.iter().map(|f| [f.x as i64, f.y as i64, f.w as i64, f.h as i64]).collect();
    let mut order: Vec<usize> = (0..faces.len()).filter(|&i| faces[i].score > score_threshold).collect();
    order.sort_by(|&a, &b| faces[b].score.partial_cmp(&faces[a].score).unwrap_or(std::cmp::Ordering::Equal));
    if top_k > 0 && top_k < order.len() {
        order.truncate(top_k);
    }
    let overlap = |a: [i64; 4], b: [i64; 4]| -> f32 {
        let (area_a, area_b) = (a[2] * a[3], b[2] * b[3]);
        if area_a + area_b <= 0 {
            return 1.;
        }
        let (x1, y1) = (a[0].max(b[0]), a[1].max(b[1]));
        let (x2, y2) = ((a[0] + a[2]).min(b[0] + b[2]), (a[1] + a[3]).min(b[1] + b[3]));
        let inter = if x2 > x1 && y2 > y1 { ((x2 - x1) * (y2 - y1)) as f64 } else { 0. };
        // OpenCV's `rectOverlap`: one less its Jaccard distance, the distance cut to a float first.
        let distance = 1. - inter / ((area_a + area_b) as f64 - inter);
        1. - distance as f32
    };
    let mut kept: Vec<usize> = Vec::new();
    for i in order {
        if kept.iter().all(|&k| overlap(boxes[i], boxes[k]) <= nms_threshold) {
            kept.push(i);
        }
    }
    kept
}
