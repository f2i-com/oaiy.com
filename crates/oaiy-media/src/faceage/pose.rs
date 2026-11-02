//! Which way a face points, from the five landmarks the face detector gives (the eyes, the nose tip, the corners of
//! the mouth), by geometry alone: no model is asked about it.
//!
//! The nose tip stands out from the plane of the eyes and mouth, so when the head turns it moves across the face
//! more than they do. With the eye line levelled first (the head's tilt to a shoulder undone), the yaw is the nose's
//! offset from the face's middle line in eye widths, and the pitch is its height within the span from the eyes to the
//! mouth. Both are turned into rough degrees with the proportions of an average face; a face's own proportions shift
//! them, so they are best read against another picture of the same face (as a check's turns are read against its
//! straight-on picture). A photo held up and turned barely moves them, as nothing in it stands out.
//!
//! Signs, in the picture as the camera took it (not mirrored): yaw is positive when the face points to the picture's
//! right, pitch when it points up.

/// An average face, in millimetres, the eyes on the x axis: half the distance between the eyes; the nose tip's
/// height below the eyes and how far it stands out; the mouth's.
const EYE_X: f64 = 31.;
const NOSE_Y: f64 = 40.;
const NOSE_Z: f64 = 22.;
const MOUTH_Y: f64 = 70.;
const MOUTH_Z: f64 = 2.;

/// Which way a face points, in degrees.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pose {
    pub yaw: f64,
    pub pitch: f64,
}

/// The pose from YuNet's landmarks: `[eye, eye, nose tip, mouth corner, mouth corner]` as x, y pairs (in whichever
/// order the eyes and the mouth's corners come). None for landmarks no face could have (eyes on one point, the
/// mouth above the eyes).
pub fn pose(landmarks: &[f32; 10]) -> Option<Pose> {
    let p = |i: usize| (landmarks[2 * i] as f64, landmarks[2 * i + 1] as f64);
    let (mut a, mut b) = (p(0), p(1));
    if b.0 < a.0 {
        std::mem::swap(&mut a, &mut b);
    }
    let (dx, dy) = (b.0 - a.0, b.1 - a.1);
    let eyes = dx.hypot(dy);
    if !(eyes > 1e-3) {
        return None;
    }
    // Level the eye line: everything turned about the eyes' midpoint by the eye line's angle.
    let (cos, sin) = (dx / eyes, dy / eyes);
    let origin = ((a.0 + b.0) / 2., (a.1 + b.1) / 2.);
    let level = |q: (f64, f64)| {
        let (x, y) = (q.0 - origin.0, q.1 - origin.1);
        (x * cos + y * sin, -x * sin + y * cos)
    };
    let nose = level(p(2));
    let (m1, m2) = (level(p(3)), level(p(4)));
    let mouth = ((m1.0 + m2.0) / 2., (m1.1 + m2.1) / 2.);
    if !(mouth.1 > 1e-3) {
        return None;
    }
    // The yaw: the nose's offset from the middle line (from the eyes' midpoint to the mouth's, at the nose's height).
    let middle = mouth.0 * (nose.1 / mouth.1).clamp(0., 1.);
    let sideways = (nose.0 - middle) / eyes;
    let yaw_scale = (NOSE_Z - MOUTH_Z * NOSE_Y / MOUTH_Y) / (2. * EYE_X);
    let yaw = (sideways / yaw_scale).atan().to_degrees();
    // The pitch: the nose's height in the eye-to-mouth span, as the average face gives it for a turn up (positive) or
    // down. For a turn of tan t, (NOSE_Y - MOUTH_Y/2 - (NOSE_Z - MOUTH_Z/2) t) / (MOUTH_Y - MOUTH_Z t).
    let r = (nose.1 - mouth.1 / 2.) / mouth.1;
    let (a0, b0) = (NOSE_Y - MOUTH_Y / 2., NOSE_Z - MOUTH_Z / 2.);
    let pitch = ((a0 - r * MOUTH_Y) / (b0 - r * MOUTH_Z)).atan().to_degrees();
    (yaw.is_finite() && pitch.is_finite()).then_some(Pose { yaw, pitch })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The average face turned by `yaw`, `pitch` and `roll` (degrees), half a metre from a webcam, as the detector would
    /// place its landmarks (pixels, y down), the eyes the other way round with `swap`.
    fn landmarks(yaw: f64, pitch: f64, roll: f64, swap: bool) -> [f32; 10] {
        landmarks_at(500., yaw, pitch, roll, swap)
    }

    /// The same, `distance` millimetres from the camera.
    fn landmarks_at(distance: f64, yaw: f64, pitch: f64, roll: f64, swap: bool) -> [f32; 10] {
        let face = [(-EYE_X, 0., 0.), (EYE_X, 0., 0.), (0., NOSE_Y, NOSE_Z), (-24., MOUTH_Y, MOUTH_Z), (24., MOUTH_Y, MOUTH_Z)];
        let (y, p, r) = (yaw.to_radians(), pitch.to_radians(), roll.to_radians());
        let mut out = [0f32; 10];
        for (i, &(x0, y0, z0)) in face.iter().enumerate() {
            // A turn to the picture's right brings the nose (z, toward the camera) to +x; a turn up brings it to -y.
            let (x1, z1) = (x0 * y.cos() + z0 * y.sin(), -x0 * y.sin() + z0 * y.cos());
            let (y2, z2) = (y0 * p.cos() - z1 * p.sin(), y0 * p.sin() + z1 * p.cos());
            let (x3, y3) = (x1 * r.cos() - y2 * r.sin(), x1 * r.sin() + y2 * r.cos());
            let depth = distance - z2;
            out[2 * i] = (320. + 600. * x3 / depth) as f32;
            out[2 * i + 1] = (200. + 600. * y3 / depth) as f32;
        }
        if swap {
            out.swap(0, 2);
            out.swap(1, 3);
        }
        out
    }

    #[test]
    fn seen_from_afar_the_average_face_reads_its_angles_exactly() {
        // Without perspective (a camera far away) the geometry is exact for the face it assumes.
        for (yaw, pitch) in [(-40., 0.), (-15., 0.), (0., 0.), (8., 0.), (25., 0.), (0., -30.), (0., -10.), (0., 20.), (15., 10.)] {
            let got = pose(&landmarks_at(100_000., yaw, pitch, 0., false)).unwrap();
            assert!((got.yaw - yaw).abs() < 0.5 && (got.pitch - pitch).abs() < 0.5, "{yaw}, {pitch}: read {got:?}");
        }
    }

    #[test]
    fn a_turn_to_either_side_reads_as_its_angle_from_the_straight_on_picture() {
        // At a webcam's distance the nose, nearer the camera, looks a little lower and the angles a little larger; read
        // against the straight-on picture (as a check reads its turns) a turn is its angle, give or take a sixth.
        let straight = pose(&landmarks(0., 0., 0., false)).unwrap();
        assert!(straight.yaw.abs() < 0.5 && straight.pitch.abs() < 6., "straight on: {straight:?}");
        let mut last = f64::NEG_INFINITY;
        for yaw in [-40., -25., -15., -8., 8., 15., 25., 40.] {
            let got = pose(&landmarks(yaw, 0., 0., false)).unwrap();
            let (turned, other) = (got.yaw - straight.yaw, got.pitch - straight.pitch);
            assert!((turned - yaw).abs() <= 1. + yaw.abs() / 6., "yaw {yaw}: read {turned} from straight on");
            assert!(other.abs() < 1.5, "yaw {yaw} reads no pitch: {other}");
            assert!(turned > last, "a larger turn reads larger");
            last = turned;
        }
    }

    #[test]
    fn a_turn_up_or_down_reads_as_its_angle_from_the_straight_on_picture() {
        let straight = pose(&landmarks(0., 0., 0., false)).unwrap();
        let mut last = f64::NEG_INFINITY;
        for pitch in [-30., -20., -10., -5., 5., 10., 20., 30.] {
            let got = pose(&landmarks(0., pitch, 0., false)).unwrap();
            let (turned, other) = (got.pitch - straight.pitch, got.yaw - straight.yaw);
            assert!((turned - pitch).abs() <= 1. + pitch.abs() / 6., "pitch {pitch}: read {turned} from straight on");
            assert!(other.abs() < 0.5, "pitch {pitch} reads no yaw: {other}");
            assert!(turned > last, "a larger turn reads larger");
            last = turned;
        }
    }

    #[test]
    fn a_tilt_to_a_shoulder_or_the_eyes_order_changes_nothing() {
        for (yaw, pitch) in [(0., 0.), (20., 0.), (-15., 10.), (10., -15.)] {
            let straight = pose(&landmarks(yaw, pitch, 0., false)).unwrap();
            for roll in [-25., -10., 10., 25.] {
                for swap in [false, true] {
                    let tilted = pose(&landmarks(yaw, pitch, roll, swap)).unwrap();
                    assert!((tilted.yaw - straight.yaw).abs() < 1.5 && (tilted.pitch - straight.pitch).abs() < 1.5, "roll {roll}: {tilted:?} against {straight:?}");
                }
            }
        }
    }

    #[test]
    fn a_turn_reads_against_the_straight_on_picture_of_the_same_face() {
        // A camera above the screen and a face not quite square to it: the turns still read as turns from there.
        let straight = pose(&landmarks(6., -8., 3., false)).unwrap();
        let left = pose(&landmarks(6. + 15., -8., 3., false)).unwrap();
        let up = pose(&landmarks(6., -8. + 12., 3., false)).unwrap();
        assert!((left.yaw - straight.yaw - 15.).abs() < 3. && (left.pitch - straight.pitch).abs() < 3., "{left:?}");
        assert!((up.pitch - straight.pitch - 12.).abs() < 3. && (up.yaw - straight.yaw).abs() < 3., "{up:?}");
    }

    #[test]
    fn landmarks_no_face_could_have_are_no_pose() {
        assert!(pose(&[10., 10., 10., 10., 10., 20., 5., 30., 15., 30.]).is_none(), "the eyes on one point");
        assert!(pose(&[0., 50., 40., 50., 20., 40., 5., 10., 35., 10.]).is_none(), "the mouth above the eyes");
    }
}
