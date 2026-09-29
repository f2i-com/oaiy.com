//! Audio for calls: resampling, WAV, and finding where the caller speaks.

/// A resampler that keeps its place across chunks (linear; phone audio does
/// not need more, and it must not click at chunk edges).
pub struct Resampler {
    from: u32,
    to: u32,
    /// Where the next output sample falls, in input samples of the next chunk
    /// (index -1 being the previous chunk's last sample, kept in `last`).
    pos: f64,
    last: Option<f32>,
}

impl Resampler {
    pub fn new(from: u32, to: u32) -> Self {
        Self { from: from.max(1), to: to.max(1), pos: 0.0, last: None }
    }

    pub fn process(&mut self, input: &[i16]) -> Vec<i16> {
        if self.from == self.to || input.is_empty() {
            return input.to_vec();
        }
        let step = self.from as f64 / self.to as f64;
        let last = self.last;
        let at = |i: isize| -> Option<f32> { if i < 0 { last } else { input.get(i as usize).map(|&s| s as f32) } };
        let mut out = Vec::with_capacity((input.len() as f64 / step) as usize + 2);
        let mut pos = self.pos;
        loop {
            let i = pos.floor() as isize;
            let frac = (pos - pos.floor()) as f32;
            let (Some(a), Some(b)) = (at(i), at(i + 1)) else { break };
            out.push((a + (b - a) * frac).round().clamp(-32768.0, 32767.0) as i16);
            pos += step;
        }
        // In the next chunk this one's last sample is index -1.
        self.pos = pos - input.len() as f64;
        self.last = input.last().map(|&s| s as f32);
        out
    }
}

/// A mono PCM16 WAV file.
pub fn wav(samples: &[i16], rate: u32) -> Vec<u8> {
    let data = samples.len() * 2;
    let mut out = Vec::with_capacity(44 + data);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&((36 + data) as u32).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&rate.to_le_bytes());
    out.extend_from_slice(&(rate * 2).to_le_bytes());
    out.extend_from_slice(&2u16.to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&(data as u32).to_le_bytes());
    for s in samples {
        out.extend_from_slice(&s.to_le_bytes());
    }
    out
}

/// A WAV file's format: (sample rate, channels, bits per sample), or None when it is not a PCM WAV.
pub fn wav_format(file: &[u8]) -> Option<(u32, u16, u16)> {
    if file.len() < 12 || &file[0..4] != b"RIFF" || &file[8..12] != b"WAVE" {
        return None;
    }
    // The chunks after the header: find "fmt ".
    let mut at = 12;
    while at + 8 <= file.len() {
        let size = u32::from_le_bytes([file[at + 4], file[at + 5], file[at + 6], file[at + 7]]) as usize;
        if &file[at..at + 4] == b"fmt " && at + 8 + 16 <= file.len() {
            let f = &file[at + 8..];
            let format = u16::from_le_bytes([f[0], f[1]]);
            // 1: PCM; 0xFFFE: extensible (PCM inside, for our purposes).
            if format != 1 && format != 0xFFFE {
                return None;
            }
            let channels = u16::from_le_bytes([f[2], f[3]]);
            let rate = u32::from_le_bytes([f[4], f[5], f[6], f[7]]);
            let bits = u16::from_le_bytes([f[14], f[15]]);
            return Some((rate, channels, bits));
        }
        at += 8 + size + (size & 1);
    }
    None
}

/// PCM16 little-endian bytes as samples (an odd last byte is dropped).
pub fn samples(bytes: &[u8]) -> Vec<i16> {
    bytes.chunks_exact(2).map(|p| i16::from_le_bytes([p[0], p[1]])).collect()
}

pub fn bytes(samples: &[i16]) -> Vec<u8> {
    samples.iter().flat_map(|s| s.to_le_bytes()).collect()
}

/// What the speech detector saw in a frame. Times are milliseconds of the
/// stream: the audio the detector has been fed, from its first sample.
#[derive(Debug, PartialEq, Eq)]
pub enum Heard {
    Nothing,
    /// The caller started speaking, at their first voiced frame.
    Started { at_ms: u64 },
    /// They have spoken long enough to mean it (`BARGE_IN_VOICED_MS` of voice): once an utterance.
    Sustained,
    /// They finished: the whole utterance, with a little from before it began,
    /// and when they spoke (their first voiced frame to the end of their last).
    Utterance { audio: Vec<i16>, start_ms: u64, end_ms: u64 },
}

/// Finds the caller's utterances in phone audio, frame by frame: an energy
/// threshold over a noise floor that follows the line. Speech starts after
/// enough voiced frames close together, and ends after a pause.
pub struct Detector {
    rate: u32,
    frame: usize,
    pending: Vec<i16>,
    noise: f32,
    speaking: bool,
    /// Recent frames' loudness, newest last (for the start decision).
    recent: std::collections::VecDeque<bool>,
    /// Audio from just before speech began.
    preroll: std::collections::VecDeque<i16>,
    utterance: Vec<i16>,
    voiced: usize,
    quiet: usize,
    /// Frames heard so far: the stream's clock.
    frames: u64,
    /// The utterance's first and last voiced frames.
    first_voiced: u64,
    last_voiced: u64,
    /// `Sustained` was said for this utterance.
    sustained: bool,
    /// While our own voice plays, the threshold rises (what echo is left after Aokie's canceller).
    pub speaking_out: bool,
}

/// 20 ms frames.
const FRAME_MS: u32 = 20;
/// Voiced frames among the last `START_WINDOW` for speech to have started (100 ms of 140).
const START_VOICED: usize = 5;
const START_WINDOW: usize = 7;
/// A pause this long ends the utterance (a breath between two sentences does not).
const END_QUIET_MS: u32 = 800;
/// Kept from before the start (a soft first sound, like "that's", is quieter than the threshold).
const PREROLL_MS: u32 = 500;
/// An utterance with less voiced audio than this is a noise, not words.
const MIN_VOICED_MS: u32 = 200;
/// This much voice over us is someone cutting in, not a cough or an "mm-hmm".
pub const BARGE_IN_VOICED_MS: u32 = 600;
/// The longest one utterance may run before it is cut and heard.
const MAX_UTTERANCE_MS: u32 = 20_000;
/// The quietest a voice can be counted (PCM16 RMS), whatever the floor.
const MIN_THRESHOLD: f32 = 350.0;

impl Detector {
    pub fn new(rate: u32) -> Self {
        let frame = (rate * FRAME_MS / 1000) as usize;
        Self {
            rate,
            frame,
            pending: Vec::new(),
            noise: 150.0,
            speaking: false,
            recent: Default::default(),
            preroll: Default::default(),
            utterance: Vec::new(),
            voiced: 0,
            quiet: 0,
            frames: 0,
            first_voiced: 0,
            last_voiced: 0,
            sustained: false,
            speaking_out: false,
        }
    }

    /// Whether the caller is in an utterance: it has started, and not yet ended.
    pub fn in_speech(&self) -> bool {
        self.speaking
    }

    fn ms_frames(&self, ms: u32) -> usize {
        (ms / FRAME_MS) as usize
    }

    /// Where a frame starts, in ms of the stream.
    fn frame_ms(frame: u64) -> u64 {
        frame * FRAME_MS as u64
    }

    /// Feed audio; what was heard in it, in order.
    pub fn push(&mut self, input: &[i16]) -> Vec<Heard> {
        self.pending.extend_from_slice(input);
        let mut heard = Vec::new();
        while self.pending.len() >= self.frame {
            let frame: Vec<i16> = self.pending.drain(..self.frame).collect();
            let h = self.frame_in(&frame);
            if h != Heard::Nothing {
                heard.push(h);
            }
        }
        heard
    }

    fn frame_in(&mut self, frame: &[i16]) -> Heard {
        let at = self.frames;
        self.frames += 1;
        let rms = (frame.iter().map(|&s| (s as f32) * (s as f32)).sum::<f32>() / frame.len() as f32).sqrt();
        let threshold = (self.noise * 2.8).max(MIN_THRESHOLD) * if self.speaking_out { 1.8 } else { 1.0 };
        let loud = rms > threshold;
        if !self.speaking {
            // The floor follows the line while no one speaks (slowly, and not up into speech).
            if !loud {
                self.noise = self.noise * 0.95 + rms.min(self.noise * 4.0 + 50.0) * 0.05;
            }
            self.recent.push_back(loud);
            if self.recent.len() > START_WINDOW {
                self.recent.pop_front();
            }
            self.preroll.extend(frame.iter().copied());
            let keep = (self.rate * PREROLL_MS / 1000) as usize;
            while self.preroll.len() > keep {
                self.preroll.pop_front();
            }
            if self.recent.iter().filter(|&&v| v).count() >= START_VOICED {
                // The window ends at this frame: the voice is from its first loud frame to its last.
                let frame_of = |i: usize| at - (self.recent.len() - 1 - i) as u64;
                self.first_voiced = self.recent.iter().position(|&v| v).map_or(at, frame_of);
                self.last_voiced = self.recent.iter().rposition(|&v| v).map_or(at, frame_of);
                self.speaking = true;
                self.sustained = false;
                self.utterance = self.preroll.drain(..).collect();
                self.voiced = START_VOICED;
                self.quiet = 0;
                self.recent.clear();
                return Heard::Started { at_ms: Self::frame_ms(self.first_voiced) };
            }
            return Heard::Nothing;
        }
        self.utterance.extend_from_slice(frame);
        if loud {
            self.voiced += 1;
            self.quiet = 0;
            self.last_voiced = at;
        } else {
            self.quiet += 1;
        }
        let too_long = self.utterance.len() >= (self.rate * MAX_UTTERANCE_MS / 1000) as usize;
        if self.quiet >= self.ms_frames(END_QUIET_MS) || too_long {
            self.speaking = false;
            self.quiet = 0;
            let audio = std::mem::take(&mut self.utterance);
            let voiced = std::mem::replace(&mut self.voiced, 0);
            if voiced >= self.ms_frames(MIN_VOICED_MS) {
                return Heard::Utterance { audio, start_ms: Self::frame_ms(self.first_voiced), end_ms: Self::frame_ms(self.last_voiced + 1) };
            }
            return Heard::Nothing;
        }
        if !self.sustained && self.voiced >= self.ms_frames(BARGE_IN_VOICED_MS) {
            self.sustained = true;
            return Heard::Sustained;
        }
        Heard::Nothing
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_wav_says_its_format() {
        assert_eq!(wav_format(&wav(&[0; 160], 16_000)), Some((16_000, 1, 16)));
        assert_eq!(wav_format(&wav(&[0; 240], 24_000)), Some((24_000, 1, 16)));
        assert_eq!(wav_format(b"RIFF\0\0\0\0WAVEdata"), None);
        assert_eq!(wav_format(b"not a wav at all"), None);
    }

    fn tone(ms: u32, rate: u32, amp: f32) -> Vec<i16> {
        (0..(rate * ms / 1000)).map(|i| (amp * (i as f32 * 0.07).sin()) as i16).collect()
    }

    fn names<'a>(heard: impl IntoIterator<Item = &'a Heard>) -> Vec<String> {
        heard.into_iter().map(|h| match h { Heard::Started { at_ms } => format!("start at {at_ms}"), Heard::Sustained => "sustained".into(), Heard::Utterance { audio, start_ms, end_ms } => format!("{} samples, {start_ms} to {end_ms}", audio.len()), Heard::Nothing => "-".into() }).collect()
    }

    #[test]
    fn a_spoken_phrase_between_pauses_is_one_utterance() {
        let rate = 24_000;
        let mut d = Detector::new(rate);
        let mut heard = Vec::new();
        heard.extend(d.push(&tone(1000, rate, 40.0)));
        heard.extend(d.push(&tone(900, rate, 6000.0)));
        heard.extend(d.push(&tone(800, rate, 40.0)));
        assert_eq!(heard.len(), 3, "{:?}", names(&heard));
        assert!(!d.in_speech(), "the pause ended it");
        // It starts where the voice does, not where it was noticed.
        assert_eq!(heard[0], Heard::Started { at_ms: 1000 });
        assert_eq!(heard[1], Heard::Sustained);
        let Heard::Utterance { audio, start_ms, end_ms } = &heard[2] else { panic!("no utterance") };
        // The phrase, its pre-roll, and the pause that ended it; its times are the voice's.
        let ms = audio.len() as u32 * 1000 / rate;
        assert!((1500..=2500).contains(&ms), "{ms} ms");
        assert_eq!((*start_ms, *end_ms), (1000, 1900));
    }

    #[test]
    fn sustained_comes_once_after_enough_voice_not_enough_time() {
        let rate = 24_000;
        let frame = (rate / 50) as usize;
        let mut d = Detector::new(rate);
        let mut heard: Vec<(u64, Heard)> = Vec::new();
        let mut fed_ms = 0u64;
        let mut feed = |d: &mut Detector, audio: Vec<i16>| {
            for f in audio.chunks(frame) {
                fed_ms += 20;
                heard.extend(d.push(f).into_iter().map(|h| (fed_ms, h)));
            }
        };
        // Voice, a pause shorter than the end of an utterance, and voice again.
        feed(&mut d, tone(1000, rate, 40.0));
        feed(&mut d, tone(400, rate, 6000.0));
        feed(&mut d, tone(400, rate, 40.0));
        feed(&mut d, tone(400, rate, 6000.0));
        feed(&mut d, tone(1000, rate, 40.0));
        assert_eq!(heard.len(), 3, "{:?}", names(heard.iter().map(|(_, h)| h)));
        assert_eq!(heard[0].1, Heard::Started { at_ms: 1000 });
        // 600 ms of voice: the first 400 ms, and 200 ms after the pause (1000 + 400 + 400 + 200).
        assert_eq!((heard[1].0, &heard[1].1), (2000, &Heard::Sustained));
        let Heard::Utterance { start_ms, end_ms, .. } = heard[2].1 else { panic!("no utterance") };
        // From the first voice to the last, not the quiet after it.
        assert_eq!((start_ms, end_ms), (1000, 2200));
    }

    #[test]
    fn a_short_phrase_is_not_sustained() {
        let rate = 24_000;
        let mut d = Detector::new(rate);
        let mut heard = d.push(&tone(1000, rate, 40.0));
        heard.extend(d.push(&tone(400, rate, 6000.0)));
        heard.extend(d.push(&tone(1000, rate, 40.0)));
        assert_eq!(heard.len(), 2, "{:?}", names(&heard));
        assert_eq!(heard[0], Heard::Started { at_ms: 1000 });
        assert!(matches!(heard[1], Heard::Utterance { start_ms: 1000, end_ms: 1400, .. }), "{:?}", names(&heard));
        // The stream's clock goes on across utterances.
        heard = d.push(&tone(700, rate, 6000.0));
        assert_eq!(heard[0], Heard::Started { at_ms: 2400 });
        assert_eq!(heard[1], Heard::Sustained);
    }

    #[test]
    fn a_click_is_not_speech() {
        let rate = 24_000;
        let mut d = Detector::new(rate);
        let mut heard = d.push(&tone(1000, rate, 40.0));
        heard.extend(d.push(&tone(60, rate, 8000.0)));
        heard.extend(d.push(&tone(1000, rate, 40.0)));
        assert!(heard.is_empty(), "{} events", heard.len());
    }

    #[test]
    fn a_noise_that_starts_like_speech_ends_without_words() {
        let rate = 24_000;
        let mut d = Detector::new(rate);
        let mut heard = d.push(&tone(1000, rate, 40.0));
        heard.extend(d.push(&tone(140, rate, 6000.0)));
        assert_eq!(heard, vec![Heard::Started { at_ms: 1000 }]);
        assert!(d.in_speech());
        // Too little voice for words: nothing is heard, and it is over.
        assert!(d.push(&tone(1000, rate, 40.0)).is_empty());
        assert!(!d.in_speech());
    }

    #[test]
    fn resampling_keeps_length_and_place_across_chunks() {
        let input: Vec<i16> = (0..2400).map(|i| ((i as f32 * 0.05).sin() * 10_000.0) as i16).collect();
        let whole = Resampler::new(24_000, 16_000).process(&input);
        let mut r = Resampler::new(24_000, 16_000);
        let mut parts = Vec::new();
        for chunk in input.chunks(960) {
            parts.extend(r.process(chunk));
        }
        assert!((whole.len() as i64 - 1600).abs() <= 2, "{}", whole.len());
        assert!((parts.len() as i64 - whole.len() as i64).abs() <= 2, "{} vs {}", parts.len(), whole.len());
        let wav = wav(&whole, 16_000);
        assert_eq!(&wav[..4], b"RIFF");
        assert_eq!(wav.len(), 44 + whole.len() * 2);
    }
}
