//! The errors of the protocol layer: what a value, a signature or a document was refused for. None of them carries a secret, a byte of a key or a token, or a
//! position in an input; a message says what was refused, and `code()` is a stable word for a test or a report to match on. (The errors of the I/O layer, which
//! can hold a status or a transport word, are `client::ClientError`.)

use core::fmt;

use crate::json::JsonError;

/// Why base64url text was refused (README section 1: "Padding, whitespace and characters outside the alphabet are refused", and Interpretation 12: a spelling whose
/// unused low bits are not zero is refused too).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum B64Error {
    /// A character outside `A-Za-z0-9_-` (padding, whitespace and the standard alphabet's `+` and `/` included).
    Alphabet,
    /// A length that no byte string has (one more than a multiple of four), or an empty text where bytes are required.
    Length,
    /// The unused low bits of the last character are not zero: another spelling of the same bytes.
    NonCanonical,
    /// A fixed-length value of the wrong decoded length.
    WrongSize,
}

/// What a function of this crate refused.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Error {
    /// JSON that is not what this crate reads (see [`JsonError`]).
    Json(JsonError),
    /// base64url that is not canonical.
    B64(B64Error),
    /// A value that is not in the form the protocol states; the member or value is named.
    Invalid(&'static str),
    /// A key, a sealed box or a signature that `oaiy-crypto` refused (small order, not canonical, did not verify, did not decrypt).
    Crypto(oaiy_crypto::Error),
    /// A signature over a document that did not verify; the document is named.
    BadSignature(&'static str),
    /// A MAC that did not verify (compared in constant time).
    BadMac(&'static str),
    /// Two values that must be the same are not (a thumbprint that does not match its key, an offer that does not name the relay it was fetched from).
    Mismatch(&'static str),
    /// A time that is outside its window; the document is named. The window is judged by the caller's clock.
    OutsideWindow(&'static str),
    /// A URI (`oaiy://pair`, `oaiy://enroll`) that is malformed or unsupported; the reason is named.
    Uri(&'static str),
}

impl Error {
    /// A stable identifier for a test, a log line or a report.
    pub const fn code(&self) -> &'static str {
        match self {
            Error::Json(_) => "json",
            Error::B64(_) => "base64url",
            Error::Invalid(_) => "invalid",
            Error::Crypto(_) => "crypto",
            Error::BadSignature(_) => "bad_signature",
            Error::BadMac(_) => "bad_mac",
            Error::Mismatch(_) => "mismatch",
            Error::OutsideWindow(_) => "outside_window",
            Error::Uri(_) => "uri",
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Json(e) => write!(f, "json: {e}"),
            Error::B64(e) => write!(f, "base64url: {e:?}"),
            Error::Invalid(w) => write!(f, "invalid: {w}"),
            Error::Crypto(e) => write!(f, "crypto: {e}"),
            Error::BadSignature(w) => write!(f, "bad signature: {w}"),
            Error::BadMac(w) => write!(f, "bad mac: {w}"),
            Error::Mismatch(w) => write!(f, "mismatch: {w}"),
            Error::OutsideWindow(w) => write!(f, "outside its window: {w}"),
            Error::Uri(w) => write!(f, "uri: {w}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<JsonError> for Error {
    fn from(e: JsonError) -> Self {
        Error::Json(e)
    }
}

impl From<B64Error> for Error {
    fn from(e: B64Error) -> Self {
        Error::B64(e)
    }
}

impl From<oaiy_crypto::Error> for Error {
    fn from(e: oaiy_crypto::Error) -> Self {
        Error::Crypto(e)
    }
}

/// `Result` of the protocol layer.
pub type Result<T> = core::result::Result<T, Error>;
