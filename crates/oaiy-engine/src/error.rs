//! The error type shared by the core and the DeepSeek engine.
//!
//! Four kinds, each carrying its detail: which file, which tensor, which
//! expert. Library code returns these and never prints; bad input from disk
//! is always an `Err`, never a panic.

use std::fmt;

#[derive(Debug)]
pub enum Error {
    /// The OS said no: a missing file, a short read, a failed write.
    Io(std::io::Error),
    /// The bytes are not what they claim to be: a malformed header, a tensor
    /// of the wrong size, JSON that does not parse.
    Format(String),
    /// The caller asked for something invalid: a buffer of the wrong size,
    /// an expert that does not exist.
    Arg(String),
    /// Valid, but not something this build implements.
    Unsupported(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io(e) => write!(f, "i/o: {e}"),
            Error::Format(m) => write!(f, "bad format: {m}"),
            Error::Arg(m) => write!(f, "invalid argument: {m}"),
            Error::Unsupported(m) => write!(f, "unsupported: {m}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        if let Error::Io(e) = self {
            Some(e)
        } else {
            None
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}

pub type Result<T> = std::result::Result<T, Error>;
