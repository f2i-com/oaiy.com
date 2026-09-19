use thiserror::Error;

pub type Result<T> = std::result::Result<T, TokenizerError>;

#[derive(Debug, Error)]
pub enum TokenizerError {
    #[error(transparent)]
    Gguf(#[from] gguf::GgufError),

    #[error("missing required GGUF metadata: {0}")]
    MissingMetadata(&'static str),

    #[error("unsupported tokenizer encoder: {0}")]
    UnsupportedEncoder(String),
}
