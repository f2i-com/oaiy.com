//! Getting models: what a file or folder is, from its own headers ([`detect`]); the catalog of the models OAIY runs and
//! their downloads ([`downloads`]), none of a file this computer has already ([`library`], by its [`sha256`]); adding a
//! picked file to the configuration ([`registry`]).

pub(crate) mod detect;
pub mod downloads;
pub(crate) mod library;
pub(crate) mod registry;
pub(crate) mod sha256;
