//! The public API: the gateway's route table ([`gateway`]), the discovery document companion apps read
//! ([`discovery`]), and `multipart/form-data` bodies ([`multipart`]).

pub(crate) mod discovery;
pub(crate) mod gateway;
pub(crate) mod multipart;
