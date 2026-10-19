//! The public API: the gateway's route table ([`gateway`]), the discovery document companion apps read
//! ([`discovery`]), `multipart/form-data` bodies ([`multipart`]), and the Playground's requests for another OAIY
//! computer's models ([`peers`]).

pub(crate) mod discovery;
pub(crate) mod gateway;
pub(crate) mod multipart;
pub(crate) mod peers;
