//! Pairing v3 (README section 10.1): a phone without FormLogic.
//!
//! - [`math`]: the secret, the typed code, the derivations, the MACs, the SAS and the approval receipt, as pure functions (vector A3).
//! - [`offer`] and [`response`]: the two documents, built, parsed and verified.

pub mod math;
pub mod offer;
pub mod response;

pub use math::{Derived, PairingKey, PairingSecret, Sas, SasEntry};
pub use offer::{Offer, OfferParams};
pub use response::{Claims, Response};
