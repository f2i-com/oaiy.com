//! Pairing v3 (README section 10.1): a phone without FormLogic.
//!
//! - [`math`]: the secret, the typed code, the derivations, the MACs, the SAS and the approval receipt, as pure functions (vector A3).
//! - [`offer`] and [`response`]: the two documents, built, parsed and verified.
//! - [`desktop`]: the desktop's half as a state machine: offer, rendezvous, verification of the response, the SAS gate, the receipt, the decision.
//! - [`phone`]: the phone's half: the key or the typed code, the proof of the relay, the offer, the response, the SAS, the receipt and the sealed token.

pub mod desktop;
pub mod math;
pub mod offer;
pub mod phone;
pub mod response;

pub use desktop::{DesktopIdentity, DesktopPairing, NewOffer, PairEvent, SasOutcome};
pub use math::{Derived, PairingKey, PairingSecret, Sas, SasEntry};
pub use offer::{Offer, OfferParams};
pub use phone::{PairingInput, PairingTarget, PhoneIdentity, PhonePairing};
pub use response::{Claims, Response};

use crate::client::ClientError;

/// What a pairing can fail with.
#[derive(Debug, Clone, PartialEq)]
pub enum PairingError {
    /// A call to the relay failed.
    Client(ClientError),
    /// A value failed a check of the protocol layer.
    Protocol(crate::Error),
    /// The relay does not know this rendezvous (never made, expired, burned: one answer for all three).
    NotFound,
    /// The rendezvous already has a response (`409 already_answered`): somebody else answered this offer. The phone shows the owner that, and does not answer.
    AlreadyAnswered,
    /// The relay is not who it was: its identity proof did not verify.
    Suspect,
    /// The call is not one this pairing can take now.
    WrongState(&'static str),
    /// A desktop call for a pairing that is not in flight.
    UnknownPairing,
    /// The approval was asked for before the owner typed the phone's code.
    SasRequired,
    /// The receipt covers the grants, the relay did not return them and the caller gave none: nothing can be verified, so nothing is stored.
    ReceiptGrantsUnknown,
    /// The receipt does not verify under the desktop key pinned from the offer (or is not dated in this pairing's window): no profile, and the token is not opened.
    ReceiptInvalid,
    /// The sealed token did not open, or is not a token.
    TokenInvalid,
    /// The secret or profile store could not be written.
    Store,
    /// Cancelled.
    Cancelled,
}

impl core::fmt::Display for PairingError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            PairingError::Client(e) => write!(f, "{e}"),
            PairingError::Protocol(e) => write!(f, "{e}"),
            PairingError::NotFound => f.write_str("the relay does not know this pairing"),
            PairingError::AlreadyAnswered => f.write_str("this offer was already answered"),
            PairingError::Suspect => f.write_str("the relay is not who it was"),
            PairingError::WrongState(w) => write!(f, "wrong state: {w}"),
            PairingError::UnknownPairing => f.write_str("no such pairing"),
            PairingError::SasRequired => f.write_str("the code on the phone has not been typed"),
            PairingError::ReceiptGrantsUnknown => f.write_str("the grants the receipt covers are not known"),
            PairingError::ReceiptInvalid => f.write_str("the approval receipt does not verify"),
            PairingError::TokenInvalid => f.write_str("the sealed token is not a token for this phone"),
            PairingError::Store => f.write_str("the pairing could not be stored"),
            PairingError::Cancelled => f.write_str("cancelled"),
        }
    }
}

impl std::error::Error for PairingError {}

impl From<ClientError> for PairingError {
    fn from(e: ClientError) -> Self {
        PairingError::Client(e)
    }
}

impl From<crate::Error> for PairingError {
    fn from(e: crate::Error) -> Self {
        PairingError::Protocol(e)
    }
}
