//! Companion-device support: trusting a phone to carry a call's audio.
//!
//! A companion app takes the live call on its own microphone and speakers. The
//! audio travels peer-to-peer over encrypted WebRTC; a relay carries only
//! signalling and never sees decoded PCM. This module owns the desktop's half
//! of the trust that makes that safe — see [`identity`].
//!
//! Deliberately generic over the plugin that brokers it. The implementation
//! this is ported from hardcodes one plugin id and leaves a TODO for the
//! second; OAIY is a plugin host, so the owner is a parameter and the gate is
//! the plugin's declared `companion.admission` capability.

pub mod identity;
pub mod pairing;
pub mod routes;
pub mod upstream;

pub use routes::{new_handle, CompanionHandle, CompanionRegistry};
pub use pairing::{
    MobilePairingClaims, MobilePairingResponse, PairingPayload, OFFER_KIND, PAIRING_TTL_SECONDS,
    RESPONSE_KIND, SCHEMA_VERSION,
};
pub use identity::{
    peer_roster_hash, ApprovedMobile, EndpointIdentity, EndpointIdentityHandle, EndpointPublicKey,
    IdentityStatus, KeyProtection, PendingMobileApproval,
};
