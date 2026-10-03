//! The relay client behind traits: HTTP, a clock and a random source, secret and profile stores, a poll store, and the status the user sees.
//!
//! - [`http`]: [`HttpClient`] (the platform's), the request and response, [`Cancel`].
//! - [`clock`]: [`Clock`], [`Rng`] and the offset of the relay's clock and the provider's.
//! - [`store`]: [`SecretStore`], [`ProfileStore`] and [`PollStore`] with in-memory and file implementations, the profile and the cursor.
//! - [`status`]: the connection state, the events, the health answer.
//! - [`relay`]: [`RelayClient`], the calls of the protocol, with the proof gate.
//! - [`poll_loop`]: [`PollLoop`], the driver of [`crate::poll`].
//! - `loopback` (feature `loopback-http`): a std-only HTTP/1.1 client for loopback, for tests and the harness.
//! - `keystore` (feature `keystore`): [`SecretStore`] over `oaiy-keystore`.

pub mod clock;
pub mod http;
#[cfg(feature = "keystore")]
pub mod keystore;
#[cfg(feature = "loopback-http")]
pub mod loopback;
pub mod pair;
pub mod poll_loop;
pub mod relay;
pub mod status;
pub mod store;

pub use clock::{Clock, OffsetClock, OsRng, ProviderClock, Rng, SystemClock};
pub use http::{Cancel, HttpClient, HttpRequest, HttpResponse, Method, TransportError};
pub use pair::{PairAck, PairCreate, PairCreated, PairFetch, PairState, ReceiptWire};
pub use poll_loop::{LoopEnd, PollHandle, PollLoop, PollLoopConfig};
pub use relay::{ClientConfig, ClientError, Hdr, PollReply, PollRequest, PostItem, PostResult, PostStatus, ProveError, RelayClient, RelayError};
pub use status::{ConnectionState, Event, Health, NullSink, RecordingSink, StatusSink};
pub use store::{
    AcceptedItem, FilePollStore, FileProfileStore, Item, MemoryPollStore, MemoryProfileStore, MemorySecretStore, PeerPin, PersistBatch, PollCursor,
    PollStore, ProfileKind, ProfileStore, RelayProfile, SecretStore, StoreError, SECRET_TOKEN,
};

use crate::enrol::{Enrolled, EnrolmentKey, Role};
use crate::keys::{VerifyKey, X25519Public};

/// Redeems an enrolment key and keeps the result in the right order: the token goes to the secret store first, the profile after it, and a failure of the second removes the
/// first, so that a half-made enrolment is never left behind (a token with no profile, or a profile whose token is not there). Returns the profile.
#[allow(clippy::too_many_arguments)]
pub fn enrol_and_store(
    client: &RelayClient,
    key: &EnrolmentKey,
    name: &str,
    ed25519: &VerifyKey,
    x25519: &X25519Public,
    secrets: &dyn SecretStore,
    profiles: &dyn ProfileStore,
    cancel: &Cancel,
) -> Result<RelayProfile, ClientError> {
    if key.role != Role::Desktop {
        return Err(ClientError::Request(crate::Error::Invalid("this client enrols desktops")));
    }
    let Enrolled { device_id, token, relay_id, time } = client.enroll(key, name, ed25519, x25519, cancel)?;
    secrets.put(SECRET_TOKEN, token.expose().as_bytes()).map_err(|_| ClientError::BadAnswer("the token could not be stored"))?;
    let profile = RelayProfile {
        kind: ProfileKind::Desktop,
        relay: key.relay.clone(),
        relay_id,
        relay_thumbprint: key.relay_thumbprint.clone(),
        device_id,
        name: crate::ids::clean_name(name, 60),
        enrolled_at: time as i64,
        app_id: None,
        grants: Vec::new(),
        peer: None,
    };
    if profiles.save(&profile).is_err() {
        let _ = secrets.delete(SECRET_TOKEN);
        return Err(ClientError::BadAnswer("the profile could not be stored"));
    }
    Ok(profile)
}
