//! The desktop's side of an owner-run relay (design: `relay.final.md`, section 4.16).
//!
//! Only the files it keeps are here so far: [`link_store`], the sibling files under `<data>/relay/`
//! that hold the relay link, which lanes use it, and the provider keys pinned for it. The client that
//! polls it, and the enrolment, come in later packages and are built on these.
//!
//! Nothing here is a web origin the desktop trusts. The provider's origin is trusted by the access
//! rules of the HTTP API because linking is the owner approving that provider; a relay is a place this
//! desktop sends its token to, not a page that may drive it, and a type that never reaches those rules is
//! how that is kept true (a test reads every file of this folder for the names of the rules).

pub mod link_store;
