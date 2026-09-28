//! Plugin events on their way to the linked account (see `waiting_request_ids`).

use std::collections::HashSet;

/// Request ids of `aokie.appointment.requested` events not yet delivered.
pub fn waiting_request_ids() -> HashSet<String> {
    HashSet::new()
}
