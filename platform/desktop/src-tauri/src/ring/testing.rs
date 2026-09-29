//! Stand-ins the tests of the ring, the call route and the plugin host share.

use std::sync::Arc;

use super::host::DeviceSource;
use super::plan::{Availability, Device, DeviceKind, Role};
use super::settings::RingSettings;

/// The Companions the owner approved, as the ring reads them.
pub struct Devices(pub Vec<Device>);

/// The Companion on the owner's own computer (the one they ticked as this computer's).
pub fn windows(id: &str) -> Device {
    Device { id: id.into(), role: Role::SecondDevice, kind: DeviceKind::Windows, call_authority: true, can_take: true, online: true, pushable: false, availability: Availability::Available }
}

/// A Companion on a second phone.
pub fn android(id: &str) -> Device {
    Device { kind: DeviceKind::Android, ..windows(id) }
}

impl DeviceSource for Devices {
    fn devices(&self, _: &RingSettings) -> Vec<Device> {
        self.0.clone()
    }

    fn label(&self, id: &str) -> String {
        format!("device {id}")
    }
}

/// The owner has set up the Companion on this computer: what a ring for the owner at their computer needs.
pub fn at_the_pc() -> Arc<Devices> {
    Arc::new(Devices(vec![windows("pc1")]))
}
