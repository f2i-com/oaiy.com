//! Who could take a transferred call: the Companions the owner approved for the phone.
//!
//! [`crate::companion::identity::ApprovedMobile`] is never the handset that carries the call (that is the
//! call's other end): it is a device that lends the call its microphone and speaker, so each is a second
//! device. Whether one is connected now is the phone plugin's to know, not this desktop's (there is no
//! relay in this build), so every approved device is taken to be reachable and the plugin offers the call
//! only to those with a live session; a ring nobody is connected to runs out and the caller is offered a
//! message. The owner says which approved device is the Companion on this computer and which are never
//! to be rung (`windowsCompanions`, `excludedDevices`).

use super::host::DeviceSource;
use super::plan::{Availability, Device, DeviceKind, Role};
use super::settings::RingSettings;

/// The approved Companions of the phone plugin `plugin`.
pub struct CompanionDevices {
    companion: crate::companion::routes::CompanionHandle,
    plugin: String,
}

impl CompanionDevices {
    pub fn new(companion: crate::companion::routes::CompanionHandle, plugin: &str) -> Self {
        Self { companion, plugin: plugin.to_string() }
    }

    fn approved(&self) -> Vec<crate::companion::identity::ApprovedMobile> {
        self.companion.identity_for(&self.plugin).map(|identity| identity.status().approved_mobiles).unwrap_or_default()
    }
}

impl DeviceSource for CompanionDevices {
    fn devices(&self, settings: &RingSettings) -> Vec<Device> {
        self.approved()
            .into_iter()
            .map(|m| m.endpoint_key.thumbprint)
            .filter(|id| !settings.excluded_devices.contains(id))
            .map(|id| Device {
                role: Role::SecondDevice,
                kind: if settings.windows_companions.contains(&id) { DeviceKind::Windows } else { DeviceKind::Android },
                call_authority: true,
                can_take: true,
                online: true,
                pushable: false,
                availability: Availability::Available,
                id,
            })
            .collect()
    }

    fn label(&self, id: &str) -> String {
        self.approved().into_iter().find(|m| m.endpoint_key.thumbprint == id).map(|m| m.display_name).filter(|n| !n.trim().is_empty()).unwrap_or_else(|| "a device".to_string())
    }
}
