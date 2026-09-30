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

type Roster = Box<dyn Fn() -> Vec<crate::companion::identity::ApprovedMobile> + Send + Sync>;

/// The approved Companions of the phone plugin `plugin`.
pub struct CompanionDevices {
    roster: Roster,
}

impl CompanionDevices {
    pub fn new(companion: crate::companion::routes::CompanionHandle, plugin: &str) -> Self {
        let plugin = plugin.to_string();
        Self::from_roster(Box::new(move || companion.identity_for(&plugin).map(|identity| identity.status().approved_mobiles).unwrap_or_default()))
    }

    /// The devices of whatever `roster` says is approved now (asked each time: a device approved or revoked is seen at once).
    pub(super) fn from_roster(roster: Roster) -> Self {
        Self { roster }
    }

    fn approved(&self) -> Vec<crate::companion::identity::ApprovedMobile> {
        (self.roster)()
    }
}

/// The devices that may take a call, from the approved Companions and what the owner said of them. The roster carries no kind (a device
/// id, a name and a key), so a Companion is a phone unless the owner ticked it as the Windows Companion on this computer, and one they said
/// never to ring is not a device at all. Every one may take calls and is taken to be reachable: whether it has a session is the plugin's to
/// know, and it offers a call only to those that do.
pub(super) fn map_devices(approved: Vec<crate::companion::identity::ApprovedMobile>, settings: &RingSettings) -> Vec<Device> {
    approved
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

impl DeviceSource for CompanionDevices {
    fn devices(&self, settings: &RingSettings) -> Vec<Device> {
        map_devices(self.approved(), settings)
    }

    fn label(&self, id: &str) -> String {
        self.approved().into_iter().find(|m| m.endpoint_key.thumbprint == id).map(|m| m.display_name).filter(|n| !n.trim().is_empty()).unwrap_or_else(|| "a device".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::companion::identity::{ApprovedMobile, EndpointKeyAlgorithm, EndpointPublicKey};

    fn mobile(thumbprint: &str, name: &str) -> ApprovedMobile {
        ApprovedMobile {
            device_id: format!("device-{thumbprint}"),
            display_name: name.into(),
            endpoint_key: EndpointPublicKey { algorithm: EndpointKeyAlgorithm::Ed25519, public_key: format!("key-{thumbprint}"), thumbprint: thumbprint.into() },
            approved_at: chrono::Utc::now(),
        }
    }

    #[test]
    fn an_approved_companion_is_a_phone_unless_the_owner_ticked_it_and_a_device_they_never_ring_is_none() {
        let roster = vec![mobile("thumb-pixel", "Pixel 6"), mobile("thumb-pc", "Office PC"), mobile("thumb-old", "Old tablet")];
        let kinds = |settings: &RingSettings| map_devices(roster.clone(), settings).into_iter().map(|d| (d.id, d.kind)).collect::<Vec<_>>();
        // As they come: every Companion is a phone, and the setup of an owner with one Companion on a second phone rings it.
        let untouched = RingSettings::default();
        assert_eq!(kinds(&untouched), vec![("thumb-pixel".to_string(), DeviceKind::Android), ("thumb-pc".to_string(), DeviceKind::Android), ("thumb-old".to_string(), DeviceKind::Android)]);
        // Ticked as the one on this computer: it is the Windows one, and only that.
        let ticked = RingSettings { windows_companions: vec!["thumb-pc".into()], ..Default::default() };
        assert_eq!(kinds(&ticked).iter().filter(|(_, k)| *k == DeviceKind::Windows).map(|(id, _)| id.as_str()).collect::<Vec<_>>(), vec!["thumb-pc"]);
        // Never rung: not a device. (Ticked and never rung: not a device either.)
        let never = RingSettings { excluded_devices: vec!["thumb-old".into(), "thumb-pc".into()], windows_companions: vec!["thumb-pc".into()], ..Default::default() };
        assert_eq!(kinds(&never).into_iter().map(|(id, _)| id).collect::<Vec<_>>(), vec!["thumb-pixel".to_string()]);
        // Every one may take a call, and is taken to be reachable (the plugin offers it only to those with a session).
        for d in map_devices(roster.clone(), &untouched) {
            assert!(d.call_authority && d.can_take && d.online && d.role == Role::SecondDevice && d.availability == Availability::Available, "{d:?}");
        }
        // No roster, no device.
        assert!(map_devices(Vec::new(), &untouched).is_empty());
    }

    #[test]
    fn the_roster_is_read_each_time_so_a_device_approved_or_revoked_is_seen_at_once_and_a_name_is_what_it_was_called() {
        let approved = std::sync::Arc::new(std::sync::Mutex::new(vec![mobile("thumb-pixel", "Pixel 6"), mobile("thumb-blank", "  ")]));
        let shared = approved.clone();
        let devices = CompanionDevices::from_roster(Box::new(move || shared.lock().unwrap().clone()));
        let settings = RingSettings::default();
        assert_eq!(devices.devices(&settings).len(), 2);
        assert_eq!(devices.label("thumb-pixel"), "Pixel 6");
        assert_eq!(devices.label("thumb-blank"), "a device", "a device with no name is not called by its key");
        assert_eq!(devices.label("thumb-gone"), "a device");
        approved.lock().unwrap().retain(|m| m.endpoint_key.thumbprint != "thumb-pixel");
        assert_eq!(devices.devices(&settings).into_iter().map(|d| d.id).collect::<Vec<_>>(), vec!["thumb-blank".to_string()]);
        approved.lock().unwrap().clear();
        assert!(devices.devices(&settings).is_empty());
    }
}
