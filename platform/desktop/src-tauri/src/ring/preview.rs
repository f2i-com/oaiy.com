//! What would happen to a caller who asked for the owner right now, in plain words, and why nobody rang when nobody did.
//!
//! The Transfers page shows [`Preview`]: it is the policy asked the question a real request would be (the same plan, with the owner's
//! settings, the devices the owner approved, whether they are at the computer and the time as they stand), without counting a try. So what
//! the page says is what a caller would get, in every state of the owner, and not a guess from the settings. The same causes explain a
//! notice to the owner when a request found nobody to ring ([`Cause`]).
//!
//! It also keeps what the phone plugin did with the calls that began while transfers were on (offered them for transfer, or did not), because
//! a plugin that never offers a call leaves the settings looking right and nothing ever ringing.

use std::sync::Mutex;

use serde::Serialize;

use super::host::Ring;
use super::plan::{Counters, Device, DeviceKind, Inputs, Now, Reason, CallFacts};
use super::settings::{PhoneRing, RingSettings};
use super::{Decision, PlanReason};

/// Why nobody could be rung, when the answer is that the owner's own setup does not let a device ring.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum Cause {
    /// No Companion is approved.
    NoCompanion,
    /// Every approved Companion is set to never ring.
    AllNever,
    /// There are phones, and phones are set to never ring.
    PhonesOff,
    /// Every Companion is one the owner marked as the one on this computer, and they are not at it.
    OnlyThisComputers,
    /// Something else in the settings keeps every device quiet right now.
    Nothing,
}

impl Cause {
    pub fn code(self) -> &'static str {
        match self {
            Cause::NoCompanion => "noCompanion",
            Cause::AllNever => "allNever",
            Cause::PhonesOff => "phonesOff",
            Cause::OnlyThisComputers => "onlyThisComputers",
            Cause::Nothing => "nothing",
        }
    }

    /// What the owner is told when somebody asked for them and this kept every device quiet (after "Someone asked for you.").
    pub fn notice_text(self) -> &'static str {
        match self {
            Cause::NoCompanion => "No Companion is approved to take a transfer, so they were offered a message.",
            Cause::AllNever => "Every approved Companion is set to never ring, so they were offered a message.",
            Cause::PhonesOff => "Your phone is set to not ring, so they were offered a message.",
            Cause::OnlyThisComputers => "The only Companion is set as the one on this computer, and you were not at it, so they were offered a message.",
            Cause::Nothing => "Nothing is set to ring right now, so they were offered a message.",
        }
    }

    /// What the Transfers page says while it holds.
    pub fn preview_text(self) -> &'static str {
        match self {
            Cause::NoCompanion => "No Companion is approved yet, so nothing can ring and every caller is offered a message.",
            Cause::AllNever => "Every approved Companion is set to never ring, so nothing can ring and every caller is offered a message.",
            Cause::PhonesOff => "Your phone is set to never ring, so nothing rings and a caller who asks for you is offered a message. Set Phones to “Ring when I am away” or “Always ring”.",
            Cause::OnlyThisComputers => {
                "The only Companion is set as the one on this computer, and it rings only while you are at the computer, which you are not now, so a caller who asks for you is offered a message. If it is a phone, untick “This is the Companion on this computer” below."
            }
            Cause::Nothing => "Nothing would ring right now, so a caller who asks for you is offered a message.",
        }
    }
}

/// What a caller who asks for the owner would get now.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Preview {
    /// Transfers are on. When they are not, nothing else here is said.
    pub enabled: bool,
    /// Something would ring.
    pub rings: bool,
    /// What would ring, by name.
    pub devices: Vec<String>,
    /// In plain words: what would ring, or why nothing would.
    pub text: String,
    /// When nothing would ring for want of a device that may: why (see [`Cause`]); else none.
    pub cause: Option<&'static str>,
    /// What the phone plugin has done with the calls while transfers were on, when that is a problem: never offered them, or not allowed.
    pub plugin: Option<String>,
}

/// Whether the phone plugin offered a call for transfer.
#[derive(Default)]
pub struct Offers {
    /// The last call that began while transfers were on: offered, or not.
    last: Option<bool>,
    /// Some call was ever offered (since this desktop started): the plugin knows the contract.
    ever: bool,
    /// The plugin refused a request for want of consent, and no request has opened a ring since.
    consent_refused: bool,
}

pub(super) type OfferLog = Mutex<Offers>;

impl Ring {
    /// A call began. `transfers_on`: the owner had transfers on. `offered`: the phone offered it for transfer (`allowTransfer`). A call that
    /// began with transfers off says nothing about the plugin.
    pub fn note_call(&self, transfers_on: bool, offered: bool) {
        if !transfers_on {
            return;
        }
        let mut offers = self.offers.lock().unwrap_or_else(|e| e.into_inner());
        offers.last = Some(offered);
        offers.ever |= offered;
    }

    /// The phone plugin refused a request for want of consent: the Companion's consent for taking calls is not granted.
    pub fn note_consent_refused(&self) {
        self.offers.lock().unwrap_or_else(|e| e.into_inner()).consent_refused = true;
    }

    /// A request opened a ring: the plugin's consent is there.
    pub(super) fn note_ring_opened(&self) {
        self.offers.lock().unwrap_or_else(|e| e.into_inner()).consent_refused = false;
    }

    /// Why the devices the owner has are all quiet.
    pub(super) fn cause(&self, settings: &RingSettings) -> Cause {
        cause_of(&self.devices(settings), settings)
    }

    /// What the phone plugin did with the calls, in words, when it is a problem for the owner who turned transfers on.
    fn plugin_line(&self, settings: &RingSettings) -> Option<String> {
        let (last, ever, consent_refused) = {
            let o = self.offers.lock().unwrap_or_else(|e| e.into_inner());
            (o.last, o.ever, o.consent_refused)
        };
        let nobody_approved = self.devices(settings).is_empty() && settings.excluded_devices.is_empty();
        let consent = "Transfers are not allowed by your phone plugin’s consent settings (the Phone page). Callers are offered a message.";
        if consent_refused {
            return Some(consent.to_string());
        }
        match last {
            None | Some(true) => None,
            Some(false) if nobody_approved => Some("No Companion is approved, so your phone plugin does not offer calls for transfer. Callers are offered a message.".to_string()),
            Some(false) if !ever => Some("Your phone plugin does not support transfers yet: it did not offer the last call for transfer. Callers are offered a message.".to_string()),
            Some(false) => Some(consent.to_string()),
        }
    }

    /// What a caller who asks for the owner would get now (see the module docs). Nothing is counted.
    pub fn preview(&self) -> Preview {
        let settings = self.settings.get();
        if !settings.enabled {
            return Preview { enabled: false, rings: false, devices: Vec::new(), text: String::new(), cause: None, plugin: None };
        }
        let plugin = self.plugin_line(&settings);
        let now_unix = self.clock().unix();
        let mut effective = settings.clone();
        effective.away = settings.away_at(now_unix);
        let inputs = Inputs {
            devices: self.devices(&settings),
            presence: self.presence(),
            relay_healthy: true,
            call: CallFacts { reason: Reason::CallerAsked, caller_asked_confirmed: true, urgent_confirmed: false, caller_is_vip: false, vip_bypasses_limits: false },
            limits: Counters::default(),
            now: Now::of(&self.clock().local()),
            settings: effective,
        };
        let (plan, _) = Ring::plan_for(&inputs);
        if plan.rings() {
            let mut devices: Vec<String> = Vec::new();
            if plan.desktop_toast {
                devices.push("this computer".to_string());
            }
            devices.extend(plan.targets().iter().map(|id| self.device_label(id)));
            let text = format!("Right now a caller who asks for you would ring: {}.", devices.join(", "));
            return Preview { enabled: true, rings: true, devices, text, cause: None, plugin };
        }
        if plan.decision == Decision::MessageOnly && plan.reason == PlanReason::QuietHours {
            return Preview { enabled: true, rings: false, devices: Vec::new(), text: "It is quiet hours now, so a caller who asks for you is offered a message.".to_string(), cause: None, plugin };
        }
        let cause = self.cause(&settings);
        Preview { enabled: true, rings: false, devices: Vec::new(), text: cause.preview_text().to_string(), cause: Some(cause.code()), plugin }
    }
}

/// Why `devices` (the approved Companions that may ring) are all quiet on `settings`.
pub(super) fn cause_of(devices: &[Device], settings: &RingSettings) -> Cause {
    if devices.is_empty() {
        return if settings.excluded_devices.is_empty() { Cause::NoCompanion } else { Cause::AllNever };
    }
    let phones = devices.iter().filter(|d| d.kind != DeviceKind::Windows).count();
    if phones == 0 {
        Cause::OnlyThisComputers
    } else if settings.phone_ring == PhoneRing::Never {
        Cause::PhonesOff
    } else {
        Cause::Nothing
    }
}
