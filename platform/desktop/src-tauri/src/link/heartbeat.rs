//! Telling the provider this desktop is still here.
//!
//! A link creates a connection record, but records go stale. Providers decide
//! whether a desktop is reachable from how recently it last spoke — FormLogic's
//! window is 90 seconds — so without a periodic ping the app links successfully
//! and then shows as offline a minute later, which is exactly what "No Desktop"
//! means.
//!
//! Descriptor-driven like everything else here: the path, the interval and the
//! field names come from the connector, so a provider with a different presence
//! endpoint needs no code. A connector that declares no heartbeat simply never
//! gets one.
//!
//! One long-lived thread that asks the store what to do on each tick, rather
//! than a thread started and stopped around each link. Linking and unlinking are
//! then ordinary state changes instead of lifecycle events something has to
//! subscribe to — and there is no way to leak a thread per link attempt.
//!
//! # The beat also says what this desktop can run
//!
//! A provider decides where a piece of work runs, and it decides it from the
//! last thing this desktop said about itself. So the beat carries a list of
//! CAPABILITY TOKENS, and the vocabulary is the provider's, out of the
//! connector: a prefix for each logic language this desktop runs, and one token
//! for "the engine is up right now".
//!
//! Two tokens, two different claims, and confusing them is the failure mode:
//!
//! * The **language** tokens say what this BUILD is. Every logic lane on this
//!   desktop — trigger conditions, binding conditions, app-logic scripts — now
//!   runs on ZIPP and nothing else, so the tokens are sent on every beat
//!   whatever the engine is doing. A provider that sees none files this desktop
//!   as a build from before the vocabulary and keeps handing it JavaScript, so
//!   dropping them when the engine goes would produce exactly the wrong answer.
//! * The **engine** token says what this desktop can do RIGHT NOW. It is sent
//!   only while the engine probe and the script host both say so, and it is the
//!   token a provider reads before it hands anything over. A desktop with a
//!   language token and no engine token takes NOTHING — which is the state a
//!   desktop with no working engine should be in.
//!
//! The two clauses are not symmetric on purpose. Dropping the engine token is
//! immediate, because a desktop whose engine has gone must stop being given
//! work at once. Adding it waits for the host's health to have HELD
//! ([`HEALTH_SETTLE`]), because a token that appears and disappears on
//! alternate beats is worse for a provider than either state on its own.

use std::time::{Duration, Instant};

use super::descriptor::{self, HeartbeatSpec};
use super::{LinkHandle, LinkedAccount};
use crate::bridge::script_host::{Health, HealthSnapshot, ScriptHost};
use crate::bridge::worker::EngineProbe;

/// How often the worker wakes to see whether a beat is due.
///
/// Independent of any connector's interval: it only has to be fine-grained
/// enough that the first beat after linking is prompt.
const TICK: Duration = Duration::from_secs(5);

/// Send the first beat this soon after a link, rather than waiting a full
/// interval — the provider's presence window starts counting immediately, and a
/// user watching the UI wants to see it come online now.
const FIRST_BEAT_DELAY: Duration = Duration::from_secs(2);

/// How long the script host's health must have HELD before the engine token is
/// added to a beat.
///
/// A host that has just come up has proved nothing, and a provider that is told
/// "up" and then "down" within one of its own cache windows has been given a
/// worse answer than either one alone. Dropping the token has no such delay:
/// the safe direction is the prompt one.
const HEALTH_SETTLE: Duration = Duration::from_secs(10);

/// The closest two beats may be when the token set changes mid-interval.
///
/// The token set going stale for a whole interval is the thing this guards
/// against — a provider holding a 45-second-old "engine up" goes on offering
/// work to a desktop that cannot take it — but a host flapping faster than this
/// must not turn the heartbeat into a beat per tick.
const MIN_FLIP_GAP: Duration = Duration::from_secs(10);

/// How often a beat will try to bring an idle script host up.
///
/// See [`ScriptHost::warm`] for why anything does this at all.
const WARM_RETRY: Duration = Duration::from_secs(60);

/// The logic language every build that reaches this code runs.
///
/// Not a list of languages — the probe's `run.languages` is that, and it widens
/// this set the moment the CLI reports another one, with no change here. This
/// is the FLOOR: a desktop whose engine is missing, broken or not yet probed
/// still has to be recognisable as a build from the ZIPP era, because the
/// alternative reading — a build from before the vocabulary — is the one that
/// keeps being handed JavaScript work it can no longer run.
const ZIPP_ERA_LANGUAGE: &str = "javascript";

pub fn spawn(store: LinkHandle) {
    std::thread::spawn(move || {
        let mut last_beat: Option<(String, Instant)> = None;
        let mut last_tokens: Option<Vec<String>> = None;
        let mut last_warm: Option<Instant> = None;
        loop {
            std::thread::sleep(TICK);
            let Some(account) = store.account() else {
                // Unlinked: forget the schedule so a future link beats promptly
                // instead of inheriting an old timer.
                last_beat = None;
                last_tokens = None;
                continue;
            };
            let Some(spec) = heartbeat_spec(&store, &account) else {
                continue;
            };
            let interval = Duration::from_secs(spec.interval_seconds);

            let host = ScriptHost::global();
            let probe = host.engine_probe();
            let health = host.health();
            let tokens = capability_tokens(&spec, probe.as_ref(), &health, Instant::now());

            if should_warm(&spec, probe.as_ref(), &health, last_warm, Instant::now()) {
                last_warm = Some(Instant::now());
                host.warm_in_background();
            }

            let key = account.account_id.clone().unwrap_or_else(|| account.base_url.clone());
            if !matches!(&last_beat, Some((k, _)) if *k == key) {
                // A different account (or none yet). Seed the schedule just
                // short of a full interval so the FIRST beat lands promptly —
                // the provider's presence window starts counting immediately,
                // and a user watching the UI wants to see it come online now.
                last_beat = Some((
                    key.clone(),
                    Instant::now()
                        .checked_sub(interval.saturating_sub(FIRST_BEAT_DELAY))
                        .unwrap_or_else(Instant::now),
                ));
                last_tokens = None;
                continue;
            }
            if !beat_due(interval, last_beat.as_ref(), last_tokens.as_deref(), &tokens, Instant::now())
            {
                continue;
            }

            let sent = send(&account, &spec, &store.instance_id(), &tokens);
            match sent {
                Ok(()) => {
                    store.note_heartbeat(None);
                    // What the provider now holds. Recorded only on success:
                    // a beat that never arrived left the OLD tokens standing
                    // there, and remembering the new ones would count the
                    // change as delivered and cancel the retry that corrects
                    // it — which is how an "engine up" from before an outage
                    // would stay up until the next interval.
                    last_tokens = Some(tokens);
                }
                // Recorded, not retried harder: a provider that is down will be
                // up again on the next tick, and a burst of retries would only
                // make a bad moment worse. The status carries the reason so the
                // panel can say why it looks offline.
                Err(e) => store.note_heartbeat(Some(e)),
            }
            last_beat = Some((key, Instant::now()));
        }
    });
}

/// Whether a beat is due on this tick.
///
/// Two reasons, and the second is what makes the engine token worth sending:
///
/// * the interval has passed, which is the ordinary presence beat;
/// * the token set has CHANGED and the last beat is at least [`MIN_FLIP_GAP`]
///   old. Waiting a whole interval would leave a provider offering work to a
///   desktop whose engine went a minute ago; beating on every change would let
///   a flapping host beat on every tick, and the gap is what stops it.
fn beat_due(
    interval: Duration,
    last_beat: Option<&(String, Instant)>,
    last_tokens: Option<&[String]>,
    tokens: &[String],
    now: Instant,
) -> bool {
    let Some((_, at)) = last_beat else {
        return false;
    };
    let age = now.saturating_duration_since(*at);
    if age >= interval {
        return true;
    }
    match last_tokens {
        Some(sent) => sent != tokens && age >= MIN_FLIP_GAP,
        // Nothing sent yet for this account: the seeded schedule decides.
        None => false,
    }
}

/// Should this tick go and bring the script host up?
///
/// Keeping the engine token ANSWERABLE is the whole reason anything here does
/// this. The host is spawned by the first job it is asked to serve, and on a
/// desktop the provider sends nothing to there is no first job — so without
/// this the token would never appear, and a provider that reads the token would
/// never send the job that would have started the host. A standoff, broken from
/// this side.
///
/// The probe is read the same way: `None` is "nobody has asked yet", which is a
/// reason TO go and find out — warming takes the probe on its way — and NOT a
/// reason to sit still. That distinction is the whole of it on a headless
/// install, where nothing else ever fills the probe cache. A CLI that has been
/// probed and cannot serve leaf scripts is left alone; so is one that is not
/// there at all, which [`ScriptHost::engine_probe`] reports as `Unavailable`
/// rather than as silence.
fn should_warm(
    spec: &HeartbeatSpec,
    probe: Option<&EngineProbe>,
    host: &HealthSnapshot,
    last_warm: Option<Instant>,
    now: Instant,
) -> bool {
    wants_engine_token(spec)
        && !matches!(host.health, Health::Ready)
        && probe.map_or(true, runs_scripts)
        && last_warm.map_or(true, |at| now.saturating_duration_since(at) >= WARM_RETRY)
}

/// Does this provider read an engine token at all?
fn wants_engine_token(spec: &HeartbeatSpec) -> bool {
    spec.capabilities_field.is_some() && spec.engine_capability.is_some()
}

/// Does the probed CLI serve the script protocol this desktop speaks?
///
/// The same requirement [`ScriptHost`] itself applies before it spawns a child,
/// asked of the remembered report rather than by probing again.
fn runs_scripts(probe: &EngineProbe) -> bool {
    matches!(probe, EngineProbe::Ready(id) if id.protocols.get("script") == Some(&1))
}

/// The capability tokens this beat declares.
///
/// Pure, so every state is pinned without a provider, an engine or a clock. A
/// provider that names no capabilities field gets no tokens and no field: it is
/// not part of this vocabulary, and sending it an empty list would be an
/// assertion it never asked for.
pub fn capability_tokens(
    spec: &HeartbeatSpec,
    probe: Option<&EngineProbe>,
    host: &HealthSnapshot,
    now: Instant,
) -> Vec<String> {
    if spec.capabilities_field.is_none() {
        return Vec::new();
    }
    let mut tokens = Vec::new();
    if let Some(prefix) = &spec.language_capability_prefix {
        for language in languages(probe) {
            tokens.push(format!("{prefix}{language}"));
        }
    }
    if let Some(engine) = &spec.engine_capability {
        if engine_is_up(probe, host, now) {
            tokens.push(engine.clone());
        }
    }
    tokens
}

/// The logic languages this desktop declares.
///
/// The probe's answer when there is one, because that is what the CLI actually
/// runs and it is what gains a second language without a line changing here.
/// Otherwise [`ZIPP_ERA_LANGUAGE`] alone — see its comment for why an engine
/// this desktop cannot reach must not make it look like a build from before the
/// vocabulary.
fn languages(probe: Option<&EngineProbe>) -> Vec<String> {
    let declared = match probe {
        Some(EngineProbe::Ready(id)) => id
            .run_languages
            .iter()
            .map(|l| l.trim())
            .filter(|l| !l.is_empty())
            .map(str::to_string)
            .collect::<Vec<_>>(),
        _ => Vec::new(),
    };
    if declared.is_empty() {
        return vec![ZIPP_ERA_LANGUAGE.to_string()];
    }
    declared
}

/// Can this desktop run logic right now?
///
/// Both halves, and both are needed. The probe answers for the CLI on disk and
/// is remembered on the file's identity, so on its own it would go on saying
/// "ready" about a `--serve` child that has crashed or hung; the host answers
/// for that child and knows nothing about whether the CLI would serve a second
/// protocol. Either one alone advertises an engine that cannot take work.
///
/// [`Health::Starting`] is not health: a host nobody has asked for anything has
/// promised nothing, and the token is a promise.
fn engine_is_up(probe: Option<&EngineProbe>, host: &HealthSnapshot, now: Instant) -> bool {
    probe.is_some_and(runs_scripts)
        && matches!(host.health, Health::Ready)
        && now.saturating_duration_since(host.since) >= HEALTH_SETTLE
}

fn heartbeat_spec(store: &LinkHandle, account: &LinkedAccount) -> Option<HeartbeatSpec> {
    descriptor::find(store.data_dir(), &account.connector_id)?.heartbeat
}

/// One beat. Blocking; the worker thread exists for this.
fn send(
    account: &LinkedAccount,
    spec: &HeartbeatSpec,
    instance_id: &str,
    capabilities: &[String],
) -> Result<(), String> {
    let url = super::oauth::join(&account.base_url, &spec.path);
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(15))
        .build()
        .map_err(|e| format!("could not build the heartbeat client: {e}"))?;

    let mut body = serde_json::Map::new();
    body.insert(
        spec.instance_id_field.clone(),
        serde_json::Value::String(instance_id.to_string()),
    );
    if let (Some(field), Some(name)) = (spec.device_name_field.as_ref(), account.account_name.as_ref())
    {
        body.insert(field.clone(), serde_json::Value::String(name.clone()));
    }
    // Sent whenever the provider names a field, empty list included: the
    // provider keeps the last list it was given, so a beat that left the field
    // out after the engine went would leave "engine up" standing there.
    if let Some(field) = spec.capabilities_field.as_ref() {
        body.insert(
            field.clone(),
            serde_json::Value::Array(
                capabilities.iter().cloned().map(serde_json::Value::String).collect(),
            ),
        );
    }

    let resp = client
        .post(&url)
        .bearer_auth(&account.credential)
        .json(&serde_json::Value::Object(body))
        .send()
        .map_err(|e| format!("could not reach {}: {e}", account.base_url))?;

    let status = resp.status();
    if status.is_success() {
        return Ok(());
    }
    // 401/403 is the one worth naming: it means the key was revoked at the
    // provider, so the link is dead and re-linking is the fix — not something a
    // user would guess from "offline".
    let detail: serde_json::Value = resp.json().unwrap_or(serde_json::Value::Null);
    let message = detail
        .get("message")
        .or_else(|| detail.get("error"))
        .and_then(|v| v.as_str())
        .unwrap_or("the provider refused the heartbeat");
    if status.as_u16() == 401 || status.as_u16() == 403 {
        return Err(format!(
            "the provider no longer accepts this desktop's key ({message}) — link again"
        ));
    }
    Err(format!("HTTP {}: {message}", status.as_u16()))
}

/// A stable per-install id, in the shape providers accept.
///
/// Letters, digits, `.`, `_`, `-` only — FormLogic rejects anything else, and
/// that character set is the common denominator across the providers seen so
/// far. Generated once and persisted: a changing id would look like a new
/// desktop on every launch and accumulate ghost rows.
pub fn new_instance_id() -> String {
    format!("oaiy-{}", uuid::Uuid::new_v4().simple())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_instance_id_is_accepted_by_the_providers_character_rule() {
        // FormLogic: ^[A-Za-z0-9._-]+$, max 128. A uuid with braces or a
        // hostname with a space would be refused and the desktop would never
        // appear online — with a 400 nobody reads.
        for _ in 0..20 {
            let id = new_instance_id();
            assert!(!id.is_empty() && id.len() <= 128, "{id}");
            assert!(
                id.bytes()
                    .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-')),
                "{id}"
            );
        }
    }

    #[test]
    fn instance_ids_are_unique_per_install() {
        assert_ne!(new_instance_id(), new_instance_id());
    }

    // --- what a beat declares ----------------------------------------------

    fn spec() -> HeartbeatSpec {
        descriptor::builtin()
            .remove(0)
            .heartbeat
            .expect("the shipped connector beats")
    }

    fn identity(languages: &[&str], protocols: &[(&str, u64)]) -> crate::bridge::worker::EngineIdentity {
        crate::bridge::worker::EngineIdentity {
            name: "zipp".into(),
            release: "v0.0.19".into(),
            version: "0.0.19".into(),
            revision: "r".into(),
            wasm_sha256: "0".repeat(64),
            run_languages: languages.iter().map(|l| l.to_string()).collect(),
            protocols: protocols.iter().map(|(k, v)| (k.to_string(), *v)).collect(),
        }
    }

    fn ready_probe() -> EngineProbe {
        EngineProbe::Ready(identity(&["javascript"], &[("run", 1), ("script", 1)]))
    }

    /// A health snapshot in `health`, `held` old.
    fn snapshot(health: Health, held: Duration) -> HealthSnapshot {
        HealthSnapshot {
            health,
            since: Instant::now().checked_sub(held).unwrap_or_else(Instant::now),
            engine: None,
            instance: 0,
            children: 0,
        }
    }

    fn settled(health: Health) -> HealthSnapshot {
        snapshot(health, HEALTH_SETTLE * 2)
    }

    fn tokens(probe: Option<&EngineProbe>, host: &HealthSnapshot) -> Vec<String> {
        capability_tokens(&spec(), probe, host, Instant::now())
    }

    #[test]
    fn a_healthy_desktop_declares_the_language_it_runs_and_that_its_engine_is_up() {
        // The exact strings the provider keys on. A mismatch of one character
        // silently disables deferral — see the descriptor test that pins them
        // against FormLogic's own constants.
        assert_eq!(
            tokens(Some(&ready_probe()), &settled(Health::Ready)),
            vec!["logic-language:javascript", "logic-engine:zipp"]
        );
    }

    #[test]
    fn a_desktop_whose_script_host_is_down_still_says_which_era_it_is_from() {
        // Only the engine token goes. The language token is what this BUILD is,
        // and a provider that sees none reads a build from before the
        // vocabulary — which keeps getting JavaScript work this desktop can no
        // longer run, since every logic lane here is now the engine.
        for down in [
            Health::Unavailable { reason: "the child exited".into() },
            // A host nobody has asked for anything has promised nothing.
            Health::Starting,
        ] {
            assert_eq!(
                tokens(Some(&ready_probe()), &settled(down.clone())),
                vec!["logic-language:javascript"],
                "{down:?}"
            );
        }
    }

    #[test]
    fn a_desktop_with_no_engine_at_all_declares_the_era_and_nothing_else() {
        // The decided reading: a language token with no engine token is a
        // ZIPP-era desktop whose engine is not reporting healthy, and the
        // provider hands it NOTHING. That is correct — it can run nothing —
        // and it is the whole point of the floor.
        let none: Vec<String> = vec!["logic-language:javascript".into()];
        assert_eq!(tokens(None, &settled(Health::Starting)), none, "never probed");
        assert_eq!(
            tokens(
                Some(&EngineProbe::Unavailable { reason: "no CLI on this machine".into() }),
                &settled(Health::Ready)
            ),
            none,
            "probed and refused"
        );
    }

    #[test]
    fn both_halves_are_required_so_neither_can_advertise_an_engine_on_its_own() {
        // The AND, asserted from both sides. The probe answers for the CLI on
        // disk and is remembered on that file's identity, so it would go on
        // saying "ready" about a `--serve` child that had crashed; the host
        // answers for the child and knows nothing about whether the CLI serves
        // the script protocol at all.
        let engine = "logic-engine:zipp".to_string();
        assert!(tokens(Some(&ready_probe()), &settled(Health::Ready)).contains(&engine));
        assert!(!tokens(None, &settled(Health::Ready)).contains(&engine));
        assert!(!tokens(Some(&ready_probe()), &settled(Health::Starting)).contains(&engine));
        // A CLI that runs flows but does not serve leaf scripts is not an
        // engine this desktop can put a condition or a script on.
        let no_script = EngineProbe::Ready(identity(&["javascript"], &[("run", 1)]));
        assert!(!tokens(Some(&no_script), &settled(Health::Ready)).contains(&engine));
    }

    #[test]
    fn a_host_that_has_only_just_come_up_is_not_advertised_until_it_has_held() {
        // A token that appears and disappears on alternate beats is worse for a
        // provider than either state: it hands work over and takes it back
        // inside its own cache window. Dropping is NOT delayed — the safe
        // direction is the prompt one.
        let engine = "logic-engine:zipp".to_string();
        let fresh = snapshot(Health::Ready, Duration::from_secs(1));
        assert!(!tokens(Some(&ready_probe()), &fresh).contains(&engine), "one second is not a state");
        let held = snapshot(Health::Ready, HEALTH_SETTLE);
        assert!(tokens(Some(&ready_probe()), &held).contains(&engine));
        let just_died = snapshot(Health::Unavailable { reason: "gone".into() }, Duration::from_millis(1));
        assert!(!tokens(Some(&ready_probe()), &just_died).contains(&engine), "dropped at once");
    }

    #[test]
    fn a_second_language_arrives_from_the_probe_with_no_change_here() {
        // The prefix is the provider's vocabulary and the languages are the
        // CLI's report. When `oaiy capabilities --json` starts naming python,
        // the heartbeat says so without a line moving in this file.
        let python = EngineProbe::Ready(identity(&["javascript", "python"], &[("run", 1), ("script", 1)]));
        assert_eq!(
            tokens(Some(&python), &settled(Health::Ready)),
            vec!["logic-language:javascript", "logic-language:python", "logic-engine:zipp"]
        );
    }

    #[test]
    fn a_provider_that_reads_no_capabilities_is_told_none() {
        // Not an empty list: a provider outside this vocabulary never asked,
        // and the field is left out of the body entirely.
        let mut s = spec();
        s.capabilities_field = None;
        s.language_capability_prefix = None;
        s.engine_capability = None;
        assert!(capability_tokens(&s, Some(&ready_probe()), &settled(Health::Ready), Instant::now()).is_empty());
    }

    // --- bringing the host up ----------------------------------------------

    fn warms(probe: Option<&EngineProbe>, host: &HealthSnapshot) -> bool {
        should_warm(&spec(), probe, host, None, Instant::now())
    }

    #[test]
    fn a_host_nobody_has_asked_anything_is_brought_up_so_the_token_can_be_true() {
        // Without this the desktop and the provider wait for each other
        // forever: no job, so no child; no child, so no engine token; no engine
        // token, so the provider sends no job.
        assert!(warms(Some(&ready_probe()), &settled(Health::Starting)));
        assert!(warms(Some(&ready_probe()), &settled(Health::Unavailable { reason: "gone".into() })));
        // A host that is already serving needs nothing.
        assert!(!warms(Some(&ready_probe()), &settled(Health::Ready)));
    }

    #[test]
    fn a_probe_nobody_has_taken_is_a_reason_to_go_and_look_not_a_reason_to_sit_still() {
        // The one that decides a headless install. Nothing there polls the
        // readiness route and no flow run arrives before the token does, so the
        // probe cache stays empty — and warming is what FILLS it, by taking the
        // probe on its way to a child. Refusing to warm until something has
        // probed would mean never warming and never probing.
        assert!(warms(None, &settled(Health::Starting)));
        // Asked and answered, though: a CLI that cannot serve leaf scripts, or
        // one that is not on this machine at all, is left alone.
        let no_script = EngineProbe::Ready(identity(&["javascript"], &[("run", 1)]));
        assert!(!warms(Some(&no_script), &settled(Health::Starting)));
        assert!(!warms(
            Some(&EngineProbe::Unavailable { reason: "no CLI on this machine".into() }),
            &settled(Health::Starting)
        ));
    }

    #[test]
    fn warming_is_rate_limited_so_a_broken_install_is_not_spawned_at_every_tick() {
        let s = spec();
        let host = settled(Health::Starting);
        let now = Instant::now();
        let just_tried = now.checked_sub(Duration::from_secs(5)).unwrap_or(now);
        assert!(!should_warm(&s, None, &host, Some(just_tried), now));
        let long_ago = now.checked_sub(WARM_RETRY).unwrap_or(now);
        assert!(should_warm(&s, None, &host, Some(long_ago), now));
    }

    #[test]
    fn a_provider_that_reads_no_engine_token_is_never_warmed_for() {
        // Nothing would read the answer, and a child is a process.
        let mut s = spec();
        s.engine_capability = None;
        assert!(!should_warm(&s, None, &settled(Health::Starting), None, Instant::now()));
    }

    // --- when a beat goes out ----------------------------------------------

    fn beat(age: Duration, sent: Option<&[String]>, now_tokens: &[String]) -> bool {
        let at = Instant::now().checked_sub(age).unwrap_or_else(Instant::now);
        beat_due(
            Duration::from_secs(45),
            Some(&("acct".to_string(), at)),
            sent,
            now_tokens,
            Instant::now(),
        )
    }

    #[test]
    fn the_ordinary_beat_is_the_interval_and_nothing_else() {
        let same: Vec<String> = vec!["logic-language:javascript".into()];
        assert!(!beat(Duration::from_secs(5), Some(&same), &same), "five seconds in");
        assert!(!beat(Duration::from_secs(44), Some(&same), &same));
        assert!(beat(Duration::from_secs(45), Some(&same), &same));
        // The bug this replaced: a schedule that re-fired every tick because
        // "not yet due" was also the condition for the prompt first beat. The
        // provider got twelve beats an interval, every interval, forever.
        assert!(!beat(Duration::from_secs(6), Some(&same), &same));
    }

    #[test]
    fn a_changed_token_set_beats_early_but_a_flapping_one_cannot_beat_every_tick() {
        let up: Vec<String> = vec!["logic-language:javascript".into(), "logic-engine:zipp".into()];
        let down: Vec<String> = vec!["logic-language:javascript".into()];
        // Waiting a whole interval would leave the provider offering work to a
        // desktop whose engine went a minute ago.
        assert!(beat(MIN_FLIP_GAP, Some(&up), &down));
        assert!(beat(MIN_FLIP_GAP, Some(&down), &up));
        // But a host flapping faster than the gap does not turn the heartbeat
        // into a beat per tick.
        assert!(!beat(Duration::from_secs(5), Some(&up), &down));
        assert!(!beat(Duration::from_secs(9), Some(&down), &up));
        // Nothing has been sent for this account yet: the seeded schedule
        // decides when the first beat goes, not the token set.
        assert!(!beat(Duration::from_secs(20), None, &up));
    }
}
