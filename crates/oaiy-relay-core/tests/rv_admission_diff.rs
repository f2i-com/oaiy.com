//! Reviewer's differential harness: the crate's admission readers on the damaged copies written by `tests/rv/admission/gen_adm.py` (read from `RV_ADM_IN`, one JSON object a line:
//! `id`, `kind` (`plugin` or `phone`), the recorded `request` body and the damaged `response` text), one verdict a line to `RV_ADM_OUT`. Ignored, and fails loudly unless both variables are set (`tools/run-differentials.ps1` runs it).

use oaiy_relay_core::admission::{MobileAdmission, MobileExpect, PluginAdmission, PluginRequest, Transport};
use oaiy_relay_core::json::{self, Json};
use oaiy_relay_core::keys::VerifyKey;

const NOW: i64 = 1_790_000_000;

fn transports(v: Option<&Json>) -> Option<Vec<Transport>> {
    v.map(|t| t.as_array().unwrap().iter().map(|x| if x.as_str() == Some("relay") { Transport::Relay } else { Transport::RelayPoll }).collect())
}

fn plugin_request(body: &Json) -> PluginRequest {
    PluginRequest {
        app_id: body.get_str("appId").unwrap().into(),
        plugin_id: body.get_str("pluginId").unwrap().into(),
        display_name: body.get_str("displayName").map(str::to_string),
        endpoint: VerifyKey::from_b64u(body.get("endpointPublicKey").unwrap().get_str("publicKey").unwrap()).unwrap(),
        approved_peers: body.get("approvedPeerKeyThumbprints").unwrap().as_array().unwrap().iter().map(|t| t.as_str().unwrap().to_string()).collect(),
        revision: body.get("peerRosterRevision").and_then(Json::as_u64).unwrap(),
        transports: transports(body.get("supportedTransports")),
    }
}

#[test]
#[ignore = "driver of tools/run-differentials.ps1: needs the generated inputs (RV_*) and python, node and php; run it through that script"]
fn rv_admission_differential() {
    let (inp, outp) = (
        std::env::var("RV_ADM_IN").expect("RV_ADM_IN is not set: run this through tools/run-differentials.ps1"),
        std::env::var("RV_ADM_OUT").expect("RV_ADM_OUT is not set: run this through tools/run-differentials.ps1"),
    );
    let text = std::fs::read_to_string(inp).unwrap();
    let mut out = String::new();
    for line in text.lines() {
        let c = json::parse(line.as_bytes()).unwrap();
        let id = c.get_str("id").unwrap();
        let req = c.get("request").unwrap();
        let response = c.get_str("response").unwrap();
        let (accept, relay) = if c.get_str("kind") == Some("plugin") {
            let r = plugin_request(req);
            match PluginAdmission::parse(response.as_bytes(), &r, NOW) {
                Ok(a) => (true, a.relay.is_some()),
                Err(_) => (false, false),
            }
        } else {
            let (app, dev, holder) = (req.get_str("appId").unwrap(), req.get_str("deviceId").unwrap(), req.get_str("holderKeyThumbprint").unwrap());
            let expect = MobileExpect { app_id: app, device_id: dev, holder_thumbprint: holder };
            match MobileAdmission::parse(response.as_bytes(), &expect, NOW) {
                Ok(a) => (true, a.relay.is_some()),
                Err(_) => (false, false),
            }
        };
        out.push_str(&format!("{{\"id\":{},\"accept\":{},\"relay\":{}}}\n", json::quote(id), accept, relay));
    }
    std::fs::write(outp, out).unwrap();
}
