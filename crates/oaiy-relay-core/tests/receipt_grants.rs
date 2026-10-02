//! A phone pairs from the receipt alone: the relay returns `receipt.grants` (the sorted set the desktop signed: `Pairing.php`, `pairing-fetch-response.schema.json`), the phone verifies the
//! signature over exactly those grants and builds its profile from them, and a receipt whose grants are missing, unsorted, repeated, unknown or altered is refused and gives no profile and no
//! token. The answer of the stub relay is the real one's (sorted, in the same shape); the damaged answers are what a hostile or broken relay would send in its place.

mod common;

use common::pair::{grants, world};
use oaiy_relay_core::client::*;
use oaiy_relay_core::json::{self, Json};
use oaiy_relay_core::pairing::phone::Outcome;
use oaiy_relay_core::pairing::{PairEvent, PairingError, PairingInput, SasOutcome};
use oaiy_relay_core::testing::stub::{Fault, StubConfig};

fn approved_answer_of(stub: &oaiy_relay_core::testing::stub::StubRelay, pid: &str) -> Json {
    let request = HttpRequest {
        method: Method::Get,
        url: format!("{}/v1/pair/{pid}", stub.public_url()),
        headers: vec![],
        body: None,
        timeout: std::time::Duration::from_secs(2),
        max_response_bytes: 1 << 20,
        cancel: Cancel::new(),
    };
    let response = HttpClient::send(stub, &request).unwrap();
    assert_eq!(response.status, 200);
    json::parse(&response.body).unwrap()
}

/// `answer` with the receipt's grants replaced (`None` removes the member).
fn with_receipt_grants(answer: &Json, grants: Option<Vec<&str>>) -> String {
    let Json::Obj(members) = answer else { panic!("an object") };
    let members = members
        .iter()
        .map(|(k, v)| {
            if k != "receipt" {
                return (k.clone(), v.clone());
            }
            let Json::Obj(receipt) = v else { panic!("a receipt") };
            let mut receipt: Vec<(String, Json)> = receipt.iter().filter(|(k, _)| k != "grants").cloned().collect();
            if let Some(g) = &grants {
                receipt.push(("grants".to_string(), Json::Arr(g.iter().map(|g| Json::str(*g)).collect())));
            }
            (k.clone(), Json::Obj(receipt))
        })
        .collect();
    Json::Obj(members).to_compact()
}

#[test]
fn a_phone_pairs_from_the_receipt_alone_and_refuses_a_receipt_whose_grants_are_not_the_signed_set() {
    let mut w = world(StubConfig { receipt_includes_grants: true, ..common::env::quick() });
    let offer = w.new_offer();
    let mut phone = w.phone(PairingInput::Key(&offer.pairing_uri), 81);
    phone.fetch_offer(&Cancel::new()).unwrap();
    let sas = phone.respond(&Cancel::new()).unwrap();
    assert!(matches!(w.deliver()[0], PairEvent::AwaitingSas { .. }));
    // The desktop approves six grants in the order it holds them (not sorted); the relay returns them sorted.
    let approved = grants();
    assert_ne!(approved, {
        let mut s = approved.clone();
        s.sort();
        s
    });
    let outcome = w.desktop.confirm_sas(&w.env.client, &w.token, &offer.pid, &sas.display(), &approved, &Cancel::new()).unwrap();
    assert!(matches!(outcome, SasOutcome::Approved { .. }));
    let honest = approved_answer_of(&w.env.stub, &offer.pid);
    let mut sorted = approved.clone();
    sorted.sort();
    let relays: Vec<&str> = honest.at_grants();
    assert_eq!(relays, sorted.iter().map(String::as_str).collect::<Vec<_>>(), "the relay returns the signed set, sorted");

    let first = sorted[0].as_str();
    let (second, third) = (sorted[1].as_str(), sorted[2].as_str());
    let reversed: Vec<&str> = sorted.iter().rev().map(String::as_str).collect();
    let mut repeated: Vec<&str> = sorted.iter().map(String::as_str).collect();
    repeated.push(first);
    let mut unknown: Vec<&str> = sorted.iter().map(String::as_str).collect();
    unknown[1] = "root_access";
    unknown.sort();
    let dropped: Vec<&str> = sorted.iter().skip(1).map(String::as_str).collect();
    let mut added: Vec<&str> = sorted.iter().map(String::as_str).collect();
    added.push("takeover"); // a grant that exists and that the desktop did not approve
    added.sort();
    let others = [second, third];
    for (what, answer, expected) in [
        ("no grants member at all", with_receipt_grants(&honest, None), PairingError::ReceiptGrantsUnknown),
        ("the grants in reverse order", with_receipt_grants(&honest, Some(reversed)), PairingError::ReceiptInvalid),
        ("a repeated grant", with_receipt_grants(&honest, Some(repeated)), PairingError::ReceiptInvalid),
        (
            "a name that is no grant (sorted, so that nothing else is wrong with it)",
            with_receipt_grants(&honest, Some(unknown)),
            PairingError::ReceiptInvalid,
        ),
        ("one grant fewer", with_receipt_grants(&honest, Some(dropped)), PairingError::ReceiptInvalid),
        ("one grant more, a real one (sorted: only the signature can tell)", with_receipt_grants(&honest, Some(added)), PairingError::ReceiptInvalid),
        ("no grants", with_receipt_grants(&honest, Some(vec![])), PairingError::ReceiptInvalid),
        ("two others", with_receipt_grants(&honest, Some(others.to_vec())), PairingError::ReceiptInvalid),
    ] {
        w.env.stub.fail_next_on("/v1/pair/", 1, Fault::Respond(200, vec![], answer));
        let result = phone.wait_outcome(None, &Cancel::new());
        assert_eq!(result.err(), Some(expected), "{what}");
    }
    // The honest answer, last: the phone pairs from the receipt alone (no list from the caller), with the grants the desktop signed.
    let paired = match phone.wait_outcome(None, &Cancel::new()).unwrap() {
        Outcome::Paired(p) => p,
        other => panic!("not paired: {:?}", std::mem::discriminant(&other)),
    };
    assert_eq!(paired.profile.grants, sorted);
}

trait AtGrants {
    fn at_grants(&self) -> Vec<&str>;
}

impl AtGrants for Json {
    fn at_grants(&self) -> Vec<&str> {
        self.get("receipt").and_then(|r| r.get("grants")).and_then(Json::as_array).expect("grants").iter().map(|g| g.as_str().unwrap()).collect()
    }
}
