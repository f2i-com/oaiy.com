//! This computer's identity on OAIY Network: a node id, a P-256 key and a self-signed certificate, made the first time
//! Network is switched on and kept in the data folder (`network/identity.json`, owner-only). A peer pins the
//! certificate's fingerprint when the two are paired; the key never leaves this computer.

use std::path::Path;

use base64::Engine as _;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// The file the identity is kept in, below the network folder.
pub const FILE: &str = "identity.json";

#[derive(Serialize, Deserialize)]
struct Stored {
    node_id: String,
    cert: String,
    key: String,
}

/// A node's identity: what its peers know it by (its id and its certificate's fingerprint), and its key.
#[derive(Clone)]
pub struct Identity {
    pub node_id: String,
    pub cert_der: Vec<u8>,
    key_der: Vec<u8>,
    pub fingerprint: String,
}

impl std::fmt::Debug for Identity {
    // (never the key)
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Identity").field("node_id", &self.node_id).field("fingerprint", &self.fingerprint).finish()
    }
}

impl Identity {
    /// A new identity: a random node id and a certificate for it, valid for a century (it is pinned, not checked
    /// against a clock or an authority).
    pub fn generate() -> Result<Identity, String> {
        let node_id = uuid::Uuid::new_v4().simple().to_string();
        let key = rcgen::KeyPair::generate_for(&rcgen::PKCS_ECDSA_P256_SHA256).map_err(|e| format!("a key: {e}"))?;
        let mut params = rcgen::CertificateParams::new(vec!["oaiy-node.invalid".to_string()]).map_err(|e| format!("a certificate: {e}"))?;
        params.distinguished_name.push(rcgen::DnType::CommonName, format!("OAIY node {node_id}"));
        params.not_before = rcgen::date_time_ymd(2025, 1, 1);
        params.not_after = rcgen::date_time_ymd(2125, 1, 1);
        let cert = params.self_signed(&key).map_err(|e| format!("a certificate: {e}"))?;
        let cert_der = cert.der().to_vec();
        Ok(Identity { fingerprint: fingerprint(&cert_der), node_id, cert_der, key_der: key.serialize_der() })
    }

    /// The identity kept in `dir`, made (and kept) where there is none.
    pub fn load_or_create(dir: &Path) -> Result<Identity, String> {
        let path = dir.join(FILE);
        if let Ok(text) = std::fs::read_to_string(&path) {
            let stored: Stored = serde_json::from_str(&text).map_err(|e| format!("{} is not an identity: {e}", path.display()))?;
            let b64 = base64::engine::general_purpose::STANDARD;
            let cert_der = b64.decode(&stored.cert).map_err(|e| format!("its certificate: {e}"))?;
            let key_der = b64.decode(&stored.key).map_err(|e| format!("its key: {e}"))?;
            return Ok(Identity { fingerprint: fingerprint(&cert_der), node_id: stored.node_id, cert_der, key_der });
        }
        let id = Identity::generate()?;
        crate::secret_file::create_private_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        let b64 = base64::engine::general_purpose::STANDARD;
        let stored = Stored { node_id: id.node_id.clone(), cert: b64.encode(&id.cert_der), key: b64.encode(&id.key_der) };
        crate::secret_file::write(&path, serde_json::to_vec_pretty(&stored).map_err(|e| e.to_string())?).map_err(|e| format!("{}: {e}", path.display()))?;
        Ok(id)
    }

    /// Its certificate, as TLS presents it.
    pub fn certificate(&self) -> rustls::pki_types::CertificateDer<'static> {
        rustls::pki_types::CertificateDer::from(self.cert_der.clone())
    }

    /// Its key, as TLS signs with it.
    pub fn private_key(&self) -> rustls::pki_types::PrivateKeyDer<'static> {
        rustls::pki_types::PrivateKeyDer::Pkcs8(rustls::pki_types::PrivatePkcs8KeyDer::from(self.key_der.clone()))
    }
}

/// A certificate's fingerprint: the SHA-256 of its DER, in lower-case hex.
pub fn fingerprint(der: &[u8]) -> String {
    Sha256::digest(der).iter().map(|b| format!("{b:02x}")).collect()
}

/// A fingerprint as a person reads one: its first 16 hex digits in groups of four.
pub fn short(fingerprint: &str) -> String {
    fingerprint.chars().take(16).collect::<Vec<_>>().chunks(4).map(|c| c.iter().collect::<String>()).collect::<Vec<_>>().join(" ")
}

/// The code both computers show while they are paired: six digits from both certificates' fingerprints, in either
/// order. A computer in the middle of the first contact has a certificate of its own, so the two screens would differ.
pub fn pairing_code(a: &str, b: &str) -> String {
    let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
    let digest = Sha256::digest(format!("oaiy-network-pairing-v1|{lo}|{hi}").as_bytes());
    let n = u32::from_be_bytes([digest[0], digest[1], digest[2], digest[3]]) % 1_000_000;
    format!("{:03} {:03}", n / 1000, n % 1000)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_identity_is_kept_and_read_back_the_same() {
        let dir = std::env::temp_dir().join(format!("oaiy-network-identity-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let made = Identity::load_or_create(&dir).unwrap();
        let again = Identity::load_or_create(&dir).unwrap();
        assert_eq!((made.node_id.clone(), made.fingerprint.clone()), (again.node_id, again.fingerprint));
        assert_eq!(made.fingerprint.len(), 64);
        assert_eq!(made.fingerprint, fingerprint(&made.cert_der));
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(dir.join(FILE)).unwrap().permissions().mode() & 0o777;
            assert_eq!(mode & 0o077, 0, "the key is the owner's alone: {mode:o}");
        }
        assert!(!format!("{made:?}").contains("key_der"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_pairing_code_is_both_certificates_and_either_order() {
        let (a, b, c) = ("aa".repeat(32), "bb".repeat(32), "cc".repeat(32));
        assert_eq!(pairing_code(&a, &b), pairing_code(&b, &a));
        assert_ne!(pairing_code(&a, &b), pairing_code(&a, &c), "a third certificate in the middle shows another code");
        let code = pairing_code(&a, &b);
        assert_eq!(code.len(), 7);
        assert!(code.chars().enumerate().all(|(i, ch)| if i == 3 { ch == ' ' } else { ch.is_ascii_digit() }));
        assert_eq!(short(&a), "aaaa aaaa aaaa aaaa");
    }
}
