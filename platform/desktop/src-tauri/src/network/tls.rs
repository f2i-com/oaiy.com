//! Mutual TLS between OAIY computers, with no certificate authority: each presents its own self-signed certificate
//! ([`super::identity`]), and each checks the other's against the fingerprint it pinned when the two were paired.
//!
//! A worker's listener accepts any client certificate whose key signs the handshake (proof that the caller holds it);
//! which routes that certificate may reach is judged per request, by whether it is a paired primary's
//! ([`super::store::Peers::by_fingerprint`]). A primary connecting to a worker accepts only the certificate it pinned,
//! or, while pairing, records the one it was shown (the pairing code is computed from it).

use std::sync::{Arc, Mutex};

use rustls::client::danger::{HandshakeSignatureValid, ServerCertVerified, ServerCertVerifier};
use rustls::crypto::{verify_tls12_signature, verify_tls13_signature, CryptoProvider, WebPkiSupportedAlgorithms};
use rustls::pki_types::{CertificateDer, ServerName, UnixTime};
use rustls::server::danger::{ClientCertVerified, ClientCertVerifier};
use rustls::{DigitallySignedStruct, DistinguishedName, SignatureScheme};

use super::identity::{fingerprint, Identity};

fn provider() -> Arc<CryptoProvider> {
    Arc::new(rustls::crypto::ring::default_provider())
}

fn algorithms() -> WebPkiSupportedAlgorithms {
    rustls::crypto::ring::default_provider().signature_verification_algorithms
}

fn tls_error(e: impl std::fmt::Display) -> String {
    format!("TLS: {e}")
}

/// A worker's listener: its certificate, TLS 1.3, a client certificate asked for (and needed for anything but
/// `health`).
pub fn server_config(id: &Identity) -> Result<Arc<rustls::ServerConfig>, String> {
    let config = rustls::ServerConfig::builder_with_provider(provider())
        .with_protocol_versions(&[&rustls::version::TLS13])
        .map_err(tls_error)?
        .with_client_cert_verifier(Arc::new(AnyClientKey { algorithms: algorithms() }))
        .with_single_cert(vec![id.certificate()], id.private_key())
        .map_err(tls_error)?;
    Ok(Arc::new(config))
}

/// What a primary accepts of a worker's certificate.
#[derive(Clone, Debug)]
pub enum Pin {
    /// Only this fingerprint (a paired worker's).
    Fingerprint(String),
    /// Any, recording it (pairing: the code both screens show comes from it).
    Record(Arc<Mutex<Option<String>>>),
}

/// A primary's TLS to a worker: its own certificate presented, the worker's checked against `pin`.
pub fn client_config(id: &Identity, pin: Pin) -> Result<rustls::ClientConfig, String> {
    rustls::ClientConfig::builder_with_provider(provider())
        .with_protocol_versions(&[&rustls::version::TLS13])
        .map_err(tls_error)?
        .dangerous()
        .with_custom_certificate_verifier(Arc::new(PinnedServer { pin, algorithms: algorithms() }))
        .with_client_auth_cert(vec![id.certificate()], id.private_key())
        .map_err(tls_error)
}

/// An HTTP client for one worker, its certificate pinned (or recorded, pairing); its own deadlines are the caller's.
pub fn client(id: &Identity, pin: Pin) -> Result<reqwest::Client, String> {
    reqwest::Client::builder()
        .use_preconfigured_tls(client_config(id, pin)?)
        .connect_timeout(std::time::Duration::from_secs(5))
        // (a worker is reached at the address it was found at; its own redirects are not followed)
        .redirect(reqwest::redirect::Policy::none())
        .no_proxy()
        .build()
        .map_err(|e| e.to_string())
}

/// Accepts any client certificate whose key signs the handshake; which routes it reaches is decided per request.
#[derive(Debug)]
struct AnyClientKey {
    algorithms: WebPkiSupportedAlgorithms,
}

impl ClientCertVerifier for AnyClientKey {
    fn offer_client_auth(&self) -> bool {
        true
    }

    fn client_auth_mandatory(&self) -> bool {
        // (`health` answers a caller with none, as a browser checking the address would be)
        false
    }

    fn root_hint_subjects(&self) -> &[DistinguishedName] {
        &[]
    }

    fn verify_client_cert(&self, _end_entity: &CertificateDer<'_>, _intermediates: &[CertificateDer<'_>], _now: UnixTime) -> Result<ClientCertVerified, rustls::Error> {
        Ok(ClientCertVerified::assertion())
    }

    fn verify_tls12_signature(&self, message: &[u8], cert: &CertificateDer<'_>, dss: &DigitallySignedStruct) -> Result<HandshakeSignatureValid, rustls::Error> {
        verify_tls12_signature(message, cert, dss, &self.algorithms)
    }

    fn verify_tls13_signature(&self, message: &[u8], cert: &CertificateDer<'_>, dss: &DigitallySignedStruct) -> Result<HandshakeSignatureValid, rustls::Error> {
        verify_tls13_signature(message, cert, dss, &self.algorithms)
    }

    fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
        self.algorithms.supported_schemes()
    }
}

/// A worker's certificate checked against the pinned fingerprint (its name and dates mean nothing here: it is not
/// signed by anyone, and a pinned key is the whole of the trust).
#[derive(Debug)]
struct PinnedServer {
    pin: Pin,
    algorithms: WebPkiSupportedAlgorithms,
}

impl ServerCertVerifier for PinnedServer {
    fn verify_server_cert(&self, end_entity: &CertificateDer<'_>, _intermediates: &[CertificateDer<'_>], _name: &ServerName<'_>, _ocsp: &[u8], _now: UnixTime) -> Result<ServerCertVerified, rustls::Error> {
        let seen = fingerprint(end_entity.as_ref());
        match &self.pin {
            Pin::Fingerprint(want) if *want == seen => Ok(ServerCertVerified::assertion()),
            Pin::Fingerprint(_) => Err(rustls::Error::General("the computer answering is not the one paired (its certificate differs)".into())),
            Pin::Record(slot) => {
                *slot.lock().unwrap_or_else(|p| p.into_inner()) = Some(seen);
                Ok(ServerCertVerified::assertion())
            }
        }
    }

    fn verify_tls12_signature(&self, message: &[u8], cert: &CertificateDer<'_>, dss: &DigitallySignedStruct) -> Result<HandshakeSignatureValid, rustls::Error> {
        verify_tls12_signature(message, cert, dss, &self.algorithms)
    }

    fn verify_tls13_signature(&self, message: &[u8], cert: &CertificateDer<'_>, dss: &DigitallySignedStruct) -> Result<HandshakeSignatureValid, rustls::Error> {
        verify_tls13_signature(message, cert, dss, &self.algorithms)
    }

    fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
        self.algorithms.supported_schemes()
    }
}

/// Whether an address is on a local network (loopback, private, link-local or a unique local IPv6 address): the only
/// addresses a worker answers or a primary calls.
pub fn local_address(ip: std::net::IpAddr) -> bool {
    use std::net::IpAddr;
    match ip {
        IpAddr::V4(v4) => v4.is_loopback() || v4.is_private() || v4.is_link_local(),
        IpAddr::V6(v6) => {
            if let Some(v4) = v6.to_ipv4_mapped() {
                return local_address(IpAddr::V4(v4));
            }
            let first = v6.segments()[0];
            v6.is_loopback() || (first & 0xffc0) == 0xfe80 || (first & 0xfe00) == 0xfc00
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_local_addresses_are_local() {
        for ip in ["127.0.0.1", "10.1.2.3", "172.16.0.9", "192.168.1.20", "169.254.3.4", "::1", "fe80::1", "fd12::5", "::ffff:192.168.1.2"] {
            assert!(local_address(ip.parse().unwrap()), "{ip}");
        }
        for ip in ["8.8.8.8", "100.64.1.1", "172.32.0.1", "2001:4860::8888", "::ffff:8.8.8.8"] {
            assert!(!local_address(ip.parse().unwrap()), "{ip}");
        }
    }

    #[test]
    fn a_config_is_made_for_each_side() {
        let id = Identity::generate().unwrap();
        assert!(server_config(&id).is_ok());
        assert!(client_config(&id, Pin::Fingerprint("ab".into())).is_ok());
        assert!(client(&id, Pin::Record(Arc::default())).is_ok());
    }
}
