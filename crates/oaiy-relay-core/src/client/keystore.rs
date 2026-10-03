//! [`SecretStore`] over `oaiy-keystore`'s `KeyStore` (feature `keystore`): the desktop's named-secret store, with its rules (`Ok(None)` is "never stored" and an error is
//! "could not read", the name is bound into the blob, a write is atomic) carried over unchanged. A name this crate uses is a valid keystore name (`relay.token`).

use oaiy_keystore::{KeyError, KeyStore, Name};
use zeroize::Zeroizing;

use super::store::{SecretStore, StoreError};

/// A [`SecretStore`] that is a keystore (`oaiy_keystore::open(&data_dir, ProviderChoice::from_env()?)` returns the box).
pub struct KeystoreSecrets(pub Box<dyn KeyStore>);

fn name(n: &str) -> Result<Name, StoreError> {
    Name::new(n).map_err(|_| StoreError("not a valid secret name".into()))
}

fn err(e: KeyError) -> StoreError {
    StoreError(e.to_string())
}

impl SecretStore for KeystoreSecrets {
    fn get(&self, n: &str) -> Result<Option<Zeroizing<Vec<u8>>>, StoreError> {
        self.0.get(&name(n)?).map_err(err)
    }

    fn put(&self, n: &str, value: &[u8]) -> Result<(), StoreError> {
        self.0.put(&name(n)?, value).map(|_| ()).map_err(err)
    }

    fn delete(&self, n: &str) -> Result<(), StoreError> {
        self.0.delete(&name(n)?).map(|_| ()).map_err(err)
    }
}
