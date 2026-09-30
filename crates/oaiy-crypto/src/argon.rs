//! `argon2id13` (design 4.1.1) with the cost ceiling of design 4.1.2.
//!
//! Argon2id v1.3, one lane, a 16-byte salt and a 32-byte output: libsodium's `crypto_pwhash(32, ..., ALG_ARGON2ID13)`, which is what
//! FormLogic's browser worker and PHP run. The cost is a value the server sends, and a hostile server chooses it, so the bounds
//! are checked **before anything is derived or allocated**: operations 3 to 10, memory 64 MiB to 256 MiB (a whole number of
//! KiB, as the design's values are; libsodium would round down), salt exactly 16 bytes. Anything else is
//! `Error::KdfParamsOutOfRange`. FormLogic's own vector with 2 passes and 8 MiB is below the floor and is checked in the unit tests
//! through a private function that has no bounds.
//!
//! The memory Argon2 works in holds values derived from the password, and the `argon2` crate's own `hash_password_into` leaves it
//! to the allocator. Here the blocks are allocated by this module and overwritten before they are released, on every path.

use argon2::{Algorithm, Argon2, Block, Params as ArgonParams, Version};
use zeroize::Zeroize;

use crate::error::Error;
use crate::zeroize::Secret;

/// The fewest passes.
pub const OPS_MIN: u64 = 3;
/// The most passes.
pub const OPS_MAX: u64 = 10;
/// The least memory, in bytes (64 MiB).
pub const MEM_MIN: u64 = 64 * 1024 * 1024;
/// The most memory, in bytes (256 MiB).
pub const MEM_MAX: u64 = 256 * 1024 * 1024;
/// The salt length.
pub const SALT_LEN: usize = 16;
/// The output length.
pub const OUTPUT_LEN: usize = 32;

/// Argon2id parameters that are inside the bounds. There is no way to make one that is not.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Params {
    ops: u32,
    mem_kib: u32,
}

impl Params {
    /// The design's default and the phrase wrapper's: 3 passes over 64 MiB.
    pub const DEFAULT: Params = Params { ops: 3, mem_kib: 64 * 1024 };

    /// Checks the bounds. `ops` in 3..=10; `mem_bytes` in 64 MiB..=256 MiB and a multiple of 1024. The arguments are `u64` so
    /// that a number read from a server's JSON cannot wrap before it is compared.
    pub fn new(ops: u64, mem_bytes: u64) -> Result<Params, Error> {
        if !(OPS_MIN..=OPS_MAX).contains(&ops) || !(MEM_MIN..=MEM_MAX).contains(&mem_bytes) || !mem_bytes.is_multiple_of(1024) {
            return Err(Error::KdfParamsOutOfRange);
        }
        Ok(Params { ops: ops as u32, mem_kib: (mem_bytes / 1024) as u32 })
    }

    /// Passes.
    pub const fn ops(&self) -> u32 {
        self.ops
    }

    /// Memory in bytes.
    pub const fn mem_bytes(&self) -> u64 {
        self.mem_kib as u64 * 1024
    }
}

/// Argon2id v1.3 of `password` with a 16-byte `salt`. Every parameter is checked first: a salt that is not 16 bytes is
/// `KdfParamsOutOfRange` as well, and nothing is allocated for a refusal.
pub fn argon2id13(password: &[u8], salt: &[u8], ops: u64, mem_bytes: u64) -> Result<Secret<32>, Error> {
    let params = Params::new(ops, mem_bytes)?;
    let salt: &[u8; SALT_LEN] = salt.try_into().map_err(|_| Error::KdfParamsOutOfRange)?;
    derive(password, salt, params)
}

/// The same, with parameters already checked.
pub fn derive(password: &[u8], salt: &[u8; SALT_LEN], params: Params) -> Result<Secret<32>, Error> {
    let count = params.mem_kib as usize;
    let mut storage: Vec<Block> = Vec::new();
    storage.try_reserve_exact(count).map_err(|_| Error::Memory)?;
    storage.resize(count, Block::default());
    let result = derive_in(password, salt, params.ops, params.mem_kib, &mut storage);
    storage.zeroize();
    result
}

/// Wipes the blocks when it goes out of scope, on success, on an error and on a panic.
struct WipeOnDrop<'a>(&'a mut [Block]);

impl Drop for WipeOnDrop<'_> {
    fn drop(&mut self) {
        wipe(self.0);
    }
}

fn wipe(blocks: &mut [Block]) {
    for block in blocks.iter_mut() {
        block.zeroize();
    }
}

/// Runs Argon2id in `storage` (which must hold `mem_kib` blocks) and wipes it. No bounds are checked here.
fn derive_in(password: &[u8], salt: &[u8; SALT_LEN], ops: u32, mem_kib: u32, storage: &mut [Block]) -> Result<Secret<32>, Error> {
    #[cfg(test)]
    CALLS.with(|calls| calls.set(calls.get() + 1));
    let guard = WipeOnDrop(storage);
    let params = ArgonParams::new(mem_kib, ops, 1, Some(OUTPUT_LEN)).map_err(|_| Error::KdfParamsOutOfRange)?;
    let argon = Argon2::new(Algorithm::Argon2id, Version::V0x13, params);
    let mut out = [0u8; OUTPUT_LEN];
    let hashed = argon.hash_password_into_with_memory(password, salt, &mut out, &mut *guard.0);
    let secret = Secret::new(out);
    out.zeroize();
    hashed.map_err(|_| Error::InvalidLength("password"))?;
    Ok(secret)
}

#[cfg(test)]
thread_local! {
    /// How many times this thread ran Argon2: the tests of "checksum before KDF" and of the bounds read it.
    static CALLS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// How many derivations this thread has run (tests only).
#[cfg(test)]
pub(crate) fn calls() -> u64 {
    CALLS.with(std::cell::Cell::get)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn is_zero(block: &Block) -> bool {
        block.as_ref().iter().all(|word| *word == 0)
    }

    fn unhex(text: &str) -> Vec<u8> {
        (0..text.len() / 2).map(|i| u8::from_str_radix(&text[2 * i..2 * i + 2], 16).unwrap()).collect()
    }

    /// The raw function with no bounds: parameters as libsodium takes them, memory in bytes.
    fn raw(password: &[u8], salt: &[u8], ops: u32, mem_bytes: u32) -> Vec<u8> {
        let salt: &[u8; 16] = salt.try_into().unwrap();
        let count = (mem_bytes / 1024) as usize;
        let mut storage = vec![Block::default(); count];
        let out = derive_in(password, salt, ops, mem_bytes / 1024, &mut storage).unwrap();
        out.expose().to_vec()
    }

    /// FormLogic's committed Argon2id vector 1 (`docs/contracts/e2ee-envelope-vectors.json`): 2 passes, 8 MiB, below the floor
    /// of the public function, so it is checked here, through the function that has no bounds. Vector 2 (3 passes, 64 MiB, the emoji)
    /// is in `tests/formlogic_vectors.rs` through the public one.
    #[test]
    fn formlogic_vector_1_below_the_floor_through_the_raw_function() {
        let out = raw(b"formlogic vector passphrase", &unhex("000102030405060708090a0b0c0d0e0f"), 2, 8 * 1024 * 1024);
        assert_eq!(out, unhex("b1558ace0a83ac40470da20fa48cad75c57fcb36b8fb3adb8480aa9584ad7032"));
        // and the public function refuses exactly those parameters
        assert_eq!(
            argon2id13(b"formlogic vector passphrase", &unhex("000102030405060708090a0b0c0d0e0f"), 2, 8 * 1024 * 1024).unwrap_err(),
            Error::KdfParamsOutOfRange
        );
    }

    /// RFC 9106 section 5.3, Argon2id: 32 KiB, 3 passes, 4 lanes, a secret and associated data. The public function has one lane and
    /// no secret, so this runs the argon2 crate directly, with this module's memory handling: it shows that the primitive under
    /// `argon2id13` is Argon2id v1.3 as the RFC defines it, on the RFC's own vector.
    #[test]
    fn rfc_9106_argon2id_vector() {
        let vector: serde_json::Value = serde_json::from_str(include_str!("../tests/vectors/public-vectors.json")).unwrap();
        let v = &vector["argon2id_rfc9106"];
        let text = |key: &str| unhex(v[key].as_str().unwrap());
        let mut builder = argon2::ParamsBuilder::new();
        builder
            .m_cost(v["m_cost_kib"].as_u64().unwrap() as u32)
            .t_cost(v["t_cost"].as_u64().unwrap() as u32)
            .p_cost(v["lanes"].as_u64().unwrap() as u32)
            .output_len(v["tag_len"].as_u64().unwrap() as usize)
            .data(argon2::AssociatedData::new(&text("ad")).unwrap());
        let params = builder.build().unwrap();
        let secret = text("secret");
        let argon = Argon2::new_with_secret(&secret, Algorithm::Argon2id, Version::V0x13, params.clone()).unwrap();
        let mut storage = vec![Block::default(); params.block_count()];
        let mut out = [0u8; 32];
        argon.hash_password_into_with_memory(&text("password"), &text("salt"), &mut out, &mut storage[..]).unwrap();
        assert_eq!(out.to_vec(), text("tag"));
    }

    /// Design 4.3 step 8 ("zeroize ... the Argon2 block"): the memory Argon2 fills holds password-derived data, and the guard wipes it.
    #[test]
    fn the_argon2_blocks_are_wiped_when_the_guard_drops() {
        let salt = [7u8; 16];
        let mut storage = vec![Block::default(); 256];
        // Fill the memory as the algorithm does, without the guard, to show the test is not vacuous.
        let params = ArgonParams::new(256, 3, 1, Some(32)).unwrap();
        let argon = Argon2::new(Algorithm::Argon2id, Version::V0x13, params);
        argon.fill_memory(b"a password", &salt, &mut storage[..]).unwrap();
        assert!(storage.iter().all(|b| !is_zero(b)), "every block holds derived values after a fill");
        {
            let _guard = WipeOnDrop(&mut storage[..]);
        }
        assert!(storage.iter().all(is_zero), "the guard wiped every block");
    }

    /// `derive_in` wipes on its own: after it returns, every block it was given is zero.
    #[test]
    fn derive_in_leaves_the_storage_zeroed() {
        let mut storage = vec![Block::default(); 256];
        let out = derive_in(b"pw", &[1u8; 16], 3, 256, &mut storage).unwrap();
        assert_eq!(out.expose().len(), 32);
        assert!(storage.iter().all(is_zero));
    }

    /// Nothing derives for parameters outside the bounds: the counter of derivations does not move.
    #[test]
    fn out_of_range_parameters_never_reach_the_derivation() {
        let before = calls();
        let salt = [0u8; 16];
        for (ops, mem) in [
            (2u64, MEM_MIN),
            (11, MEM_MIN),
            (3, MEM_MIN - 1024),
            (3, MEM_MAX + 1024),
            (3, MEM_MIN + 1),
            (0, 0),
            (u64::MAX, u64::MAX),
            (3, u64::from(u32::MAX) * 1024),
        ] {
            assert_eq!(argon2id13(b"pw", &salt, ops, mem).unwrap_err(), Error::KdfParamsOutOfRange, "{ops} {mem}");
        }
        for len in [0usize, 1, 15, 17, 32] {
            assert_eq!(argon2id13(b"pw", &vec![0u8; len], 3, MEM_MIN).unwrap_err(), Error::KdfParamsOutOfRange, "salt {len}");
        }
        assert_eq!(calls(), before);
    }

    #[test]
    fn the_bounds_are_inclusive() {
        assert!(Params::new(3, MEM_MIN).is_ok());
        assert!(Params::new(10, MEM_MAX).is_ok());
        assert!(Params::new(3, MEM_MAX).is_ok());
        assert!(Params::new(10, MEM_MIN).is_ok());
        assert_eq!(Params::DEFAULT, Params::new(3, 64 * 1024 * 1024).unwrap());
    }
}
