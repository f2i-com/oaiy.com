//! A person's age from a camera picture, for an age check: a face detector finds the faces, and an age model reads the
//! one face there should be, with the picture around it as the body. Loaded for one job and gone with it.
pub mod mivolo;
pub mod prep;

#[cfg(test)]
mod tests;
