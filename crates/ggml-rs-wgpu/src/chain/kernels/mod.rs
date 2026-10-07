//! The kernels' sources (WGSL) and the functions that make one for a size, in files by what they compute; [`super`]
//! names every one of them as its own.

mod attention;
mod attention_coop;
mod basic;
mod image;
mod matmul;
mod media;
mod qsa;
mod recurrent;
mod streams;

pub(crate) use attention::*;
pub(in crate::chain) use attention_coop::*;
pub(in crate::chain) use basic::*;
pub(in crate::chain) use image::*;
pub(in crate::chain) use matmul::*;
pub(in crate::chain) use media::*;
pub(in crate::chain) use qsa::*;
pub(in crate::chain) use recurrent::*;
pub(in crate::chain) use streams::*;
