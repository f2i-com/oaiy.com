//! The importance-weighted formats: IQ4_NL and IQ4_XS (a non-linear 4-bit codebook), and the grid formats ([`iq`]:
//! IQ1_S to IQ3_S, groups of weights looked up in ggml's tables, [`iq_tables`]).

pub mod iq;
pub mod iq4_nl;
pub mod iq4_xs;
pub mod iq_tables;
