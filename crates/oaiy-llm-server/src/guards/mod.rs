//! What keeps a reply in bounds without reading the user's text: degenerate repetition ([`repetition`]), the tool-call
//! phase's precision ([`tool_phase`]), and a resident reviewer of tool calls ([`observer`]).

pub mod observer;
pub(crate) mod repetition;
pub(crate) mod tool_phase;
