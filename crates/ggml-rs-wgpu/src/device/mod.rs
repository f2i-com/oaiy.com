//! The device: its own functions ([`gpu`]: pipelines, bind groups, pools of scratch, waits, read-backs), its queue
//! ([`feed`]), writes to its buffers ([`upload`]), the memory there is ([`memory`]), and the watchdog on its waits
//! ([`watch`]).

pub(crate) mod feed;
pub(crate) mod gpu;
pub mod memory;
pub(crate) mod upload;
pub mod watch;
