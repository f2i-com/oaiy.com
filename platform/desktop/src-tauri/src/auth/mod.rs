//! The access model.
//!
//! One model for everything that reaches the local API: every non-public route is classified into a
//! scope (or one of a few small special classes) in [`routes`], and a route with no classification is
//! refused. See the route table's own documentation for the classes.

pub mod export;
pub mod routes;

#[cfg(test)]
mod route_coverage;
