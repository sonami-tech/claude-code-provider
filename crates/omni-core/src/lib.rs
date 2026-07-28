//! omni-core
//! Canonical types and traits for pluggable frontends and providers.
//! The "connect anything to anything" glue. Minimal and stable.

pub mod bootstrap;
pub mod canonical;
pub mod native_anthropic;
pub mod traits;
pub mod version;

pub use bootstrap::*;
pub use canonical::*;
pub use native_anthropic::*;
pub use traits::*;
pub use version::*;
