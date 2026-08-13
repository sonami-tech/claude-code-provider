//! Outbound HTTP timeouts shared by every provider.
//!
//! Two families, kept as separate constants:
//! - **upstream**: inference POST (chat / messages / responses), including the
//!   streamed body.
//! - **oauth**: token-endpoint POSTs only (Claude / Grok / Codex refresh).
//!
//! Provider crates must use these names. Do not restate the seconds at the
//! `Client::builder` site. Fingerprint wire values such as Claude's
//! `x-stainless-timeout: 600` are not ours; they stay in the fingerprint
//! module.

use std::time::Duration;

/// Total time for one upstream inference request, including streamed body.
///
/// Matches xAI's documented reasoning-stream timeout (3600s) and grok-shell's
/// live `inference_idle_timeout_secs`. This is a total cap, not an idle gap.
pub const UPSTREAM_REQUEST_TIMEOUT: Duration = Duration::from_secs(3600);

/// TCP/TLS connect budget for an upstream inference request.
pub const UPSTREAM_CONNECT_TIMEOUT: Duration = Duration::from_secs(15);

/// Total time for an OAuth token refresh POST. Not an inference timeout.
pub const OAUTH_REQUEST_TIMEOUT: Duration = Duration::from_secs(30);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn upstream_request_timeout_is_3600s() {
        assert_eq!(UPSTREAM_REQUEST_TIMEOUT, Duration::from_secs(3600));
    }

    #[test]
    fn families_stay_distinct() {
        assert!(UPSTREAM_REQUEST_TIMEOUT > UPSTREAM_CONNECT_TIMEOUT);
        assert!(UPSTREAM_REQUEST_TIMEOUT > OAUTH_REQUEST_TIMEOUT);
        assert_eq!(UPSTREAM_CONNECT_TIMEOUT, Duration::from_secs(15));
        assert_eq!(OAUTH_REQUEST_TIMEOUT, Duration::from_secs(30));
    }
}
