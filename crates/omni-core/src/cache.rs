//! Internal prompt-cache intent.
//!
//! Inbound parsers store official cache fields here. Providers emit each
//! backend's official cache fields from this shape. Canonical types stay
//! internal; Omni does not invent a client-facing cache dialect.

use serde::{Deserialize, Serialize};

/// Request-level cache intent lifted from official inbound fields.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
pub struct CanonicalCacheIntent {
    /// Chat/Responses `prompt_cache_key`, or Chat header `x-grok-conv-id`
    /// when no body key is present.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub routing_identity: Option<String>,
    /// OpenAI `prompt_cache_options.mode`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mode: Option<CanonicalCacheMode>,
    /// Requested lifetime when the client sent an official TTL.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ttl: Option<CanonicalCacheTtl>,
    /// OpenAI `prompt_cache_retention`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub legacy_retention: Option<CanonicalCacheRetention>,
    /// Anthropic top-level automatic `cache_control`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub automatic: Option<CanonicalCacheMark>,
}

impl CanonicalCacheIntent {
    pub fn is_empty(&self) -> bool {
        self.routing_identity.is_none()
            && self.mode.is_none()
            && self.ttl.is_none()
            && self.legacy_retention.is_none()
            && self.automatic.is_none()
    }

    /// Duration used for clamp-down. Prefer an explicit TTL; otherwise map
    /// legacy retention. `None` means the client did not ask for a lifetime.
    pub fn requested_duration_secs(&self) -> Option<u64> {
        if let Some(ttl) = self.ttl {
            return Some(ttl.duration_secs());
        }
        self.legacy_retention
            .map(CanonicalCacheRetention::duration_secs)
    }
}

/// OpenAI `prompt_cache_options.mode`.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CanonicalCacheMode {
    Implicit,
    Explicit,
}

/// Discrete official TTL values (not a continuous range).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum CanonicalCacheTtl {
    #[serde(rename = "5m")]
    FiveMinutes,
    #[serde(rename = "30m")]
    ThirtyMinutes,
    #[serde(rename = "1h")]
    OneHour,
}

impl CanonicalCacheTtl {
    pub const CLAUDE: [Self; 2] = [Self::FiveMinutes, Self::OneHour];
    pub const CODEX_GPT56: [Self; 1] = [Self::ThirtyMinutes];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::FiveMinutes => "5m",
            Self::ThirtyMinutes => "30m",
            Self::OneHour => "1h",
        }
    }

    pub fn duration_secs(self) -> u64 {
        match self {
            Self::FiveMinutes => 5 * 60,
            Self::ThirtyMinutes => 30 * 60,
            Self::OneHour => 60 * 60,
        }
    }

    /// Longest offered value that is ≤ `self`. No round-up.
    pub fn clamp_down(self, offered: &[Self]) -> Option<Self> {
        clamp_duration(self.duration_secs(), offered.iter().copied())
    }
}

/// OpenAI `prompt_cache_retention`.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum CanonicalCacheRetention {
    #[serde(rename = "in_memory")]
    InMemory,
    #[serde(rename = "24h")]
    TwentyFourHours,
}

impl CanonicalCacheRetention {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::InMemory => "in_memory",
            Self::TwentyFourHours => "24h",
        }
    }

    /// Conservative duration for clamp-down. `in_memory` is the short
    /// (~5 minute) end of the documented range so we never retain longer
    /// than asked. `24h` is 24 hours.
    pub fn duration_secs(self) -> u64 {
        match self {
            Self::InMemory => 5 * 60,
            Self::TwentyFourHours => 24 * 60 * 60,
        }
    }
}

/// Block- or tool-level cache breakpoint. `ttl: None` still marks the block.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct CanonicalCacheMark {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ttl: Option<CanonicalCacheTtl>,
}

impl CanonicalCacheMark {
    pub fn breakpoint() -> Self {
        Self { ttl: None }
    }
}

/// Pick the longest offered TTL whose duration is ≤ `requested_secs`.
pub fn clamp_duration(
    requested_secs: u64,
    offered: impl IntoIterator<Item = CanonicalCacheTtl>,
) -> Option<CanonicalCacheTtl> {
    offered
        .into_iter()
        .filter(|ttl| ttl.duration_secs() <= requested_secs)
        .max_by_key(|ttl| ttl.duration_secs())
}

/// Pick the longest offered retention whose duration is ≤ `requested_secs`.
pub fn clamp_retention(
    requested_secs: u64,
    offered: impl IntoIterator<Item = CanonicalCacheRetention>,
) -> Option<CanonicalCacheRetention> {
    offered
        .into_iter()
        .filter(|ret| ret.duration_secs() <= requested_secs)
        .max_by_key(|ret| ret.duration_secs())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clamp_down_picks_longest_offered_value_that_is_not_longer() {
        // WHY: docs/cache-translation.md locked examples. These are discrete
        // enums; 1h against {5m,30m} must become 30m, never 5m.
        assert_eq!(
            CanonicalCacheTtl::OneHour.clamp_down(&[
                CanonicalCacheTtl::FiveMinutes,
                CanonicalCacheTtl::ThirtyMinutes
            ]),
            Some(CanonicalCacheTtl::ThirtyMinutes)
        );
        assert_eq!(
            CanonicalCacheTtl::ThirtyMinutes.clamp_down(&CanonicalCacheTtl::CLAUDE),
            Some(CanonicalCacheTtl::FiveMinutes)
        );
        assert_eq!(
            CanonicalCacheTtl::FiveMinutes.clamp_down(&CanonicalCacheTtl::CODEX_GPT56),
            None
        );
    }
}
