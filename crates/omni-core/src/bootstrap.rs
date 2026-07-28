//! Provider bootstrap result: uniform shape returned by each provider crate.
//!
//! The edge loops registered factories and inserts these into its provider map.
//! Detect / init / catalog / extras policy live in provider crates; the edge only
//! wires HTTP, auth, stats, and dispatch.

use std::sync::Arc;

use crate::native_anthropic::AnthropicNativeSurface;
use crate::traits::LlmProvider;

/// Product of a provider factory after version selection and startup validation.
pub struct BootstrappedProvider {
    /// Canonical path (`LlmProvider`).
    pub provider: Arc<dyn LlmProvider>,
    /// Optional native Anthropic surface (Claude). None for translated-only backends.
    pub anthropic_native: Option<Arc<dyn AnthropicNativeSurface>>,
    /// Model catalog entries for `/v1/models` (without `owned_by`; edge stamps it).
    pub models: Vec<serde_json::Value>,
    /// Alias → canonical model id pairs for routing.
    pub aliases: Vec<(String, String)>,
    /// Allowlist for `provider_extras` keys. Reject when false.
    pub extras_allowed: fn(&str) -> bool,
}
