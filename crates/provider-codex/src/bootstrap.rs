//! Codex provider bootstrap: detect, init, catalog, extras.
//!
//! Owned by provider-codex so the omni edge only registers a factory.
//! One active pin only (issue #12).

use std::sync::Arc;

use omni_core::BootstrappedProvider;
use tracing::info;

use crate::{CodexProvider, codex_extra_allowed};

/// Provider id for routing / stats / registration.
pub const PROVIDER_ID: &str = "codex";

pub fn detected() -> bool {
    CodexProvider::detected()
}

pub fn detection_source() -> Option<String> {
    CodexProvider::detection_source()
}

pub fn extras_allowed(key: &str) -> bool {
    codex_extra_allowed(key)
}

/// Bootstrap Codex from env / CODEX_HOME on the active pin.
pub fn bootstrap() -> anyhow::Result<BootstrappedProvider> {
    let provider = init_provider()?;
    from_provider(provider)
}

/// Build a bootstrapped entry around an already-constructed provider (tests).
pub fn from_provider(provider: CodexProvider) -> anyhow::Result<BootstrappedProvider> {
    let models = provider
        .models_list()
        .into_iter()
        .map(|model| {
            serde_json::to_value(model)
                .map_err(|e| anyhow::anyhow!("codex model catalog serialize: {e}"))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    let aliases = provider.model_aliases();
    Ok(BootstrappedProvider {
        provider: Arc::new(provider),
        anthropic_native: None,
        models,
        aliases,
        extras_allowed,
    })
}

fn init_provider() -> anyhow::Result<CodexProvider> {
    let version = CodexProvider::pinned_version();
    let provider = CodexProvider::new().map_err(anyhow::Error::from)?;
    // Transport (ChatGPT WS vs REST) is inferred per-request from CODEX_HOME
    // config/auth, matching Claude/Grok (no operator mode flag).
    // Hard-fail at launch when settings cannot be loaded (invalid CODEX_HOME, etc.).
    let source = CodexProvider::startup_source_summary()
        .map_err(|e| anyhow::anyhow!("codex: cannot load settings for enabled provider: {e}"))?;
    info!(
        version,
        source = %source,
        "initializing codex provider"
    );
    Ok(provider)
}
