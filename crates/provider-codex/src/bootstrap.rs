//! Codex provider bootstrap: detect, version resolve, init, catalog, extras.
//!
//! Owned by provider-codex so the omni edge only registers a factory.

use std::sync::Arc;

use omni_core::{BootstrappedProvider, VersionSelector};
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

/// Bootstrap Codex from a version selector and env / CODEX_HOME.
pub fn bootstrap(selector: &VersionSelector) -> anyhow::Result<BootstrappedProvider> {
    let provider = init_provider(selector)?;
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

fn init_provider(selector: &VersionSelector) -> anyhow::Result<CodexProvider> {
    let version = omni_core::resolve_version(CodexProvider::version_catalog(), selector)
        .map_err(|e| anyhow::anyhow!("codex: cannot resolve version selector: {e}"))?;
    let selector_desc = describe_version_selector(selector, version.version);
    let provider = CodexProvider::new()
        .map_err(anyhow::Error::from)?
        .with_version(version.version)
        .map_err(anyhow::Error::from)?;
    // Transport (ChatGPT WS vs REST) is inferred per-request from CODEX_HOME
    // config/auth, matching Claude/Grok (no operator mode flag).
    // Hard-fail at launch when settings cannot be loaded (invalid CODEX_HOME, etc.).
    let source = CodexProvider::startup_source_summary()
        .map_err(|e| anyhow::anyhow!("codex: cannot load settings for enabled provider: {e}"))?;
    info!(
        version = version.version,
        selector = %selector_desc,
        source = %source,
        "initializing codex provider"
    );
    Ok(provider)
}

fn describe_version_selector(selector: &VersionSelector, chose: &str) -> String {
    match selector {
        VersionSelector::Latest => format!("latest → chose={chose}"),
        VersionSelector::Exact(v) => format!("exact={v} → chose={chose}"),
        VersionSelector::MatchSystem(v) => {
            format!("match-system detected={v} → chose={chose}")
        }
        VersionSelector::MatchSystemExact(v) => {
            format!("match-system-exact detected={v} → chose={chose}")
        }
    }
}
