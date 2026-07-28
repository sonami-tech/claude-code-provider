//! Claude provider bootstrap: detect, version/profile resolve, init, catalog, extras.
//!
//! Owned by provider-claude so the omni edge only registers a factory and never
//! resolves `FingerprintProfile` or hand-writes init bodies.

use std::sync::Arc;

use omni_common::env_nonempty;
use omni_core::{BootstrappedProvider, VersionSelector};
use tracing::{info, warn};

use crate::ClaudeProvider;
use crate::credentials::Credentials;
use crate::fingerprint::{
    FingerprintProfile, default_profile, resolve_profile, valid_profile_selectors,
};

/// Provider id for routing / stats / registration.
pub const PROVIDER_ID: &str = "claude";

/// Whether Claude is auto-detectable from env / credentials.
pub fn detected() -> bool {
    ClaudeProvider::detected()
}

/// Short reason claude is auto-detectable (for startup logs).
pub fn detection_source() -> Option<String> {
    ClaudeProvider::detection_source()
}

/// Claude rejects all `provider_extras` keys (no extras surface).
pub fn extras_allowed(_key: &str) -> bool {
    false
}

/// Bootstrap Claude from a version selector and env (custom gateway or CLI creds).
pub fn bootstrap(selector: &VersionSelector) -> anyhow::Result<BootstrappedProvider> {
    let provider = init_provider(selector)?;
    let models = model_catalog_values(provider.profile())?;
    let aliases = model_aliases(provider.profile());
    let provider = Arc::new(provider);
    Ok(BootstrappedProvider {
        provider: provider.clone(),
        anthropic_native: Some(provider),
        models,
        aliases,
        extras_allowed,
    })
}

/// Build a bootstrapped entry around an already-constructed provider (tests).
pub fn from_provider(provider: ClaudeProvider) -> anyhow::Result<BootstrappedProvider> {
    let models = model_catalog_values(provider.profile())?;
    let aliases = model_aliases(provider.profile());
    let provider = Arc::new(provider);
    Ok(BootstrappedProvider {
        provider: provider.clone(),
        anthropic_native: Some(provider),
        models,
        aliases,
        extras_allowed,
    })
}

fn model_catalog_values(profile: &FingerprintProfile) -> anyhow::Result<Vec<serde_json::Value>> {
    profile
        .models_list()
        .into_iter()
        .map(|model| {
            serde_json::to_value(model)
                .map_err(|e| anyhow::anyhow!("claude model catalog serialize: {e}"))
        })
        .collect()
}

fn model_aliases(profile: &FingerprintProfile) -> Vec<(String, String)> {
    let mut out = Vec::new();
    for model in profile.models {
        out.push((model.canonical.to_string(), model.canonical.to_string()));
        out.push((model.cli_name.to_string(), model.canonical.to_string()));
        for alias in model.aliases {
            out.push((alias.to_string(), model.canonical.to_string()));
        }
    }
    for model in profile.model_wire_overrides {
        out.push((model.model.to_string(), model.model.to_string()));
    }
    out
}

fn init_provider(selector: &VersionSelector) -> anyhow::Result<ClaudeProvider> {
    let profile = resolve_claude_profile(selector)?;
    let selector_desc = describe_version_selector(selector, profile.claude_cli_version);
    if let Some(base_url) = env_nonempty("OMNI_CLAUDE_BASE_URL") {
        let authorization_bearer = env_nonempty("OMNI_CLAUDE_AUTH_TOKEN")
            .is_some()
            .then(|| "OMNI_CLAUDE_AUTH_TOKEN".to_string());
        let api_key = env_nonempty("OMNI_CLAUDE_API_KEY")
            .is_some()
            .then(|| "OMNI_CLAUDE_API_KEY".to_string());
        let custom_headers = std::env::var_os("OMNI_CLAUDE_CUSTOM_HEADERS")
            .is_some()
            .then(|| "OMNI_CLAUDE_CUSTOM_HEADERS".to_string());
        let auth = describe_env_auth_winner(&[
            ("OMNI_CLAUDE_AUTH_TOKEN", "bearer"),
            ("OMNI_CLAUDE_API_KEY", "x-api-key"),
            ("OMNI_CLAUDE_CUSTOM_HEADERS", "custom-headers"),
        ]);
        info!(
            profile = profile.claude_cli_version,
            selector = %selector_desc,
            base_url = %base_url,
            auth = %auth,
            "initializing claude provider (custom gateway via OMNI_CLAUDE_BASE_URL)"
        );
        return ClaudeProvider::new_for_custom_gateway_env(
            profile,
            base_url,
            authorization_bearer,
            api_key,
            custom_headers,
        )
        .map_err(anyhow::Error::from);
    }

    if let Some(base_url) = env_nonempty("ANTHROPIC_BASE_URL") {
        let authorization_bearer = env_nonempty("ANTHROPIC_AUTH_TOKEN")
            .is_some()
            .then(|| "ANTHROPIC_AUTH_TOKEN".to_string());
        let api_key = env_nonempty("ANTHROPIC_API_KEY")
            .is_some()
            .then(|| "ANTHROPIC_API_KEY".to_string());
        let custom_headers = std::env::var_os("ANTHROPIC_CUSTOM_HEADERS")
            .is_some()
            .then(|| "ANTHROPIC_CUSTOM_HEADERS".to_string());
        let auth = describe_env_auth_winner(&[
            ("ANTHROPIC_AUTH_TOKEN", "bearer"),
            ("ANTHROPIC_API_KEY", "x-api-key"),
            ("ANTHROPIC_CUSTOM_HEADERS", "custom-headers"),
        ]);
        info!(
            profile = profile.claude_cli_version,
            selector = %selector_desc,
            base_url = %base_url,
            auth = %auth,
            "initializing claude provider (custom gateway via ANTHROPIC_BASE_URL)"
        );
        return ClaudeProvider::new_for_custom_gateway_env(
            profile,
            base_url,
            authorization_bearer,
            api_key,
            custom_headers,
        )
        .map_err(anyhow::Error::from);
    }

    let creds_path = Credentials::default_path();
    let via = if std::env::var_os("CLAUDE_CREDENTIALS_PATH").is_some() {
        " via CLAUDE_CREDENTIALS_PATH"
    } else {
        ""
    };
    let path_disp = display_path_for_log(&creds_path);
    if !creds_path.is_file() {
        anyhow::bail!(
            "claude: credentials file missing at {path_disp}{via}; cannot start with claude enabled"
        );
    }
    Credentials::load_fresh(&creds_path).with_context(|| {
        format!("claude: credentials at {path_disp}{via} are unreadable or invalid")
    })?;
    info!(
        profile = profile.claude_cli_version,
        selector = %selector_desc,
        auth = %format!("auth={path_disp}{via} (present)"),
        "initializing claude provider"
    );
    ClaudeProvider::new_with_profile(profile).map_err(anyhow::Error::from)
}

fn resolve_claude_profile(
    selector: &VersionSelector,
) -> anyhow::Result<&'static FingerprintProfile> {
    match selector {
        VersionSelector::Latest => Ok(default_profile()),
        VersionSelector::Exact(v) | VersionSelector::MatchSystemExact(v) => {
            resolve_profile(v).ok_or_else(|| {
                anyhow::anyhow!(
                    "claude: no fingerprint profile matches version {v:?} (exact-or-fail); known selectors: {}",
                    valid_profile_selectors()
                )
            })
        }
        VersionSelector::MatchSystem(v) => Ok(resolve_profile(v).unwrap_or_else(|| {
            let newest = default_profile();
            warn!(
                installed = %v,
                chose = newest.claude_cli_version,
                "claude: no exact profile for installed version; using newest (default) profile"
            );
            newest
        })),
    }
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

fn describe_env_auth_winner(candidates: &[(&str, &str)]) -> String {
    for (env_key, kind) in candidates {
        let present = if env_key.contains("HEADERS") {
            std::env::var_os(env_key).is_some()
        } else {
            env_nonempty(env_key).is_some()
        };
        if present {
            return format!("auth=env:{env_key} ({kind}, set)");
        }
    }
    "auth=none (no gateway auth env set)".into()
}

fn display_path_for_log(path: &std::path::Path) -> String {
    std::fs::canonicalize(path)
        .unwrap_or_else(|_| path.to_path_buf())
        .display()
        .to_string()
}

use anyhow::Context;
