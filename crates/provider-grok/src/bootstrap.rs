//! Grok provider bootstrap: detect, init, catalog, extras.
//!
//! Owned by provider-grok so the omni edge only registers a factory.
//! One active pin only (issue #12).

use std::sync::Arc;

use anyhow::Context;
use omni_common::env_nonempty;
use omni_core::BootstrappedProvider;
use tracing::info;

use crate::credentials::{GrokCredentials, GrokCredentialsError};
use crate::{GrokProvider, grok_extra_allowed};

/// Provider id for routing / stats / registration.
pub const PROVIDER_ID: &str = "grok";

pub fn detected() -> bool {
    GrokProvider::detected()
}

pub fn detection_source() -> Option<String> {
    GrokProvider::detection_source()
}

pub fn extras_allowed(key: &str) -> bool {
    grok_extra_allowed(key)
}

/// Bootstrap Grok from env (custom endpoint or CLI path) on the active pin.
pub fn bootstrap() -> anyhow::Result<BootstrappedProvider> {
    let provider = init_provider()?;
    from_provider(provider)
}

/// Build a bootstrapped entry around an already-constructed provider (tests).
pub fn from_provider(provider: GrokProvider) -> anyhow::Result<BootstrappedProvider> {
    let models = provider
        .models_list()
        .into_iter()
        .map(|model| {
            serde_json::to_value(model)
                .map_err(|e| anyhow::anyhow!("grok model catalog serialize: {e}"))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    let aliases = provider
        .model_aliases()
        .into_iter()
        .map(|(a, c)| (a.to_string(), c.to_string()))
        .collect();
    Ok(BootstrappedProvider {
        provider: Arc::new(provider),
        anthropic_native: None,
        models,
        aliases,
        extras_allowed,
    })
}

fn init_provider() -> anyhow::Result<GrokProvider> {
    let version = GrokProvider::pinned_version();
    let provider = GrokProvider::new(None).map_err(anyhow::Error::from)?;
    if let Some(base_url) = env_nonempty("OMNI_GROK_BASE_URL") {
        let auth = describe_env_auth_winner(&[
            ("OMNI_GROK_AUTH_TOKEN", "bearer-token"),
            ("OMNI_GROK_API_KEY", "api-key"),
            ("OMNI_GROK_CUSTOM_HEADERS", "custom-headers"),
        ]);
        info!(
            version,
            base_url = %base_url,
            auth = %auth,
            "initializing grok provider (custom endpoint via OMNI_GROK_BASE_URL)"
        );
        return Ok(provider.with_custom_auth_env(
            base_url,
            Some("OMNI_GROK_AUTH_TOKEN".into()),
            Some("OMNI_GROK_API_KEY".into()),
            Some("OMNI_GROK_CUSTOM_HEADERS".into()),
        ));
    }

    if let Some(base_url) = env_nonempty("GROK_MODELS_BASE_URL") {
        let auth = if env_nonempty("XAI_API_KEY").is_some() {
            "auth=env:XAI_API_KEY (set)"
        } else {
            "auth=none (XAI_API_KEY unset)"
        };
        info!(
            version,
            base_url = %base_url,
            auth,
            "initializing grok provider (custom endpoint via GROK_MODELS_BASE_URL)"
        );
        return Ok(provider.with_base_url(base_url).with_custom_auth(
            None,
            Some("XAI_API_KEY".into()),
            vec![],
        ));
    }

    validate_cli_credentials_at_startup()?;
    let auth = GrokCredentials::describe_cli_auth_source();
    info!(
        version,
        base_url = "https://cli-chat-proxy.grok.com",
        auth = %auth,
        "initializing grok provider (CLI path; credentials re-read per request)"
    );
    Ok(provider)
}

fn validate_cli_credentials_at_startup() -> anyhow::Result<()> {
    if let Some(p) = std::env::var_os("XAI_CREDENTIALS_PATH") {
        let path = std::path::PathBuf::from(p);
        if !path.is_file() {
            anyhow::bail!(
                "grok: XAI_CREDENTIALS_PATH={} is missing or not a file; cannot start with grok enabled",
                path.display()
            );
        }
        GrokCredentials::load_fresh(&path).with_context(|| {
            format!(
                "grok: credentials at {} (XAI_CREDENTIALS_PATH) are unreadable or invalid",
                path.display()
            )
        })?;
        return Ok(());
    }

    if let Some(cli_path) = GrokCredentials::grok_cli_path()
        && cli_path.is_file()
    {
        match GrokCredentials::load_fresh(&cli_path) {
            Ok(_) => return Ok(()),
            Err(GrokCredentialsError::MissingToken) => {
                // Fall through to static key, same as runtime CLI resolve.
            }
            Err(e) => {
                anyhow::bail!(
                    "grok: credentials at {} are unreadable or invalid: {e}",
                    cli_path.display()
                );
            }
        }
    }

    let static_path = GrokCredentials::default_path();
    if static_path.is_file() {
        GrokCredentials::load_fresh(&static_path).with_context(|| {
            format!(
                "grok: credentials at {} are unreadable or invalid",
                static_path.display()
            )
        })?;
        return Ok(());
    }

    let auth = GrokCredentials::describe_cli_auth_source();
    anyhow::bail!(
        "grok: no credentials file found for CLI path ({auth}); cannot start with grok enabled"
    )
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
