//! Build the outbound header set for api.anthropic.com requests, mimicking
//! the claude CLI wire fingerprint.
//!
//! **CRITICAL INVARIANT (Claude Code fingerprint exactness):**
//! The single active Claude Code pin must reproduce that version's wire
//! fingerprint **byte-for-byte** - the version string, `anthropic-beta` flags,
//! stainless versions, the `x-anthropic-billing-header` cch checksum (when the
//! pin uses cch), the model catalog, wire defaults, and identity preamble
//! injection. This exactness is the entire point of provider-claude: an
//! inexact fingerprint is eventually rejected by Anthropic's subscription
//! OAuth gate. "Close" is a failure, not a partial success.
//!
//! All code that contributes to the serialized request body or the header
//! set for /v1/messages lives ONLY in this crate (fingerprint + translate
//! wire types + identity prepend). It never leaks into omni-common or
//! omni-core.
//!
//! Active baseline: Claude Code 2.1.221 (captured 2026-08-04). Single pin only
//! (issue #12). No cch field on this pin; cch algorithms remain for vectors and
//! future pins that reintroduce checksums.
//!
//! Ported/adapted directly from reference-src-claude/upstream/fingerprint.rs
//! (the authoritative source for the invariant).

use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use ring::digest;
use uuid::Uuid;

use crate::credentials::Credentials;
use crate::models::{
    MODEL_CATALOG, ModelDef, ModelInfo, models_list_from_catalog, resolve_model_in_catalog,
};

/// Static identity Omni claims on the wire. These values must move together
/// when re-baselining against a new Claude Code release.
#[derive(Debug, Clone, Copy)]
pub struct FingerprintProfile {
    pub name: &'static str,
    pub claude_cli_version: &'static str,
    pub stainless_package_version: &'static str,
    pub stainless_runtime_version: &'static str,
    pub entrypoint: &'static str,
    pub beta_reply: &'static str,
    pub model_beta_overrides: &'static [ModelBetaOverride],
    pub system_preamble: &'static str,
    pub models: &'static [ModelDef],
    pub preserve_explicit_model: bool,
    pub wire_defaults: WireDefaults,
    pub model_wire_overrides: &'static [ModelWireOverride],
    billing: BillingScheme,
}

#[derive(Debug, Clone, Copy)]
pub struct ModelBetaOverride {
    pub model: &'static str,
    pub beta_reply: &'static str,
}

#[derive(Debug, Clone, Copy)]
pub struct WireDefaults {
    pub max_tokens: u32,
    pub opus_max_tokens: u32,
    pub temperature: Option<f32>,
    pub output_effort: Option<&'static str>,
}

#[derive(Debug, Clone, Copy)]
pub struct ModelWireOverride {
    pub model: &'static str,
    pub max_tokens: u32,
    pub temperature: Option<f32>,
    pub output_effort: Option<&'static str>,
}

#[derive(Debug, Clone, Copy)]
struct BillingScheme {
    suffix_algorithm: BillingSuffixAlgorithm,
    seed: &'static str,
    sample_indices: &'static [usize],
    cch: BillingCchMode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BillingSuffixAlgorithm {
    Sha256Utf16SampleV1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)] // Static / FinalBodyChecksum* kept for algorithm tests and future pins
enum BillingCchMode {
    Static(&'static str),
    FinalBodyChecksum,
    FinalBodyChecksumSkipModelsAndMaxTokens,
    /// No `cch=` segment in the billing header at all. Observed on Claude Code
    /// 2.1.186+: the header ends at `cc_entrypoint=<entrypoint>;` with no
    /// trailing checksum field. The body is sent unmodified.
    None,
}

impl FingerprintProfile {
    pub fn user_agent(&self) -> String {
        format!(
            "claude-cli/{} (external, {})",
            self.claude_cli_version, self.entrypoint
        )
    }

    pub fn resolve_model(&self, input: &str) -> Option<&'static ModelDef> {
        resolve_model_in_catalog(input, self.models)
    }

    pub fn outbound_model(&self, input: &str, model: &ModelDef) -> String {
        if self.preserve_explicit_model && self.is_explicit_claude_model(input) {
            input.to_string()
        } else {
            model.canonical.to_string()
        }
    }

    /// Whether `input` is a real, Anthropic-acceptable Claude model id that
    /// should be forwarded verbatim (an explicit version pin) rather than
    /// resolved to the profile canonical.
    fn is_explicit_claude_model(&self, input: &str) -> bool {
        if !input.starts_with("claude-") {
            return false;
        }
        self.models.iter().any(|model| input == model.canonical)
            || self
                .model_wire_overrides
                .iter()
                .any(|override_| override_.model == input)
    }

    pub fn beta_reply_for_model(&self, model: &str) -> &'static str {
        self.model_beta_overrides
            .iter()
            .find(|override_| override_.model == model)
            .map(|override_| override_.beta_reply)
            .unwrap_or(self.beta_reply)
    }

    pub fn wire_defaults_for_model(&self, model: &str) -> WireDefaults {
        if let Some(override_) = self
            .model_wire_overrides
            .iter()
            .find(|override_| override_.model == model)
        {
            return WireDefaults {
                max_tokens: override_.max_tokens,
                opus_max_tokens: override_.max_tokens,
                temperature: override_.temperature,
                output_effort: override_.output_effort,
            };
        }
        if model.contains("opus") {
            return WireDefaults {
                max_tokens: self.wire_defaults.opus_max_tokens,
                ..self.wire_defaults
            };
        }
        self.wire_defaults
    }

    pub fn models_list(&self) -> Vec<ModelInfo> {
        models_list_from_catalog(self.models)
    }

    pub fn billing_header_text(&self, first_user_text: &str) -> String {
        if matches!(self.billing.cch, BillingCchMode::None) {
            // 2.1.186+: no trailing cch field; header ends at cc_entrypoint.
            return format!(
                "x-anthropic-billing-header: cc_version={}.{}; cc_entrypoint={};",
                self.claude_cli_version,
                self.billing_suffix(first_user_text),
                self.entrypoint,
            );
        }
        format!(
            "x-anthropic-billing-header: cc_version={}.{}; cc_entrypoint={}; cch={};",
            self.claude_cli_version,
            self.billing_suffix(first_user_text),
            self.entrypoint,
            self.billing.cch.placeholder()
        )
    }

    pub fn finalize_body_json(
        &self,
        body: &serde_json::Value,
        ctx: &RequestContext,
    ) -> Result<Vec<u8>, serde_json::Error> {
        let bytes = serde_json::to_vec(body)?;
        Ok(self.finalize_body_bytes(bytes, ctx))
    }

    fn finalize_body_bytes(&self, bytes: Vec<u8>, _ctx: &RequestContext) -> Vec<u8> {
        match self.billing.cch {
            BillingCchMode::Static(_) | BillingCchMode::None => bytes,
            BillingCchMode::FinalBodyChecksum => {
                self.finalize_body_cch_checksum(bytes, claude_code_cch_checksum)
            }
            BillingCchMode::FinalBodyChecksumSkipModelsAndMaxTokens => self
                .finalize_body_cch_checksum(
                    bytes,
                    claude_code_cch_checksum_skip_models_and_max_tokens,
                ),
        }
    }

    fn finalize_body_cch_checksum(
        &self,
        mut bytes: Vec<u8>,
        checksum_fn: fn(&[u8]) -> u64,
    ) -> Vec<u8> {
        let Some(offset) = self.find_billing_cch_placeholder(&bytes) else {
            return bytes;
        };
        let checksum = checksum_fn(&bytes);
        let replacement = format!("{checksum:05x}");
        debug_assert_eq!(replacement.len(), 5);
        bytes[offset..offset + 5].copy_from_slice(replacement.as_bytes());
        bytes
    }

    fn find_billing_cch_placeholder(&self, bytes: &[u8]) -> Option<usize> {
        let system_start = find_subslice(bytes, br#""system":"#)?;
        let prefix = format!(
            "x-anthropic-billing-header: cc_version={}.",
            self.claude_cli_version
        );
        let tail = format!("; cc_entrypoint={}; cch=00000;", self.entrypoint);
        let search = &bytes[system_start..];
        let mut cursor = 0;
        while cursor < search.len() {
            let prefix_rel = find_subslice(&search[cursor..], prefix.as_bytes())?;
            let prefix_pos = system_start + cursor + prefix_rel;
            let suffix_pos = prefix_pos + prefix.len() + 3;
            let suffix_end = suffix_pos + tail.len();
            if suffix_end <= bytes.len()
                && bytes[prefix_pos + prefix.len()..suffix_pos]
                    .iter()
                    .all(u8::is_ascii_hexdigit)
                && bytes[suffix_pos..suffix_end] == *tail.as_bytes()
            {
                let cch_rel = tail.find("00000").expect("tail contains cch placeholder");
                return Some(suffix_pos + cch_rel);
            }
            cursor += prefix_rel + prefix.len();
        }
        None
    }

    fn billing_suffix(&self, first_user_text: &str) -> String {
        match self.billing.suffix_algorithm {
            BillingSuffixAlgorithm::Sha256Utf16SampleV1 => claude_code_version_suffix_v1(
                first_user_text,
                self.claude_cli_version,
                self.billing.seed,
                self.billing.sample_indices,
            ),
        }
    }
}

impl BillingCchMode {
    fn placeholder(self) -> &'static str {
        match self {
            BillingCchMode::Static(value) => value,
            BillingCchMode::FinalBodyChecksum
            | BillingCchMode::FinalBodyChecksumSkipModelsAndMaxTokens => "00000",
            // No cch segment is emitted; the placeholder is never used.
            BillingCchMode::None => "",
        }
    }
}

const BILLING_SUFFIX_SEED_V1: &str = "59cf53e54c78";
const BILLING_SUFFIX_INDICES_V1: [usize; 3] = [4, 7, 20];
#[allow(dead_code)]
const BILLING_SCHEME_V1_CCH_00000: BillingScheme = BillingScheme {
    suffix_algorithm: BillingSuffixAlgorithm::Sha256Utf16SampleV1,
    seed: BILLING_SUFFIX_SEED_V1,
    sample_indices: &BILLING_SUFFIX_INDICES_V1,
    cch: BillingCchMode::Static("00000"),
};
#[allow(dead_code)]
const BILLING_SCHEME_V1_CCH_XXH64_BODY: BillingScheme = BillingScheme {
    suffix_algorithm: BillingSuffixAlgorithm::Sha256Utf16SampleV1,
    seed: BILLING_SUFFIX_SEED_V1,
    sample_indices: &BILLING_SUFFIX_INDICES_V1,
    cch: BillingCchMode::FinalBodyChecksum,
};
#[cfg(test)]
const BILLING_SCHEME_V1_CCH_XXH64_SKIP_MODELS_AND_MAX_TOKENS: BillingScheme = BillingScheme {
    suffix_algorithm: BillingSuffixAlgorithm::Sha256Utf16SampleV1,
    seed: BILLING_SUFFIX_SEED_V1,
    sample_indices: &BILLING_SUFFIX_INDICES_V1,
    cch: BillingCchMode::FinalBodyChecksumSkipModelsAndMaxTokens,
};
// 2.1.186: the version suffix is still computed (cc_version=...a80), but the
// billing header carries no cch field and the body is not rewritten. Verified
// 2026-06-22 against two independent live captures (mitmproxy reverse proxy and
// the drift checker's capture server): the header ends at `cc_entrypoint=sdk-cli;`.
const BILLING_SCHEME_V1_NO_CCH: BillingScheme = BillingScheme {
    suffix_algorithm: BillingSuffixAlgorithm::Sha256Utf16SampleV1,
    seed: BILLING_SUFFIX_SEED_V1,
    sample_indices: &BILLING_SUFFIX_INDICES_V1,
    cch: BillingCchMode::None,
};

pub const ANTHROPIC_VERSION: &str = "2023-06-01";
/// Anthropic's OAuth-subscription gate expects this canonical Claude Code
/// identifier in the system block array after the billing marker.
///
/// Verified empirically 2026-05-10: any other prefix, suffix, casing, or
/// preceding whitespace fails. Only block-array form allows additional
/// content; flat-string form must equal this sentence verbatim.
pub const CLAUDE_CODE_SYSTEM_PREAMBLE: &str =
    "You are Claude Code, Anthropic's official CLI for Claude.";

/// Active-pin default beta list (default-model resolution to opus).
pub const BETA_DEFAULT: &str = "claude-code-20250219,oauth-2025-04-20,context-1m-2025-08-07,interleaved-thinking-2025-05-14,thinking-token-count-2026-05-13,context-management-2025-06-27,prompt-caching-scope-2026-01-05,mid-conversation-system-2026-04-07,effort-2025-11-24,fallback-credit-2026-06-01,extended-cache-ttl-2025-04-11";
/// Active-pin explicit opus beta list.
pub const BETA_OPUS: &str = "claude-code-20250219,oauth-2025-04-20,interleaved-thinking-2025-05-14,thinking-token-count-2026-05-13,context-management-2025-06-27,prompt-caching-scope-2026-01-05,mid-conversation-system-2026-04-07,effort-2025-11-24,fallback-credit-2026-06-01,extended-cache-ttl-2025-04-11";
/// Active-pin sonnet beta list (matches mid-conversation, no fallback-credit).
pub const BETA_SONNET: &str = "claude-code-20250219,oauth-2025-04-20,interleaved-thinking-2025-05-14,thinking-token-count-2026-05-13,context-management-2025-06-27,prompt-caching-scope-2026-01-05,mid-conversation-system-2026-04-07,effort-2025-11-24,extended-cache-ttl-2025-04-11";
/// Active-pin haiku beta list.
pub const BETA_HAIKU: &str = "oauth-2025-04-20,interleaved-thinking-2025-05-14,thinking-token-count-2026-05-13,context-management-2025-06-27,prompt-caching-scope-2026-01-05,claude-code-20250219,extended-cache-ttl-2025-04-11";
/// Active-pin fable beta list.
pub const BETA_FABLE: &str = "claude-code-20250219,oauth-2025-04-20,interleaved-thinking-2025-05-14,thinking-token-count-2026-05-13,context-management-2025-06-27,prompt-caching-scope-2026-01-05,mid-conversation-system-2026-04-07,effort-2025-11-24,fallback-credit-2026-06-01,extended-cache-ttl-2025-04-11";

const MODEL_BETA_OVERRIDES: &[ModelBetaOverride] = &[
    ModelBetaOverride {
        model: "claude-fable-5",
        beta_reply: BETA_FABLE,
    },
    ModelBetaOverride {
        model: "claude-opus-5",
        beta_reply: BETA_OPUS,
    },
    ModelBetaOverride {
        model: "claude-sonnet-5",
        beta_reply: BETA_SONNET,
    },
    ModelBetaOverride {
        model: "claude-haiku-4-5",
        beta_reply: BETA_HAIKU,
    },
    ModelBetaOverride {
        model: "claude-haiku-4-5-20251001",
        beta_reply: BETA_HAIKU,
    },
    ModelBetaOverride {
        model: "haiku",
        beta_reply: BETA_HAIKU,
    },
];

// Active-pin wire: opus-5 + sonnet-5 64k/no-temp/high; haiku 32k/no-temp/no-effort.
// Fable carries xhigh from prior capture (uncaptured on this pin account).
const MODEL_WIRE_OVERRIDES: &[ModelWireOverride] = &[
    ModelWireOverride {
        model: "claude-fable-5",
        max_tokens: 64_000,
        temperature: None,
        output_effort: Some("xhigh"),
    },
    ModelWireOverride {
        model: "claude-opus-5",
        max_tokens: 64_000,
        temperature: None,
        output_effort: Some("high"),
    },
    ModelWireOverride {
        model: "claude-sonnet-5",
        max_tokens: 64_000,
        temperature: None,
        output_effort: Some("high"),
    },
    ModelWireOverride {
        model: "claude-haiku-4-5",
        max_tokens: 32_000,
        temperature: None,
        output_effort: None,
    },
    ModelWireOverride {
        model: "claude-haiku-4-5-20251001",
        max_tokens: 32_000,
        temperature: None,
        output_effort: None,
    },
];

/// Active-pin wire defaults: temperature omitted; non-override fallback 32k/high-effort.
pub const WIRE_DEFAULTS: WireDefaults = WireDefaults {
    max_tokens: 32_000,
    opus_max_tokens: 64_000,
    temperature: None,
    output_effort: Some("high"),
};

pub const DEFAULT_PROFILE_NAME: &str = "cc-2.1.221-sdk-cli";

// Captured 2026-08-04 against installed Claude Code 2.1.221 via the shared
// tools.capture framework (mitmproxy reverse proxy + real claude CLI, clean tmpfs
// HOME), for default, explicit opus, sonnet, and haiku. This is the sole active
// pin (issue #12). Wire shape matches the 2.1.220 catalog/betas with only the
// version string moved (UA + billing cc_version). No cch field.
// Captured cc_version=2.1.221.116 for prompt "Say OK".
pub const PROFILE_CLAUDE_2_1_221_SDK_CLI: FingerprintProfile = FingerprintProfile {
    name: DEFAULT_PROFILE_NAME,
    claude_cli_version: "2.1.221",
    stainless_package_version: "0.94.0",
    stainless_runtime_version: "v26.3.0",
    entrypoint: "sdk-cli",
    beta_reply: BETA_DEFAULT,
    model_beta_overrides: MODEL_BETA_OVERRIDES,
    system_preamble: CLAUDE_CODE_SYSTEM_PREAMBLE,
    models: MODEL_CATALOG,
    preserve_explicit_model: true,
    wire_defaults: WIRE_DEFAULTS,
    model_wire_overrides: MODEL_WIRE_OVERRIDES,
    billing: BILLING_SCHEME_V1_NO_CCH,
};

pub fn default_profile() -> &'static FingerprintProfile {
    &PROFILE_CLAUDE_2_1_221_SDK_CLI
}

pub fn is_claude_code_billing_header(text: &str) -> bool {
    // Two accepted shapes:
    //   <= 2.1.175: ...; cc_entrypoint=<ep>; cch=<checksum>;
    //   >= 2.1.186: ...; cc_entrypoint=<ep>;     (no trailing cch field)
    text.starts_with("x-anthropic-billing-header: cc_version=") && text.contains("; cc_entrypoint=")
}

/// What kind of request this is - controls minor header variations.
#[derive(Debug, Clone, Copy)]
pub enum RequestKind {
    /// A user-facing reply request. Default beta list.
    Reply,
}

/// Per-call ephemeral context. Session ID stays stable across a logical
/// "session"; client_request_id is regenerated per HTTP call.
#[derive(Debug, Clone)]
pub struct RequestContext {
    pub session_id: Uuid,
    pub client_request_id: Uuid,
    pub retry_count: u32,
    pub kind: RequestKind,
    pub model: Option<String>,
}

impl RequestContext {
    pub fn new_reply() -> Self {
        Self {
            session_id: Uuid::new_v4(),
            client_request_id: Uuid::new_v4(),
            retry_count: 0,
            kind: RequestKind::Reply,
            model: None,
        }
    }

    pub fn with_session(mut self, session_id: Uuid) -> Self {
        self.session_id = session_id;
        self
    }

    pub fn with_model(mut self, model: String) -> Self {
        self.model = Some(model);
        self
    }

    pub fn next_attempt(&mut self) {
        self.retry_count += 1;
        self.client_request_id = Uuid::new_v4();
    }
}

/// Build the full outbound header set for a Messages call.
///
/// Header names are emitted lowercase because HTTP/2 requires lowercase and
/// HTTP/1.1 is case-insensitive. Anthropic does not appear to care about case.
pub fn build_headers(
    creds: &Credentials,
    ctx: &RequestContext,
    profile: &FingerprintProfile,
) -> HeaderMap {
    build_headers_with_profile(creds, ctx, profile)
}

fn build_headers_with_profile(
    creds: &Credentials,
    ctx: &RequestContext,
    profile: &FingerprintProfile,
) -> HeaderMap {
    let mut h = HeaderMap::new();

    insert(&mut h, "accept", "application/json");

    let bearer = format!("Bearer {}", creds.access_token);
    insert(&mut h, "authorization", &bearer);

    insert(&mut h, "content-type", "application/json");

    insert(&mut h, "user-agent", &profile.user_agent());

    insert(
        &mut h,
        "x-claude-code-session-id",
        &ctx.session_id.to_string(),
    );

    insert(&mut h, "x-stainless-arch", "x64");
    insert(&mut h, "x-stainless-lang", "js");
    insert(&mut h, "x-stainless-os", "Linux");
    insert(
        &mut h,
        "x-stainless-package-version",
        profile.stainless_package_version,
    );
    insert(
        &mut h,
        "x-stainless-retry-count",
        &ctx.retry_count.to_string(),
    );
    insert(&mut h, "x-stainless-runtime", "node");
    insert(
        &mut h,
        "x-stainless-runtime-version",
        profile.stainless_runtime_version,
    );
    insert(&mut h, "x-stainless-timeout", "600");

    let beta = match ctx.kind {
        RequestKind::Reply => ctx
            .model
            .as_deref()
            .map(|model| profile.beta_reply_for_model(model))
            .unwrap_or(profile.beta_reply),
    };
    insert(&mut h, "anthropic-beta", beta);

    insert(&mut h, "anthropic-dangerous-direct-browser-access", "true");
    insert(&mut h, "anthropic-version", ANTHROPIC_VERSION);
    insert(&mut h, "x-app", "cli");
    insert(
        &mut h,
        "x-client-request-id",
        &ctx.client_request_id.to_string(),
    );

    h
}

/// Claude Code's body marker appends a three-hex-character suffix to the CLI
/// version. The sampled positions are JavaScript string indices, so non-BMP
/// characters count as two UTF-16 code units. Claude Code joins the sampled
/// one-code-unit strings before hashing, so sampled surrogate halves can pair
/// with each other exactly as a JavaScript string would during UTF-8 encoding.
#[cfg(test)]
pub fn claude_code_version_suffix(first_user_text: &str, claude_cli_version: &str) -> String {
    claude_code_version_suffix_v1(
        first_user_text,
        claude_cli_version,
        BILLING_SUFFIX_SEED_V1,
        &BILLING_SUFFIX_INDICES_V1,
    )
}

fn claude_code_version_suffix_v1(
    first_user_text: &str,
    claude_cli_version: &str,
    seed: &str,
    sample_indices: &[usize],
) -> String {
    let mut input = Vec::new();
    input.extend_from_slice(seed.as_bytes());
    let code_units: Vec<u16> = first_user_text.encode_utf16().collect();
    let mut sampled_units = Vec::with_capacity(sample_indices.len());
    for index in sample_indices {
        if let Some(unit) = code_units.get(*index) {
            sampled_units.push(*unit);
        } else {
            sampled_units.push(b'0' as u16);
        }
    }
    append_javascript_utf8(&mut input, &sampled_units);
    input.extend_from_slice(claude_cli_version.as_bytes());

    let digest = digest::digest(&digest::SHA256, &input);
    let mut suffix = String::with_capacity(3);
    for byte in digest.as_ref().iter().take(2) {
        suffix.push_str(&format!("{byte:02x}"));
    }
    suffix.truncate(3);
    suffix
}

fn append_javascript_utf8(out: &mut Vec<u8>, units: &[u16]) {
    let mut idx = 0;
    while idx < units.len() {
        let unit = units[idx];
        let scalar = if (0xd800..=0xdbff).contains(&unit) {
            if let Some(low) = units.get(idx + 1) {
                if (0xdc00..=0xdfff).contains(low) {
                    idx += 2;
                    0x10000 + (((unit as u32 - 0xd800) << 10) | (*low as u32 - 0xdc00))
                } else {
                    idx += 1;
                    char::REPLACEMENT_CHARACTER as u32
                }
            } else {
                idx += 1;
                char::REPLACEMENT_CHARACTER as u32
            }
        } else if (0xdc00..=0xdfff).contains(&unit) {
            idx += 1;
            char::REPLACEMENT_CHARACTER as u32
        } else {
            idx += 1;
            unit as u32
        };

        let ch = char::from_u32(scalar).unwrap_or(char::REPLACEMENT_CHARACTER);
        let mut buf = [0; 4];
        out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
    }
}

const CCH_XXH64_SEED: u64 = 0x4d659218e32a3268;
const XXH64_PRIME1: u64 = 11_400_714_785_074_694_791;
const XXH64_PRIME2: u64 = 14_029_467_366_897_019_727;
const XXH64_PRIME3: u64 = 1_609_587_929_392_839_161;
const XXH64_PRIME4: u64 = 9_650_029_242_287_828_579;
const XXH64_PRIME5: u64 = 2_870_177_450_012_600_261;

fn claude_code_cch_checksum(bytes: &[u8]) -> u64 {
    xxh64(bytes, CCH_XXH64_SEED) & 0xfffff
}

fn claude_code_cch_checksum_skip_models_and_max_tokens(bytes: &[u8]) -> u64 {
    xxh64(
        &body_for_cch_skip_models_and_max_tokens(bytes),
        CCH_XXH64_SEED,
    ) & 0xfffff
}

fn body_for_cch_skip_models_and_max_tokens(bytes: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(bytes.len());
    let mut cursor = 0;
    while let Some((range_start, range_end)) = find_next_max_tokens_range(bytes, cursor) {
        append_with_model_values_removed(&mut out, &bytes[cursor..range_start]);
        cursor = range_end;
    }
    append_with_model_values_removed(&mut out, &bytes[cursor..]);
    out
}

fn find_next_max_tokens_range(bytes: &[u8], start: usize) -> Option<(usize, usize)> {
    let found = find_subslice(&bytes[start..], br#""max_tokens":"#)? + start;
    let value_start = found + br#""max_tokens":"#.len();
    let mut value_end = value_start;
    while value_end < bytes.len() && bytes[value_end].is_ascii_digit() {
        value_end += 1;
    }
    if value_end == value_start {
        return Some((found, value_end));
    }
    if found > start && bytes[found - 1] == b',' {
        Some((found - 1, value_end))
    } else if value_end < bytes.len() && bytes[value_end] == b',' {
        Some((found, value_end + 1))
    } else {
        Some((found, value_end))
    }
}

fn append_with_model_values_removed(out: &mut Vec<u8>, bytes: &[u8]) {
    let mut cursor = 0;
    while let Some(rel) = find_subslice(&bytes[cursor..], br#""model":""#) {
        let start = cursor + rel;
        let value_start = start + br#""model":""#.len();
        let Some(value_end) = find_json_string_end(bytes, value_start) else {
            break;
        };
        out.extend_from_slice(&bytes[cursor..value_start]);
        cursor = value_end;
    }
    out.extend_from_slice(&bytes[cursor..]);
}

fn find_json_string_end(bytes: &[u8], start: usize) -> Option<usize> {
    let mut idx = start;
    while idx < bytes.len() {
        match bytes[idx] {
            b'\\' => idx = idx.saturating_add(2),
            b'"' => return Some(idx),
            _ => idx += 1,
        }
    }
    None
}

fn xxh64(bytes: &[u8], seed: u64) -> u64 {
    let mut offset = 0;
    let mut h64;

    if bytes.len() >= 32 {
        let mut v1 = seed.wrapping_add(XXH64_PRIME1).wrapping_add(XXH64_PRIME2);
        let mut v2 = seed.wrapping_add(XXH64_PRIME2);
        let mut v3 = seed;
        let mut v4 = seed.wrapping_sub(XXH64_PRIME1);

        while offset <= bytes.len() - 32 {
            v1 = xxh64_round(v1, read_u64_le(bytes, offset));
            v2 = xxh64_round(v2, read_u64_le(bytes, offset + 8));
            v3 = xxh64_round(v3, read_u64_le(bytes, offset + 16));
            v4 = xxh64_round(v4, read_u64_le(bytes, offset + 24));
            offset += 32;
        }

        h64 = v1
            .rotate_left(1)
            .wrapping_add(v2.rotate_left(7))
            .wrapping_add(v3.rotate_left(12))
            .wrapping_add(v4.rotate_left(18));
        h64 = xxh64_merge_round(h64, v1);
        h64 = xxh64_merge_round(h64, v2);
        h64 = xxh64_merge_round(h64, v3);
        h64 = xxh64_merge_round(h64, v4);
    } else {
        h64 = seed.wrapping_add(XXH64_PRIME5);
    }

    h64 = h64.wrapping_add(bytes.len() as u64);

    while offset + 8 <= bytes.len() {
        let k1 = xxh64_round(0, read_u64_le(bytes, offset));
        h64 ^= k1;
        h64 = h64
            .rotate_left(27)
            .wrapping_mul(XXH64_PRIME1)
            .wrapping_add(XXH64_PRIME4);
        offset += 8;
    }

    if offset + 4 <= bytes.len() {
        h64 ^= (read_u32_le(bytes, offset) as u64).wrapping_mul(XXH64_PRIME1);
        h64 = h64
            .rotate_left(23)
            .wrapping_mul(XXH64_PRIME2)
            .wrapping_add(XXH64_PRIME3);
        offset += 4;
    }

    while offset < bytes.len() {
        h64 ^= (bytes[offset] as u64).wrapping_mul(XXH64_PRIME5);
        h64 = h64.rotate_left(11).wrapping_mul(XXH64_PRIME1);
        offset += 1;
    }

    xxh64_avalanche(h64)
}

fn xxh64_round(acc: u64, input: u64) -> u64 {
    acc.wrapping_add(input.wrapping_mul(XXH64_PRIME2))
        .rotate_left(31)
        .wrapping_mul(XXH64_PRIME1)
}

fn xxh64_merge_round(acc: u64, val: u64) -> u64 {
    let mut acc = acc ^ xxh64_round(0, val);
    acc = acc.wrapping_mul(XXH64_PRIME1).wrapping_add(XXH64_PRIME4);
    acc
}

fn xxh64_avalanche(mut h64: u64) -> u64 {
    h64 ^= h64 >> 33;
    h64 = h64.wrapping_mul(XXH64_PRIME2);
    h64 ^= h64 >> 29;
    h64 = h64.wrapping_mul(XXH64_PRIME3);
    h64 ^= h64 >> 32;
    h64
}

fn read_u64_le(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(
        bytes[offset..offset + 8]
            .try_into()
            .expect("xxh64 chunk length must be 8"),
    )
}

fn read_u32_le(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(
        bytes[offset..offset + 4]
            .try_into()
            .expect("xxh64 chunk length must be 4"),
    )
}

fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() {
        return Some(0);
    }
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

fn insert(h: &mut HeaderMap, name: &'static str, value: &str) {
    let n = HeaderName::from_static(name);
    if let Ok(v) = HeaderValue::from_str(value) {
        h.insert(n, v);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_creds() -> Credentials {
        Credentials {
            access_token: "sk-ant-oat01-test-token".into(),
            expires_at_ms: None,
            subscription_type: Some("max".into()),
        }
    }

    /// Synthetic cch-emitting profile for rewrite-path tests only.
    /// The active pin uses NO_CCH; these tests guard the algorithm still in tree.
    fn cch_rewrite_profile() -> FingerprintProfile {
        FingerprintProfile {
            name: "test-cch-rewrite",
            claude_cli_version: "2.1.175",
            stainless_package_version: "0.94.0",
            stainless_runtime_version: "v24.3.0",
            entrypoint: "sdk-cli",
            beta_reply: BETA_DEFAULT,
            model_beta_overrides: &[],
            system_preamble: CLAUDE_CODE_SYSTEM_PREAMBLE,
            models: MODEL_CATALOG,
            preserve_explicit_model: true,
            wire_defaults: WIRE_DEFAULTS,
            model_wire_overrides: &[],
            billing: BILLING_SCHEME_V1_CCH_XXH64_SKIP_MODELS_AND_MAX_TOKENS,
        }
    }

    #[test]
    fn header_set_matches_claude_baseline() {
        // WHY: active pin must keep the captured Claude Code header name set;
        // missing or extra headers fail the OAuth subscription gate.
        assert_profile_header_set_matches_baseline(default_profile());
    }

    #[test]
    fn active_pin_uses_captured_beta_list_per_model() {
        // WHY: per-model anthropic-beta bytes are load-bearing. A silent swap of
        // sonnet onto the default (context-1m) list would be an inexact fingerprint.
        let profile = default_profile();
        let creds = fixture_creds();
        let cases = [
            ("claude-fable-5", BETA_FABLE),
            ("claude-opus-5", BETA_OPUS),
            ("claude-sonnet-5", BETA_SONNET),
            ("claude-haiku-4-5", BETA_HAIKU),
            ("claude-haiku-4-5-20251001", BETA_HAIKU),
        ];
        for (model, expected_beta) in cases {
            let ctx = RequestContext::new_reply().with_model(model.to_string());
            let beta = build_headers(&creds, &ctx, profile)
                .get("anthropic-beta")
                .unwrap()
                .to_str()
                .unwrap()
                .to_string();
            assert_eq!(beta, expected_beta, "unexpected beta list for {model}");
        }
    }

    #[test]
    fn bare_aliases_resolve_to_correct_per_model_beta_on_active_pin() {
        // WHY: bare alias ("sonnet"/"haiku"/"opus") must get that model's captured
        // beta, not DEFAULT. Requires outbound_model canonicalize before headers.
        let profile = default_profile();
        let creds = fixture_creds();
        let cases = [
            ("fable", BETA_FABLE, false),
            ("opus", BETA_OPUS, false),
            ("sonnet", BETA_SONNET, false),
            ("haiku", BETA_HAIKU, false),
        ];
        for (alias, expected_beta, has_context_1m) in cases {
            let model_def = profile.resolve_model(alias).unwrap();
            let outbound = profile.outbound_model(alias, model_def);
            let ctx = RequestContext::new_reply().with_model(outbound);
            let beta = build_headers(&creds, &ctx, profile)
                .get("anthropic-beta")
                .unwrap()
                .to_str()
                .unwrap()
                .to_string();
            assert_eq!(beta, expected_beta, "alias {alias} got the wrong beta list");
            assert_eq!(
                beta.contains("context-1m"),
                has_context_1m,
                "alias {alias} context-1m presence mismatch"
            );
        }
    }

    fn assert_profile_header_set_matches_baseline(profile: &FingerprintProfile) {
        // WHY: active-pin header names AND static values must match capture.
        // Name-set equality catches missing/extra headers; value locks catch silent
        // fingerprint drift that still returns HTTP 200.
        let creds = fixture_creds();
        let ctx = RequestContext::new_reply();
        let h = build_headers(&creds, &ctx, profile);

        let expected_names = [
            "accept",
            "authorization",
            "content-type",
            "user-agent",
            "x-claude-code-session-id",
            "x-stainless-arch",
            "x-stainless-lang",
            "x-stainless-os",
            "x-stainless-package-version",
            "x-stainless-retry-count",
            "x-stainless-runtime",
            "x-stainless-runtime-version",
            "x-stainless-timeout",
            "anthropic-beta",
            "anthropic-dangerous-direct-browser-access",
            "anthropic-version",
            "x-app",
            "x-client-request-id",
        ];
        let mut names: Vec<&str> = h.keys().map(|k| k.as_str()).collect();
        names.sort();
        let mut expected = expected_names.to_vec();
        expected.sort();
        assert_eq!(
            names, expected,
            "header name set drifted on {}",
            profile.name
        );

        assert_eq!(h.get("anthropic-version").unwrap(), "2023-06-01");
        assert_eq!(
            h.get("anthropic-dangerous-direct-browser-access").unwrap(),
            "true"
        );
        assert_eq!(h.get("x-app").unwrap(), "cli");
        assert_eq!(h.get("x-stainless-arch").unwrap(), "x64");
        assert_eq!(h.get("x-stainless-lang").unwrap(), "js");
        assert_eq!(h.get("x-stainless-os").unwrap(), "Linux");
        assert_eq!(h.get("x-stainless-runtime").unwrap(), "node");
        assert_eq!(
            h.get("x-stainless-package-version")
                .unwrap()
                .to_str()
                .unwrap(),
            profile.stainless_package_version,
            "x-stainless-package-version drifted on {}",
            profile.name
        );
        assert_eq!(
            h.get("x-stainless-runtime-version")
                .unwrap()
                .to_str()
                .unwrap(),
            profile.stainless_runtime_version,
            "x-stainless-runtime-version drifted on {}",
            profile.name
        );

        // No-model reply path: exact default beta bytes (order included).
        let beta = h.get("anthropic-beta").unwrap().to_str().unwrap();
        assert_eq!(
            beta, profile.beta_reply,
            "default-reply beta drifted from profile.beta_reply on {}",
            profile.name
        );
        assert!(
            beta.contains("oauth-2025-04-20"),
            "beta list missing oauth-2025-04-20: {beta}"
        );
        assert!(
            beta.contains("claude-code-20250219"),
            "beta list missing claude-code-20250219: {beta}"
        );

        let auth = h.get("authorization").unwrap().to_str().unwrap();
        assert!(auth.starts_with("Bearer sk-ant-oat01-"));
        let ua = h.get("user-agent").unwrap().to_str().unwrap();
        assert_eq!(
            ua,
            profile.user_agent(),
            "user-agent drifted from profile.user_agent() on {}",
            profile.name
        );
    }

    #[test]
    fn next_attempt_increments_retry_count_and_rotates_request_id() {
        let mut ctx = RequestContext::new_reply();
        let first_id = ctx.client_request_id;
        ctx.next_attempt();
        assert_eq!(ctx.retry_count, 1);
        assert_ne!(ctx.client_request_id, first_id);
    }

    #[test]
    fn active_pin_matches_refreshed_claude_code_baseline() {
        // WHY: issue #12 ships exactly one pin. These bytes are the gate: UA,
        // stainless, catalog ids, and per-model beta lists must match capture.
        let profile = default_profile();
        assert_eq!(profile.name, "cc-2.1.221-sdk-cli");
        assert_eq!(profile.claude_cli_version, "2.1.221");
        assert_eq!(profile.stainless_package_version, "0.94.0");
        assert_eq!(profile.stainless_runtime_version, "v26.3.0");
        assert_eq!(
            profile.user_agent(),
            "claude-cli/2.1.221 (external, sdk-cli)"
        );
        assert_eq!(
            profile.resolve_model("fable").unwrap().canonical,
            "claude-fable-5"
        );
        assert_eq!(
            profile.resolve_model("opus").unwrap().canonical,
            "claude-opus-5"
        );
        assert_eq!(
            profile.resolve_model("sonnet").unwrap().canonical,
            "claude-sonnet-5"
        );
        assert_eq!(
            profile.resolve_model("haiku").unwrap().canonical,
            "claude-haiku-4-5-20251001"
        );
        assert!(profile.beta_reply.contains("fallback-credit-2026-06-01"));
        assert_eq!(profile.beta_reply_for_model("claude-opus-5"), BETA_OPUS);
        assert_eq!(profile.beta_reply_for_model("claude-sonnet-5"), BETA_SONNET);
        // Wire golden: opus/sonnet 64k no-temp high; haiku 32k no-temp no-effort.
        let opus_w = profile.wire_defaults_for_model("claude-opus-5");
        assert_eq!(opus_w.max_tokens, 64_000);
        assert_eq!(opus_w.temperature, None);
        assert_eq!(opus_w.output_effort, Some("high"));
        let sonnet_w = profile.wire_defaults_for_model("claude-sonnet-5");
        assert_eq!(sonnet_w.max_tokens, 64_000);
        assert_eq!(sonnet_w.temperature, None);
        assert_eq!(sonnet_w.output_effort, Some("high"));
        let haiku_w = profile.wire_defaults_for_model("claude-haiku-4-5");
        assert_eq!(haiku_w.max_tokens, 32_000);
        assert_eq!(haiku_w.temperature, None);
        assert_eq!(haiku_w.output_effort, None);
        assert!(
            !profile.billing_header_text("Say OK").contains("cch="),
            "active pin must use no-cch billing header"
        );
    }

    #[test]
    fn active_pin_catalog_names_are_unique() {
        let profile = default_profile();
        assert!(!profile.name.is_empty());
        assert!(!profile.claude_cli_version.is_empty());
        assert!(!profile.models.is_empty());
        assert!(crate::models::catalog_contains_unique_names(profile.models));
    }

    #[test]
    fn billing_suffix_matches_claude_code_probe() {
        // Historical suffix vectors lock the algorithm across past versions.
        // Active pin: 2.1.221 / "Say OK" -> 116; header has no cch field.
        assert_eq!(claude_code_version_suffix("Say OK", "2.1.142"), "73b");
        assert_eq!(claude_code_version_suffix("Say OK", "2.1.150"), "5bd");
        assert_eq!(claude_code_version_suffix("Say OK", "2.1.154"), "cea");
        assert_eq!(claude_code_version_suffix("Say OK", "2.1.161"), "d2b");
        assert_eq!(claude_code_version_suffix("Say OK", "2.1.162"), "b87");
        assert_eq!(claude_code_version_suffix("Say OK", "2.1.165"), "492");
        assert_eq!(claude_code_version_suffix("Say OK", "2.1.175"), "174");
        assert_eq!(claude_code_version_suffix("Say OK", "2.1.186"), "a80");
        assert_eq!(claude_code_version_suffix("Say OK", "2.1.197"), "c8e");
        assert_eq!(claude_code_version_suffix("Say OK", "2.1.207"), "aa4");
        assert_eq!(claude_code_version_suffix("Say OK", "2.1.211"), "08c");
        assert_eq!(claude_code_version_suffix("Say OK", "2.1.220"), "01b");
        assert_eq!(claude_code_version_suffix("Say OK", "2.1.221"), "116");
        assert_eq!(
            default_profile().billing_header_text("Say OK"),
            "x-anthropic-billing-header: cc_version=2.1.221.116; cc_entrypoint=sdk-cli;"
        );
        // Synthetic cch profile still emits sentinel form for rewrite tests.
        assert_eq!(
            cch_rewrite_profile().billing_header_text("Say OK"),
            "x-anthropic-billing-header: cc_version=2.1.175.174; cc_entrypoint=sdk-cli; cch=00000;"
        );
    }

    #[test]
    fn billing_cch_stays_on_known_safe_sentinel() {
        let profile = default_profile();
        let header = profile.billing_header_text("Say OK");
        assert!(
            !header.contains("cch="),
            "active pin must omit cch: {header}"
        );
        assert!(
            header.ends_with("; cc_entrypoint=sdk-cli;"),
            "active pin unexpected header tail: {header}"
        );
        let cch_header = cch_rewrite_profile().billing_header_text("Say OK");
        assert!(
            cch_header.contains("cch=00000;"),
            "cch rewrite profile must emit sentinel: {cch_header}"
        );
    }

    #[test]
    fn finalized_body_writes_profile_checksum() {
        // WHY: cch rewrite must replace the 00000 sentinel in-place without
        // changing body length. Active pin has no cch; synthetic profile guards
        // the algorithm kept for vectors and future pins that reintroduce cch.
        let profile = cch_rewrite_profile();
        let ctx = RequestContext::new_reply();
        let body = serde_json::json!({
            "system": [
                {
                    "type": "text",
                    "text": profile.billing_header_text("Say OK"),
                }
            ],
            "messages": []
        });
        let placeholder = serde_json::to_vec(&body).unwrap();
        let bytes = profile.finalize_body_json(&body, &ctx).unwrap();
        let json = String::from_utf8(bytes).unwrap();
        let expected = format!(
            "{:05x}",
            claude_code_cch_checksum_skip_models_and_max_tokens(&placeholder)
        );

        assert!(json.contains(&format!("cch={expected};")));
        assert_eq!(json.len(), placeholder.len());
        assert!(!json.contains("cch=00000;"));
    }

    #[test]
    fn omni_serialized_body_cch_snapshot_stays_stable() {
        let profile = default_profile();
        let ctx = RequestContext::new_reply();
        let body = serde_json::json!({
            "model": "claude-haiku-4-5",
            "max_tokens": 4096,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Say OK"}
                    ]
                }
            ],
            "system": [
                {
                    "type": "text",
                    "text": profile.billing_header_text("Say OK"),
                },
                {
                    "type": "text",
                    "text": profile.system_preamble,
                }
            ],
            "stream": false
        });
        let json = String::from_utf8(profile.finalize_body_json(&body, &ctx).unwrap()).unwrap();

        assert!(
            json.contains("cc_entrypoint=sdk-cli;"),
            "active body missing entrypoint terminator"
        );
        assert!(
            !json.contains("cch="),
            "active pin body unexpectedly contains a cch field: {json}"
        );

        // Rewrite algorithm snapshot on synthetic cch profile.
        let cch_profile = cch_rewrite_profile();
        let cch_body = serde_json::json!({
            "model": "claude-haiku-4-5",
            "max_tokens": 4096,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Say OK"}
                    ]
                }
            ],
            "system": [
                {
                    "type": "text",
                    "text": cch_profile.billing_header_text("Say OK"),
                },
                {
                    "type": "text",
                    "text": cch_profile.system_preamble,
                }
            ],
            "stream": false
        });
        let cch_json =
            String::from_utf8(cch_profile.finalize_body_json(&cch_body, &ctx).unwrap()).unwrap();
        let marker = "cc_entrypoint=sdk-cli; cch=";
        let idx = cch_json
            .find(marker)
            .expect("snapshot body missing cch marker");
        let got = &cch_json[idx + marker.len()..idx + marker.len() + 5];
        assert_eq!(
            got, "527d7",
            "cch rewrite snapshot changed (re-derive literal)"
        );
    }

    #[test]
    fn finalized_body_is_deterministic() {
        let profile = default_profile();
        let ctx = RequestContext::new_reply();
        let body = serde_json::json!({
            "system": [
                {
                    "type": "text",
                    "text": profile.billing_header_text("Say OK"),
                }
            ],
            "messages": []
        });

        assert_eq!(
            profile.finalize_body_json(&body, &ctx).unwrap(),
            profile.finalize_body_json(&body, &ctx).unwrap()
        );
    }

    #[test]
    fn finalized_body_does_not_rewrite_user_text_sentinel() {
        let profile = cch_rewrite_profile();
        let ctx = RequestContext::new_reply();
        let user_text = "leave user cch=00000 untouched";
        let body = serde_json::json!({
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text}
                    ]
                }
            ],
            "system": [
                {
                    "type": "text",
                    "text": profile.billing_header_text("Say OK"),
                }
            ]
        });
        let bytes = profile.finalize_body_json(&body, &ctx).unwrap();
        let json = String::from_utf8(bytes).unwrap();

        assert!(json.contains(user_text));
        assert_eq!(json.matches("cch=00000").count(), 1);
        assert!(json.contains(
            "x-anthropic-billing-header: cc_version=2.1.175.174; cc_entrypoint=sdk-cli; cch="
        ));
        assert!(!json.contains("cc_entrypoint=sdk-cli; cch=00000;"));
    }

    #[test]
    fn finalized_body_without_billing_sentinel_is_unchanged() {
        let profile = default_profile();
        let ctx = RequestContext::new_reply();
        let body = serde_json::json!({
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Say OK"}
                    ]
                }
            ]
        });
        let expected = serde_json::to_vec(&body).unwrap();
        assert_eq!(profile.finalize_body_json(&body, &ctx).unwrap(), expected);
    }

    #[test]
    fn finalized_body_preserves_non_sentinel_cch() {
        let profile = default_profile();
        let ctx = RequestContext::new_reply();
        let body = serde_json::json!({
            "system": [
                {
                    "type": "text",
                    "text": "x-anthropic-billing-header: cc_version=2.1.142.73b; cc_entrypoint=sdk-cli; cch=abcde;",
                }
            ],
            "messages": []
        });
        let expected = serde_json::to_vec(&body).unwrap();
        assert_eq!(profile.finalize_body_json(&body, &ctx).unwrap(), expected);
    }

    #[test]
    fn finalized_body_rewrites_only_first_billing_sentinel() {
        let profile = cch_rewrite_profile();
        let ctx = RequestContext::new_reply();
        let body = serde_json::json!({
            "system": [
                {
                    "type": "text",
                    "text": profile.billing_header_text("Say OK"),
                },
                {
                    "type": "text",
                    "text": profile.billing_header_text("Say OK"),
                }
            ],
            "messages": []
        });
        let bytes = profile.finalize_body_json(&body, &ctx).unwrap();
        let json = String::from_utf8(bytes).unwrap();

        assert_eq!(json.matches("x-anthropic-billing-header:").count(), 2);
        assert_eq!(json.matches("cc_entrypoint=sdk-cli; cch=00000;").count(), 1);
    }

    #[test]
    fn static_cch_mode_preserves_sentinel() {
        let profile = FingerprintProfile {
            name: "test-static",
            claude_cli_version: "2.1.142",
            stainless_package_version: "0.94.0",
            stainless_runtime_version: "v24.3.0",
            entrypoint: "sdk-cli",
            beta_reply: BETA_DEFAULT,
            model_beta_overrides: &[],
            system_preamble: CLAUDE_CODE_SYSTEM_PREAMBLE,
            models: MODEL_CATALOG,
            preserve_explicit_model: false,
            wire_defaults: WIRE_DEFAULTS,
            model_wire_overrides: &[],
            billing: BILLING_SCHEME_V1_CCH_00000,
        };
        let ctx = RequestContext::new_reply();
        let body = serde_json::json!({
            "system": [
                {
                    "type": "text",
                    "text": profile.billing_header_text("Say OK"),
                }
            ],
            "messages": []
        });
        let bytes = profile.finalize_body_json(&body, &ctx).unwrap();
        let json = String::from_utf8(bytes).unwrap();

        assert_eq!(json, serde_json::to_string(&body).unwrap());
        assert!(json.contains("cch=00000;"));
    }

    #[test]
    fn cch_checksum_matches_recovered_claude_code_captures() {
        let cases = [
            (
                "3bc55",
                r#"{"model":"claude-haiku-4-5","messages":[{"role":"user","content":[{"type":"text","text":"Say OK"}]}],"system":[{"type":"text","text":"x-anthropic-billing-header: cc_version=2.1.142.73b; cc_entrypoint=sdk-cli; cch=3bc55;"}],"max_tokens":1,"stream":true}"#,
            ),
            (
                "06b67",
                r#"{"model":"claude-haiku-4-5","messages":[{"role":"user","content":[{"type":"text","text":"Say OK"}]}],"system":[{"type":"text","text":"x-anthropic-billing-header: cc_version=2.1.142.73b; cc_entrypoint=sdk-cli; cch=06b67;"}],"max_tokens":2,"stream":true}"#,
            ),
            (
                "9bce0",
                r#"{"model":"claude-haiku-4-5-20251001","max_tokens":1,"messages":[{"role":"user","content":[{"type":"text","text":"factor"}]}],"system":[{"type":"text","text":"x-anthropic-billing-header: cc_version=2.1.142.73b; cc_entrypoint=sdk-cli; cch=9bce0;"},{"type":"text","text":"You are Claude Code, Anthropic's official CLI for Claude.","cache_control":{"type":"ephemeral"}}],"stream":true}"#,
            ),
            (
                "4dc19",
                r#"{"system":[{"type":"text","text":"x-anthropic-billing-header: cc_version=2.1.142.73b; cc_entrypoint=sdk-cli; cch=4dc19;"},{"type":"text","text":"You are Claude Code, Anthropic's official CLI for Claude.","cache_control":{"type":"ephemeral"}}],"model":"claude-haiku-4-5-20251001","max_tokens":1,"messages":[{"role":"user","content":[{"type":"text","text":"factor"}]}],"stream":true}"#,
            ),
            (
                "7afbb",
                r#"{"model":"claude-haiku-4-5-20251001","max_tokens":1,"messages":[{"role":"user","content":[{"type":"text","text":"factor"}]}],"system":[{"type":"text","text":"x-anthropic-billing-header: cc_version=2.1.142.73b; cc_entrypoint=sdk-cli; cch=7afbb;"},{"type":"text","text":"x-anthropic-billing-header: cc_version=2.1.142.73b; cc_entrypoint=sdk-cli; cch=00000;"},{"type":"text","text":"You are Claude Code, Anthropic's official CLI for Claude.","cache_control":{"type":"ephemeral"}}],"stream":true}"#,
            ),
            (
                "c159b",
                r#"{"model":"claude-haiku-4-5-20251001","max_tokens":1,"messages":[{"role":"user","content":[{"type":"text","text":"WATCHPOINT_MARKER_CCP_CCH_7a9d3f41"}]}],"system":[{"type":"text","text":"x-anthropic-billing-header: cc_version=2.1.142.73b; cc_entrypoint=sdk-cli; cch=c159b;"},{"type":"text","text":"You are Claude Code, Anthropic's official CLI for Claude.","cache_control":{"type":"ephemeral"}}],"stream":true}"#,
            ),
        ];

        for (expected, final_body) in cases {
            let placeholder_body =
                final_body.replacen(&format!("cch={expected};"), "cch=00000;", 1);
            assert_eq!(
                format!(
                    "{:05x}",
                    claude_code_cch_checksum(placeholder_body.as_bytes())
                ),
                expected
            );
        }
    }

    #[test]
    fn cch_matches_real_2_1_162_clean_room_capture_vectors() {
        // WHY: committed capture vectors prove xxh64 + finalize match live Claude
        // Code wire cch over rich body shapes (not only synthetic bodies).
        let vectors = [
            (
                "claude-haiku-4-5",
                include_str!(
                    "../../../tools/providers/claude/fingerprint/vectors/vector-2.1.162-claude-haiku-4-5.json"
                ),
            ),
            (
                "claude-sonnet-4-6",
                include_str!(
                    "../../../tools/providers/claude/fingerprint/vectors/vector-2.1.162-claude-sonnet-4-6.json"
                ),
            ),
            (
                "claude-opus-4-8",
                include_str!(
                    "../../../tools/providers/claude/fingerprint/vectors/vector-2.1.162-claude-opus-4-8.json"
                ),
            ),
        ];
        for (model, body) in vectors {
            let marker = "cc_entrypoint=sdk-cli; cch=";
            let idx = body
                .find(marker)
                .unwrap_or_else(|| panic!("no billing cch marker in {model} vector"));
            let start = idx + marker.len();
            let embedded = &body[start..start + 5];
            let placeholder_body = body.replacen(
                &format!("{marker}{embedded};"),
                &format!("{marker}00000;"),
                1,
            );
            assert_ne!(
                placeholder_body, body,
                "{model}: cch substitution was a no-op"
            );
            assert_eq!(
                format!(
                    "{:05x}",
                    claude_code_cch_checksum(placeholder_body.as_bytes())
                ),
                embedded,
                "cch != real Claude Code 2.1.162 cch for the {model} capture vector"
            );
        }
    }

    #[test]
    fn cch_matches_real_2_1_165_clean_room_capture_vectors() {
        let vectors = [
            (
                "claude-haiku-4-5",
                include_str!(
                    "../../../tools/providers/claude/fingerprint/vectors/vector-2.1.165-claude-haiku-4-5.json"
                ),
            ),
            (
                "claude-sonnet-4-6",
                include_str!(
                    "../../../tools/providers/claude/fingerprint/vectors/vector-2.1.165-claude-sonnet-4-6.json"
                ),
            ),
            (
                "claude-opus-4-8",
                include_str!(
                    "../../../tools/providers/claude/fingerprint/vectors/vector-2.1.165-claude-opus-4-8.json"
                ),
            ),
        ];
        for (model, body) in vectors {
            let marker = "cc_entrypoint=sdk-cli; cch=";
            let idx = body
                .find(marker)
                .unwrap_or_else(|| panic!("no billing cch marker in {model} vector"));
            let start = idx + marker.len();
            let embedded = &body[start..start + 5];
            let placeholder_body = body.replacen(
                &format!("{marker}{embedded};"),
                &format!("{marker}00000;"),
                1,
            );
            assert_ne!(
                placeholder_body, body,
                "{model}: cch substitution was a no-op"
            );
            assert_eq!(
                format!(
                    "{:05x}",
                    claude_code_cch_checksum(placeholder_body.as_bytes())
                ),
                embedded,
                "cch != real Claude Code 2.1.165 cch for the {model} capture vector"
            );
        }
    }

    #[test]
    fn cch_matches_real_2_1_175_clean_room_capture_vectors() {
        let vectors = [
            (
                "claude-fable-5",
                include_str!(
                    "../../../tools/providers/claude/fingerprint/vectors/vector-2.1.175-claude-fable-5.json"
                ),
            ),
            (
                "claude-haiku-4-5",
                include_str!(
                    "../../../tools/providers/claude/fingerprint/vectors/vector-2.1.175-claude-haiku-4-5.json"
                ),
            ),
            (
                "claude-sonnet-4-6",
                include_str!(
                    "../../../tools/providers/claude/fingerprint/vectors/vector-2.1.175-claude-sonnet-4-6.json"
                ),
            ),
            (
                "claude-opus-4-8",
                include_str!(
                    "../../../tools/providers/claude/fingerprint/vectors/vector-2.1.175-claude-opus-4-8.json"
                ),
            ),
        ];
        for (model, body) in vectors {
            let marker = "cc_entrypoint=sdk-cli; cch=";
            let idx = body
                .find(marker)
                .unwrap_or_else(|| panic!("no billing cch marker in {model} vector"));
            let start = idx + marker.len();
            let embedded = &body[start..start + 5];
            let placeholder_body = body.replacen(
                &format!("{marker}{embedded};"),
                &format!("{marker}00000;"),
                1,
            );
            assert_ne!(
                placeholder_body, body,
                "{model}: cch substitution was a no-op"
            );
            assert_eq!(
                format!(
                    "{:05x}",
                    claude_code_cch_checksum_skip_models_and_max_tokens(
                        placeholder_body.as_bytes()
                    )
                ),
                embedded,
                "cch != real Claude Code 2.1.175 cch for the {model} capture vector"
            );
        }
    }

    #[test]
    fn xxh64_matches_independent_small_input_vectors() {
        assert_eq!(xxh64(b"", CCH_XXH64_SEED), 0xb8b30e7de65b46c5);
        assert_eq!(xxh64(b"abc", CCH_XXH64_SEED), 0xdfc4f4d6913699b6);
        assert_eq!(xxh64(b"hello", CCH_XXH64_SEED), 0xfc8105d2d40e53f1);
        assert_eq!(
            xxh64(b"123456789abcdef", CCH_XXH64_SEED),
            0xd491c6f888304d64
        );
    }

    #[test]
    fn billing_header_detector_accepts_real_nonzero_cch() {
        assert!(is_claude_code_billing_header(
            "x-anthropic-billing-header: cc_version=2.1.142.73b; cc_entrypoint=sdk-cli; cch=e5ba6;"
        ));
    }

    #[test]
    fn billing_suffix_uses_zero_for_missing_positions() {
        assert_eq!(claude_code_version_suffix("", "2.1.142"), "1aa");
        assert_eq!(claude_code_version_suffix("abc", "2.1.142"), "1aa");
    }

    #[test]
    fn billing_suffix_uses_utf16_code_units() {
        assert_eq!(
            claude_code_version_suffix("abc😀efghijklmnopqrstuv", "2.1.142"),
            "db0"
        );
    }

    #[test]
    fn billing_suffix_treats_sampled_surrogates_like_javascript_string_indices() {
        assert_eq!(claude_code_version_suffix("abcd😀😀", "2.1.142"), "052");
    }

    #[test]
    fn billing_header_text_end_to_end_matches_suffix_oracle_for_varied_first_text() {
        let profile = default_profile();
        let ver = profile.claude_cli_version;
        let inputs = [
            "",
            "Say OK",
            "abc",
            "0123456789abcdefghijuvwxyz",
            "héllo wörld with nön-ascii café 99",
            "abc😀efghijklmnopqrstuv",
            "abcd😀😀",
        ];
        for input in inputs {
            let expected_suffix = claude_code_version_suffix(input, ver);
            assert_eq!(expected_suffix.len(), 3, "suffix len for {input:?}");
            assert!(
                expected_suffix.chars().all(|c| c.is_ascii_hexdigit()),
                "suffix not hex for {input:?}: {expected_suffix}"
            );
            let header = profile.billing_header_text(input);
            assert_eq!(
                header,
                format!(
                    "x-anthropic-billing-header: cc_version={ver}.{expected_suffix}; \
                     cc_entrypoint=sdk-cli;"
                ),
                "billing_header_text diverged from suffix oracle for first_user_text {input:?}"
            );
        }

        let cch_profile = cch_rewrite_profile();
        let cver = cch_profile.claude_cli_version;
        for input in inputs {
            let expected_suffix = claude_code_version_suffix(input, cver);
            let header = cch_profile.billing_header_text(input);
            assert_eq!(
                header,
                format!(
                    "x-anthropic-billing-header: cc_version={cver}.{expected_suffix}; \
                     cc_entrypoint=sdk-cli; cch=00000;"
                ),
                "cch billing_header_text diverged from suffix oracle for {input:?}"
            );
        }
    }
}
