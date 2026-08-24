//! Inbound parse of official prompt-cache fields into canonical intent.
//!
//! Chat Completions, Responses, and Anthropic Messages each keep their own
//! official fields. Invalid values on the inbound dialect are 400. Clamp-down
//! happens later, at outbound translation.

use serde_json::{Map, Value};

use omni_core::{
    CanonicalCacheIntent, CanonicalCacheMark, CanonicalCacheMode, CanonicalCacheRetention,
    CanonicalCacheTtl,
};

/// Top-level OpenAI cache keys that must not remain as provider extras.
pub const OPENAI_CACHE_EXTRA_KEYS: &[&str] = &[
    "prompt_cache_key",
    "prompt_cache_options",
    "prompt_cache_retention",
];

/// Lift `prompt_cache_key` from a JSON value (flattened extras or a body key).
///
/// Empty / whitespace / JSON null is absent. A non-string is a 400: this is a
/// routing identity, not a silent drop.
pub fn parse_prompt_cache_key(value: Option<&Value>) -> Result<Option<String>, String> {
    match value {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(s)) => {
            let trimmed = s.trim();
            if trimmed.is_empty() {
                Ok(None)
            } else {
                Ok(Some(trimmed.to_string()))
            }
        }
        Some(_) => Err("prompt_cache_key must be a string".into()),
    }
}

/// Header `x-grok-conv-id` → routing identity. Empty / whitespace is absent.
pub fn parse_grok_conv_id_header(value: Option<&str>) -> Result<Option<String>, String> {
    match value {
        None => Ok(None),
        Some(s) => {
            let trimmed = s.trim();
            if trimmed.is_empty() {
                Ok(None)
            } else if trimmed.bytes().any(|b| b == b'\n' || b == b'\r' || b == 0) {
                Err("x-grok-conv-id must not contain CR, LF, or NUL".into())
            } else {
                Ok(Some(trimmed.to_string()))
            }
        }
    }
}

/// Body key wins when both header and body are present. Never forward both.
pub fn merge_routing_identity(
    body_key: Option<String>,
    header_key: Option<String>,
) -> Option<String> {
    body_key.or(header_key)
}

pub fn parse_prompt_cache_options(
    value: Option<&Value>,
) -> Result<(Option<CanonicalCacheMode>, Option<CanonicalCacheTtl>), String> {
    match value {
        None | Some(Value::Null) => Ok((None, None)),
        Some(Value::Object(obj)) => {
            let mode = match obj.get("mode") {
                None | Some(Value::Null) => None,
                Some(Value::String(s)) => Some(parse_cache_mode(s)?),
                Some(_) => return Err("prompt_cache_options.mode must be a string".into()),
            };
            let ttl = match obj.get("ttl") {
                None | Some(Value::Null) => None,
                Some(Value::String(s)) => Some(parse_openai_ttl(s)?),
                Some(_) => return Err("prompt_cache_options.ttl must be a string".into()),
            };
            Ok((mode, ttl))
        }
        Some(_) => Err("prompt_cache_options must be an object".into()),
    }
}

pub fn parse_prompt_cache_retention(
    value: Option<&Value>,
) -> Result<Option<CanonicalCacheRetention>, String> {
    match value {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(s)) => match s.as_str() {
            "in_memory" => Ok(Some(CanonicalCacheRetention::InMemory)),
            "24h" => Ok(Some(CanonicalCacheRetention::TwentyFourHours)),
            other => Err(format!(
                "prompt_cache_retention must be \"in_memory\" or \"24h\", got {other:?}"
            )),
        },
        Some(_) => Err("prompt_cache_retention must be a string".into()),
    }
}

pub fn parse_prompt_cache_breakpoint(
    value: Option<&Value>,
) -> Result<Option<CanonicalCacheMark>, String> {
    match value {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Object(obj)) => {
            match obj.get("mode") {
                Some(Value::String(s)) if s == "explicit" => {}
                Some(Value::String(s)) => {
                    return Err(format!(
                        "prompt_cache_breakpoint.mode must be \"explicit\", got {s:?}"
                    ));
                }
                Some(_) => return Err("prompt_cache_breakpoint.mode must be a string".into()),
                None => return Err("prompt_cache_breakpoint.mode is required".into()),
            }
            Ok(Some(CanonicalCacheMark::breakpoint()))
        }
        Some(_) => Err("prompt_cache_breakpoint must be an object".into()),
    }
}

pub fn parse_openai_cache_intent(
    extras: &Value,
    header_conv_id: Option<&str>,
) -> Result<Option<CanonicalCacheIntent>, String> {
    if extras.get("prompt_cache_breakpoint").is_some() {
        return Err("prompt_cache_breakpoint is only valid on content parts".into());
    }
    let body_key = parse_prompt_cache_key(extras.get("prompt_cache_key"))?;
    let header_key = parse_grok_conv_id_header(header_conv_id)?;
    let routing_identity = merge_routing_identity(body_key, header_key);
    let (mode, ttl) = parse_prompt_cache_options(extras.get("prompt_cache_options"))?;
    let legacy_retention = parse_prompt_cache_retention(extras.get("prompt_cache_retention"))?;
    let intent = CanonicalCacheIntent {
        routing_identity,
        mode,
        ttl,
        legacy_retention,
        automatic: None,
    };
    if intent.is_empty() {
        Ok(None)
    } else {
        Ok(Some(intent))
    }
}

pub fn parse_anthropic_cache_control(
    value: Option<&Value>,
) -> Result<Option<CanonicalCacheMark>, String> {
    match value {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Object(obj)) => {
            match obj.get("type").and_then(Value::as_str) {
                Some("ephemeral") => {}
                Some(other) => {
                    return Err(format!(
                        "cache_control.type must be \"ephemeral\", got {other:?}"
                    ));
                }
                None => return Err("cache_control.type is required".into()),
            }
            let ttl = match obj.get("ttl") {
                None | Some(Value::Null) => None,
                Some(Value::String(s)) => Some(parse_anthropic_ttl(s)?),
                Some(_) => return Err("cache_control.ttl must be a string".into()),
            };
            Ok(Some(CanonicalCacheMark { ttl }))
        }
        Some(_) => Err("cache_control must be an object".into()),
    }
}

pub fn reject_anthropic_prompt_cache_key(body: &Value) -> Result<(), String> {
    if body.get("prompt_cache_key").is_some() {
        return Err("prompt_cache_key is not a valid Anthropic Messages field".into());
    }
    Ok(())
}

pub fn strip_openai_cache_keys(obj: &Map<String, Value>) -> Map<String, Value> {
    obj.iter()
        .filter(|(key, _)| !OPENAI_CACHE_EXTRA_KEYS.contains(&key.as_str()))
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect()
}

fn parse_cache_mode(s: &str) -> Result<CanonicalCacheMode, String> {
    match s {
        "implicit" => Ok(CanonicalCacheMode::Implicit),
        "explicit" => Ok(CanonicalCacheMode::Explicit),
        other => Err(format!(
            "prompt_cache_options.mode must be \"implicit\" or \"explicit\", got {other:?}"
        )),
    }
}

fn parse_openai_ttl(s: &str) -> Result<CanonicalCacheTtl, String> {
    match s {
        "30m" => Ok(CanonicalCacheTtl::ThirtyMinutes),
        other => Err(format!(
            "prompt_cache_options.ttl must be \"30m\", got {other:?}"
        )),
    }
}

fn parse_anthropic_ttl(s: &str) -> Result<CanonicalCacheTtl, String> {
    match s {
        "5m" => Ok(CanonicalCacheTtl::FiveMinutes),
        "1h" => Ok(CanonicalCacheTtl::OneHour),
        other => Err(format!(
            "cache_control.ttl must be \"5m\" or \"1h\", got {other:?}"
        )),
    }
}
