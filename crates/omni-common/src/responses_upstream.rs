//! OpenAI-Responses *upstream* protocol machinery (shared, provider-neutral).
//!
//! This is the single source of truth for the highest-wire-risk surface: the
//! pure SSE framing + the streaming event parser + the non-stream
//! response->canonical mapper for backends that speak the OpenAI Responses wire
//! (`response.created`, `response.output_text.delta`, ..., `response.completed`,
//! with NO `[DONE]` sentinel - that is a Chat Completions convention only).
//!
//! It was extracted verbatim from provider-codex so that any provider talking
//! the same wire at a different host (e.g. Grok CLI path) can reuse
//! the exact same parsing, guaranteeing wire parity. To stay decoupled from any
//! one provider, the three Codex-specific couplings are parameterized:
//!   1. the canonical metadata `provider` tag -> [`response_to_canonical`] takes
//!      a `provider_tag: &str`,
//!   2. error-string redaction -> abstracted behind the [`ErrorRedactor`] trait,
//!   3. the literal "codex" substrings in error messages -> reworded to
//!      provider-neutral text ("Responses stream ...").
//!
//! Request BODY builders stay per-provider; only the response/stream protocol
//! lives here.

use std::collections::HashMap;

use omni_core::{
    CanonicalResponse, CanonicalResponseMetadata, CanonicalStreamEvent, CanonicalToolCall,
    CanonicalUsage, ProviderError,
};
use serde_json::Value;

/// Hard caps on SSE framing to bound memory against a hostile or broken
/// upstream. A single line may not exceed [`MAX_SSE_LINE_BYTES`]; the
/// accumulated `data:` payload of one event may not exceed
/// [`MAX_SSE_EVENT_BYTES`]. These match the original Codex values exactly so
/// the framing behavior is identical across providers.
pub const MAX_SSE_LINE_BYTES: usize = 1024 * 1024;
pub const MAX_SSE_EVENT_BYTES: usize = 8 * 1024 * 1024;

/// Redacts secrets out of error strings before they are surfaced to a caller.
///
/// The parser and the non-stream mapper run upstream payloads through this
/// before wrapping them in [`ProviderError::Upstream`], so credentials that an
/// upstream may echo back in an error never leak. Each provider supplies its
/// own implementation (it knows which header/query secrets to scrub); the
/// shared protocol code only needs the `redact` operation.
///
/// The bounds (`Clone + Default + Debug`) match how [`ResponsesStreamParser`]
/// uses the redactor: it stores it in a field, is `#[derive(Default)]`, and
/// clones it per stream.
pub trait ErrorRedactor: Clone + Default + std::fmt::Debug {
    /// Return `input` with any known secrets replaced. Must be lossless with
    /// respect to non-secret content (callers rely on the redacted text still
    /// describing the error).
    fn redact(&self, input: &str) -> String;
}

/// Prefix scrubber for secrets in an upstream error body. Scans for each marker
/// prefix and replaces from the marker to the next delimiter (whitespace /
/// quote / comma) with `<redacted>`. This catches known-prefix secrets even when
/// no resolved credentials are in scope; providers layer their exact captured
/// secrets on top for tokens that carry no known prefix.
///
/// Each provider passes its own marker set: Grok and Codex scrub
/// `["sk-", "xai-", "eyJ"]`; Claude scrubs `["sk-", "eyJ"]` (no xAI keys reach
/// the Anthropic path, and `sk-` already covers Claude OAuth `sk-ant-oat01-...`
/// and custom-gateway `sk-...` keys). `eyJ` covers JWT bearers.
pub fn redact_prefixed_secrets(input: &str, markers: &[&str]) -> String {
    let mut out = input.to_string();
    for marker in markers {
        while let Some(pos) = out.find(marker) {
            let end = out[pos..]
                .find(|c: char| c.is_whitespace() || c == '"' || c == '\'' || c == ',')
                .map(|i| pos + i)
                .unwrap_or(out.len());
            out.replace_range(pos..end, "<redacted>");
        }
    }
    out
}

/// Map a non-stream OpenAI-Responses payload to a [`CanonicalResponse`].
///
/// `fallback_model` is used when the payload omits `model`. `provider_tag` is
/// stamped into [`CanonicalResponseMetadata::provider`] so the caller can tell
/// which backend produced the response. `error_redactor` scrubs secrets from
/// the error body when the upstream reports `status == "failed"`.
pub fn response_to_canonical(
    value: &Value,
    fallback_model: &str,
    provider_tag: &str,
    error_redactor: &impl ErrorRedactor,
) -> Result<CanonicalResponse, ProviderError> {
    if value.get("status").and_then(|v| v.as_str()) == Some("failed") {
        return Err(ProviderError::upstream(
            error_redactor.redact(&value.to_string()),
        ));
    }

    let model = value
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or(fallback_model)
        .to_string();
    let response_id = value.get("id").and_then(|v| v.as_str()).map(str::to_string);
    let mut content = String::new();
    let mut refusal = String::new();
    let mut tool_calls = Vec::new();
    let mut annotations = Vec::new();
    if let Some(items) = value.get("output").and_then(|v| v.as_array()) {
        for item in items {
            match item.get("type").and_then(|v| v.as_str()) {
                Some("message") => {
                    if let Some(parts) = item.get("content").and_then(|v| v.as_array()) {
                        for part in parts {
                            if let Some(text) = part.get("text").and_then(|v| v.as_str()) {
                                content.push_str(text);
                                if let Some(part_annotations) =
                                    part.get("annotations").and_then(|v| v.as_array())
                                {
                                    annotations.extend(part_annotations.iter().cloned());
                                }
                            } else if let Some(refusal_text) =
                                part.get("refusal").and_then(|v| v.as_str())
                            {
                                refusal.push_str(refusal_text);
                            }
                        }
                    }
                }
                Some("function_call") => {
                    let id = item
                        .get("call_id")
                        .or_else(|| item.get("id"))
                        .and_then(|v| v.as_str())
                        .unwrap_or("call_unknown")
                        .to_string();
                    let name = item
                        .get("name")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    let arguments = item
                        .get("arguments")
                        .and_then(|v| v.as_str())
                        .unwrap_or("{}")
                        .to_string();
                    tool_calls.push(CanonicalToolCall {
                        id,
                        name,
                        arguments,
                    });
                }
                _ => {}
            }
        }
    }
    if content.is_empty()
        && let Some(text) = value.get("output_text").and_then(|v| v.as_str())
    {
        content.push_str(text);
    }

    let usage = response_usage(value).unwrap_or_default();
    let finish_reason = match response_status(value) {
        Some("incomplete") => Some(response_incomplete_reason(value).to_string()),
        _ if !tool_calls.is_empty() => Some("tool_calls".to_string()),
        _ => Some("stop".to_string()),
    };

    Ok(CanonicalResponse {
        model,
        content,
        refusal: if refusal.is_empty() {
            None
        } else {
            Some(refusal)
        },
        tool_calls,
        finish_reason,
        usage,
        id: response_id.clone(),
        annotations,
        metadata: Some(CanonicalResponseMetadata {
            id: response_id,
            system_fingerprint: value
                .get("system_fingerprint")
                .and_then(|v| v.as_str())
                .map(str::to_string),
            service_tier: value
                .get("service_tier")
                .and_then(|v| v.as_str())
                .map(str::to_string),
            provider: Some(provider_tag.to_string()),
            raw: None,
        }),
        reasoning: Vec::new(),
    })
}

/// Incremental SSE framer for the Responses wire.
///
/// Feeds raw response bytes (which may split a UTF-8 char, a line, or an event
/// across chunks) and yields complete [`ResponsesSseEvent`]s. Handles `\n`,
/// `\r\n`, and bare `\r` line endings, ignores comment (`:`) lines, and rejects
/// lines/events that exceed the byte caps. Call [`finish`](Self::finish) at end
/// of stream to flush any buffered trailing event.
#[derive(Debug, Default)]
pub struct ResponsesSseBuffer {
    line: Vec<u8>,
    last_was_cr: bool,
    event: Option<String>,
    data: Vec<String>,
    event_bytes: usize,
}

/// One framed SSE event: an optional `event:` name and the joined `data:` body.
#[derive(Debug)]
pub struct ResponsesSseEvent {
    pub event: Option<String>,
    pub data: String,
}

impl ResponsesSseBuffer {
    /// Push a chunk of raw bytes, returning any events completed by it.
    pub fn push(&mut self, bytes: &[u8]) -> Result<Vec<ResponsesSseEvent>, String> {
        let mut events = Vec::new();
        for line in self.complete_lines(bytes)? {
            self.process_line(line, &mut events)?;
        }
        Ok(events)
    }

    fn complete_lines(&mut self, bytes: &[u8]) -> Result<Vec<String>, String> {
        let mut lines = Vec::new();
        for byte in bytes {
            if self.last_was_cr {
                self.last_was_cr = false;
                if *byte == b'\n' {
                    continue;
                }
            }
            match *byte {
                b'\n' => lines.push(self.take_line()?),
                b'\r' => {
                    lines.push(self.take_line()?);
                    self.last_was_cr = true;
                }
                byte => {
                    self.line.push(byte);
                    if self.line.len() > MAX_SSE_LINE_BYTES {
                        return Err(format!(
                            "Responses stream line exceeded {} bytes",
                            MAX_SSE_LINE_BYTES
                        ));
                    }
                }
            }
        }
        Ok(lines)
    }

    fn take_line(&mut self) -> Result<String, String> {
        String::from_utf8(std::mem::take(&mut self.line))
            .map_err(|e| format!("Responses stream line was not UTF-8: {e}"))
    }

    fn process_line(
        &mut self,
        line: String,
        events: &mut Vec<ResponsesSseEvent>,
    ) -> Result<(), String> {
        if line.is_empty() {
            if let Some(event) = self.take_event() {
                events.push(event);
            }
            return Ok(());
        }
        if line.starts_with(':') {
            return Ok(());
        }
        if let Some(value) = line.strip_prefix("event:") {
            self.event = Some(value.trim().to_string());
        } else if let Some(value) = line.strip_prefix("data:") {
            let value = value.trim_start();
            self.event_bytes = self.event_bytes.saturating_add(value.len());
            if self.event_bytes > MAX_SSE_EVENT_BYTES {
                return Err(format!(
                    "Responses stream event exceeded {} bytes",
                    MAX_SSE_EVENT_BYTES
                ));
            }
            self.data.push(value.to_string());
        }
        Ok(())
    }

    /// Flush a trailing event that was not terminated by a blank line.
    pub fn finish(&mut self) -> Result<Option<ResponsesSseEvent>, String> {
        if !self.line.is_empty() {
            let line = self.take_line()?;
            let mut events = Vec::new();
            self.process_line(line, &mut events)?;
            if let Some(event) = events.into_iter().next() {
                return Ok(Some(event));
            }
        }
        Ok(self.take_event())
    }

    fn take_event(&mut self) -> Option<ResponsesSseEvent> {
        if self.event.is_none() && self.data.is_empty() {
            return None;
        }
        let event = ResponsesSseEvent {
            event: self.event.take(),
            data: std::mem::take(&mut self.data).join("\n"),
        };
        self.event_bytes = 0;
        Some(event)
    }
}

#[derive(Debug, Clone, Default)]
struct StreamToolCall {
    id: Option<String>,
    name: Option<String>,
    emitted_open: bool,
    arguments: String,
    emitted_arguments_len: usize,
    canonical_index: u32,
}

/// How no-match free-claim behaves when a terminal/full item carries a call_id
/// that is not yet on any slot.
#[derive(Debug, Clone, Copy)]
enum FreeClaimPolicy {
    /// completed/incomplete: claim only missing or empty-untagged at bare index.
    StrictEmptyOrMissing,
    /// stream item.done: may attach id to untagged (incl. non-empty) at bare index
    /// so metadata can arrive after argument deltas.
    AllowUntaggedNonEmpty,
}

/// Stateful parser that turns framed Responses SSE events into canonical
/// stream events.
///
/// Construct with [`new`](Self::new), then feed each [`ResponsesSseEvent`] to
/// [`handle_event`](Self::handle_event). The parser accumulates text/refusal
/// deltas, assembles parallel function-call arguments (absorbing gateways that
/// repeat the full arguments), and on a terminal `response.completed` /
/// `response.incomplete` emits any remaining content, a [`Usage`] event, and a
/// single [`Finish`]. A `[DONE]` sentinel or `response.failed` / `error` event
/// is surfaced as a redacted [`ProviderError::Upstream`].
///
/// [`Usage`]: CanonicalStreamEvent::Usage
/// [`Finish`]: CanonicalStreamEvent::Finish
#[derive(Debug, Default)]
pub struct ResponsesStreamParser<R: ErrorRedactor> {
    tool_calls: HashMap<u32, StreamToolCall>,
    next_tool_index: u32,
    saw_tool_call: bool,
    emitted_text: HashMap<(u32, &'static str), String>,
    completed: bool,
    provider_tag: String,
    error_redactor: R,
}

impl<R: ErrorRedactor> ResponsesStreamParser<R> {
    /// Create a parser stamping `provider_tag` into emitted response metadata
    /// and using `error_redactor` to scrub surfaced errors.
    pub fn new(provider_tag: &str, error_redactor: R) -> Self {
        Self {
            provider_tag: provider_tag.to_string(),
            error_redactor,
            ..Default::default()
        }
    }

    /// Whether a terminal event (completed/incomplete) has been observed.
    pub fn completed(&self) -> bool {
        self.completed
    }

    fn redact(&self, input: &str) -> String {
        self.error_redactor.redact(input)
    }

    /// Parse one framed SSE event into zero or more canonical stream events.
    pub fn handle_event(
        &mut self,
        event: ResponsesSseEvent,
    ) -> Vec<Result<CanonicalStreamEvent, ProviderError>> {
        let event_type = event.event.as_deref().unwrap_or_default();
        if event.data.trim() == "[DONE]" {
            return vec![Err(ProviderError::upstream(
                "Responses stream sent Chat [DONE] sentinel without a terminal response event",
            ))];
        }
        let value: Value = match serde_json::from_str(&event.data) {
            Ok(value) => value,
            Err(e) => {
                return vec![Err(ProviderError::upstream(self.redact(&format!(
                    "decode Responses stream event {event_type}: {e}: {}",
                    event.data
                ))))];
            }
        };
        let kind = value
            .get("type")
            .and_then(|v| v.as_str())
            .unwrap_or(event_type);
        match kind {
            "response.created" => self.handle_response_metadata(&value),
            "response.output_text.delta" | "response.refusal.delta" => self
                .emit_text_delta(
                    response_output_index(&value),
                    if kind == "response.refusal.delta" {
                        "refusal"
                    } else {
                        "text"
                    },
                    value
                        .get("delta")
                        .and_then(|v| v.as_str())
                        .unwrap_or_default(),
                )
                .into_iter()
                .map(Ok)
                .collect(),
            "response.output_text.done" => self.handle_text_done(&value, "text"),
            "response.refusal.done" => self.handle_text_done(&value, "refusal"),
            "response.output_item.added" => self.handle_output_item_added(&value),
            "response.function_call_arguments.delta" => self.handle_function_args_delta(&value),
            "response.function_call_arguments.done" => self.handle_function_args_done(&value),
            "response.output_item.done" => self.handle_output_item_done(&value),
            "response.completed" => self.handle_completed(&value),
            "response.incomplete" => self.handle_incomplete(&value),
            "response.failed" | "error" => {
                vec![Err(ProviderError::upstream(
                    self.redact(&value.to_string()),
                ))]
            }
            _ => Vec::new(),
        }
    }

    fn handle_response_metadata(
        &self,
        value: &Value,
    ) -> Vec<Result<CanonicalStreamEvent, ProviderError>> {
        let id = value
            .get("response")
            .and_then(|v| v.get("id"))
            .or_else(|| value.get("id"))
            .and_then(|v| v.as_str())
            .map(str::to_string);
        let response = response_payload(value);
        let metadata = CanonicalResponseMetadata {
            id,
            system_fingerprint: response
                .get("system_fingerprint")
                .and_then(|v| v.as_str())
                .map(str::to_string),
            service_tier: response
                .get("service_tier")
                .and_then(|v| v.as_str())
                .map(str::to_string),
            provider: Some(self.provider_tag.clone()),
            raw: None,
        };
        if metadata.id.is_none()
            && metadata.system_fingerprint.is_none()
            && metadata.service_tier.is_none()
        {
            Vec::new()
        } else {
            vec![Ok(CanonicalStreamEvent::ResponseMetadata(metadata))]
        }
    }

    fn emit_text_delta(
        &mut self,
        output_index: u32,
        channel: &'static str,
        delta: &str,
    ) -> Vec<CanonicalStreamEvent> {
        if delta.is_empty() {
            return Vec::new();
        }
        self.emitted_text
            .entry((output_index, channel))
            .or_default()
            .push_str(delta);
        let delta = delta.to_string();
        if channel == "refusal" {
            vec![CanonicalStreamEvent::RefusalDelta(delta)]
        } else {
            vec![CanonicalStreamEvent::TextDelta(delta)]
        }
    }

    fn handle_text_done(
        &mut self,
        value: &Value,
        field: &'static str,
    ) -> Vec<Result<CanonicalStreamEvent, ProviderError>> {
        let final_text = value
            .get(field)
            .or_else(|| value.get("text"))
            .or_else(|| value.get("content"))
            .and_then(|v| v.as_str())
            .unwrap_or_default();
        let output_index = response_output_index(value);
        self.emit_final_text(output_index, field, final_text)
    }

    fn emit_final_text(
        &mut self,
        output_index: u32,
        field: &'static str,
        final_text: &str,
    ) -> Vec<Result<CanonicalStreamEvent, ProviderError>> {
        if final_text.is_empty() {
            return Vec::new();
        }
        let emitted = self
            .emitted_text
            .get(&(output_index, field))
            .map(String::as_str)
            .unwrap_or_default();
        if !final_text.starts_with(emitted) {
            return vec![Err(ProviderError::upstream(self.redact(&format!(
                "Responses stream {field}.done text did not extend prior text deltas"
            ))))];
        }
        let suffix = &final_text[emitted.len()..];
        self.emit_text_delta(output_index, field, suffix)
            .into_iter()
            .map(Ok)
            .collect()
    }

    fn handle_output_item_added(
        &mut self,
        value: &Value,
    ) -> Vec<Result<CanonicalStreamEvent, ProviderError>> {
        let Some(item) = value.get("item") else {
            return Vec::new();
        };
        if item.get("type").and_then(|v| v.as_str()) != Some("function_call") {
            return Vec::new();
        }
        let output_index = value
            .get("output_index")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u32;
        // Stream announcement: bare output_index is authoritative for following
        // deltas. Do not re-key by call_id here (that would desync later deltas).
        let canonical_index = self.ensure_tool_call(output_index);
        let call = self.tool_calls.entry(output_index).or_default();
        call.id = item
            .get("call_id")
            .or_else(|| item.get("id"))
            .and_then(|v| v.as_str())
            .map(str::to_string);
        call.name = item
            .get("name")
            .and_then(|v| v.as_str())
            .map(str::to_string);
        self.saw_tool_call = true;
        if let Some(arguments) = item.get("arguments").and_then(|v| v.as_str()) {
            if let Some(err) = self.merge_full_tool_arguments(output_index, arguments) {
                return vec![Err(err)];
            }
        }
        let mut events = self.emit_tool_open_if_ready(output_index);
        events.extend(self.emit_pending_tool_args(output_index, canonical_index));
        events.into_iter().map(Ok).collect::<Vec<_>>()
    }

    fn handle_function_args_delta(
        &mut self,
        value: &Value,
    ) -> Vec<Result<CanonicalStreamEvent, ProviderError>> {
        let output_index = value
            .get("output_index")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u32;
        let delta = value
            .get("delta")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string();
        self.saw_tool_call = true;
        let canonical_index = self.ensure_tool_call(output_index);
        if !delta.is_empty() {
            let already = self
                .tool_calls
                .get(&output_index)
                .map(|call| call.arguments.clone())
                .unwrap_or_default();
            if !already.is_empty() && delta == already {
                // Some Responses-compatible gateways repeat the full arguments
                // as the first delta after announcing them on output_item.added.
            } else if delta.starts_with(&already) && delta.len() > already.len() {
                self.append_tool_arguments(output_index, &delta[already.len()..]);
            } else {
                self.append_tool_arguments(output_index, &delta);
            }
        }
        let mut events = self.emit_tool_open_if_ready(output_index);
        events.extend(self.emit_pending_tool_args(output_index, canonical_index));
        events.into_iter().map(Ok).collect()
    }

    fn handle_function_args_done(
        &mut self,
        value: &Value,
    ) -> Vec<Result<CanonicalStreamEvent, ProviderError>> {
        let output_index = value
            .get("output_index")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u32;
        let Some(arguments) = value.get("arguments").and_then(|v| v.as_str()) else {
            return Vec::new();
        };
        let already = self
            .tool_calls
            .get(&output_index)
            .map(|call| call.arguments.clone())
            .unwrap_or_default();
        if arguments == already {
            return Vec::new();
        }
        if arguments.len() <= already.len() || !arguments.starts_with(&already) {
            return vec![Err(ProviderError::upstream(self.redact(
				"Responses stream function_call_arguments.done arguments did not extend prior argument deltas",
			)))];
        }
        let delta = arguments[already.len()..].to_string();
        let canonical_index = self.ensure_tool_call(output_index);
        self.append_tool_arguments(output_index, &delta);
        let mut events = self.emit_tool_open_if_ready(output_index);
        events.extend(self.emit_pending_tool_args(output_index, canonical_index));
        events.into_iter().map(Ok).collect()
    }

    fn handle_output_item_done(
        &mut self,
        value: &Value,
    ) -> Vec<Result<CanonicalStreamEvent, ProviderError>> {
        let Some(item) = value.get("item") else {
            return Vec::new();
        };
        if item.get("type").and_then(|v| v.as_str()) != Some("function_call") {
            return Vec::new();
        }
        let output_index = value
            .get("output_index")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u32;
        // Full-arg item.done shares call_id re-key + validate-before-mutate with
        // the terminal path, but free-claim may attach to untagged non-empty at
        // this stream index (metadata can arrive after deltas).
        self.apply_function_call_full_item(
            output_index,
            item,
            FreeClaimPolicy::AllowUntaggedNonEmpty,
        )
    }

    fn handle_completed(
        &mut self,
        value: &Value,
    ) -> Vec<Result<CanonicalStreamEvent, ProviderError>> {
        if response_status(value) == Some("failed") {
            return vec![Err(ProviderError::upstream(
                self.redact(&value.to_string()),
            ))];
        }
        self.completed = true;
        let mut events = self.handle_response_metadata(value);
        events.extend(self.emit_terminal_output(value));
        if let Some(usage) = response_usage(value) {
            events.push(Ok(CanonicalStreamEvent::Usage(usage)));
        }
        events.push(Ok(CanonicalStreamEvent::Finish {
            finish_reason: self.finish_reason(),
        }));
        events
    }

    fn handle_incomplete(
        &mut self,
        value: &Value,
    ) -> Vec<Result<CanonicalStreamEvent, ProviderError>> {
        self.completed = true;
        let mut events = self.handle_response_metadata(value);
        events.extend(self.emit_terminal_output(value));
        if let Some(usage) = response_usage(value) {
            events.push(Ok(CanonicalStreamEvent::Usage(usage)));
        }
        events.push(Ok(CanonicalStreamEvent::Finish {
            finish_reason: Some(response_incomplete_reason(value).to_string()),
        }));
        events
    }

    fn emit_terminal_output(
        &mut self,
        value: &Value,
    ) -> Vec<Result<CanonicalStreamEvent, ProviderError>> {
        let mut events = Vec::new();
        let Some(items) = response_payload(value)
            .get("output")
            .and_then(|v| v.as_array())
        else {
            return events;
        };
        for (position, item) in items.iter().enumerate() {
            let output_index = item
                .get("output_index")
                .and_then(|v| v.as_u64())
                .and_then(|value| u32::try_from(value).ok())
                .unwrap_or(position as u32);
            match item.get("type").and_then(|v| v.as_str()) {
                Some("message") => {
                    if let Some(parts) = item.get("content").and_then(|v| v.as_array()) {
                        for part in parts {
                            events.extend(self.emit_terminal_content_part(output_index, part));
                        }
                    }
                }
                Some("function_call") => {
                    events.extend(self.emit_terminal_function_call(output_index, item));
                }
                _ => {}
            }
        }
        events
    }

    fn emit_terminal_content_part(
        &mut self,
        output_index: u32,
        part: &Value,
    ) -> Vec<Result<CanonicalStreamEvent, ProviderError>> {
        let kind = part.get("type").and_then(|v| v.as_str());
        if kind == Some("refusal") || part.get("refusal").is_some() {
            let final_text = part
                .get("refusal")
                .and_then(|v| v.as_str())
                .unwrap_or_default();
            return self.emit_final_text(output_index, "refusal", final_text);
        }
        if kind == Some("output_text") || part.get("text").is_some() {
            let final_text = part
                .get("text")
                .and_then(|v| v.as_str())
                .unwrap_or_default();
            let mut events = self.emit_final_text(output_index, "text", final_text);
            if let Some(annotations) = part.get("annotations").and_then(|v| v.as_array())
                && !annotations.is_empty()
            {
                events.push(Ok(CanonicalStreamEvent::OutputAnnotations(
                    annotations.to_vec(),
                )));
            }
            return events;
        }
        Vec::new()
    }

    fn emit_terminal_function_call(
        &mut self,
        output_index: u32,
        item: &Value,
    ) -> Vec<Result<CanonicalStreamEvent, ProviderError>> {
        // Class-K measured site (completed/incomplete): unique call_id re-key
        // with strict free-claim (missing/empty-untagged only).
        self.apply_function_call_full_item(
            output_index,
            item,
            FreeClaimPolicy::StrictEmptyOrMissing,
        )
    }

    /// Shared full-arg function_call apply for terminal completed/incomplete and
    /// stream `output_item.done`. Sequence: resolve by call_id → ensure → content
    /// merge (validate-before-mutate) → fill missing id/name → emit from the
    /// **resolved** index. Flight `call_id_match` (if logged) must sample the
    /// original `output_index` before resolve.
    fn apply_function_call_full_item(
        &mut self,
        output_index: u32,
        item: &Value,
        free_claim: FreeClaimPolicy,
    ) -> Vec<Result<CanonicalStreamEvent, ProviderError>> {
        let terminal_call_id = item
            .get("call_id")
            .or_else(|| item.get("id"))
            .and_then(|v| v.as_str());
        let terminal_name = item.get("name").and_then(|v| v.as_str());

        let merge_index =
            match self.resolve_tool_merge_index(output_index, terminal_call_id, free_claim) {
                Ok(idx) => idx,
                Err(err) => return vec![Err(err)],
            };

        let canonical_index = self.ensure_tool_call(merge_index);
        if let Some(arguments) = item.get("arguments").and_then(|v| v.as_str()) {
            if let Some(err) = self.merge_full_tool_arguments(merge_index, arguments) {
                return vec![Err(err)];
            }
        }
        {
            let call = self.tool_calls.entry(merge_index).or_default();
            if call.id.is_none() {
                call.id = terminal_call_id.map(str::to_string);
            }
            if call.name.is_none() {
                call.name = terminal_name.map(str::to_string);
            }
        }
        self.saw_tool_call = true;
        let mut events = self.emit_tool_open_if_ready(merge_index);
        events.extend(self.emit_pending_tool_args(merge_index, canonical_index));
        events.into_iter().map(Ok).collect()
    }

    fn ensure_tool_call(&mut self, output_index: u32) -> u32 {
        if let Some(call) = self.tool_calls.get(&output_index) {
            return call.canonical_index;
        }
        let canonical_index = self.next_tool_index;
        self.next_tool_index += 1;
        self.tool_calls.insert(
            output_index,
            StreamToolCall {
                canonical_index,
                ..Default::default()
            },
        );
        canonical_index
    }

    /// Choose the `tool_calls` map key for a full/terminal arg merge.
    ///
    /// When `terminal_call_id` is present:
    /// - unique existing `id == call_id` → that key (ignore bare `output_index`)
    /// - multi-match → fail-loud, zero mutations
    /// - no match → free-claim at `output_index` per `free_claim` policy
    ///
    /// When absent: bare `output_index` (legacy path).
    fn resolve_tool_merge_index(
        &mut self,
        output_index: u32,
        terminal_call_id: Option<&str>,
        free_claim: FreeClaimPolicy,
    ) -> Result<u32, ProviderError> {
        let Some(cid) = terminal_call_id else {
            return Ok(output_index);
        };

        let matches: Vec<u32> = self
            .tool_calls
            .iter()
            .filter(|(_, call)| call.id.as_deref() == Some(cid))
            .map(|(&idx, _)| idx)
            .collect();

        match matches.as_slice() {
            [only] => Ok(*only),
            [] => {
                // Pure decision only: do not attach id or ensure here. Claiming
                // id before content merge would leave a poisoned call_id if
                // merge_full_tool_arguments later fails (item.done conflict).
                // apply_function_call_full_item attaches id after content accepts.
                let free = match (free_claim, self.tool_calls.get(&output_index)) {
                    (_, None) => true,
                    // Stream item.done: metadata may land after deltas filled an
                    // untagged buffer at this same index.
                    (FreeClaimPolicy::AllowUntaggedNonEmpty, Some(call)) => call.id.is_none(),
                    // Terminal class-K: never claim untagged non-empty (prefix-
                    // compatible JSON is common across tools).
                    (FreeClaimPolicy::StrictEmptyOrMissing, Some(call)) => {
                        call.id.is_none() && call.arguments.is_empty()
                    }
                };
                if !free {
                    return Err(self.terminal_function_call_args_error());
                }
                Ok(output_index)
            }
            _ => Err(self.terminal_function_call_args_error()),
        }
    }

    fn terminal_function_call_args_error(&self) -> ProviderError {
        ProviderError::upstream(self.redact(
            "Responses stream terminal function_call arguments did not extend prior argument deltas",
        ))
    }

    fn emit_tool_open(&mut self, output_index: u32) -> Vec<CanonicalStreamEvent> {
        let canonical_index = self.ensure_tool_call(output_index);
        let call = self.tool_calls.entry(output_index).or_default();
        if call.emitted_open {
            return Vec::new();
        }
        call.emitted_open = true;
        vec![CanonicalStreamEvent::ToolCallDelta {
            index: canonical_index,
            id: call.id.clone(),
            name: call.name.clone(),
            arguments_delta: String::new(),
        }]
    }

    fn emit_tool_open_if_ready(&mut self, output_index: u32) -> Vec<CanonicalStreamEvent> {
        let Some(call) = self.tool_calls.get(&output_index) else {
            return Vec::new();
        };
        if call.emitted_open || call.id.is_none() || call.name.is_none() {
            return Vec::new();
        }
        self.emit_tool_open(output_index)
    }

    fn append_tool_arguments(&mut self, output_index: u32, delta: &str) {
        if delta.is_empty() {
            return;
        }
        if let Some(call) = self.tool_calls.get_mut(&output_index) {
            call.arguments.push_str(delta);
        }
    }

    /// Content merge only (slot already chosen). Validate before any mutation:
    /// equal → no-op; terminal is prefix-extension of buffer → append suffix;
    /// else fail-loud with zero mutations.
    fn merge_full_tool_arguments(
        &mut self,
        merge_index: u32,
        arguments: &str,
    ) -> Option<ProviderError> {
        if arguments.is_empty() {
            return None;
        }
        let already = self
            .tool_calls
            .get(&merge_index)
            .map(|call| call.arguments.clone())
            .unwrap_or_default();
        if arguments == already {
            return None;
        }
        if arguments.len() > already.len() && arguments.starts_with(&already) {
            self.append_tool_arguments(merge_index, &arguments[already.len()..]);
            return None;
        }
        Some(self.terminal_function_call_args_error())
    }

    fn emit_pending_tool_args(
        &mut self,
        output_index: u32,
        canonical_index: u32,
    ) -> Vec<CanonicalStreamEvent> {
        let Some(call) = self.tool_calls.get_mut(&output_index) else {
            return Vec::new();
        };
        if !call.emitted_open || call.emitted_arguments_len >= call.arguments.len() {
            return Vec::new();
        }
        let delta = call.arguments[call.emitted_arguments_len..].to_string();
        call.emitted_arguments_len = call.arguments.len();
        vec![CanonicalStreamEvent::ToolCallDelta {
            index: canonical_index,
            id: None,
            name: None,
            arguments_delta: delta,
        }]
    }

    fn finish_reason(&self) -> Option<String> {
        Some(if self.saw_tool_call {
            "tool_calls".to_string()
        } else {
            "stop".to_string()
        })
    }
}

/// Extract token usage from a Responses payload (looking under `response.usage`
/// first, then top-level `usage`). Returns `None` when no usage is present.
pub fn response_usage(value: &Value) -> Option<CanonicalUsage> {
    let usage = value
        .get("response")
        .and_then(|v| v.get("usage"))
        .or_else(|| value.get("usage"))?;
    let input_audio_tokens = usage
        .get("input_tokens_details")
        .and_then(|v| v.get("audio_tokens"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let output_audio_tokens = usage
        .get("output_tokens_details")
        .and_then(|v| v.get("audio_tokens"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    Some(CanonicalUsage {
        input_tokens: usage
            .get("input_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0),
        output_tokens: usage
            .get("output_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0),
        cache_read: usage
            .get("input_tokens_details")
            .and_then(|v| v.get("cached_tokens"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0),
        cache_creation: 0,
        reasoning_tokens: usage
            .get("output_tokens_details")
            .and_then(|v| v.get("reasoning_tokens"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0),
        audio_tokens: input_audio_tokens + output_audio_tokens,
        input_audio_tokens,
        output_audio_tokens,
        accepted_prediction_tokens: usage
            .get("output_tokens_details")
            .and_then(|v| v.get("accepted_prediction_tokens"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0),
        rejected_prediction_tokens: usage
            .get("output_tokens_details")
            .and_then(|v| v.get("rejected_prediction_tokens"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0),
        ..CanonicalUsage::default()
    })
}

/// Read the response `status` ("completed", "incomplete", "failed", ...),
/// looking under a `response` envelope first.
pub fn response_status(value: &Value) -> Option<&str> {
    response_payload(value)
        .get("status")
        .and_then(|v| v.as_str())
}

/// Unwrap the `response` envelope when present, else return `value` itself.
/// Terminal stream events nest the payload under `response`; the non-stream
/// body is the payload directly.
pub fn response_payload(value: &Value) -> &Value {
    value.get("response").unwrap_or(value)
}

fn response_output_index(value: &Value) -> u32 {
    value
        .get("output_index")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as u32
}

/// Map an `incomplete` response to its canonical finish reason. The wire
/// `max_output_tokens` reason is normalized to the canonical `length`; any
/// other reason (e.g. `content_filter`) is preserved verbatim.
pub fn response_incomplete_reason(value: &Value) -> &str {
    let reason = value
        .get("response")
        .and_then(|v| v.get("incomplete_details"))
        .and_then(|v| v.get("reason"))
        .or_else(|| {
            value
                .get("incomplete_details")
                .and_then(|v| v.get("reason"))
        })
        .and_then(|v| v.as_str())
        .unwrap_or("max_output_tokens");
    if reason == "max_output_tokens" {
        "length"
    } else {
        reason
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const TAG: &str = "test-provider";

    /// A redactor that scrubs a single known token. Used to prove the parser
    /// and mapper still run errors through redaction (the wire-safety
    /// guarantee), without pulling in any provider-specific secret detection.
    #[derive(Clone, Debug, Default)]
    struct TokenRedactor;

    impl ErrorRedactor for TokenRedactor {
        fn redact(&self, input: &str) -> String {
            input.replace("sk-secret", "<redacted>")
        }
    }

    fn parser() -> ResponsesStreamParser<TokenRedactor> {
        ResponsesStreamParser::new(TAG, TokenRedactor)
    }

    #[test]
    fn redact_prefixed_secrets_honors_marker_set_and_delimiter() {
        // WHY: providers pass different marker sets (Claude omits `xai-`). A
        // marker NOT in the set must survive verbatim, or Claude would scrub
        // substrings it never intended to. Present markers must scrub from the
        // prefix up to the next delimiter (space/quote/comma), no further.
        let markers = ["sk-", "eyJ"];
        let out = redact_prefixed_secrets(
            r#"{"a":"sk-leak","b":"xai-keep","c":"eyJtok end"}"#,
            &markers,
        );
        assert!(!out.contains("sk-leak"), "prefixed secret leaked: {out}");
        assert!(!out.contains("eyJtok"), "jwt bearer leaked: {out}");
        assert!(
            out.contains("xai-keep"),
            "marker not in set must be untouched: {out}"
        );
        // Non-secret structure around the scrubbed spans is preserved.
        assert!(out.contains("<redacted>"));
        assert!(out.contains(r#""b":"xai-keep""#));
    }

    /// Feed a single SSE event (by name + JSON data) through a fresh framer and
    /// parser, returning the canonical events. WHY: exercises the real framing
    /// + parse path, not a hand-built event, so the two stay in lock-step.
    fn run_event(
        parser: &mut ResponsesStreamParser<TokenRedactor>,
        event_name: &str,
        data: Value,
    ) -> Vec<Result<CanonicalStreamEvent, ProviderError>> {
        let mut buffer = ResponsesSseBuffer::default();
        let chunk = format!("event: {event_name}\ndata: {data}\n\n");
        let framed = buffer.push(chunk.as_bytes()).expect("frame event");
        let mut out = Vec::new();
        for event in framed {
            out.extend(parser.handle_event(event));
        }
        out
    }

    fn ok_events(
        results: Vec<Result<CanonicalStreamEvent, ProviderError>>,
    ) -> Vec<CanonicalStreamEvent> {
        results.into_iter().map(|r| r.expect("ok event")).collect()
    }

    // WHY: text deltas are the most common path; they must surface as ordered
    // TextDelta events so the framing layer can reconstruct the assistant
    // message. Mirrors codex send_stream_maps_responses_text_usage_and_finish.
    #[test]
    fn text_deltas_accumulate_into_text_events() {
        let mut parser = parser();
        let a = ok_events(run_event(
            &mut parser,
            "response.output_text.delta",
            json!({"type":"response.output_text.delta","delta":"Hel"}),
        ));
        let b = ok_events(run_event(
            &mut parser,
            "response.output_text.delta",
            json!({"type":"response.output_text.delta","delta":"lo"}),
        ));
        assert_eq!(a, vec![CanonicalStreamEvent::TextDelta("Hel".into())]);
        assert_eq!(b, vec![CanonicalStreamEvent::TextDelta("lo".into())]);
    }

    // WHY: some upstreams send only the final output_text.done with no prior
    // deltas; we must still emit the full text once. Mirrors codex
    // send_stream_uses_output_text_done_when_no_deltas.
    #[test]
    fn output_text_done_without_deltas_emits_full_text() {
        let mut parser = parser();
        let events = ok_events(run_event(
            &mut parser,
            "response.output_text.done",
            json!({"type":"response.output_text.done","text":"complete"}),
        ));
        assert_eq!(
            events,
            vec![CanonicalStreamEvent::TextDelta("complete".into())]
        );
    }

    // WHY: when deltas precede a .done that repeats the full text, only the
    // missing suffix may be emitted, never a duplicate. Mirrors codex
    // send_stream_emits_output_text_done_suffix_without_duplicate.
    #[test]
    fn output_text_done_emits_only_missing_suffix() {
        let mut parser = parser();
        let mut text = String::new();
        for event in ok_events(run_event(
            &mut parser,
            "response.output_text.delta",
            json!({"type":"response.output_text.delta","delta":"Hel"}),
        )) {
            if let CanonicalStreamEvent::TextDelta(d) = event {
                text.push_str(&d);
            }
        }
        for event in ok_events(run_event(
            &mut parser,
            "response.output_text.done",
            json!({"type":"response.output_text.done","text":"Hello"}),
        )) {
            if let CanonicalStreamEvent::TextDelta(d) = event {
                text.push_str(&d);
            }
        }
        assert_eq!(text, "Hello");
    }

    // WHY: a function call announced via output_item.added then streamed via
    // argument deltas must assemble into one tool call: an opening
    // ToolCallDelta carrying id+name, then argument-only deltas in order, and a
    // terminal Finish reason of "tool_calls". This is the core multi-turn-tools
    // contract. Mirrors codex send_stream_maps_responses_tool_call_deltas.
    #[test]
    fn function_call_argument_deltas_assemble_into_tool_call() {
        let mut parser = parser();
        let mut events = Vec::new();
        events.extend(ok_events(run_event(
            &mut parser,
            "response.output_item.added",
            json!({"type":"response.output_item.added","output_index":0,
				"item":{"type":"function_call","call_id":"call_1","name":"lookup","arguments":""}}),
        )));
        events.extend(ok_events(run_event(
			&mut parser,
			"response.function_call_arguments.delta",
			json!({"type":"response.function_call_arguments.delta","output_index":0,"delta":"{\"q\""}),
		)));
        events.extend(ok_events(run_event(
			&mut parser,
			"response.function_call_arguments.delta",
			json!({"type":"response.function_call_arguments.delta","output_index":0,"delta":":\"sf\"}"}),
		)));
        events.extend(ok_events(run_event(
			&mut parser,
			"response.completed",
			json!({"type":"response.completed","response":{"status":"completed","usage":{"input_tokens":5,"output_tokens":6}}}),
		)));
        assert_eq!(
            events,
            vec![
                CanonicalStreamEvent::ToolCallDelta {
                    index: 0,
                    id: Some("call_1".into()),
                    name: Some("lookup".into()),
                    arguments_delta: String::new(),
                },
                CanonicalStreamEvent::ToolCallDelta {
                    index: 0,
                    id: None,
                    name: None,
                    arguments_delta: "{\"q\"".into(),
                },
                CanonicalStreamEvent::ToolCallDelta {
                    index: 0,
                    id: None,
                    name: None,
                    arguments_delta: ":\"sf\"}".into(),
                },
                CanonicalStreamEvent::Usage(CanonicalUsage {
                    input_tokens: 5,
                    output_tokens: 6,
                    ..Default::default()
                }),
                CanonicalStreamEvent::Finish {
                    finish_reason: Some("tool_calls".into())
                }
            ]
        );
    }

    // WHY: certain Responses gateways repeat the FULL arguments as the first
    // delta right after announcing them on output_item.added. The parser must
    // absorb that repeat and emit the arguments exactly once, or downstream
    // JSON would be doubled and unparseable. Mirrors codex
    // send_stream_does_not_duplicate_arguments_repeated_after_item_added.
    #[test]
    fn arguments_repeated_after_item_added_are_absorbed() {
        let mut parser = parser();
        let mut events = Vec::new();
        events.extend(ok_events(run_event(
			&mut parser,
			"response.output_item.added",
			json!({"type":"response.output_item.added","output_index":0,
				"item":{"type":"function_call","call_id":"call_1","name":"lookup","arguments":"{\"q\":\"sf\"}"}}),
		)));
        events.extend(ok_events(run_event(
			&mut parser,
			"response.function_call_arguments.delta",
			json!({"type":"response.function_call_arguments.delta","output_index":0,"delta":"{\"q\":\"sf\"}"}),
		)));
        let arguments: String = events
            .into_iter()
            .filter_map(|event| match event {
                CanonicalStreamEvent::ToolCallDelta {
                    arguments_delta, ..
                } if !arguments_delta.is_empty() => Some(arguments_delta),
                _ => None,
            })
            .collect();
        assert_eq!(arguments, r#"{"q":"sf"}"#);
    }

    // WHY: argument deltas may arrive BEFORE the output_item.added that carries
    // id+name; the opening ToolCallDelta must be withheld until id+name exist,
    // then the buffered args flushed. Otherwise a tool call with no name would
    // reach the client. Mirrors codex
    // send_stream_buffers_tool_arguments_until_metadata_arrives.
    #[test]
    fn tool_arguments_buffer_until_metadata_arrives() {
        let mut parser = parser();
        let mut events = Vec::new();
        events.extend(ok_events(run_event(
			&mut parser,
			"response.function_call_arguments.delta",
			json!({"type":"response.function_call_arguments.delta","output_index":3,"delta":"{\"q\""}),
		)));
        events.extend(ok_events(run_event(
			&mut parser,
			"response.function_call_arguments.delta",
			json!({"type":"response.function_call_arguments.delta","output_index":3,"delta":":\"sf\"}"}),
		)));
        events.extend(ok_events(run_event(
            &mut parser,
            "response.output_item.added",
            json!({"type":"response.output_item.added","output_index":3,
				"item":{"type":"function_call","call_id":"call_late","name":"lookup","arguments":""}}),
        )));
        assert_eq!(
            events[0],
            CanonicalStreamEvent::ToolCallDelta {
                index: 0,
                id: Some("call_late".into()),
                name: Some("lookup".into()),
                arguments_delta: String::new(),
            }
        );
        assert_eq!(
            events[1],
            CanonicalStreamEvent::ToolCallDelta {
                index: 0,
                id: None,
                name: None,
                arguments_delta: "{\"q\":\"sf\"}".into(),
            }
        );
    }

    // WHY: sparse upstream output_index values (e.g. 2) must map to a dense,
    // zero-based canonical tool index so clients see contiguous tool calls.
    // Mirrors codex send_stream_maps_sparse_response_output_indexes_to_dense_tool_indexes.
    #[test]
    fn sparse_output_indexes_map_to_dense_tool_indexes() {
        let mut parser = parser();
        let mut events = Vec::new();
        events.extend(ok_events(run_event(
            &mut parser,
            "response.output_item.added",
            json!({"type":"response.output_item.added","output_index":2,
				"item":{"type":"function_call","call_id":"call_sparse","name":"lookup","arguments":""}}),
        )));
        events.extend(ok_events(run_event(
            &mut parser,
            "response.function_call_arguments.delta",
            json!({"type":"response.function_call_arguments.delta","output_index":2,"delta":"{}"}),
        )));
        assert_eq!(
            events[0],
            CanonicalStreamEvent::ToolCallDelta {
                index: 0,
                id: Some("call_sparse".into()),
                name: Some("lookup".into()),
                arguments_delta: String::new(),
            }
        );
        assert_eq!(
            events[1],
            CanonicalStreamEvent::ToolCallDelta {
                index: 0,
                id: None,
                name: None,
                arguments_delta: "{}".into(),
            }
        );
    }

    // WHY: response.completed must close the stream with a Usage event (so the
    // framing layer can report tokens) followed by exactly one Finish. Mirrors
    // the tail of codex send_stream_maps_responses_text_usage_and_finish.
    #[test]
    fn completed_emits_usage_then_finish() {
        let mut parser = parser();
        let events = ok_events(run_event(
            &mut parser,
            "response.completed",
            json!({"type":"response.completed","response":{"status":"completed","usage":{"input_tokens":3,"output_tokens":4}}}),
        ));
        assert_eq!(
            events,
            vec![
                CanonicalStreamEvent::Usage(CanonicalUsage {
                    input_tokens: 3,
                    output_tokens: 4,
                    ..Default::default()
                }),
                CanonicalStreamEvent::Finish {
                    finish_reason: Some("stop".into())
                }
            ]
        );
        assert!(parser.completed());
    }

    // WHY: an incomplete response must preserve the wire reason as the finish
    // reason (here content_filter) rather than collapsing to "stop", so the
    // caller learns WHY generation stopped. Mirrors codex
    // send_stream_preserves_incomplete_content_filter_reason.
    #[test]
    fn incomplete_preserves_reason_as_finish_reason() {
        let mut parser = parser();
        let events = ok_events(run_event(
            &mut parser,
            "response.incomplete",
            json!({"type":"response.incomplete","response":{"status":"incomplete","incomplete_details":{"reason":"content_filter"}}}),
        ));
        assert_eq!(
            events.last().unwrap(),
            &CanonicalStreamEvent::Finish {
                finish_reason: Some("content_filter".into())
            }
        );
    }

    // WHY: response.failed (and the bare `error` event) must become a
    // ProviderError::Upstream, AND the error body must be redacted so a leaked
    // upstream credential is never surfaced. Mirrors codex
    // send_stream_redacts_failed_response_event.
    #[test]
    fn failed_event_becomes_redacted_upstream_error() {
        let mut parser = parser();
        let results = run_event(
            &mut parser,
            "response.failed",
            json!({"type":"response.failed","response":{"status":"failed"},"error":{"message":"bad sk-secret token"}}),
        );
        let err = results.into_iter().next().unwrap().unwrap_err().to_string();
        assert!(matches!(
            ProviderError::upstream(String::new()),
            ProviderError::Upstream { .. }
        ));
        assert!(!err.contains("sk-secret"), "leaked secret: {err}");
        assert!(err.contains("<redacted>"), "redaction missing: {err}");
    }

    // WHY: completed-with-status-failed is a second failure shape (the failure
    // rides inside the completed envelope) and must also redact and error.
    // Mirrors codex send_stream_treats_completed_failed_status_as_error.
    #[test]
    fn completed_with_failed_status_is_redacted_error() {
        let mut parser = parser();
        let results = run_event(
            &mut parser,
            "response.completed",
            json!({"type":"response.completed","response":{"status":"failed","error":"boom sk-secret"}}),
        );
        let err = results.into_iter().next().unwrap().unwrap_err().to_string();
        assert!(!err.contains("sk-secret"), "leaked secret: {err}");
        assert!(err.contains("<redacted>"));
    }

    // WHY: the Responses wire has NO [DONE] sentinel (that is Chat Completions
    // only). Receiving one means a mislabeled/Chat stream reached the Responses
    // parser and must be rejected loudly, not silently treated as success.
    // Mirrors codex send_stream_rejects_chat_done_sentinel_on_responses_wire.
    #[test]
    fn done_sentinel_is_rejected_as_error() {
        let mut parser = parser();
        let mut buffer = ResponsesSseBuffer::default();
        let framed = buffer.push(b"data: [DONE]\n\n").expect("frame");
        let mut results = Vec::new();
        for event in framed {
            results.extend(parser.handle_event(event));
        }
        let err = results.into_iter().next().unwrap().unwrap_err().to_string();
        assert!(err.contains("[DONE] sentinel"), "{err}");
    }

    // WHY: response bytes can split a multi-byte UTF-8 char across network
    // chunks; the framer must reassemble before decoding so text is never
    // corrupted. Mirrors codex send_stream_preserves_split_utf8_lines.
    #[test]
    fn buffer_reassembles_utf8_split_across_chunks() {
        let mut parser = parser();
        let mut buffer = ResponsesSseBuffer::default();
        let line =
			b"event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"delta\":\"h\xc3\xa9llo \xf0\x9f\x8c\x8e\"}\n\n";
        let split = 40;
        let mut results = Vec::new();
        for event in buffer.push(&line[..split]).expect("first chunk") {
            results.extend(parser.handle_event(event));
        }
        for event in buffer.push(&line[split..]).expect("second chunk") {
            results.extend(parser.handle_event(event));
        }
        let events = ok_events(results);
        assert_eq!(
            events[0],
            CanonicalStreamEvent::TextDelta("héllo 🌎".into())
        );
    }

    // WHY: SSE permits bare \r line endings; the framer must treat \r, \n, and
    // \r\n identically so a CR-only stream still parses. Mirrors codex
    // send_stream_accepts_bare_cr_sse_line_endings.
    #[test]
    fn buffer_accepts_bare_cr_line_endings() {
        let mut parser = parser();
        let mut buffer = ResponsesSseBuffer::default();
        let body = "event: response.output_text.delta\r\
data: {\"type\":\"response.output_text.delta\",\"delta\":\"ok\"}\r\
\r\
event: response.completed\r\
data: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\"}}\r\
\r";
        let mut results = Vec::new();
        for event in buffer.push(body.as_bytes()).expect("frame") {
            results.extend(parser.handle_event(event));
        }
        if let Some(event) = buffer.finish().expect("finish") {
            results.extend(parser.handle_event(event));
        }
        let events = ok_events(results);
        assert_eq!(events[0], CanonicalStreamEvent::TextDelta("ok".into()));
        assert_eq!(
            events.last().unwrap(),
            &CanonicalStreamEvent::Finish {
                finish_reason: Some("stop".into())
            }
        );
    }

    // WHY: an oversized accumulated event must be rejected to bound memory
    // against a hostile/broken upstream. Mirrors codex
    // send_stream_rejects_oversized_sse_event.
    #[test]
    fn buffer_rejects_oversized_event() {
        let mut buffer = ResponsesSseBuffer::default();
        let line = format!("data: {}\n", "x".repeat(1024));
        let body = line.repeat((MAX_SSE_EVENT_BYTES / 1024) + 2);
        let err = buffer.push(body.as_bytes()).unwrap_err();
        assert!(err.contains("event exceeded"), "{err}");
    }

    // WHY: an oversized single line must likewise be rejected before it is
    // buffered without bound.
    #[test]
    fn buffer_rejects_oversized_line() {
        let mut buffer = ResponsesSseBuffer::default();
        let body = "x".repeat(MAX_SSE_LINE_BYTES + 1);
        let err = buffer.push(body.as_bytes()).unwrap_err();
        assert!(err.contains("line exceeded"), "{err}");
    }

    // WHY: the non-stream mapper is the second entry point onto this wire. It
    // must extract id/model/content/tool_calls/usage and stamp the caller's
    // provider_tag (not a hardcoded "codex") and finish reason. Mirrors codex
    // responses_output_maps_to_canonical.
    #[test]
    fn response_to_canonical_maps_full_payload_and_tags_provider() {
        let value = json!({
            "id": "resp_backend",
            "model": "gpt-5.5",
            "service_tier": "default",
            "system_fingerprint": "fp_x",
            "status": "completed",
            "output": [
                {"type":"message","content":[{"type":"output_text","text":"hello","annotations":[{"type":"url_citation","url":"https://e.test"}]}]},
                {"type":"function_call","call_id":"call_1","name":"lookup","arguments":"{}"}
            ],
            "usage": {
                "input_tokens": 3,
                "output_tokens": 4,
                "input_tokens_details": {"cached_tokens": 1, "audio_tokens": 5},
                "output_tokens_details": {"reasoning_tokens": 6, "audio_tokens": 7}
            }
        });
        let resp = response_to_canonical(&value, "fallback", TAG, &TokenRedactor).unwrap();
        assert_eq!(resp.id.as_deref(), Some("resp_backend"));
        assert_eq!(resp.model, "gpt-5.5");
        assert_eq!(resp.content, "hello");
        assert!(resp.refusal.is_none());
        assert_eq!(resp.tool_calls.len(), 1);
        assert_eq!(resp.tool_calls[0].id, "call_1");
        assert_eq!(resp.tool_calls[0].name, "lookup");
        assert_eq!(resp.tool_calls[0].arguments, "{}");
        assert_eq!(resp.usage.input_tokens, 3);
        assert_eq!(resp.usage.cache_read, 1);
        assert_eq!(resp.usage.reasoning_tokens, 6);
        assert_eq!(resp.usage.audio_tokens, 12);
        assert_eq!(resp.usage.input_audio_tokens, 5);
        assert_eq!(resp.usage.output_audio_tokens, 7);
        assert_eq!(resp.annotations[0]["url"], "https://e.test");
        let meta = resp.metadata.as_ref().unwrap();
        assert_eq!(meta.service_tier.as_deref(), Some("default"));
        assert_eq!(meta.system_fingerprint.as_deref(), Some("fp_x"));
        assert_eq!(meta.provider.as_deref(), Some(TAG));
        assert_eq!(resp.finish_reason.as_deref(), Some("tool_calls"));
    }

    // WHY: a refusal + incomplete (content_filter) non-stream payload must map
    // refusal to the refusal field and preserve the incomplete reason as the
    // finish reason. Mirrors codex
    // responses_output_maps_refusal_and_content_filter_to_canonical.
    #[test]
    fn response_to_canonical_maps_refusal_and_incomplete_reason() {
        let value = json!({
            "model": "gpt-5.5",
            "status": "incomplete",
            "incomplete_details": {"reason": "content_filter"},
            "output": [
                {"type":"message","content":[{"type":"refusal","refusal":"No thanks"}]}
            ],
            "usage": {"input_tokens": 1, "output_tokens": 2}
        });
        let resp = response_to_canonical(&value, "fallback", TAG, &TokenRedactor).unwrap();
        assert_eq!(resp.content, "");
        assert_eq!(resp.refusal.as_deref(), Some("No thanks"));
        assert_eq!(resp.finish_reason.as_deref(), Some("content_filter"));
    }

    // WHY: a non-stream payload with status "failed" must become a redacted
    // upstream error, never a successful CanonicalResponse.
    #[test]
    fn response_to_canonical_failed_is_redacted_error() {
        let value = json!({"status":"failed","error":"boom sk-secret"});
        let err = response_to_canonical(&value, "fallback", TAG, &TokenRedactor)
            .unwrap_err()
            .to_string();
        assert!(!err.contains("sk-secret"), "leaked secret: {err}");
        assert!(err.contains("<redacted>"));
    }

    // WHY: incomplete with the wire reason max_output_tokens must normalize to
    // the canonical "length" finish reason.
    #[test]
    fn incomplete_reason_normalizes_max_output_tokens_to_length() {
        let value = json!({"response":{"status":"incomplete","incomplete_details":{"reason":"max_output_tokens"}}});
        assert_eq!(response_incomplete_reason(&value), "length");
    }

    // --- Class-K: terminal function_call full-arg merge by unique call_id ---

    /// Announce a function_call at `output_index` and stream its full args as one delta.
    fn seed_tool_with_args(
        parser: &mut ResponsesStreamParser<TokenRedactor>,
        output_index: u32,
        call_id: &str,
        name: &str,
        arguments: &str,
    ) -> Vec<CanonicalStreamEvent> {
        let mut events = Vec::new();
        events.extend(ok_events(run_event(
            parser,
            "response.output_item.added",
            json!({
                "type":"response.output_item.added",
                "output_index": output_index,
                "item":{
                    "type":"function_call",
                    "call_id": call_id,
                    "name": name,
                    "arguments":""
                }
            }),
        )));
        if !arguments.is_empty() {
            events.extend(ok_events(run_event(
                parser,
                "response.function_call_arguments.delta",
                json!({
                    "type":"response.function_call_arguments.delta",
                    "output_index": output_index,
                    "delta": arguments
                }),
            )));
        }
        events
    }

    /// Map tool call id → concatenated argument deltas from a stream of events.
    fn tool_args_by_id(events: &[CanonicalStreamEvent]) -> HashMap<String, String> {
        let mut id_by_index: HashMap<u32, String> = HashMap::new();
        let mut args_by_index: HashMap<u32, String> = HashMap::new();
        for event in events {
            if let CanonicalStreamEvent::ToolCallDelta {
                index,
                id,
                arguments_delta,
                ..
            } = event
            {
                if let Some(id) = id {
                    id_by_index.insert(*index, id.clone());
                }
                args_by_index
                    .entry(*index)
                    .or_default()
                    .push_str(arguments_delta);
            }
        }
        id_by_index
            .into_iter()
            .map(|(idx, id)| (id, args_by_index.remove(&idx).unwrap_or_default()))
            .collect()
    }

    fn first_upstream_err(
        results: Vec<Result<CanonicalStreamEvent, ProviderError>>,
    ) -> ProviderError {
        results
            .into_iter()
            .find_map(|r| r.err())
            .expect("expected an upstream error")
    }

    /// Prove buffers at the given stream indexes were not mutated: a full
    /// `function_call_arguments.done` with the original text must equal-absorb
    /// (no error, no new arg delta). Uses the bare-index done path so call_id
    /// routing cannot mask a wrong-slot mutation.
    fn assert_slots_unmutated(
        parser: &mut ResponsesStreamParser<TokenRedactor>,
        slots: &[(u32, &str)],
    ) {
        for &(output_index, original) in slots {
            let results = run_event(
                parser,
                "response.function_call_arguments.done",
                json!({
                    "type":"response.function_call_arguments.done",
                    "output_index": output_index,
                    "arguments": original
                }),
            );
            let events = ok_events(results);
            let extra_args: String = events
                .into_iter()
                .filter_map(|e| match e {
                    CanonicalStreamEvent::ToolCallDelta {
                        arguments_delta, ..
                    } if !arguments_delta.is_empty() => Some(arguments_delta),
                    _ => None,
                })
                .collect();
            assert!(
                extra_args.is_empty(),
                "slot {output_index} mutated after fail-loud; equal-absorb should emit nothing, got {extra_args:?}"
            );
        }
    }

    // WHY: measured class-K failure — stream deltas fill correct slots by
    // output_index, but response.completed lists tools in a different order
    // (array position becomes the merge key). Without call_id re-key, full args
    // land on the wrong buffer and hard-error. Routing by unique call_id must
    // identity-map args and succeed.
    #[test]
    fn terminal_completed_shuffled_indexes_merge_by_call_id() {
        let mut parser = parser();
        let mut events = Vec::new();
        events.extend(seed_tool_with_args(
            &mut parser,
            0,
            "call_a",
            "tool_a",
            r#"{"a":1}"#,
        ));
        events.extend(seed_tool_with_args(
            &mut parser,
            1,
            "call_b",
            "tool_b",
            r#"{"b":2}"#,
        ));
        // Completed lists call_b then call_a (rotated relative to stream indexes).
        events.extend(ok_events(run_event(
            &mut parser,
            "response.completed",
            json!({
                "type":"response.completed",
                "response":{
                    "status":"completed",
                    "output":[
                        {"type":"function_call","call_id":"call_b","name":"tool_b","arguments":"{\"b\":2}"},
                        {"type":"function_call","call_id":"call_a","name":"tool_a","arguments":"{\"a\":1}"}
                    ]
                }
            }),
        )));
        let by_id = tool_args_by_id(&events);
        assert_eq!(by_id.get("call_a").map(String::as_str), Some(r#"{"a":1}"#));
        assert_eq!(by_id.get("call_b").map(String::as_str), Some(r#"{"b":2}"#));
    }

    // WHY: same-length but different JSON bodies must not be treated as a
    // content conflict when completed routes them to the wrong bare index;
    // unique call_id must re-home each body onto the slot that already holds
    // that id (equal-absorb / success).
    #[test]
    fn terminal_same_len_different_content_routes_by_call_id() {
        let mut parser = parser();
        let mut events = Vec::new();
        // Same length: {"x":1} and {"y":2} are both 7 bytes.
        events.extend(seed_tool_with_args(
            &mut parser,
            0,
            "call_x",
            "tool_x",
            r#"{"x":1}"#,
        ));
        events.extend(seed_tool_with_args(
            &mut parser,
            1,
            "call_y",
            "tool_y",
            r#"{"y":2}"#,
        ));
        events.extend(ok_events(run_event(
            &mut parser,
            "response.completed",
            json!({
                "type":"response.completed",
                "response":{
                    "status":"completed",
                    "output":[
                        {"type":"function_call","call_id":"call_y","name":"tool_y","arguments":"{\"y\":2}"},
                        {"type":"function_call","call_id":"call_x","name":"tool_x","arguments":"{\"x\":1}"}
                    ]
                }
            }),
        )));
        let by_id = tool_args_by_id(&events);
        assert_eq!(by_id.get("call_x").map(String::as_str), Some(r#"{"x":1}"#));
        assert_eq!(by_id.get("call_y").map(String::as_str), Some(r#"{"y":2}"#));
    }

    // WHY: after correct call_id routing, a true content conflict on the same
    // call must still fail loud and leave buffers unchanged (no soft drop, no
    // partial overwrite).
    #[test]
    fn terminal_same_call_id_content_conflict_fails_without_mutation() {
        let mut parser = parser();
        let _ = seed_tool_with_args(&mut parser, 0, "call_1", "lookup", r#"{"q":"sf"}"#);
        let results = run_event(
            &mut parser,
            "response.completed",
            json!({
                "type":"response.completed",
                "response":{
                    "status":"completed",
                    "output":[
                        {"type":"function_call","call_id":"call_1","name":"lookup","arguments":"{\"q\":\"nyc\"}"}
                    ]
                }
            }),
        );
        let err = first_upstream_err(results);
        assert!(
            err.to_string().contains("terminal function_call arguments"),
            "{err}"
        );
        assert_slots_unmutated(&mut parser, &[(0, r#"{"q":"sf"}"#)]);
    }

    // WHY: an unknown terminal call_id must not steal a slot already tagged with
    // a different id (would corrupt multi-tool identity). Fail loud, no mutate.
    #[test]
    fn terminal_no_match_onto_other_tagged_slot_fails_without_mutation() {
        let mut parser = parser();
        let _ = seed_tool_with_args(&mut parser, 0, "call_a", "tool_a", r#"{"a":1}"#);
        let _ = seed_tool_with_args(&mut parser, 1, "call_b", "tool_b", r#"{"b":2}"#);
        let results = run_event(
            &mut parser,
            "response.completed",
            json!({
                "type":"response.completed",
                "response":{
                    "status":"completed",
                    "output":[
                        {"type":"function_call","call_id":"call_orphan","name":"tool_z","arguments":"{\"z\":9}"}
                    ]
                }
            }),
        );
        let err = first_upstream_err(results);
        assert!(
            err.to_string().contains("terminal function_call arguments"),
            "{err}"
        );
        assert_slots_unmutated(&mut parser, &[(0, r#"{"a":1}"#), (1, r#"{"b":2}"#)]);
    }

    // WHY: untagged non-empty buffers exist when deltas precede metadata. Claiming
    // them via equal/prefix heuristics is wrong-slot guessing (prefix-compatible
    // JSON is common). No-match must fail loud and leave the buffer alone.
    #[test]
    fn terminal_no_match_onto_untagged_nonempty_fails_without_mutation() {
        let mut parser = parser();
        // Deltas first: creates untagged non-empty slot at index 0.
        let _ = ok_events(run_event(
            &mut parser,
            "response.function_call_arguments.delta",
            json!({
                "type":"response.function_call_arguments.delta",
                "output_index": 0,
                "delta": r#"{"pre":1}"#
            }),
        ));
        // Terminal presents a *different* call_id that matches nothing, targeting
        // the same bare index (array position 0).
        let results = run_event(
            &mut parser,
            "response.completed",
            json!({
                "type":"response.completed",
                "response":{
                    "status":"completed",
                    "output":[
                        {"type":"function_call","call_id":"call_new","name":"lookup","arguments":"{\"pre\":1}"}
                    ]
                }
            }),
        );
        let err = first_upstream_err(results);
        assert!(
            err.to_string().contains("terminal function_call arguments"),
            "{err}"
        );
        assert_slots_unmutated(&mut parser, &[(0, r#"{"pre":1}"#)]);
    }

    // WHY: terminal-only tools (no prior stream slot) or empty untagged slots must
    // still attach the call_id at the bare index and merge, or non-stream-like
    // completed payloads would fail-loud incorrectly.
    #[test]
    fn terminal_no_match_missing_or_empty_untagged_attaches_and_merges() {
        // Missing slot: completed alone.
        {
            let mut p = parser();
            let events = ok_events(run_event(
                &mut p,
                "response.completed",
                json!({
                    "type":"response.completed",
                    "response":{
                        "status":"completed",
                        "output":[
                            {"type":"function_call","call_id":"call_solo","name":"lookup","arguments":"{\"q\":1}"}
                        ]
                    }
                }),
            ));
            let by_id = tool_args_by_id(&events);
            assert_eq!(
                by_id.get("call_solo").map(String::as_str),
                Some(r#"{"q":1}"#)
            );
        }

        // Empty untagged: an empty delta ensures a map key with id=None and empty
        // buffer; free-claim must attach the terminal call_id and merge args.
        {
            let mut p = parser();
            let _ = ok_events(run_event(
                &mut p,
                "response.function_call_arguments.delta",
                json!({
                    "type":"response.function_call_arguments.delta",
                    "output_index": 0,
                    "delta": ""
                }),
            ));
            let events = ok_events(run_event(
                &mut p,
                "response.completed",
                json!({
                    "type":"response.completed",
                    "response":{
                        "status":"completed",
                        "output":[
                            {"type":"function_call","call_id":"call_free","name":"lookup","arguments":"{\"q\":2}"}
                        ]
                    }
                }),
            ));
            let by_id = tool_args_by_id(&events);
            assert_eq!(
                by_id.get("call_free").map(String::as_str),
                Some(r#"{"q":2}"#)
            );
        }
    }

    // WHY: three-tool rotation is the realistic multi-tool Codex shape; every
    // completed entry must land on its own call_id buffer with no hard error.
    #[test]
    fn terminal_three_tool_completed_rotation_identity_maps() {
        let mut parser = parser();
        let mut events = Vec::new();
        events.extend(seed_tool_with_args(
            &mut parser,
            0,
            "call_a",
            "tool_a",
            r#"{"a":1}"#,
        ));
        events.extend(seed_tool_with_args(
            &mut parser,
            1,
            "call_b",
            "tool_b",
            r#"{"b":2}"#,
        ));
        events.extend(seed_tool_with_args(
            &mut parser,
            2,
            "call_c",
            "tool_c",
            r#"{"c":3}"#,
        ));
        // Rotation: c, a, b at completed positions 0,1,2.
        events.extend(ok_events(run_event(
            &mut parser,
            "response.completed",
            json!({
                "type":"response.completed",
                "response":{
                    "status":"completed",
                    "output":[
                        {"type":"function_call","call_id":"call_c","name":"tool_c","arguments":"{\"c\":3}"},
                        {"type":"function_call","call_id":"call_a","name":"tool_a","arguments":"{\"a\":1}"},
                        {"type":"function_call","call_id":"call_b","name":"tool_b","arguments":"{\"b\":2}"}
                    ]
                }
            }),
        )));
        let by_id = tool_args_by_id(&events);
        assert_eq!(by_id.get("call_a").map(String::as_str), Some(r#"{"a":1}"#));
        assert_eq!(by_id.get("call_b").map(String::as_str), Some(r#"{"b":2}"#));
        assert_eq!(by_id.get("call_c").map(String::as_str), Some(r#"{"c":3}"#));
    }

    // WHY: duplicate call_id tags are a corrupted stream; first-match-wins would
    // silently pick a buffer. Multi-match must fail loud with zero mutations.
    #[test]
    fn terminal_multi_match_same_call_id_fails_without_mutation() {
        let mut parser = parser();
        let _ = seed_tool_with_args(&mut parser, 0, "call_dup", "tool_a", r#"{"a":1}"#);
        let _ = seed_tool_with_args(&mut parser, 1, "call_dup", "tool_b", r#"{"b":2}"#);
        let results = run_event(
            &mut parser,
            "response.completed",
            json!({
                "type":"response.completed",
                "response":{
                    "status":"completed",
                    "output":[
                        {"type":"function_call","call_id":"call_dup","name":"tool_a","arguments":"{\"a\":1}"}
                    ]
                }
            }),
        );
        let err = first_upstream_err(results);
        assert!(
            err.to_string().contains("terminal function_call arguments"),
            "{err}"
        );
        assert_slots_unmutated(&mut parser, &[(0, r#"{"a":1}"#), (1, r#"{"b":2}"#)]);
    }

    // WHY: equal full args on completed must no-op (already streamed), and a
    // terminal that is a pure prefix-extension must append only the suffix —
    // both pre-existing contracts must stay green under the shared merge helper.
    #[test]
    fn terminal_equal_absorb_and_prefix_extend_remain() {
        // Equal absorb.
        let mut p_eq = parser();
        let mut events = seed_tool_with_args(&mut p_eq, 0, "call_1", "lookup", r#"{"q":"sf"}"#);
        events.extend(ok_events(run_event(
            &mut p_eq,
            "response.completed",
            json!({
                "type":"response.completed",
                "response":{
                    "status":"completed",
                    "output":[
                        {"type":"function_call","call_id":"call_1","name":"lookup","arguments":"{\"q\":\"sf\"}"}
                    ]
                }
            }),
        )));
        let by_id = tool_args_by_id(&events);
        assert_eq!(
            by_id.get("call_1").map(String::as_str),
            Some(r#"{"q":"sf"}"#),
            "equal absorb must not double args"
        );

        // Prefix extend: stream partial, completed supplies the rest.
        let mut p_ext = parser();
        let mut events = seed_tool_with_args(&mut p_ext, 0, "call_2", "lookup", r#"{"q":"#);
        events.extend(ok_events(run_event(
            &mut p_ext,
            "response.completed",
            json!({
                "type":"response.completed",
                "response":{
                    "status":"completed",
                    "output":[
                        {"type":"function_call","call_id":"call_2","name":"lookup","arguments":"{\"q\":\"sf\"}"}
                    ]
                }
            }),
        )));
        let by_id = tool_args_by_id(&events);
        assert_eq!(
            by_id.get("call_2").map(String::as_str),
            Some(r#"{"q":"sf"}"#),
            "prefix extend must append only the missing suffix"
        );
    }

    // WHY: free-claim must not attach call_id before content validation. An
    // item.done conflict on untagged non-empty must leave the slot untagged so
    // later unique-match routing is not poisoned by a failed claim.
    #[test]
    fn item_done_content_conflict_does_not_poison_call_id() {
        let mut parser = parser();
        let _ = ok_events(run_event(
            &mut parser,
            "response.function_call_arguments.delta",
            json!({
                "type":"response.function_call_arguments.delta",
                "output_index": 0,
                "delta": r#"{"q":"sf"}"#
            }),
        ));
        let results = run_event(
            &mut parser,
            "response.output_item.done",
            json!({
                "type":"response.output_item.done",
                "output_index": 0,
                "item":{
                    "type":"function_call",
                    "call_id":"call_poison",
                    "name":"lookup",
                    "arguments":"{\"q\":\"nyc\"}"
                }
            }),
        );
        let err = first_upstream_err(results);
        assert!(
            err.to_string().contains("terminal function_call arguments"),
            "{err}"
        );
        // Strict terminal free-claim must still see untagged non-empty → fail.
        // If call_id were poisoned on, unique match would equal-absorb wrongly.
        let results = run_event(
            &mut parser,
            "response.completed",
            json!({
                "type":"response.completed",
                "response":{
                    "status":"completed",
                    "output":[
                        {"type":"function_call","call_id":"call_poison","name":"lookup","arguments":"{\"q\":\"sf\"}"}
                    ]
                }
            }),
        );
        let err = first_upstream_err(results);
        assert!(
            err.to_string().contains("terminal function_call arguments"),
            "poisoned call_id would unique-match; expected strict free-claim fail: {err}"
        );
        assert_slots_unmutated(&mut parser, &[(0, r#"{"q":"sf"}"#)]);
    }

    // WHY: stream item.done may arrive after deltas filled an untagged buffer at
    // the same output_index. Free-claim must allow attaching call_id to that
    // non-empty untagged slot (metadata-late path), unlike strict terminal
    // completed free-claim which refuses untagged non-empty.
    #[test]
    fn item_done_attaches_call_id_to_untagged_nonempty_same_index() {
        let mut parser = parser();
        let _ = ok_events(run_event(
            &mut parser,
            "response.function_call_arguments.delta",
            json!({
                "type":"response.function_call_arguments.delta",
                "output_index": 0,
                "delta": r#"{"q":"sf"}"#
            }),
        ));
        let events = ok_events(run_event(
            &mut parser,
            "response.output_item.done",
            json!({
                "type":"response.output_item.done",
                "output_index": 0,
                "item":{
                    "type":"function_call",
                    "call_id":"call_late",
                    "name":"lookup",
                    "arguments":"{\"q\":\"sf\"}"
                }
            }),
        ));
        let by_id = tool_args_by_id(&events);
        assert_eq!(
            by_id.get("call_late").map(String::as_str),
            Some(r#"{"q":"sf"}"#)
        );
    }

    // WHY: item.done full-arg path must re-key by unique call_id and emit from
    // the resolved slot (not bare wrong index), matching terminal class-K so a
    // mis-indexed done cannot corrupt sibling tool buffers.
    #[test]
    fn item_done_routes_full_args_by_call_id() {
        let mut parser = parser();
        let mut events = Vec::new();
        events.extend(seed_tool_with_args(
            &mut parser,
            0,
            "call_a",
            "tool_a",
            r#"{"a":1}"#,
        ));
        events.extend(seed_tool_with_args(
            &mut parser,
            1,
            "call_b",
            "tool_b",
            r#"{"b":2}"#,
        ));
        // Wrong bare index (0) but call_b's full args — must re-key to slot 1.
        events.extend(ok_events(run_event(
            &mut parser,
            "response.output_item.done",
            json!({
                "type":"response.output_item.done",
                "output_index": 0,
                "item":{
                    "type":"function_call",
                    "call_id":"call_b",
                    "name":"tool_b",
                    "arguments":"{\"b\":2}"
                }
            }),
        )));
        let by_id = tool_args_by_id(&events);
        assert_eq!(by_id.get("call_a").map(String::as_str), Some(r#"{"a":1}"#));
        assert_eq!(by_id.get("call_b").map(String::as_str), Some(r#"{"b":2}"#));
        assert_slots_unmutated(&mut parser, &[(0, r#"{"a":1}"#), (1, r#"{"b":2}"#)]);
    }

    // WHY: incomplete shares emit_terminal_function_call → the same call_id
    // re-resolve helper. One incomplete multi-tool rotation proves the shared
    // path is wired (measured prod site is completed; incomplete inherits).
    #[test]
    fn terminal_incomplete_shares_call_id_merge_helper() {
        let mut parser = parser();
        let mut events = Vec::new();
        events.extend(seed_tool_with_args(
            &mut parser,
            0,
            "call_a",
            "tool_a",
            r#"{"a":1}"#,
        ));
        events.extend(seed_tool_with_args(
            &mut parser,
            1,
            "call_b",
            "tool_b",
            r#"{"b":2}"#,
        ));
        events.extend(ok_events(run_event(
            &mut parser,
            "response.incomplete",
            json!({
                "type":"response.incomplete",
                "response":{
                    "status":"incomplete",
                    "incomplete_details":{"reason":"max_output_tokens"},
                    "output":[
                        {"type":"function_call","call_id":"call_b","name":"tool_b","arguments":"{\"b\":2}"},
                        {"type":"function_call","call_id":"call_a","name":"tool_a","arguments":"{\"a\":1}"}
                    ]
                }
            }),
        )));
        let by_id = tool_args_by_id(&events);
        assert_eq!(by_id.get("call_a").map(String::as_str), Some(r#"{"a":1}"#));
        assert_eq!(by_id.get("call_b").map(String::as_str), Some(r#"{"b":2}"#));
        assert_eq!(
            events.last(),
            Some(&CanonicalStreamEvent::Finish {
                finish_reason: Some("length".into())
            })
        );
    }
}
