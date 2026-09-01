# Claude Rebaseline Procedure

Use this when a new Claude Code release appears or a current pin is rejected
upstream.

## Breaking change (issue #12)

Omni ships **one live pin** per provider. Multi-version CLI flags
(`--claude-version`, `--grok-version`, `--codex-version`,
`--match-system`, `--match-system-exact`) and env counterparts
(`OMNI_*_VERSION`, `OMNI_MATCH_SYSTEM*`) are removed. Rebaseline **overwrites**
the single pin; it does not append a profile ladder. For older wire fingerprints,
use an older Omni release from git history.

## Safety

- Raw mitmproxy `.flow` files contain live OAuth bearer tokens.
- Keep raw flows on RAM-backed tmpfs only.
- Never commit `.flow` files, credentials, extracted bearer tokens, or local
  reports containing unredacted auth.
- Use clean HOME/CWD captures so project or user instruction files are not
  copied into request bodies or reports.
- Do not run live capture without explicit operator approval.
- Refresh capture additionally mutates only the staged credential copy to force
  expiry. Never edit the real credential file for capture.

## Tools

- Shared capture CLI: `python3 -m tools.capture` (source of truth for live
  capture, refresh capture, staging, MITM, extraction, and cleanup)
- `tools/providers/claude/fingerprint/check_claude_code_drift.py`
- `tools/providers/claude/fingerprint/capture_baseline.sh` (thin wrapper; prefer the shared CLI for new work)
- `tools/providers/claude/fingerprint/extract_flow.py` (compatibility wrapper around `tools.capture extract flow`)
- `tools/providers/claude/fingerprint/BASELINE_HEADERS.md`
- `tools/providers/claude/fingerprint/CCH_ALGORITHM.md`
- `tools/providers/claude/fingerprint/vectors/`

Live extraction requires the `mitmproxy` Python package in the same interpreter
as `tools.capture` (for example `uv run --with mitmproxy python -m tools.capture`).

## Procedure

1. Detect drift:

   ```sh
   uv run --script tools/providers/claude/fingerprint/check_claude_code_drift.py
   ```

   Continue only if `status` is not `ok`, or if a provider rejection requires a
   fresh capture despite a matching version.

2. Capture live traffic on tmpfs (requires `OMNI_CAPTURE_LIVE=1` or `--live-capture`):

   ```sh
   uv run --with mitmproxy python -m tools.capture capture run \
     --provider claude --mode general --live-capture \
     --models opus sonnet haiku
   ```

   The legacy wrapper remains for compatibility:

   ```sh
   tools/providers/claude/fingerprint/capture_baseline.sh \
     claude-fable-5 claude-haiku-4-5 claude-sonnet-5 claude-opus-5
   ```

   Both helpers start mitmdump as a reverse proxy to `https://api.anthropic.com`,
   copy only Claude credentials into a clean tmpfs HOME, drive the installed
   `claude` CLI from that clean HOME/CWD, extracts a redacted structural
   Markdown report, and removes the tmpfs workdir (including staged credential
   copies) unless `KEEP_FLOW=1`. `KEEP_FLOW=1` retains the workdir and raw flow
   on tmpfs and prints warnings.

   Refresh validation proves Anthropic API-host traffic through the reverse
   proxy. Separate auth-host proof awaits a stable observed auth endpoint.

   Refresh capture command:

   ```sh
   uv run --with mitmproxy python -m tools.capture capture run \
     --provider claude --mode refresh --live-capture --refresh-capture
   ```

3. Analyze the extract:

   - Confirm `POST /v1/messages?beta=true`.
   - Record send-order headers.
   - Compare `anthropic-beta`, stainless package/runtime versions, and
     `anthropic-version`.
   - Confirm `model`, `max_tokens`, `temperature`, `thinking` /
     `output_config`, `metadata`, `context_management`, `stream`, and system
     block structure.
   - Confirm default model from the no-`--model` capture.
   - Obtain this pin's catalog from captured `POST /v1/messages` model ids
     (`python3 -m tools.capture catalog --provider claude --flow-file ...`).
     If that listing is empty, stop. Do not keep the previous pin's catalog.
   - Confirm all pinned catalog models are accepted.
   - Confirm the billing suffix and cch behavior.
   - If any checksum or body mutation cannot be reproduced exactly, do not
     promote the new pin.

4. Update code (overwrite the single active pin; do not append a profile ladder):

   - Overwrite the active `FingerprintProfile` in
     `crates/provider-claude/src/fingerprint.rs` with the new capture.
   - Overwrite the single model catalog in `crates/provider-claude/src/models.rs`
     from this pin's listing only.
   - Rename shared constants if the pin version changes (do not keep historical
     `*_VERSION` symbol names that disagree with the live pin).
   - Update active-pin goldens and local vectors. History lives in git tags and
     older Omni releases, not in-tree multi-profile selection.

5. Regenerate clean-room cch vectors:

   ```sh
   uv run --script tools/providers/claude/fingerprint/check_claude_code_drift.py \
     --emit-vectors tools/providers/claude/fingerprint/vectors
   ```

6. Update docs:

   - `tools/providers/claude/fingerprint/BASELINE_HEADERS.md`
   - `docs/providers/claude/README.md` only if structure or invariant changed.
   - `docs/providers/README.md` only if shared capture policy changed.

7. Verify:

   ```sh
   cargo test -p provider-claude
   cargo test --workspace
   cargo clippy --workspace --all-targets -- -D warnings
   ```

8. Optional live smoke, only with approval:

   ```sh
   OMNI_LIVE_TESTS=1 cargo test -p provider-claude claude_send_exercises_full_fingerprint_path
   ```

## Done Criteria

- Drift checker agrees with the pinned version and cch.
- Captured fields are represented in source.
- Recovered vectors are local to this repo and covered by Rust tests.
- Default workspace tests pass without credentials or network.

## Current 2.1.257 Status

On 2026-09-01, Claude Code 2.1.257 was captured and model behavior was verified
for default, `opus`, `sonnet`, `haiku`, `fable`, and `claude-fable-5` flows.
Headers use SDK package `0.112.1`, runtime `v26.3.0`, Anthropic version
`2023-06-01`, and `claude-cli/2.1.257 (external, sdk-cli)`.

2.1.257 is the current active pin. Drift versus 2.1.232:

1. CLI version string (UA + billing `cc_version`).
2. Catalog adds `claude-fable-5-1` (alias `fable`). Explicit `claude-fable-5`
   is still accepted on the wire.

Per-model betas, wire defaults, stainless package/runtime, identity preamble,
and the no-`x-client-request-id` header set are otherwise live-confirmed
unchanged.

Like 2.1.186/197/207/211/220/221/228/232 it emits the billing header with no
`cch=` field, ending at `cc_entrypoint=sdk-cli;`. The `cc_version` suffix
algorithm is unchanged: the existing Sha256Utf16SampleV1 suffix reproduces the
captured `cc_version=2.1.257.27e` exactly, and the live drift checker agrees
against the installed CLI. Because there is no checksum to recompute, this
no-cch profile ships no clean-room cch vectors.
