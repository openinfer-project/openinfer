# Chat Completions E2E

> **TL;DR:** PR #630 review follow-up now strictly covers streamed content, a unique terminal `[DONE]`, and the ordered usage-only chunk contract with correct token accounting.
>
> **Last touched:** 2026-07

## Preparation

- **Read**:
  - `docs/index.md` — routed this as frontend subsystem work.
  - `docs/subsystems/frontend/simulated-inference-engine.md` — confirmed `openinfer-sim` is the CPU-only frontend validation harness and its metadata fixture is intentionally minimal.
  - PR #630 review — identified four follow-ups: assert streamed content, enforce terminal `[DONE]`, validate the usage-only chunk contract, and remove author-local provenance.
  - `openinfer-sim/tests/frontend_e2e.rs` — confirmed the current file has completions E2E tests but no chat completions tests and no `chat_template`.
- **Relevant history**:
  - `docs/subsystems/frontend/simulated-inference-engine.md` — chat/text frontend construction still requires local tokenizer metadata even when the simulated engine never loads real weights.
- **Plan**:
  1. Patch `openinfer-sim/tests/frontend_e2e.rs` to add the three chat-completions E2E tests from the downloaded PR note, adjusted only for the existing local helper style if needed.
  2. Add the minimal `chat_template` field to `TINY_TOKENIZER_CONFIG_JSON` so `/v1/chat/completions` can exercise the chat formatting path.
  3. Run `cargo test -p openinfer-sim --test frontend_e2e`.
  4. Run `cargo clippy -p openinfer-sim --tests`.
- **Risks / open questions**:
  - The local frontend contract may have drifted from the PR note; if assertions fail, the tests may need to reflect the current intended OpenAI-compatible response shape rather than copying the diff blindly.
  - The tiny tokenizer and simulated engine must produce deterministic visible text after chat-template rendering; the release E2E payload is the gate for the fixture choice.

## Execution Log

### Step 1: Add chat-completions tests and fixture template

- Patched `openinfer-sim/tests/frontend_e2e.rs`.
- Added three tests:
  - `chat_completions_returns_correct_format`
  - `chat_completions_streaming_emits_role_content_and_done`
  - `chat_completions_usage_with_stream_options`
- Added a minimal `chat_template` to `TINY_TOKENIZER_CONFIG_JSON`.
- Result: code edited; validation pending.

### Step 2: Run `cargo test -p openinfer-sim --test frontend_e2e`

- First attempt failed before compilation while Cargo fetched git dependencies:
  `failed to clone ... dynamo ... unexpected return value from ssl handshake -9806`.
- Retrying with `CARGO_NET_GIT_FETCH_WITH_CLI=true`, as suggested by Cargo.
- Result: pass. `8 passed; 0 failed`, including the three new chat-completions tests.

### Step 3: Run `cargo clippy -p openinfer-sim --tests`

- Running after the E2E pass, with `CARGO_NET_GIT_FETCH_WITH_CLI=true` for consistency.
- Result: pass. Clippy finished with no warnings from `openinfer-sim`; Cargo only printed the existing future-incompat note for `proc-macro-error2`.

### Step 4: Address PR #630 review

- Planned changes:
  - preserve SSE payload order and require exactly one terminal `[DONE]`;
  - concatenate and assert streamed `delta.content`;
  - require one usage-only chunk after finish and validate its counters;
  - remove the author-local Downloads path from this document.
- The first focused content test exposed the tokenizer's expected leading-space decode: each simulated `alpha` token streams as `" alpha"`; the exact concatenated assertion is therefore `" alpha alpha"`.
- Result:
  - Rebased onto upstream `main` at `6b6b6c37` without conflicts.
  - `cargo test --release -p openinfer-sim --test frontend_e2e`: pass, `12 passed; 0 failed` on the rebased branch.
  - `cargo clippy -p openinfer-sim --tests`: pass; only the existing dependency future-incompatibility note remains.
  - `cargo fmt --all --check`: pass.

## Debrief

- **Outcome**: Implemented the chat-completions E2E coverage and addressed PR #630's requested changes. The streaming tests now fail if content disappears, `[DONE]` is duplicated or non-terminal, or the usage-only chunk is malformed, misplaced, or miscounted.
- **Pitfalls encountered**:
  - The first `cargo test` attempt failed during git dependency fetch for `dynamo` with `unexpected return value from ssl handshake -9806`.
  - Retrying with `CARGO_NET_GIT_FETCH_WITH_CLI=true` let Cargo fetch git dependencies through the system git client and proceed.
- **Lessons learned**:
  - For this frontend test path, `openinfer-sim` still needs chat-capable tokenizer metadata even though it does not load model weights.
  - SSE tests must preserve data-event order; filtering `[DONE]` before assertions makes terminal-contract regressions invisible.
- **Follow-ups**:
  - Resolve the four PR #630 review threads after the strengthened tests are pushed.
