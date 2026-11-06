# CubicAI Lab: desktop companion to the CubicAI Android app

Rust workspace (edition 2024): `crates/gguf` (cubic-gguf, no UI) and `app` (cubicai-lab, egui/eframe 0.37 on wgpu). Personal use. Stages planned: 1 chat + Engine Room and Anatomy (done), 3 X-ray + token fork (done), 4 Steering Lab (done), then Fine-tune Studio, and merging with the phone's PC gateway.

## Build / check
- `cargo build --release`; `cargo clippy --workspace --all-targets` should stay clean.
- Screens are checked offscreen with egui_kittest (`app/src/shots.rs`):
  - `SHOTS=<dir> MODEL=<name part> TENSOR=<tensor> [LIGHT=1] cargo test --release shots -- --nocapture`
  - Read the PNGs rather than opening windows on the desktop.
- Tests:
  - `cargo test --workspace --release` runs about 89 offline tests in roughly a second, and must stay green.
  - Test helpers live in `app/src/tests/mod.rs`: `tiny_model` (a GGUF built with `cubic_gguf::Writer`), `TempDir` and `mock_http`.
  - Decoder fixtures come from gguf-py (`crates/gguf/tests/fixtures/make.py`).
  - Live tests are `#[ignore]`: `net_*` (network/Ollama) and `net_real_llama` (runs downloaded llama.cpp, so the user runs it).

## egui 0.37 notes
- `App::ui(&mut self, ui, frame)` replaces `update`.
- Panels are `egui::Panel::left/right/top/bottom(id).show(ui, ..)`; `CentralPanel::no_frame()`. `show_inside` is deprecated.
- `Id::unique` replaces `Id::new`.
- egui panics on a glyph no installed font has:
  - UI text uses only what Roboto/Chakra cover (Latin-1, dashes, quotes, bullet, ellipsis);
  - `egui/monochrome_emoji_fonts` is on for model replies and egui_commonmark's copy icon.
- eframe hands `ui` over unpainted; `ui::frame` fills the page colour first.

## Design (same rules as the phone)
- Cobalt + amber `Palette` (theme.rs), with Chakra Petch headings and Roboto body.
- Cut-corner surfaces: `widgets::surface/card/button/chip/segmented/switch`. No rounded corners.
- Cards: `widgets::card` subtracts its own margins from the width, so content never overflows the right edge. Cap card widths with `available_width().min(..)`.
- Destructive actions use `Kind::Danger` / `p.error`.

## Engine
- llama.cpp releases (`engine/release.rs`):
  - assets are picked by variant key **and** CPU architecture (`arch_tag`; there are arm64 CUDA builds);
  - CUDA builds also fetch the matching `cudart-` archive unless the system has the runtime;
  - installs go to `<data>/llama/<tag>-<variant>/`, and the previous build folder is removed.
- llama-server flags:
  - always: `-np 1 --no-webui --alias`;
  - GPU layers: `-ngl auto` (fit on) or `-ngl N -fit off`;
  - experts in RAM: `-ncmoe N`;
  - `-fa on|off|auto`, `-ctk/-ctv`;
  - `--mmproj` or `--no-mmproj`.
- Readiness: `/health` returns 503 while loading and 200 when ready. The child dies with the app (PR_SET_PDEATHSIG).
- Memory plan (`engine/fit.rs`):
  - Mirrors llama.cpp's placement: the last N layers go on the GPU, and the output head once ngl > n_layer. Token embeddings (and Gemma's per-layer embeddings) stay in RAM. `-ncmoe` keeps the experts of the first N layers in RAM.
  - KV cache comes from tensor shapes, the SWA pattern and `shared_kv_layers`; recurrent (SSM) layers have none.
  - The plan is approximate.
- Ollama store: read in place (manifests → `sha256-` blobs). Ollama-packaged models (`a.`/`v.`/`mm.` towers inside, gpt-oss `attn_out` names) may not load in upstream llama.cpp; the library marks them "Ollama layout".
- Never run downloaded llama.cpp binaries from a script without the user's say-so.

## Anatomy
- Files are memory-mapped.
- On open, a background pass samples 8 rows of every tensor (spread colours and the layer fingerprint); this takes about 5–9 s on /mnt/storage.
- Tensor detail:
  - histogram from up to 4096 sampled rows;
  - a 512×512 RMS overview with progress and cancel;
  - a 48×48 zoom window of real values.
- IQ1/IQ2/IQ3 codebook types are not decoded yet (`can_dequantize`).

## X-ray (stage 3)
- `engine/xray.rs` holds the pure logic plus the client; `ui/xray.rs` is the screen.
- Generation:
  - `/apply-template` turns the messages into the prompt;
  - `/completion` streams with `n_probs` 10. Each partial chunk carries one token's `completion_probabilities`: `id`, `bytes`, `logprob` and `top_logprobs` (`prob`/`top_probs` with `post_sampling_probs`). This was checked in server-task.cpp at b12170.
- Token text: build it from `bytes`, since a token can be half a UTF-8 character.
- Forks: `Tree::continuation` sends `[template, id, id, …]`, so nothing is re-tokenised. `Tree::locate` maps a position in a branch's text to the branch and step that own it.
- The sampler lens mirrors llama.cpp's default chain (top-k → top-p → min-p → temperature) over the visible top-10. It is exact only while the cut-offs fall inside those ten.
- Only the local llama-server is supported; Ollama and cloud providers don't give per-token alternatives this way.

## Steering (stage 4)
- `cubic_gguf::forward`: CPU forward pass for llama/qwen2/qwen3, all prompts in one batch.
  - The shared beginning is computed once.
  - Every weight row is decoded once per layer, and an AVX2 kernel does 2 rows × 4 tokens per step (~25-35 tok/s for 8B Q4 on 16 threads).
  - Rope covers norm/neox, `rope_freqs` (Llama 3.1) and YaRN/linear scaling.
  - Checked against Ollama/llama.cpp next-token logprobs to about 0.02.
- Hidden states are the residual after each block, at each prompt's last token.
- Tokenising and templating go through the running llama-server (`/apply-template`, `/tokenize` with `parse_special`), so the tokens match chat exactly.
- llama.cpp control vectors:
  - `direction.N` (F32, n_embd) is added after 0-indexed block N;
  - N=0 is rejected, and N may be at most n_layer;
  - they load at start-up only, so each strength is a restart.
- We bake a file per (vector, window, strength): only the window, pre-scaled. It is passed with a plain `--control-vector`, because `--control-vector-scaled` splits on ':', which breaks Windows paths.
- Strength scale: `scale = strength × 1.5 / window width` (`STRENGTH_UNIT`), on raw mean differences.
  - On Llama 3.1 8B, layers 14-23: strength ±1 (scale 0.15) is clearly steered and fluent, ±1.7 (0.25) is all caps / nihilism, and ±0.7 (0.1) barely shows.
- Vectors live in `<data>/steering/<id>.{json,gguf}`, and baked files in `steering/baked/`.
- `RunPreset.steer` keeps one on for chat, and `App::start_model` bakes it.
- Comparisons and builds advance in `ui::steer::pump`, called every frame from `App::ui`. After a comparison, the model is restarted with its own preset.

## Data
- Paths come from `directories` (`dev/martin/cubicai-lab`).
- `settings.json` is in config.
- Chats are one JSON file each in `<data>/chats`, in the phone's camelCase format plus `tps`.
- Keys go in the keyring service `cubicai-lab`; the fallback is a 0600 `secrets.json`. Never put a secret in the repo.
- Behaviours in `personas.rs` are generated from the phone's `Personas.kt`; keep them in sync.
