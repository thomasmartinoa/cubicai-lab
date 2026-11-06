# CubicAI Lab: desktop companion to the CubicAI Android app

Rust workspace (edition 2024): `crates/gguf` (cubic-gguf, no UI), `crates/gateway` (cubic-gateway, the phone's PC gateway) and `app` (cubicai-lab, egui/eframe 0.37 on wgpu). Personal use. Stages: 1 chat + Engine Room and Anatomy (done), 3 X-ray + token fork (done), 4 Steering Lab (done), 5 Fine-tune Studio (done; verified live), 6 the phone's gateway rewritten in Rust (the user's choice over keeping Python): 6a core (done), 6b ComfyUI media, 6c voice calls, 6d the Lab as its panel + switch-over.

## Build / check
- `cargo build --release`; `cargo clippy --workspace --all-targets` should stay clean.
- Screens are checked offscreen with egui_kittest (`app/src/shots.rs`):
  - `SHOTS=<dir> MODEL=<name part> TENSOR=<tensor> [LIGHT=1] cargo test --release shots -- --nocapture`
  - Read the PNGs rather than opening windows on the desktop.
- Tests:
  - `cargo test --workspace --release` runs about 135 offline tests in roughly a second, and must stay green.
  - Test helpers live in `app/src/tests/mod.rs`: `tiny_model` (a GGUF built with `cubic_gguf::Writer`), `TempDir` and `mock_http`.
  - Decoder fixtures come from gguf-py (`crates/gguf/tests/fixtures/make.py`).
  - Live tests are `#[ignore]`: `net_*` (network/Ollama), and `net_real_llama` / `net_real_train`. The last two run downloaded llama.cpp, Python and Unsloth; the user has allowed me to run them. `net_real_train` (Llama 3.1 8B, RTX 3070 Ti 8 GB): 18 steps in 44 s, loss 3.72 -> 0.13, base "The capital of Japan is Tokyo." vs tuned "Ahoy! Tokyo is the capital of Japan.". The Fine-tune shots replay that run (`AHOY_FACTS`, `REAL_LOSSES` in shots.rs).

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

## Fine-tune (stage 5)
- **Environment.** `engine/train.rs` `Env` lives in `<data>/tuning/env`.
  - uv comes from the PATH, or is downloaded from astral-sh/uv releases.
  - Setup runs `uv venv --python 3.12 --seed`, then `uv pip install --torch-backend=auto unsloth` (CUDA wheels to match the driver).
  - The system Python here is 3.14, too new for torch.
  - No Mac support: Unsloth needs CUDA.
- **Trainer.** `engine/train.py` is embedded and written to each run folder.
  - It prints `CUBIC {json}` events: status, progress, loss, done, error. Everything else is log.
  - The loss chart (`ui/tune.rs` `smoothed`) draws a centred moving average over the faint raw points, and short runs (under 20 steps) only the raw line. A trailing EMA lagged: on the 18-step run it ended near 0.3 while the loss was 0.13.
  - A `STOP` file in the run folder ends training and keeps the adapter.
  - TRL argument names vary between versions, hence `only_known` and the `processing_class`/`tokenizer` switch.
  - "Replies only" uses Unsloth's `train_on_responses_only` with the Llama 3, ChatML and Mistral markers.
- **Converter.** `engine/lora.rs` is our own PEFT → GGUF converter, so llama.cpp's Python scripts aren't needed.
  - Each pair becomes `blk.N.<attn_q|…|ffn_down>.weight.lora_a` with ne [in, r], and `.lora_b` with ne [r, out].
  - Metadata: `general.type=adapter`, `adapter.type=lora`, `adapter.lora.alpha`, and the base's architecture.
  - Llama arch: B rows of q/k are permuted like `LlamaModel.permute` (n_head for q, n_head_kv for k). Qwen isn't permuted.
  - rsLoRA stores alpha × sqrt(r), because llama.cpp divides by r.
  - Shapes are checked against the base GGUF.
- **Base model.** Training uses Unsloth's bnb-4bit HF twin of the GGUF (`suggest_base`; the user can edit the repo).
- **Before/after.** One llama-server started with `--lora` switches the adapter per request (`"lora": [{"id": 0, "scale": s}]`), so there is no restart.
- **Chat.** `RunPreset.lora` (adapter id) loads it in chat.
- **GPU.** Training stops our server and unloads Ollama first.
- **Storage.** Datasets are `<data>/tuning/datasets/<id>.json`. Each adapter gets `<data>/tuning/adapters/<id>/` with adapter.json, data.jsonl, train.json, adapter/ (PEFT) and lora.gguf.
- **Drafting.** It goes through `App::provider_request` + `llm::collect`, 25 examples per request, dropping repeats.

## Gateway (stage 6, `crates/gateway`)
- A Rust port of `cubicai/server/cubicai_server.py` speaking exactly its protocol, so the phone needs no change and no re-pairing: same files in `~/.config/cubicai-server` (cert.pem, key.pem, token, settings.json), same fingerprint, same `/v1/cubicai/status` JSON (the GTK panel can read it). Binary `cubicai-gateway serve|pair|remote on/off/status|rotate`.
- Threads, no async. rustls (ring, TLS 1.3 only), our own small HTTP/1.1 (`http.rs`, keep-alive, Content-Length bodies, SSE), ureq for Ollama/SearXNG/pages.
- Rules kept from Python: drop disallowed peers before TLS and 403 per request; 64 slots, 12 per address; 1 s sleep + 401 on a bad token; error replies drain the body (≤1 MiB) and close; chat replies are SSE with `Connection: close`; Ollama tool-call arguments are re-encoded like Python's json.dumps (`python_dumps`).
- Page fetching for search is guarded by `search::PublicOnly`, a ureq resolver that refuses unless every resolved address is public (`access::is_global`); ureq connects only to those addresses and resolves each redirect again, proxies off.
- The long Ollama agent limits only connect (10 s) and time to first byte (900 s), not the whole stream. Model info is cached only when /api/show succeeded (Python also cached failures).
- Not ported yet: media routes answer "isn't in this gateway yet" (generators empty), voice reports not ready. So the systemd service stays on Python until 6b/6c.
- Tests: `tests/units.rs` (rules, against Python's own outputs), `tests/e2e.rs` (real TLS client pinning the fingerprint, fake Ollama/SearXNG: search, keep-alive, slots, hang-up stops Ollama). `real_matches_python` (ignored) runs beside the live Python gateway on 8443 with the same identity: model lists identical, same event stream, live search (8 sources, ~13 s).

## Data
- Paths come from `directories` (`dev/martin/cubicai-lab`).
- `settings.json` is in config.
- Chats are one JSON file each in `<data>/chats`, in the phone's camelCase format plus `tps`.
- Keys go in the keyring service `cubicai-lab`; the fallback is a 0600 `secrets.json`. Never put a secret in the repo.
- Behaviours in `personas.rs` are generated from the phone's `Personas.kt`; keep them in sync.
