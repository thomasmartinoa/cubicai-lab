# CubicAI Lab: desktop companion to the CubicAI Android app

Rust workspace (edition 2024): `crates/gguf` (cubic-gguf, no UI), `crates/gateway` (cubic-gateway, the phone's PC gateway) and `app` (cubicai-lab, egui/eframe 0.37 on wgpu). Personal use. Stages: 1 chat + Engine Room and Anatomy (done), 3 X-ray + token fork (done), 4 Steering Lab (done), 5 Fine-tune Studio (done; verified live), 6 the phone's gateway rewritten in Rust (the user's choice over keeping Python): 6a core (done), 6b ComfyUI media (done; verified live), 6c voice calls (done; verified live), 6d the Lab as its panel + switch-over (done 2026-11-07: the `cubicai-server` service runs the Rust gateway). The Python gateway was removed from the CubicAI repo the same day; it lives on only in git history there and as recorded answers here. 7 (the user's pick, 2026-11-07, in order): 7a the Arena (done; verified live), 7b the dataset builder in Fine-tune (done; verified live), 7c the report card after training (done; verified live).

## Build / check
- `cargo build --release`; `cargo clippy --workspace --all-targets` should stay clean.
- Screens are checked offscreen with egui_kittest (`app/src/shots.rs`):
  - `SHOTS=<dir> MODEL=<name part> TENSOR=<tensor> [LIGHT=1] [SIZE=960x760] cargo test --release shots -- --nocapture` (SIZE checks a narrow window: the top bar drops the server words, then the brand words, then scrolls its tabs; side panels shrink with the window; Phone and Settings are a centred column)
  - Read the PNGs rather than opening windows on the desktop.
- Tests:
  - `cargo test --workspace --release` runs about 191 offline tests in roughly a second, and must stay green.
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

## Dataset builder (stage 7b, `engine/datakit.rs`, `ui/build.rs`)
- `Example` has an `id` (assigned on load, `ensure_ids`), `score`/`note` (AI grade), `held` (test split: left out of `to_jsonl`, used by the report card via `held_prompts`) and `tokens` (exact count). `Dataset.pairs` are DPO pairs (`pairs_jsonl`: conversational prompt/chosen/rejected).
- Versions: `Store::snapshot` before every bulk change (each job, imports, cleaning, the split, restore), in `<id>/versions`, newest 20 kept.
- Jobs (`ui/build.rs`): `start_job` snapshots, then runs on a `Task` pushing `Out` pieces (Add, Grades, Changes, Tokens keyed by example id) that `pump` applies to the live dataset each frame, so edits made meanwhile survive. AI work uses the drafting provider and its Settings model (8 per rewrite batch, 10 per grade batch; 3 failed batches end the job).
- Grading asks for the reason before the score and to check facts itself: llama3.1:8b still called 7 × 6 = 42 wrong and 2 + 2 = 5 right, gemma4:e4b got all 12 (`net_builder` uses gemma4).
- Hugging Face via datasets-server (`/splits`, `/rows` ≤100 per call, the token from Settings for gated sets). Documents: `pdf-extract` (behind `catch_unwind`), HTML stripped, ~3500-character pieces.
- Near duplicates: MinHash (64 hashes, 16 bands) over word 3-grams, then exact Jaccard. Topic map: TF-IDF + k-means++ (fixed seed) + PCA by power iteration; words seen once are kept below 100 examples.
- Token counts use the running llama-server's `/apply-template` + `/tokenize` (so they include the template).
- DPO: `Params.method` "dpo" + `beta`; train.py uses TRL's `DPOTrainer` with `ref_model=None` (the LoRA off is the reference), `PatchDPOTrainer` when Unsloth has it, no replies-only step. `net_real_dpo`: 24 pairs, 18 steps in 96 s, loss 0.693 → 0.009, base "The capital of Japan is Tokyo." vs tuned "Ahoy! Tokyo is the capital of Japan.".
- `Adapter.dataset_id` links an adapter to its dataset (for the report card).

## Report card (stage 7c, `arena::report_run`/`report`, `ui/arena.rs` `report_card`)
- In Fine-tune's Try it. `report_run`: contestants `Adapter{scale 0}` (base) and `{scale 1}` (tuned) on the same model and adapter file, so one llama-server answers both (`lora` per request); the dataset's `held_prompts` (≤ `REPORT_MAX` 40) with the held-back reply as reference; temperature 0, blind, `REPORT_RUBRIC` ("closer to the reference in content, tone and format, while still being correct"), `report_for` = the adapter id. It is an ordinary Arena run (open it there).
- `report`: wins/ties from verdicts (failed judgements left out), the judge's averages, `similarity` to the reference.
- A judge on Ollama gets the GPU: `pump_judge` stops our server first (the user's model comes back at the end). Before that, gemma4 judged on the CPU (127 MiB in VRAM) and a 6-question card took 422 s; now 70 s (12 s answering).
- `net_report_card`: 24 Ahoy examples, 6 held back (`hold_out` 0.25, random), trained on 18 in ~45 s; tuned won 6 of 6 both runs, judge 9.3 vs 6.5, shared wording ~0.75–0.82 vs ~0.56. The Try-it shot replays the first run.

## Gateway (stage 6, `crates/gateway`)
- A Rust port of `cubicai/server/cubicai_server.py` speaking exactly its protocol, so the phone needs no change and no re-pairing: same files in `~/.config/cubicai-server` (cert.pem, key.pem, token, settings.json), same fingerprint, same `/v1/cubicai/status` JSON (the GTK panel can read it). Binary `cubicai-gateway serve|pair|remote on/off/status|rotate|voice-setup|install`.
- Threads, no async. rustls (ring, TLS 1.3 only), our own small HTTP/1.1 (`http.rs`, keep-alive, Content-Length bodies, SSE), ureq for Ollama/SearXNG/pages.
- Rules kept from Python: drop disallowed peers before TLS and 403 per request; 64 slots, 12 per address; 1 s sleep + 401 on a bad token; error replies drain the body (≤1 MiB) and close; chat replies are SSE with `Connection: close`; Ollama tool-call arguments are re-encoded like Python's json.dumps (`python_dumps`).
- Page fetching for search is guarded by `search::PublicOnly`, a ureq resolver that refuses unless every resolved address is public (`access::is_global`); ureq connects only to those addresses and resolves each redirect again, proxies off.
- The long Ollama agent limits only connect (10 s) and time to first byte (900 s), not the whole stream. Model info is cached only when /api/show succeeded (Python also cached failures).
- Media (6b): `workflow.rs` (comfy_workflow.py: UI workflow → API prompt), `media.rs` (the 13 `GENERATORS`, prompt building, the generator list with the workflow's own defaults, saving on the PC), `jobs.rs` (`Comfy`: the `cubicai-comfyui` unit started on demand with reset-failed + a second try, stopped after `CUBICAI_COMFY_IDLE` if we started it, `/free` + `free_ollama(partial_only)` before a chat; `Jobs`: one worker, a Condvar per job for the event streams, `preview_seq` so a preview is sent once, a janitor that empties our ComfyUI's history when idle and deletes results not collected within the hour; our own small WebSocket client for ComfyUI's progress and previews). Starting the media manager clears ComfyUI's `cubicai/` folders, the temp folder and our unit's queue, so it must not start beside the running service (`Config.media: None` in `real_ollama_chat_and_search`).
- Python quirks kept: `round()` is banker's rounding (`round_ties_even`), seeds are taken modulo the node's max+1 in u128 (KSampler's max is 2^64−1), serde_json has `preserve_order` (object_info order decides widget order; also Python's key order in events).
- Voice (6c, `src/voice/`): `text` (speakable words, sentence cutter; Python counts characters, so the thresholds do), `phonemes` (libespeak-ng by dlopen behind a mutex, phonemizer's punctuation preserve/restore and word clean-up ported exactly), `asr` (NeMo log-mel features with our own radix-2 FFT and librosa's Slaney bank, greedy TDT via the `Decoder` trait), `tts` (kokoro-onnx: vocab table, ≤510-phoneme batches, styles from the uncompressed npz, librosa trim), `models` (ONNX Runtime through `ort` 2.0.0-rc.15 with `load-dynamic`: `voice-setup` puts libonnxruntime.so.1.31.1 in the voice folder; one environment, telemetry off; VAD single-threaded, ASR/TTS without spinning), `call` (the protocol; the connection thread owns the TLS stream, reads with a 10 ms timeout and writes what the ears/mouth threads queued; speech is tagged with the cancel epoch so a barge-in drops it unsent). `Engine` trait so calls can be tested with a scripted engine; `Voice` loads on the first call, unloads after `CUBICAI_VOICE_IDLE`, caches 40 previews.
- Measured: models load in 2.2 s (Python ~7 s); Kokoro's audio is bit-identical to Python's; transcripts identical; a real-time call gets the final transcript 0.61 s after speech ends (600 ms of it the turn pause) and the first sentence voiced 0.25 s after it arrives.
- Service (6d, `src/local.rs`): `install` copies the binary to `~/.local/bin/cubicai-gateway` (copy + rename, so a running one keeps its file), writes `cubicai-server.service` with the Python unit's exact sandbox (home read-only: the voice models and ONNX Runtime are only read), writes `cubicai-comfyui.service` (`COMFY_UNIT`) only if there's none, and enables + restarts the service. There is no going back to Python any more. `local::get` reads the status over TLS pinned to the fingerprint, with the token, as the phone does.
- Tests: `tests/units.rs` (rules, against Python's own outputs), `tests/e2e.rs` (real TLS client pinning the fingerprint, fake Ollama/SearXNG: search, keep-alive, slots, hang-up stops Ollama). `real_ollama_chat_and_search` (ignored) runs on a spare port with the real identity, Ollama and SearXNG: the model list, a chat's event shape (content, usage, [DONE]: what Python sent too), a live search (8 sources, ~11 s). `tests/media.rs`: a fake ComfyUI (HTTP + WebSocket with pings, fragments, a binary preview, an `../etc` path in its history) driving a job end to end, errors, cancel, expiry and saving; `real_comfy_makes_a_picture` (ignored) makes a FLUX.2 Klein picture on the real ComfyUI (~13 s with a cold start) in its own cache folder and stops the unit again if it was off. `tests/voice.rs`: speakable/segmenter/phoneme values from Python, TDT decoding with a scripted decoder, trim/npz/PCM, WebSocket frames, and a whole call over real TLS with a scripted engine (handshake + frames in one write, captions, final, sentences + PCM, cancel dropping queued speech, ping, close); `real_call_with_the_real_models` (ignored) streams Kokoro's voice in real time as the microphone. `tests/voice_golden.rs` (ignored; needs the voice models): text, phonemes in 7 languages, features vs onnx-asr, Kokoro audio and Parakeet transcripts against Python's. `tests/golden.rs` (ignored; needs the saved workflows): converter on all 21 saved workflows, every generator's built prompt, sizes and the list match Python's exactly. Python's answers are recorded in `tests/golden/recorded` (the workflows' contents only as SHA-256 per node, since they hold the user's prompts; a mismatch prints the Rust node). `GOLDEN_RECORD=<dir with cubicai_server.py and its .venv>` re-records them from an old checkout of the CubicAI repo (`git worktree add <dir> 2017145`), running `tests/golden/*.py`.

## Arena (stage 7a, `engine/arena.rs`, `ui/arena.rs`)
- A `Run` holds 2–4 contestants (`Who`: `Library` GGUF on our llama-server with its preset, `Adapter` at a scale (0 = base), `Provider` = Ollama or a cloud provider with a model) and `Item`s (prompt, optional reference, an `Answer` per contestant, your `Vote`, the judge's `Verdict`, a blind `order`). Saved in `<data>/arena/runs/<id>.json`, prompt sets in `arena/sets/`.
- Running (`ui::arena::pump`, every frame): each cloud contestant answers its prompts one after another, all cloud contestants at once; local ones (llama.cpp, adapters, Ollama) share one GPU lane in `local_order` (Ollama first, then ours grouped by model so each loads once), contestant by contestant. The GPU is handed over first (`hand_gpu_to_ollama`: Ollama's other models unloaded and the next one loaded warm, our server stopped for Ollama; Ollama emptied before our server starts), so load time isn't timed. Afterwards the server goes back to the model it ran before, with its preset (`restore_server`).
- Events are stamped on a forwarding thread, so first-token times aren't rounded to frames. Speed: llama-server's `timings` when given, else tokens after the first over the time after it, only from 8 tokens. GPU memory: our server's process, or Ollama's runner (also named llama-server: told apart by `/proc/<pid>/cmdline`, since its `exe` link belongs to the ollama user).
- `llm::Request.extra` carries temperature/max_tokens and the adapter's per-request `lora`.
- The judge (any provider, temperature 0) sees the working answers as A, B, … in the item's blind order with no names, the reference if any, and the rubric; it answers JSON scores 1–10 + best + why (`parse_verdict` takes fences, string numbers, "Answer B", "tie"). Leaderboard: Elo (K 32, start 1000) from your votes in time order, a pick beats each other working answer, tie / all bad draw; judge agreement counts prompts where both decided.
- Differences: LCS over words (whitespace kept with each word, case ignored); over 4M cells it compares lines. Quant siblings: `split_quant` reads the quant from the name (Q*/IQ*/TQ*, F16/BF16/MXFP4, after `-` `.` `_` or `:`, split-file suffix dropped).
- `net_arena` (ignored): llama3.1:8b on llama.cpp (LLAMA_BIN, default the b12173 the other live tests installed in /tmp/cubic-llama) vs the same on Ollama vs qwen3.5:4b, judged by gemma4:e4b; ~90 s, deletes its run.

## Phone panel (stage 6d, `app/src/phone.rs`, `ui/phone.rs`)
- Replaces the GTK "CubicAI Server" window (`cubicai/server/cubicai_control.py`): Status (service on/off, start at login, which gateway the unit runs with Install / Update / Use the Rust gateway, requests, Ollama's loaded models with Unload, SearXNG container, context length), Pair (QR, address, copy link, Tailscale remote access, Unpair all with a confirm, the ufw hint) and Activity (right now, recent, devices with router names from `getent hosts`).
- `Panel::poll` gathers everything off the UI thread every 2 s while the tab is open (systemctl, docker, Ollama, `local::get`); actions run through `Panel::act` and come back as toasts. Settings (`remote`, `num_ctx`) are written straight to settings.json, which the gateway reads per request.
- QR: the `qrcode` crate (level M, 2-module quiet zone), black on white whatever the theme; a test decodes it with `rqrr` back to the exact link. Its segments differ from Python's `qrcode`, so the matrices differ but both scan. On Hyprland the Pair view asks for an opaque window (runtime rule, class `cubicai-lab`).
- `cubicai-lab phone|pair|activity` opens there; the "CubicAI Server" launcher (`~/.local/share/applications/dev.martin.cubicai.Server.desktop`) runs `~/.local/bin/cubicai-lab` with those (copy the release build there after rebuilding).
- Screenshots use sample data and a throwaway identity, never the real token. `live_panel_reads_the_real_gateway` (ignored) checks it on this PC.

## Data
- Paths come from `directories` (`dev/martin/cubicai-lab`).
- `settings.json` is in config.
- Chats are one JSON file each in `<data>/chats`, in the phone's camelCase format plus `tps`.
- Keys go in the keyring service `cubicai-lab`; the fallback is a 0600 `secrets.json`. Never put a secret in the repo.
- Behaviours in `personas.rs` are generated from the phone's `Personas.kt`; keep them in sync.
