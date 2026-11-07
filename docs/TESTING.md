# Testing CubicAI Lab

```
cargo test --workspace --release              # every offline test, a few seconds
cargo clippy --workspace --all-targets        # must stay clean
```

The repository doesn't use rustfmt; please don't run `cargo fmt` over it.

## Offline tests (191)

- **GGUF:**
  - Every weight decoder is checked against gguf-py's output, using fixtures in `crates/gguf/tests/fixtures`.
  - A small model written byte by byte checks the parser, the layer layout, the KV-cache shape, the statistics and the zoom windows against exact values.
- **Engine:**
  - llama-server arguments and failure messages;
  - a full start → ready → chat → stop run against a stand-in `llama-server`;
  - the memory plan (placement, KV cache with sliding windows, MoE experts in RAM);
  - Ollama store and folder scanning;
  - Hugging Face file grouping and benchmark output;
  - choosing the release asset, and an offline install with the CUDA runtime, unpacking and resumed downloads.
- **Chat:**
  - the SSE stream (content, reasoning, usage, speed, dropped connections, errors);
  - requests to a mock server;
  - the phone's chat files read and written back unchanged;
  - settings fallbacks;
  - behaviours kept in sync with the phone's.
- **X-ray:**
  - the server's probability chunks (both shapes);
  - uncertainty in bits;
  - the sampler lens against hand-worked examples;
  - branch trees with nested forks and exact-token continuation prompts;
  - the screen's own run → fork → continue flow against a mock llama-server.
- **Steering:**
  - the CPU forward pass: batching and shared beginnings change nothing, it is causal and position-aware, quantised and Qwen variants, and steering lands after the right layer;
  - the direction maths and separation;
  - control-vector files in llama.cpp's format (no layer 0, only the steered window, scaled);
  - the vector store;
  - a full build on a small model through a stand-in llama-server;
  - server arguments, and presets saved before steering existed.
- **Fine-tune:**
  - safetensors in F32/F16/BF16;
  - PEFT → GGUF names, and Llama's q/k row order checked against convert_hf_to_gguf's numpy formula;
  - adapter files in llama.cpp's shape rules, and adapters for the wrong model refused;
  - datasets (validity, JSONL, forgiving parsing of drafted replies, chats as examples, held-back examples left out, versions);
  - the builder: imports in every common shape (and preference pairs), Hugging Face names and rows, documents cut into pieces, exact and near duplicates, length limits, personal details found and replaced, grades and rewrites read from the shapes models write, topic clusters, Arena votes as examples and pairs, and the test split;
  - the DPO trainer config;
  - the trainer's events, and a whole run with a stand-in trainer (stopping, errors, conversion);
  - the per-request adapter switch.
- **Arena:**
  - word differences (joined back exactly, case ignored, long answers by line) and how similar two answers are;
  - stable blind orders, the GPU lane's order (Ollama first, one model at a time), and speed from timings or tokens;
  - the judge's prompt (only working answers, blind letters, no names) and verdicts in the shapes models write;
  - Elo from votes, draws, failed answers left out, the judge's averages and agreement;
  - quantisations read from names, and runs and prompt sets saved and loaded;
  - the report card's run (base and tuned on one server, blind, the held-back reply as the reference) and its count of wins, scores and shared wording.
- **Gateway:**
  - who may connect (home network, the PC's own IPv6 prefixes, Tailscale only with remote on), and which addresses count as public;
  - the pairing link and tool-call arguments, byte for byte against the Python gateway;
  - Ollama's stream turned into the phone's events, search queries, page text, and pages on the home network never being fetched;
  - end to end over real TLS with the certificate pinned like the phone does: the token, keep-alive, slot limits, a chat with web search, and hanging up stopping the model;
  - voice calls with a scripted engine over real TLS: hearing the user (captions, the final transcript), speaking a streamed reply sentence by sentence, a barge-in dropping speech already queued; and the voice's text, phoneme and decoding rules against values from the Python gateway;
  - the Phone panel: times and rows as the old panel showed them, settings written where the gateway reads them, the pairing QR decoded back to the exact link, and which gateway a service unit runs;
  - making things through a stand-in ComfyUI: the generator list, a picture's progress, preview and download, nothing left behind, ComfyUI's own errors, cancelling, results expiring after an hour, and saving on the PC.

## Live tests (`--ignored`, run by hand)

These use the network, this PC's GPU, Ollama, or downloaded llama.cpp and Python. Each cleans up after itself (temporary folders, runs deleted, settings changed only in memory).

```
cargo test --release net_ -- --ignored --nocapture --skip net_real    # GitHub release + install, Hugging Face, Ollama stream
LLAMA_DIR=/tmp/llama MODEL_PATH=/path/model.gguf \
  cargo test --release net_real_llama -- --ignored --nocapture        # real llama.cpp: install, load, chat, X-ray + fork,
                                                                      # our forward pass vs llama.cpp, a steering vector, benchmark
LLAMA_DIR=/tmp/llama MODEL_PATH=/path/llama-3.1-8b.gguf \
  cargo test --release net_real_train -- --ignored --nocapture        # sets up the trainer, trains an adapter, checks it in llama.cpp
LLAMA_DIR=/tmp/llama MODEL_PATH=/path/llama-3.1-8b.gguf \
  cargo test --release net_real_dpo -- --ignored --nocapture          # the same habit taught by preference pairs only (DPO)
cargo test --release net_builder -- --ignored --nocapture             # the dataset builder's jobs on Ollama, Hugging Face and llama.cpp
cargo test --release net_report_card -- --ignored --nocapture         # trains with 6 questions held back, then the report card on them
cargo test --release net_arena -- --ignored --nocapture               # llama.cpp vs Ollama vs another model, judged
cargo test -p cubic-gateway --release real_ollama -- --ignored --nocapture
                                                                      # the gateway on a spare port with the real Ollama and SearXNG
cargo test -p cubicai-lab --release live_panel -- --ignored --nocapture
                                                                      # the Phone panel reading this PC's real gateway
cargo test -p cubic-gateway --release real_comfy_makes_a_picture -- --ignored --nocapture
                                                                      # a real picture through ComfyUI (starts and stops its unit)
cargo test -p cubic-gateway --release --test golden -- --ignored      # workflow conversion and prompts identical to the Python gateway's
cargo test -p cubic-gateway --release --test voice real_call -- --ignored --nocapture
                                                                      # a real-time voice call on the real models (needs voice-setup)
cargo test -p cubic-gateway --release --test voice_golden -- --ignored --test-threads 1
                                                                      # phonemes, features, Kokoro audio and transcripts vs Python's
```

`net_arena`, `net_builder` and `net_report_card` take the llama.cpp folder from `LLAMA_BIN` (the folder holding `llama-server`) and use Ollama's `llama3.1:8b` and `gemma4:e4b`. `net_builder` takes the grading model from `OLLAMA_MODEL`.

Measured on an RTX 3070 Ti (8 GB):

| Test | Result |
|---|---|
| `net_real_train` | 18 steps in 44 s, loss 3.72 → 0.13; "The capital of Japan is Tokyo." becomes "Ahoy! Tokyo is the capital of Japan." |
| `net_real_dpo` | 24 pairs, 18 steps in 96 s, loss 0.693 → 0.009, same habit learned from preferences alone |
| `net_report_card` | trained on 18, 6 held back; tuned won 6 of 6, judge 9.3 vs 6.5; about 70 s for the card |
| `net_builder` | grading 12, fixing, rewording, questions from a document, 150 Hugging Face rows, 30 exact token counts: about a minute |
| `net_arena` | three models on two prompts plus judging, about 90 s |

## Screenshots

```
SHOTS=/tmp/s MODEL=gpt-oss TENSOR=blk.5.ffn_down_exps.weight \
  cargo test --release shots -- --nocapture                           # renders every screen offscreen to PNGs
```

The screens are drawn with sample data held in memory (the Fine-tune ones replay the real training run above), so nothing personal appears and nothing is saved. `docs/screenshots` holds the ones the README uses.
