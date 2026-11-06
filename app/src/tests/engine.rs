use super::{mock_http, tiny_info, TempDir, EMBD, LAYERS};
use crate::engine::gpu::{GpuInfo, Vendor};
use crate::engine::library::{self, Source};
use crate::engine::release::{self, Asset, Release, Variant};
use crate::engine::server::{self, Launch, Server, State};
use crate::engine::{bench, fit, hf, Task};
use crate::store::{FlashAttn, RunPreset};
use std::path::Path;
use std::time::{Duration, Instant};

// --- llama-server arguments ----------------------------------------------------------------------

fn launch_args(p: &RunPreset, mmproj: Option<&Path>) -> Vec<String> {
    Server::args_for(&Launch {
        bin_dir: Path::new("/x"),
        model_key: "k",
        model_name: "my model",
        model: Path::new("/models/a b.gguf"),
        mmproj,
        preset: p,
        port: 8193,
    })
}

fn has(args: &[String], pair: &[&str]) -> bool {
    args.windows(pair.len()).any(|w| w.iter().zip(pair).all(|(a, b)| a == b))
}

#[test]
fn default_arguments_let_llama_cpp_fit() {
    let a = launch_args(&RunPreset::default(), None);
    for pair in [
        &["-m", "/models/a b.gguf"][..],
        &["--host", "127.0.0.1"],
        &["--port", "8193"],
        &["-c", "16384"],
        &["-np", "1"],
        &["-fa", "auto"],
        &["-ctk", "f16"],
        &["-ctv", "f16"],
        &["-ngl", "auto"],
        &["--alias", "my model"],
    ] {
        assert!(has(&a, pair), "missing {pair:?} in {a:?}");
    }
    assert!(a.contains(&"--no-mmproj".to_string()));
    assert!(a.contains(&"--no-webui".to_string()));
    assert!(!a.contains(&"-fit".to_string()), "fit stays on unless layers are fixed");
}

#[test]
fn manual_settings_turn_fitting_off() {
    let p = RunPreset {
        gpu_layers: Some(20),
        cpu_moe_layers: Some(5),
        threads: Some(8),
        flash_attn: FlashAttn::Off,
        kv_type: "q8_0".into(),
        extra_args: r#"--temp 0.7 --system-prompt-file "/a b/c.txt""#.into(),
        ..Default::default()
    };
    let a = launch_args(&p, Some(Path::new("/m/mmproj.gguf")));
    for pair in [&["-ngl", "20"][..], &["-fit", "off"], &["-ncmoe", "5"], &["-t", "8"], &["-fa", "off"], &["-ctk", "q8_0"], &["--mmproj", "/m/mmproj.gguf"], &["--temp", "0.7"], &["--system-prompt-file", "/a b/c.txt"]] {
        assert!(has(&a, pair), "missing {pair:?} in {a:?}");
    }
    let no_vision = launch_args(&RunPreset { use_mmproj: false, ..Default::default() }, Some(Path::new("/m/mmproj.gguf")));
    assert!(no_vision.contains(&"--no-mmproj".to_string()));
}

#[test]
fn extra_arguments_split_like_a_shell() {
    assert_eq!(server::shell_words(""), Vec::<String>::new());
    assert_eq!(server::shell_words("  -a  1 "), vec!["-a", "1"]);
    assert_eq!(server::shell_words(r#"--x "two words" 'single q' """#), vec!["--x", "two words", "single q", ""]);
}

#[test]
fn failures_are_explained() {
    assert!(server::explain_failure("ggml_backend_cuda_buffer_type_alloc_buffer: cudaMalloc failed: out of memory").contains("GPU memory"));
    assert!(server::explain_failure("error loading model: unknown model architecture: 'gemma9'").contains("can't load this model"));
    assert!(server::explain_failure("libcublas.so.13: cannot open shared object file").contains("library is missing"));
    assert!(server::explain_failure("bind: Address already in use").contains("port is taken"));
    assert!(server::explain_failure("something else").contains("something else"));
}

// --- llama-server lifecycle, with a stand-in executable ------------------------------------------

/// A fake llama-server (Python, stdlib only): logs, answers /health 503 then 200, streams a reply.
#[cfg(unix)]
fn fake_server(dir: &Path, crash: bool) {
    use std::os::unix::fs::PermissionsExt;
    let script = if crash {
        "#!/bin/sh\necho 'load_tensors: loading model'\necho 'cudaMalloc failed: out of memory' >&2\nexit 1\n".to_string()
    } else {
        r#"#!/usr/bin/env python3
import sys, time, json, http.server
args = sys.argv[1:]
port = int(args[args.index('--port') + 1]); alias = args[args.index('--alias') + 1]
print('main: loading model', ' '.join(args), flush=True)
start = time.time()
class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        ok = time.time() - start > 0.6
        self.send_response(200 if ok else 503); self.end_headers()
        self.wfile.write(b'{"status":"ok"}' if ok else b'{"error":"loading"}')
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        self.send_response(200); self.send_header('Content-Type', 'text/event-stream'); self.end_headers()
        for d in [{'content': 'Hello from '}, {'content': alias}]:
            self.wfile.write(('data: ' + json.dumps({'choices': [{'delta': d}]}) + '\n\n').encode())
        self.wfile.write(('data: ' + json.dumps({'choices': [{'delta': {}, 'finish_reason': 'stop'}], 'timings': {'predicted_per_second': 99.0}}) + '\n\n').encode())
        self.wfile.write(b'data: [DONE]\n\n')
http.server.HTTPServer(('127.0.0.1', port), H).serve_forever()
"#
        .to_string()
    };
    let exe = release::server_exe(dir);
    std::fs::write(&exe, script).unwrap();
    std::fs::set_permissions(&exe, std::fs::Permissions::from_mode(0o755)).unwrap();
}

#[cfg(unix)]
fn wait_for(s: &mut Server, secs: u64, f: impl Fn(&State) -> bool) -> State {
    let t = Instant::now();
    while t.elapsed() < Duration::from_secs(secs) {
        s.poll();
        let st = s.state();
        if f(&st) {
            return st;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    s.state()
}

#[cfg(unix)]
#[test]
fn server_starts_answers_and_stops() {
    let dir = TempDir::new("server");
    fake_server(&dir.0, false);
    let mut s = Server::new();
    let preset = RunPreset::default();
    let model = dir.join("m.gguf");
    s.start(Launch { bin_dir: &dir.0, model_key: "key1", model_name: "tiny", model: &model, mmproj: None, preset: &preset, port: 18193 }).unwrap();
    assert_eq!(s.state(), State::Loading, "loading until /health says ok");
    let st = wait_for(&mut s, 15, |st| *st != State::Loading);
    assert_eq!(st, State::Ready, "log: {:?}", s.log_lines());
    assert!(s.is_ready_with("key1") && !s.is_ready_with("other"));
    assert!(s.ready_after.is_some());
    assert!(s.log_lines()[0].starts_with("$ llama-server -m"));

    // Chat through it like the app does.
    let (tx, rx) = std::sync::mpsc::channel();
    let req = crate::llm::Request { base_url: s.base_url(), key: String::new(), model: "tiny".into(), messages: vec![serde_json::json!({"role":"user","content":"hi"})], local: true };
    crate::llm::stream(req, 1, tx, Default::default(), egui::Context::default());
    let mut reply = String::new();
    let mut tps = 0.0;
    loop {
        match rx.recv_timeout(Duration::from_secs(10)).unwrap().1 {
            crate::llm::Ev::Content(c) => reply += &c,
            crate::llm::Ev::Speed { gen_tps } => tps = gen_tps,
            crate::llm::Ev::Done => break,
            crate::llm::Ev::Error(e) => panic!("{e}"),
            _ => {}
        }
    }
    assert_eq!(reply, "Hello from tiny");
    assert_eq!(tps, 99.0);

    let pid = s.pid().unwrap();
    s.stop();
    assert_eq!(s.state(), State::Stopped);
    assert!(s.pid().is_none());
    std::thread::sleep(Duration::from_millis(100));
    assert!(!Path::new(&format!("/proc/{pid}")).exists() || std::fs::read_to_string(format!("/proc/{pid}/stat")).unwrap_or_default().contains(") Z"), "child still running");
}

#[cfg(unix)]
#[test]
fn a_crashing_server_reports_why() {
    let dir = TempDir::new("crash");
    fake_server(&dir.0, true);
    let mut s = Server::new();
    let preset = RunPreset::default();
    let model = dir.join("m.gguf");
    s.start(Launch { bin_dir: &dir.0, model_key: "k", model_name: "m", model: &model, mmproj: None, preset: &preset, port: 18293 }).unwrap();
    std::thread::sleep(Duration::from_millis(300)); // let the log readers catch the last lines
    match wait_for(&mut s, 10, |st| matches!(st, State::Failed(_))) {
        State::Failed(why) => assert!(why.contains("GPU memory"), "{why}"),
        other => panic!("{other:?}"),
    }
}

#[test]
fn starting_without_llama_cpp_says_install_it() {
    let dir = TempDir::new("nobin");
    let mut s = Server::new();
    let preset = RunPreset::default();
    let e = s.start(Launch { bin_dir: &dir.0, model_key: "k", model_name: "m", model: Path::new("/m.gguf"), mmproj: None, preset: &preset, port: 18393 }).unwrap_err();
    assert!(e.contains("isn't installed"));
}

// --- Memory plan --------------------------------------------------------------------------------

fn gib(x: f64) -> u64 {
    (x * (1u64 << 30) as f64) as u64
}

#[test]
fn placement_follows_llama_cpp() {
    let d = TempDir::new("fit");
    let info = tiny_info(&d, false);
    let p = RunPreset { ctx: 4096, ..Default::default() };
    let n = LAYERS as u32;
    let total: u64 = info.gguf.total_tensor_bytes();
    let embd: u64 = info.anatomy.before.iter().map(|&i| info.gguf.tensors[i].n_bytes).sum();
    let head: u64 = info.anatomy.after.iter().map(|&i| info.gguf.tensors[i].n_bytes).sum();

    let cpu_only = fit::estimate(&info, &p, n + 1, 0, false);
    assert_eq!((cpu_only.weights_gpu, cpu_only.kv_gpu, cpu_only.compute, cpu_only.overhead), (0, 0, 0, 0));
    assert_eq!(cpu_only.weights_cpu, total);

    let all = fit::estimate(&info, &p, n + 1, 0, true);
    assert_eq!(all.weights_cpu, embd, "token embeddings stay in RAM");
    assert_eq!(all.weights_gpu + all.weights_cpu, total);
    assert_eq!(all.kv_cpu, 0);

    let layers_only = fit::estimate(&info, &p, n, 0, true);
    assert_eq!(layers_only.weights_cpu, embd + head, "the head joins only after every layer");

    // Weights on the GPU grow with every layer offloaded.
    let mut last = 0;
    for l in 0..=n + 1 {
        let e = fit::estimate(&info, &p, l, 0, true);
        assert!(e.weights_gpu >= last);
        last = e.weights_gpu;
    }
    assert_eq!(fit::estimate(&info, &p, 99, 0, true).gpu_layers, n + 1, "clamped");
}

#[test]
fn kv_cache_size_from_shapes_windows_and_type() {
    let d = TempDir::new("kv");
    let info = tiny_info(&d, false);
    let n = LAYERS as u32 + 1;
    // K and V rows: 64 + 64 per token per layer; half the layers are capped at 128 + ubatch cells.
    let p = RunPreset { ctx: 8192, ubatch: 512, ..Default::default() };
    let e = fit::estimate(&info, &p, n, 0, true);
    let full = 4 * 8192 * 128 * 2;
    let windowed = 4 * (128 + 512) * 128 * 2;
    assert_eq!(e.kv_gpu, full + windowed);
    let q8 = fit::estimate(&info, &RunPreset { kv_type: "q8_0".into(), ..p.clone() }, n, 0, true);
    assert_eq!(q8.kv_gpu, ((full + windowed) as f64 * 34.0 / 64.0) as u64, "q8_0 is 34/64 of f16");
    let fa_off = fit::estimate(&info, &RunPreset { flash_attn: FlashAttn::Off, ..p.clone() }, n, 0, true);
    assert!(fa_off.compute > e.compute, "no flash attention needs the score matrix");
    assert!(fit::kv_bytes("q4_0") < fit::kv_bytes("q8_0") && fit::kv_bytes("q8_0") < fit::kv_bytes("f16"));
}

#[test]
fn auto_plan_fits_the_budget() {
    let d = TempDir::new("plan");
    let info = tiny_info(&d, false);
    let p = RunPreset { ctx: 4096, ..Default::default() };
    let n = LAYERS as u32;
    assert_eq!(fit::auto_plan(&info, &p, gib(64.0)), (n + 1, 0), "everything fits a big GPU");
    assert_eq!(fit::auto_plan(&info, &p, 1).0, 0, "nothing fits 1 byte");
    let half = fit::estimate(&info, &p, n / 2, 0, true).gpu_total();
    let (l, m) = fit::auto_plan(&info, &p, half);
    assert_eq!((l, m), (n / 2, 0));
    assert!(fit::estimate(&info, &p, l + 1, 0, true).gpu_total() > half);
}

#[test]
fn moe_experts_go_to_ram_first() {
    let d = TempDir::new("moe");
    let info = tiny_info(&d, true);
    assert_eq!(info.anatomy.hyper.experts, 8);
    assert!(info.active_params < info.anatomy.params, "only 2 of 8 experts run");
    let p = RunPreset { ctx: 4096, ..Default::default() };
    let n = LAYERS as u32;
    let all = fit::estimate(&info, &p, n + 1, 0, true);
    let three = fit::estimate(&info, &p, n + 1, 3, true);
    let per_layer_experts = all.weights_gpu - fit::estimate(&info, &p, n + 1, 1, true).weights_gpu;
    assert!(per_layer_experts > 0);
    assert_eq!(all.weights_gpu - three.weights_gpu, 3 * per_layer_experts);
    assert_eq!(three.kv_gpu, all.kv_gpu, "the KV cache stays on the GPU");
    // A budget between: all layers on the GPU with some experts in RAM.
    let budget = fit::estimate(&info, &p, n + 1, 4, true).gpu_total();
    assert_eq!(fit::auto_plan(&info, &p, budget), (n + 1, 4));
    assert_eq!(fit::min_cpu_moe(&info, &p, budget), Some(4));
    let dense = TempDir::new("dense");
    assert_eq!(fit::min_cpu_moe(&tiny_info(&dense, false), &p, budget), None);
}

// --- Model library ------------------------------------------------------------------------------

#[test]
fn split_names() {
    assert_eq!(library::split_base("Qwen3-235B-Q4_K_M-00001-of-00003").as_deref(), Some("Qwen3-235B-Q4_K_M"));
    assert_eq!(library::split_base("model-1-of-3"), None);
    assert_eq!(library::split_base("model"), None);
}

#[test]
fn folders_group_splits_and_find_the_projector() {
    let d = TempDir::new("lib");
    std::fs::create_dir_all(d.join("big/sub")).unwrap();
    for (f, n) in [("small.gguf", 10), ("big/sub/huge-00001-of-00002.gguf", 100), ("big/sub/huge-00002-of-00002.gguf", 50), ("big/sub/mmproj-F16.gguf", 5), ("notes.txt", 1)] {
        std::fs::write(d.join(f), vec![0u8; n]).unwrap();
    }
    let mut out = vec![];
    library::scan_folder(&d.0, &mut out, 0);
    out.sort_by(|a, b| a.name.cmp(&b.name));
    assert_eq!(out.iter().map(|m| m.name.as_str()).collect::<Vec<_>>(), ["huge", "small"]);
    let huge = &out[0];
    assert_eq!(huge.parts.len(), 2);
    assert_eq!(huge.size, 150);
    assert!(huge.parts[0].ends_with("huge-00001-of-00002.gguf"));
    assert!(huge.mmproj.as_ref().unwrap().ends_with("mmproj-F16.gguf"));
    assert!(out[1].mmproj.is_none());
}

#[test]
fn ollama_store_names_and_blobs() {
    let d = TempDir::new("ollama");
    let blob = |digest: &str, bytes: &[u8]| std::fs::write(d.join(&format!("blobs/sha256-{digest}")), bytes).unwrap();
    std::fs::create_dir_all(d.join("blobs")).unwrap();
    let manifest = |path: &str, model: &str, proj: Option<&str>| {
        let mut layers = vec![serde_json::json!({"mediaType": "application/vnd.ollama.image.model", "digest": format!("sha256:{model}"), "size": 1234})];
        if let Some(p) = proj {
            layers.push(serde_json::json!({"mediaType": "application/vnd.ollama.image.projector", "digest": format!("sha256:{p}"), "size": 5}));
        }
        let f = d.join(&format!("manifests/{path}"));
        std::fs::create_dir_all(f.parent().unwrap()).unwrap();
        std::fs::write(f, serde_json::json!({"layers": layers}).to_string()).unwrap();
    };
    blob("aaa", b"x");
    blob("bbb", b"y");
    blob("ccc", b"z");
    manifest("registry.ollama.ai/library/gemma4/e4b", "aaa", Some("ccc"));
    manifest("registry.ollama.ai/someone/tool/latest", "bbb", None);
    manifest("hf.co/user/Repo-GGUF/Q4_K_M", "bbb", None);
    manifest("registry.ollama.ai/library/missing/7b", "fff", None);
    manifest(&format!("registry.ollama.ai/library/gemma4/{}", "a".repeat(64)), "aaa", None);
    let mut out = vec![];
    library::scan_ollama(&d.0, &mut out);
    let mut names: Vec<&str> = out.iter().map(|m| m.name.as_str()).collect();
    names.sort();
    assert_eq!(names, ["gemma4:e4b", "hf.co/user/Repo-GGUF:Q4_K_M", "someone/tool:latest"]);
    let g = out.iter().find(|m| m.name == "gemma4:e4b").unwrap();
    assert_eq!(g.size, 1234);
    assert!(g.mmproj.as_ref().unwrap().ends_with("sha256-ccc"));
    assert_eq!(g.source, Source::Ollama("gemma4:e4b".into()));
}

#[test]
fn reading_a_model_and_its_split_parts() {
    let d = TempDir::new("info");
    let info = tiny_info(&d, false);
    assert_eq!(info.anatomy.layers.len(), LAYERS);
    assert_eq!(info.anatomy.hyper.embd, EMBD);
    assert_eq!(info.quant_label(), "Q8_0");
    assert!(!info.ollama_packaging);
    // The same file listed twice as two parts: tensors from both, offsets per part.
    let path = d.join("dense.gguf");
    let m = library::Model { key: "k".into(), name: "n".into(), parts: vec![path.clone(), path], size: 0, mmproj: None, source: Source::Folder };
    let two = library::read_info(&m).unwrap();
    assert_eq!(two.gguf.tensors.len(), 2 * info.gguf.tensors.len());
    let bad = library::Model { parts: vec![d.join("nope.gguf")], ..m };
    assert!(library::read_info(&bad).is_err());
}

// --- Hugging Face and benchmark parsing ---------------------------------------------------------

#[test]
fn hugging_face_files_are_grouped() {
    let files = hf::group_files(vec![
        ("Q4_K_M/model-Q4_K_M-00002-of-00002.gguf".into(), 20),
        ("Q4_K_M/model-Q4_K_M-00001-of-00002.gguf".into(), 30),
        ("model-Q8_0.gguf".into(), 70),
        ("mmproj-model-f16.gguf".into(), 3),
    ]);
    assert_eq!(files.len(), 3);
    let split = files.iter().find(|f| f.parts.len() == 2).unwrap();
    assert_eq!(split.size, 50);
    assert!(split.path.ends_with("00001-of-00002.gguf"), "the first part opens the model");
    assert!(files.iter().find(|f| f.path.starts_with("mmproj")).unwrap().is_mmproj);
    assert_eq!(hf::file_url("u/r", "a/b.gguf"), "https://huggingface.co/u/r/resolve/main/a/b.gguf");
}

#[test]
fn benchmark_output() {
    let out = r#"[
      {"build_commit": "abc", "n_prompt": 512, "n_gen": 0, "avg_ts": 2500.5},
      {"build_commit": "abc", "n_prompt": 0, "n_gen": 128, "avg_ts": 101.25}
    ]"#;
    assert_eq!(bench::parse_output(&format!("ggml_cuda_init: found 1 CUDA devices\n{out}"), "").unwrap(), (2500.5, 101.25));
    let e = bench::parse_output("", "loading\nerror: failed to load model\n\n").unwrap_err();
    assert!(e.contains("failed to load model"), "{e}");
    assert!(bench::parse_output("[]", "").is_err());
}

// --- llama.cpp releases ---------------------------------------------------------------------------

fn asset(name: &str) -> Asset {
    Asset { name: name.into(), url: format!("https://x/{name}"), size: 1 }
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
#[test]
fn picks_the_build_for_this_machine() {
    let rel = Release {
        tag: "b9".into(),
        assets: [
            "llama-b9-bin-ubuntu-cuda-13.5-arm64.tar.gz",
            "llama-b9-bin-ubuntu-cuda-13.5-x64.tar.gz",
            "cudart-llama-b9-bin-ubuntu-cuda-13.5-arm64.tar.gz",
            "cudart-llama-b9-bin-ubuntu-cuda-13.5-x64.tar.gz",
            "llama-b9-bin-ubuntu-vulkan-x64.tar.gz",
            "llama-b9-bin-ubuntu-x64.tar.gz",
            "llama-b9-bin-win-cuda-13.5-x64.zip",
        ]
        .map(asset)
        .to_vec(),
    };
    assert_eq!(rel.build_for(Variant::Cuda13).unwrap().name, "llama-b9-bin-ubuntu-cuda-13.5-x64.tar.gz");
    assert_eq!(rel.cudart_for(Variant::Cuda13).unwrap().name, "cudart-llama-b9-bin-ubuntu-cuda-13.5-x64.tar.gz");
    assert_eq!(rel.build_for(Variant::Vulkan).unwrap().name, "llama-b9-bin-ubuntu-vulkan-x64.tar.gz");
    assert_eq!(rel.build_for(Variant::Cpu).unwrap().name, "llama-b9-bin-ubuntu-x64.tar.gz");
    assert!(rel.cudart_for(Variant::Vulkan).is_none());
    assert!(rel.build_for(Variant::Cuda12).is_none());
    assert!(!Variant::Metal.available());
}

#[test]
fn recommends_by_gpu_and_driver() {
    let gpu = |vendor, driver| GpuInfo { vendor, name: "g".into(), total: 8 << 30, used: 0, driver_major: driver, processes: vec![] };
    assert_eq!(release::recommended(&gpu(Vendor::Nvidia, Some(590))), Variant::Cuda13);
    assert_eq!(release::recommended(&gpu(Vendor::Nvidia, Some(550))), Variant::Cuda12);
    assert_eq!(release::recommended(&gpu(Vendor::Amd, None)), Variant::Vulkan);
    #[cfg(not(target_os = "macos"))]
    assert_eq!(release::recommended(&gpu(Vendor::None, None)), Variant::Cpu);
}

fn tar_gz(files: &[(&str, &[u8])]) -> Vec<u8> {
    let mut b = tar::Builder::new(flate2::write::GzEncoder::new(vec![], flate2::Compression::fast()));
    for (name, data) in files {
        let mut h = tar::Header::new_gnu();
        h.set_size(data.len() as u64);
        h.set_mode(0o755);
        h.set_cksum();
        b.append_data(&mut h, name, *data).unwrap();
    }
    b.into_inner().unwrap().finish().unwrap()
}

#[test]
fn unpacks_tar_gz_and_zip() {
    let d = TempDir::new("unpack");
    std::fs::write(d.join("a.tar.gz"), tar_gz(&[("llama-b1/llama-server", b"bin"), ("llama-b1/libggml.so", b"lib")])).unwrap();
    release::unpack(&d.join("a.tar.gz"), &d.join("out")).unwrap();
    assert_eq!(std::fs::read(d.join("out/llama-b1/llama-server")).unwrap(), b"bin");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        assert_ne!(std::fs::metadata(d.join("out/llama-b1/llama-server")).unwrap().permissions().mode() & 0o111, 0, "stays executable");
    }
    let mut z = zip::ZipWriter::new(std::io::Cursor::new(vec![]));
    z.start_file("bin/llama-server.exe", zip::write::SimpleFileOptions::default()).unwrap();
    std::io::Write::write_all(&mut z, b"exe").unwrap();
    std::fs::write(d.join("b.zip"), z.finish().unwrap().into_inner()).unwrap();
    release::unpack(&d.join("b.zip"), &d.join("zout")).unwrap();
    assert_eq!(std::fs::read(d.join("zout/bin/llama-server.exe")).unwrap(), b"exe");
    assert!(release::unpack(&d.join("missing.zip"), &d.join("x")).is_err());
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
#[test]
fn installs_a_build_with_its_cuda_runtime_offline() {
    let build = tar_gz(&[("llama-b9/llama-server", b"server"), ("llama-b9/llama-bench", b"bench"), ("llama-b9/libllama.so", b"lib")]);
    let cudart = tar_gz(&[("libcudart.so.13", b"rt"), ("libcublas.so.13", b"blas")]);
    let (b2, c2) = (build.clone(), cudart.clone());
    let port = mock_http(move |req: &str| {
        let body = if req.contains("cudart-") { c2.clone() } else { b2.clone() };
        let mut r = format!("HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", body.len()).into_bytes();
        r.extend(body);
        r
    });
    let url = |n: &str| format!("http://127.0.0.1:{port}/{n}");
    let rel = Release {
        tag: "b9".into(),
        assets: vec![
            Asset { name: "llama-b9-bin-ubuntu-cuda-13.5-x64.tar.gz".into(), url: url("llama.tgz"), size: build.len() as u64 },
            Asset { name: "cudart-llama-b9-bin-ubuntu-cuda-13.5-x64.tar.gz".into(), url: url("cudart-x.tgz"), size: cudart.len() as u64 },
        ],
    };
    let d = TempDir::new("install");
    let task = Task::new("install");
    let bin = release::install(&rel, Variant::Cuda13, true, &d.0, &task).unwrap();
    assert!(bin.ends_with("b9-cuda-13/llama-b9"), "{}", bin.display());
    assert!(release::server_exe(&bin).is_file() && release::bench_exe(&bin).is_file());
    assert!(bin.join("libcudart.so.13").is_file(), "the CUDA runtime sits next to the binaries");
    assert!(server::bin_dir_valid(&bin));
    assert_eq!(release::find_server(&d.0).unwrap(), bin);
    assert_eq!(task.progress.load(std::sync::atomic::Ordering::Relaxed), 1000);
    // Downloads were removed after unpacking.
    assert!(!d.join(&rel.assets[0].name).exists());

    // A size that doesn't match what arrived is refused.
    let short = Release { assets: vec![Asset { size: build.len() as u64 + 10, ..rel.assets[0].clone() }], ..rel.clone() };
    let e = release::install(&short, Variant::Cuda13, false, &d.0, &Task::new("again")).unwrap_err();
    assert!(e.contains("incomplete"), "{e}");
}

#[test]
fn downloads_resume_from_a_part_file() {
    let port = mock_http(|req: &str| {
        if let Some(r) = req.lines().find_map(|l| l.strip_prefix("Range: bytes=").or_else(|| l.strip_prefix("range: bytes="))) {
            let from: usize = r.trim_end_matches('-').parse().unwrap();
            let body = &b"0123456789"[from..];
            let mut out = format!("HTTP/1.1 206 Partial Content\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", body.len()).into_bytes();
            out.extend(body);
            out
        } else {
            b"HTTP/1.1 200 OK\r\nContent-Length: 10\r\nConnection: close\r\n\r\n0123456789".to_vec()
        }
    });
    let d = TempDir::new("resume");
    let file = d.join("f.bin");
    std::fs::write(d.join("f.bin.part"), b"0123").unwrap();
    release::download(&format!("http://127.0.0.1:{port}/f"), &file, |_| {}, &Task::new("dl")).unwrap();
    assert_eq!(std::fs::read(&file).unwrap(), b"0123456789");
    assert!(!d.join("f.bin.part").exists());
}
