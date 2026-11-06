//! `cargo run --release --example dump -- <file.gguf> [tensor-name rows out.f32]`
//! Prints metadata and the tensor directory; with a tensor name, writes its first rows as raw f32
//! (for checking the decoders against gguf-py).
use cubic_gguf::{stats, Anatomy, Gguf};
use std::io::Write;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let f = std::fs::File::open(&args[1]).unwrap();
    let map = unsafe { memmap2::Mmap::map(&f).unwrap() };
    let g = Gguf::parse(&map, Some(map.len() as u64)).unwrap();
    if args.len() >= 5 {
        let t = g.tensors.iter().find(|t| t.name == args[2]).expect("no such tensor");
        let start = g.tensor_start(t) as usize;
        let data = &map[start..start + t.n_bytes as usize];
        let rows: usize = args[3].parse().unwrap();
        let r = stats::RowReader::new(t, data).expect("can't decode");
        let mut buf = vec![0f32; r.row_len];
        let mut out = std::fs::File::create(&args[4]).unwrap();
        for i in 0..rows.min(r.rows as usize) {
            assert!(r.read(i as u64, &mut buf));
            for v in &buf {
                out.write_all(&v.to_le_bytes()).unwrap();
            }
        }
        println!("{} {} {:?} wrote {} rows of {}", t.name, t.ty, t.dims, rows, r.row_len);
        return;
    }
    println!("GGUF v{} data@{} align {}", g.version, g.data_offset, g.alignment);
    for (k, v) in &g.metadata {
        if !k.starts_with("tokenizer.ggml.") || matches!(v, cubic_gguf::Value::Str(_)) {
            println!("  {k} = {}", v.display());
        }
    }
    let a = Anatomy::build(&g);
    println!("params {} active {} bytes {} layers {}", a.params, a.active_params(&g), a.bytes, a.layers.len());
    println!("mix {:?}", a.type_mix.iter().map(|(t, b, c)| format!("{t}:{}MB/{c}", b >> 20)).collect::<Vec<_>>());
    println!("hyper {:?}", a.hyper);
    let show: Vec<usize> = a.before.iter().chain(a.layers.first().map(|l| &l.tensors).into_iter().flatten()).chain(a.after.iter()).chain(a.extra.iter().take(10)).copied().collect();
    for i in show {
        let t = &g.tensors[i];
        println!("  {:40} {:8} [{}] {}", t.name, t.ty.name(), t.shape_text(), t.n_bytes);
    }
    if a.layers.len() > 1 {
        let l = &a.layers[a.layers.len() / 2];
        println!(" layer {} attn={} exps={}:", l.index, l.has_attention, l.has_experts);
        for &i in &l.tensors {
            let t = &g.tensors[i];
            println!("  {:40} {:8} [{}]", t.name, t.ty.name(), t.shape_text());
        }
    }
    let t0 = std::time::Instant::now();
    let mut n = 0;
    for t in &g.tensors {
        let start = g.tensor_start(t) as usize;
        if let Some(s) = stats::sample_stats(t, &map[start..start + t.n_bytes as usize], 16) {
            n += 1;
            if n <= 3 { println!("  stats {}: mean {:.5} std {:.5} absmax {:.4}", t.name, s.mean, s.std, s.absmax); }
        }
    }
    println!("sampled {n}/{} tensors in {:?}", g.tensors.len(), t0.elapsed());
}
