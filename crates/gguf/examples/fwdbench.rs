//! `fwdbench <model.gguf> <prompts> <tokens>`: speed of the CPU forward pass on a batch.
use cubic_gguf::forward::Model;
use cubic_gguf::stats::Progress;
use cubic_gguf::Gguf;
use std::time::Instant;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let f = std::fs::File::open(&a[1]).unwrap();
    let map = unsafe { memmap2::Mmap::map(&f).unwrap() };
    let g = Gguf::parse(&map, Some(map.len() as u64)).unwrap();
    let (n, len): (usize, usize) = (a[2].parse().unwrap(), a[3].parse().unwrap());
    let prompts: Vec<Vec<u32>> = (0..n).map(|s| (0..len).map(|i| (1000 + s * 37 + i * 101) as u32 % 30000 + 100).collect()).collect();
    let m = Model::new(&g, vec![&map[..]]).unwrap();
    let t = Instant::now();
    let h = m.run(&prompts, false, &Progress::default()).unwrap();
    let s = t.elapsed().as_secs_f64();
    let flops = 2.0 * (g.total_tensor_bytes() as f64 / 0.56) * (n * len) as f64; // ~params × tokens, rough
    println!("{} prompts × {len} tokens = {} tokens in {s:.1} s ({:.0} tok/s, ~{:.0} GFLOP/s), layers {}", n, n * len, (n * len) as f64 / s, flops / s / 1e9, h.last[0].len());
}
