//! `next <model.gguf> "<text>"`: the model's next-token odds from the CPU forward pass.
//! Tokenises plain text by greedy longest match on the vocabulary (byte-level BPE spaces = Ġ),
//! which is exact for simple English where every word is one token.

use cubic_gguf::forward::{top_k, Model};
use cubic_gguf::parse::read_string_array;
use cubic_gguf::stats::Progress;
use cubic_gguf::Gguf;
use std::collections::HashMap;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let f = std::fs::File::open(&args[1]).unwrap();
    let map = unsafe { memmap2::Mmap::map(&f).unwrap() };
    let g = Gguf::parse(&map, Some(map.len() as u64)).unwrap();
    let vocab = read_string_array(&map, "tokenizer.ggml.tokens").unwrap().unwrap();
    let index: HashMap<&str, u32> = vocab.iter().enumerate().map(|(i, s)| (s.as_str(), i as u32)).collect();
    let text = args[2].replace(' ', "\u{120}");
    let mut ids = vec![];
    if let Some(bos) = g.get_u64("tokenizer.ggml.bos_token_id")
        && g.get("tokenizer.ggml.add_bos_token").and_then(|v| v.as_bool()).unwrap_or(true)
    {
        ids.push(bos as u32);
    }
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        let mut found = None;
        for j in (i + 1..=chars.len()).rev() {
            let s: String = chars[i..j].iter().collect();
            if let Some(&id) = index.get(s.as_str()) {
                found = Some((id, j));
                break;
            }
        }
        let (id, j) = found.expect("untokenisable text");
        ids.push(id);
        i = j;
    }
    println!("tokens: {:?}", ids.iter().map(|&i| vocab[i as usize].as_str()).collect::<Vec<_>>());
    let m = Model::new(&g, vec![&map[..]]).unwrap();
    println!("{:?}", m.cfg);
    let t = Instant::now();
    let h = m.run(&[ids.clone()], true, &Progress::default()).unwrap();
    println!("{} tokens in {:.2} s", ids.len(), t.elapsed().as_secs_f32());
    for (id, p) in top_k(&h.logits[0], 8) {
        println!("{:>8.3}  {:6.2}%  {:?}", p.ln(), p * 100.0, vocab[id].replace('\u{120}', " "));
    }
    let norms: Vec<String> = h.last[0].iter().map(|v| format!("{:.0}", v.iter().map(|x| x * x).sum::<f32>().sqrt())).collect();
    println!("residual norm by layer: {}", norms.join(" "));
}
