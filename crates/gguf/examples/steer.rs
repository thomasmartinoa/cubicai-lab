//! `steer <llama3.gguf> <scale>...`: builds a joyful↔gloomy steering vector with the CPU forward
//! pass and writes a short greedy reply at each scale (applied over the best layer window), to
//! check that the vectors steer. Llama 3 template; greedy longest-match tokenising.

use cubic_gguf::forward::{top_k, Model};
use cubic_gguf::parse::read_string_array;
use cubic_gguf::stats::Progress;
use cubic_gguf::steer::{best_range, directions, norm};
use cubic_gguf::Gguf;
use std::collections::HashMap;
use std::time::Instant;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let f = std::fs::File::open(&a[1]).unwrap();
    let map = unsafe { memmap2::Mmap::map(&f).unwrap() };
    let g = Gguf::parse(&map, Some(map.len() as u64)).unwrap();
    let vocab = read_string_array(&map, "tokenizer.ggml.tokens").unwrap().unwrap();
    let index: HashMap<&str, u32> = vocab.iter().enumerate().map(|(i, s)| (s.as_str(), i as u32)).collect();
    let tok = |text: &str| -> Vec<u32> {
        let chars: Vec<char> = text.replace(' ', "\u{120}").replace('\n', "\u{10a}").chars().collect();
        let mut ids = vec![];
        let mut i = 0;
        while i < chars.len() {
            let (id, j) = (i + 1..=chars.len().min(i + 24))
                .rev()
                .find_map(|j| index.get(chars[i..j].iter().collect::<String>().as_str()).map(|&id| (id, j)))
                .unwrap_or_else(|| panic!("can't tokenise {:?}", &chars[i..]));
            ids.push(id);
            i = j;
        }
        ids
    };
    let detok = |id: usize| vocab[id].replace('\u{120}', " ").replace('\u{10a}', "\n");
    let chat = |user: &str, opener: &str| {
        tok(&format!("<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{opener}"))
    };
    let m = Model::new(&g, vec![&map[..]]).unwrap();
    let questions = ["Tell me about your weekend.", "What do you think of rain?", "Describe your home.", "How was work today?", "What is your favourite food?", "Talk about the future."];
    let openers = ["I", "Well,", "The", "Honestly"];
    let (mut pos, mut neg) = (vec![], vec![]);
    for q in questions {
        for o in openers {
            pos.push(chat(&format!("Act as if you are extremely joyful and cheerful. {q}"), o));
            neg.push(chat(&format!("Act as if you are extremely gloomy and miserable. {q}"), o));
        }
    }
    let t = Instant::now();
    let all: Vec<Vec<u32>> = pos.iter().chain(&neg).cloned().collect();
    let h = m.run(&all, false, &Progress::default()).unwrap();
    let n = pos.len();
    println!("{} prompts, {} tokens, hidden states in {:.1} s", all.len(), all.iter().map(Vec::len).sum::<usize>(), t.elapsed().as_secs_f32());
    let d = directions(&h.last[..n], &h.last[n..]);
    let (lo, hi) = best_range(&d.separation);
    println!("separation: {}", d.separation.iter().map(|s| format!("{s:.1}")).collect::<Vec<_>>().join(" "));
    println!("norms:      {}", d.layers.iter().map(|v| format!("{:.1}", norm(v))).collect::<Vec<_>>().join(" "));
    println!("best range {lo}..={hi}");
    let mut dirs = d.layers.clone();
    for (l, v) in dirs.iter_mut().enumerate() {
        if l < lo || l > hi {
            v.clear();
        }
    }
    for s in &a[2..] {
        let scale: f32 = s.parse().unwrap();
        let mut ids = chat("Tell me about your day in two sentences.", "");
        let mut out = String::new();
        for _ in 0..28 {
            let h = m.run_steered(&[ids.clone()], true, Some((&dirs, scale)), &Progress::default()).unwrap();
            let next = top_k(&h.logits[0], 1)[0].0;
            if vocab[next].starts_with("<|") {
                break;
            }
            out += &detok(next);
            ids.push(next as u32);
        }
        println!("scale {scale:+.2}: {}", out.replace('\n', " "));
    }
}
