//! LoRA adapters: a PEFT adapter (`adapter_model.safetensors` + `adapter_config.json`, as Unsloth
//! saves it) converted to the GGUF adapter llama.cpp loads with `--lora`. Our own converter, so no
//! llama.cpp Python scripts or base-model download are needed.
//!
//! llama.cpp's rules (llama-adapter.cpp): `general.type = adapter`, `adapter.type = lora`,
//! `adapter.lora.alpha`, the model's architecture, and per weight `<gguf name>.lora_a` with
//! ne = [in, r] and `.lora_b` with ne = [r, out]. It scales by alpha / r.

use cubic_gguf::{Gguf, Writer};
use serde_json::Value;

#[derive(Debug)]
pub struct Tensor {
    pub name: String,
    /// PyTorch order (outermost first).
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

/// Every tensor of a safetensors file, as f32.
pub fn read_safetensors(bytes: &[u8]) -> Result<Vec<Tensor>, String> {
    let bad = |m: &str| format!("Not a safetensors file: {m}");
    let n = u64::from_le_bytes(bytes.get(..8).ok_or_else(|| bad("too short"))?.try_into().unwrap()) as usize;
    let header = bytes.get(8..8usize.checked_add(n).ok_or_else(|| bad("header size"))?).ok_or_else(|| bad("header cut short"))?;
    let h: Value = serde_json::from_slice(header).map_err(|e| bad(&e.to_string()))?;
    let data = &bytes[8 + n..];
    let mut out = vec![];
    for (name, t) in h.as_object().ok_or_else(|| bad("header isn't an object"))? {
        if name == "__metadata__" {
            continue;
        }
        let shape: Vec<usize> = t["shape"].as_array().ok_or_else(|| bad("no shape"))?.iter().filter_map(|d| d.as_u64().map(|d| d as usize)).collect();
        let offs: Vec<usize> = t["data_offsets"].as_array().ok_or_else(|| bad("no offsets"))?.iter().filter_map(|d| d.as_u64().map(|d| d as usize)).collect();
        let raw = data.get(offs[0]..offs[1]).ok_or_else(|| bad(&format!("{name} is cut short")))?;
        let count: usize = shape.iter().product();
        let vals: Vec<f32> = match t["dtype"].as_str().unwrap_or("") {
            "F32" => raw.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect(),
            "F16" => raw.as_chunks::<2>().0.iter().map(|c| f16_to_f32(u16::from_le_bytes(*c))).collect(),
            "BF16" => raw.as_chunks::<2>().0.iter().map(|c| f32::from_bits((u16::from_le_bytes(*c) as u32) << 16)).collect(),
            other => return Err(format!("{name} is stored as {other}, which isn't supported")),
        };
        if vals.len() != count {
            return Err(bad(&format!("{name} has {} values for shape {shape:?}", vals.len())));
        }
        out.push(Tensor { name: name.clone(), shape, data: vals });
    }
    out.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(out)
}

fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) as u32) << 31;
    let exp = ((h >> 10) & 0x1f) as u32;
    let frac = (h & 0x3ff) as u32;
    let bits = match exp {
        0 if frac == 0 => sign,
        0 => {
            // Subnormal: normalise.
            let mut e = 127 - 15 + 1;
            let mut f = frac;
            while f & 0x400 == 0 {
                f <<= 1;
                e -= 1;
            }
            sign | (e << 23) | ((f & 0x3ff) << 13)
        }
        0x1f => sign | 0x7f80_0000 | (frac << 13),
        _ => sign | ((exp + 127 - 15) << 23) | (frac << 13),
    };
    f32::from_bits(bits)
}

/// Writes a safetensors file (tests only).
#[cfg(test)]
pub fn write_safetensors(tensors: &[Tensor]) -> Vec<u8> {
    let mut header = serde_json::Map::new();
    let mut data = vec![];
    for t in tensors {
        let start = data.len();
        for v in &t.data {
            data.extend(v.to_le_bytes());
        }
        header.insert(t.name.clone(), serde_json::json!({"dtype": "F32", "shape": t.shape, "data_offsets": [start, data.len()]}));
    }
    let h = Value::Object(header).to_string().into_bytes();
    let mut out = (h.len() as u64).to_le_bytes().to_vec();
    out.extend(h);
    out.extend(data);
    out
}

#[derive(Clone, Debug, PartialEq)]
pub struct AdapterConfig {
    pub rank: u32,
    pub alpha: f32,
    pub rslora: bool,
    pub base_model: String,
}

pub fn read_config(json: &str) -> Result<AdapterConfig, String> {
    let v: Value = serde_json::from_str(json).map_err(|e| format!("adapter_config.json: {e}"))?;
    let rank = v["r"].as_u64().ok_or("adapter_config.json has no rank (r)")? as u32;
    Ok(AdapterConfig {
        rank,
        alpha: v["lora_alpha"].as_f64().map(|a| a as f32).unwrap_or(rank as f32),
        rslora: v["use_rslora"].as_bool().unwrap_or(false),
        base_model: v["base_model_name_or_path"].as_str().unwrap_or("").to_string(),
    })
}

/// `…layers.7.self_attn.q_proj.lora_A(.default).weight` → ("blk.7.attn_q.weight", is A).
pub fn gguf_name(peft: &str) -> Result<(String, bool), String> {
    let rest = peft.split_once("layers.").map(|x| x.1).ok_or_else(|| format!("{peft}: only per-layer weights can be converted"))?;
    let mut parts = rest.split('.');
    let layer: usize = parts.next().and_then(|l| l.parse().ok()).ok_or_else(|| format!("{peft}: no layer number"))?;
    let _block = parts.next();
    let proj = parts.next().unwrap_or("");
    let which = parts.next().unwrap_or("");
    let target = match proj {
        "q_proj" => "attn_q",
        "k_proj" => "attn_k",
        "v_proj" => "attn_v",
        "o_proj" => "attn_output",
        "gate_proj" => "ffn_gate",
        "up_proj" => "ffn_up",
        "down_proj" => "ffn_down",
        other => return Err(format!("{peft}: {other} isn't a weight llama.cpp can adapt here")),
    };
    let a = match which {
        "lora_A" => true,
        "lora_B" => false,
        other => return Err(format!("{peft}: unexpected part {other}")),
    };
    Ok((format!("blk.{layer}.{target}.weight"), a))
}

/// Llama's Hugging Face weights keep each head's rotary halves apart; GGUF interleaves them
/// (convert_hf_to_gguf's `permute`). For a LoRA that is a reordering of B's rows.
pub fn permute_rows(rows: &[f32], n_rows: usize, width: usize, n_head: usize) -> Vec<f32> {
    let hd = n_rows / n_head;
    let half = hd / 2;
    let mut out = vec![0f32; rows.len()];
    for h in 0..n_head {
        for j in 0..half {
            for i in 0..2 {
                let dst = h * hd + j * 2 + i;
                let src = h * hd + i * half + j;
                out[dst * width..(dst + 1) * width].copy_from_slice(&rows[src * width..(src + 1) * width]);
            }
        }
    }
    out
}

pub struct Converted {
    pub gguf: Vec<u8>,
    pub pairs: usize,
    pub params: u64,
}

/// Converts a PEFT adapter for the given base model (its GGUF header supplies the architecture,
/// head counts and the shapes every pair must match).
pub fn convert(tensors: &[Tensor], cfg: &AdapterConfig, base: &Gguf) -> Result<Converted, String> {
    let arch = base.arch().to_string();
    let n_head = base.arch_u64("attention.head_count").unwrap_or(1) as usize;
    let n_head_kv = base.arch_u64("attention.head_count_kv").unwrap_or(n_head as u64) as usize;
    // Llama (and Mistral, which GGUF calls llama) needs the q/k reordering; Qwen doesn't.
    let permute = arch == "llama";
    // llama.cpp scales by alpha / r; rank-stabilised LoRA means alpha / sqrt(r).
    let alpha = if cfg.rslora { cfg.alpha * (cfg.rank as f32).sqrt() } else { cfg.alpha };

    let mut pairs: std::collections::BTreeMap<String, (Option<&Tensor>, Option<&Tensor>)> = Default::default();
    for t in tensors {
        let (name, is_a) = gguf_name(&t.name)?;
        let e = pairs.entry(name).or_default();
        if is_a { e.0 = Some(t) } else { e.1 = Some(t) }
    }
    if pairs.is_empty() {
        return Err("The adapter has no weights.".into());
    }
    let mut w = Writer::new();
    w.str("general.architecture", &arch).str("general.type", "adapter").str("adapter.type", "lora").f32("adapter.lora.alpha", alpha);
    let mut params = 0u64;
    for (name, (a, b)) in &pairs {
        let (Some(a), Some(b)) = (a, b) else { return Err(format!("{name}: half of the LoRA pair is missing")) };
        let base_t = base.tensors.iter().find(|t| &t.name == name).ok_or_else(|| format!("{name} isn't in the base model: is this adapter for a different model?"))?;
        let (r, inp) = (a.shape[0], a.shape[1]);
        let (out, rb) = (b.shape[0], b.shape[1]);
        if r != rb {
            return Err(format!("{name}: A has rank {r} but B has {rb}"));
        }
        // GGUF dims are innermost first: the weight is [in, out].
        if base_t.dims.first().copied() != Some(inp as u64) || base_t.dims.get(1).copied() != Some(out as u64) {
            return Err(format!(
                "{name}: the adapter is {inp}→{out} but the model's weight is {:?}. The adapter was trained on a different model.",
                base_t.dims
            ));
        }
        let b_data = match name.rsplit('.').nth(1) {
            Some("attn_q") if permute => permute_rows(&b.data, out, r, n_head),
            Some("attn_k") if permute => permute_rows(&b.data, out, r, n_head_kv),
            _ => b.data.clone(),
        };
        w.tensor_f32(&format!("{name}.lora_a"), &[inp as u64, r as u64], |i| a.data[i as usize]);
        w.tensor_f32(&format!("{name}.lora_b"), &[r as u64, out as u64], |i| b_data[i as usize]);
        params += (a.data.len() + b.data.len()) as u64;
    }
    Ok(Converted { gguf: w.finish(), pairs: pairs.len(), params })
}
