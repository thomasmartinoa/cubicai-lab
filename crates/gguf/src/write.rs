//! Writing GGUF files: metadata plus tensors whose bytes the caller supplies. Used for test models
//! now, and later for control vectors and adapters made in the app.

use crate::types::GgmlType;

#[derive(Default)]
pub struct Writer {
    kv: Vec<u8>,
    n_kv: u64,
    tensors: Vec<(String, Vec<u64>, GgmlType, Vec<u8>)>,
    alignment: u64,
}

fn string(out: &mut Vec<u8>, v: &str) {
    out.extend((v.len() as u64).to_le_bytes());
    out.extend(v.as_bytes());
}

impl Writer {
    pub fn new() -> Self {
        Writer { alignment: 32, ..Default::default() }
    }

    fn key(&mut self, k: &str, ty: u32) -> &mut Vec<u8> {
        string(&mut self.kv, k);
        self.kv.extend(ty.to_le_bytes());
        self.n_kv += 1;
        &mut self.kv
    }

    pub fn u32(&mut self, k: &str, v: u32) -> &mut Self {
        self.key(k, 4).extend(v.to_le_bytes());
        self
    }

    pub fn u64(&mut self, k: &str, v: u64) -> &mut Self {
        self.key(k, 10).extend(v.to_le_bytes());
        self
    }

    pub fn f32(&mut self, k: &str, v: f32) -> &mut Self {
        self.key(k, 6).extend(v.to_le_bytes());
        self
    }

    pub fn bool(&mut self, k: &str, v: bool) -> &mut Self {
        self.key(k, 7).push(v as u8);
        self
    }

    pub fn str(&mut self, k: &str, v: &str) -> &mut Self {
        let kv = self.key(k, 8);
        string(kv, v);
        self
    }

    pub fn strs(&mut self, k: &str, v: &[&str]) -> &mut Self {
        let kv = self.key(k, 9);
        kv.extend(8u32.to_le_bytes());
        kv.extend((v.len() as u64).to_le_bytes());
        for x in v {
            string(kv, x);
        }
        self
    }

    pub fn bools(&mut self, k: &str, v: &[bool]) -> &mut Self {
        let kv = self.key(k, 9);
        kv.extend(7u32.to_le_bytes());
        kv.extend((v.len() as u64).to_le_bytes());
        kv.extend(v.iter().map(|&b| b as u8));
        self
    }

    pub fn u32s(&mut self, k: &str, v: &[u32]) -> &mut Self {
        let kv = self.key(k, 9);
        kv.extend(4u32.to_le_bytes());
        kv.extend((v.len() as u64).to_le_bytes());
        for x in v {
            kv.extend(x.to_le_bytes());
        }
        self
    }

    /// An F32 tensor; `f(i)` gives element i in storage order (row by row).
    pub fn tensor_f32(&mut self, name: &str, dims: &[u64], f: impl Fn(u64) -> f32) -> &mut Self {
        let n: u64 = dims.iter().product();
        let data = (0..n).flat_map(|i| f(i).to_le_bytes()).collect();
        self.tensors.push((name.into(), dims.to_vec(), GgmlType::F32, data));
        self
    }

    /// A tensor of any type from its encoded bytes (`dims` in ggml order: row length first).
    pub fn tensor_raw(&mut self, name: &str, dims: &[u64], ty: GgmlType, data: Vec<u8>) -> &mut Self {
        self.tensors.push((name.into(), dims.to_vec(), ty, data));
        self
    }

    pub fn finish(&self) -> Vec<u8> {
        let align = self.alignment.max(1);
        let mut out = b"GGUF".to_vec();
        out.extend(3u32.to_le_bytes());
        out.extend((self.tensors.len() as u64).to_le_bytes());
        out.extend(self.n_kv.to_le_bytes());
        out.extend(&self.kv);
        let mut offset = 0u64;
        let mut offsets = vec![];
        for (name, dims, ty, data) in &self.tensors {
            string(&mut out, name);
            out.extend((dims.len() as u32).to_le_bytes());
            for d in dims {
                out.extend(d.to_le_bytes());
            }
            out.extend(ty.id().to_le_bytes());
            out.extend(offset.to_le_bytes());
            offsets.push(offset);
            offset = (offset + data.len() as u64).div_ceil(align) * align;
        }
        out.resize((out.len() as u64).div_ceil(align) as usize * align as usize, 0);
        let base = out.len();
        for ((_, _, _, data), off) in self.tensors.iter().zip(offsets) {
            out.resize(base + off as usize, 0);
            out.extend(data);
        }
        out
    }
}
