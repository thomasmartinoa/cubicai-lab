//! Numbers and pictures of a tensor's weights: summary statistics from a sample of rows, a pooled
//! magnitude map of the whole matrix, and full-resolution windows into it.

use crate::dequant::{can_dequantize, dequantize};
use crate::parse::TensorInfo;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

pub const HIST_BINS: usize = 96;

#[derive(Clone, Debug, Default)]
pub struct Stats {
    pub count: u64,
    pub mean: f64,
    pub std: f64,
    pub min: f32,
    pub max: f32,
    pub absmax: f32,
    /// Share of weights that are exactly zero.
    pub zeros: f64,
    /// Histogram over [-range, range].
    pub hist: Vec<u32>,
    pub range: f32,
    /// Rows looked at, out of all.
    pub rows_seen: u64,
    pub rows_total: u64,
}

struct Acc {
    n: u64,
    sum: f64,
    sq: f64,
    min: f32,
    max: f32,
    zeros: u64,
}

impl Acc {
    fn new() -> Self {
        Acc { n: 0, sum: 0.0, sq: 0.0, min: f32::INFINITY, max: f32::NEG_INFINITY, zeros: 0 }
    }

    fn add(&mut self, row: &[f32]) {
        let (mut s, mut q) = (0f64, 0f64);
        for &v in row {
            if !v.is_finite() {
                continue;
            }
            s += v as f64;
            q += (v as f64) * (v as f64);
            self.min = self.min.min(v);
            self.max = self.max.max(v);
            if v == 0.0 {
                self.zeros += 1;
            }
            self.n += 1;
        }
        self.sum += s;
        self.sq += q;
    }

    fn finish(&self, hist: Vec<u32>, range: f32, rows_seen: u64, rows_total: u64) -> Stats {
        let n = self.n.max(1) as f64;
        let mean = self.sum / n;
        let var = (self.sq / n - mean * mean).max(0.0);
        Stats {
            count: self.n,
            mean,
            std: var.sqrt(),
            min: if self.n > 0 { self.min } else { 0.0 },
            max: if self.n > 0 { self.max } else { 0.0 },
            absmax: if self.n > 0 { self.min.abs().max(self.max.abs()) } else { 0.0 },
            zeros: self.zeros as f64 / n,
            hist,
            range,
            rows_seen,
            rows_total,
        }
    }
}

/// Reads tensor rows. `data` is exactly the tensor's bytes.
pub struct RowReader<'a> {
    pub t: &'a TensorInfo,
    data: &'a [u8],
    row_bytes: usize,
    pub row_len: usize,
    pub rows: u64,
}

impl<'a> RowReader<'a> {
    pub fn new(t: &'a TensorInfo, data: &'a [u8]) -> Option<Self> {
        if !can_dequantize(t.ty) {
            return None;
        }
        let row_len = t.row_len() as usize;
        let row_bytes = t.ty.bytes_for(row_len as u64)? as usize;
        let rows = t.n_rows();
        if (data.len() as u64) < row_bytes as u64 * rows {
            return None;
        }
        Some(RowReader { t, data, row_bytes, row_len, rows })
    }

    pub fn read(&self, row: u64, out: &mut [f32]) -> bool {
        let start = row as usize * self.row_bytes;
        dequantize(self.t.ty, &self.data[start..start + self.row_bytes], &mut out[..self.row_len])
    }
}

/// Statistics from up to `max_rows` rows spread evenly through the tensor (all rows if fewer).
pub fn sample_stats(t: &TensorInfo, data: &[u8], max_rows: u64) -> Option<Stats> {
    let r = RowReader::new(t, data)?;
    let step = (r.rows / max_rows.max(1)).max(1);
    let mut buf = vec![0f32; r.row_len];
    let mut acc = Acc::new();
    let mut picked = Vec::new();
    let mut row = 0;
    while row < r.rows {
        if r.read(row, &mut buf) {
            acc.add(&buf);
            picked.push(row);
        }
        row += step;
    }
    // Histogram over ±(a robust range): absmax would let one outlier squash everything into one bin.
    let range = robust_range(&acc, &r, &picked, &mut buf);
    let mut hist = vec![0u32; HIST_BINS];
    for &row in &picked {
        r.read(row, &mut buf);
        bin(&buf, range, &mut hist);
    }
    Some(acc.finish(hist, range, picked.len() as u64, r.rows))
}

fn robust_range(acc: &Acc, _r: &RowReader, _rows: &[u64], _buf: &mut [f32]) -> f32 {
    let n = acc.n.max(1) as f64;
    let mean = acc.sum / n;
    let std = (acc.sq / n - mean * mean).max(0.0).sqrt();
    let absmax = acc.min.abs().max(acc.max.abs());
    let r = ((mean.abs() + 5.0 * std) as f32).min(absmax);
    if r > 0.0 && r.is_finite() { r } else { absmax.max(1e-6) }
}

fn bin(row: &[f32], range: f32, hist: &mut [u32]) {
    let k = hist.len() as f32 / (2.0 * range);
    for &v in row {
        if v.is_finite() {
            let i = ((v + range) * k).floor();
            let i = (i.max(0.0) as usize).min(hist.len() - 1);
            hist[i] += 1;
        }
    }
}

/// The whole matrix (one slice of a 3-D expert tensor) pooled down to at most `max_w` × `max_h`
/// cells of root-mean-square magnitude, with per-column and per-row RMS profiles.
#[derive(Clone, Debug)]
pub struct Overview {
    pub w: usize,
    pub h: usize,
    /// Row-major, h rows of w cells.
    pub rms: Vec<f32>,
    /// RMS of each input column (dims[0]), pooled to w.
    pub col_rms: Vec<f32>,
    /// RMS of each output row, pooled to h.
    pub row_rms: Vec<f32>,
    pub stats: Stats,
    /// Matrix size of the slice: rows (outputs) × cols (inputs).
    pub rows: u64,
    pub cols: u64,
}

pub struct Progress {
    /// Parts per thousand done.
    pub done: AtomicU32,
    pub cancel: AtomicBool,
}

impl Default for Progress {
    fn default() -> Self {
        Progress { done: AtomicU32::new(0), cancel: AtomicBool::new(false) }
    }
}

/// Rows of one slice: for a [cols, rows, slices] tensor, slice `s` is rows s*rows..(s+1)*rows.
pub fn slice_rows(t: &TensorInfo) -> (u64, u64) {
    let rows = t.dims.get(1).copied().unwrap_or(1);
    let slices = t.dims.iter().skip(2).product::<u64>().max(1);
    (rows, slices)
}

pub fn overview(t: &TensorInfo, data: &[u8], slice: u64, max_w: usize, max_h: usize, p: &Progress) -> Option<Overview> {
    let r = RowReader::new(t, data)?;
    let (rows, slices) = slice_rows(t);
    let slice = slice.min(slices - 1);
    let cols = r.row_len;
    let w = cols.min(max_w).max(1);
    let h = (rows as usize).min(max_h).max(1);
    let mut sq = vec![0f64; w * h];
    let mut cnt = vec![0u32; w * h];
    let mut col_sq = vec![0f64; w];
    let mut row_sq = vec![0f64; h];
    let mut buf = vec![0f32; cols];
    let mut acc = Acc::new();
    let first = slice * rows;
    // Histogram range from a quick sample first, so the full pass can bin as it goes.
    let quick = sample_stats(t, data, 64)?;
    let range = quick.range;
    let mut hist = vec![0u32; HIST_BINS];
    // Precompute which pooled column each input column falls in.
    let col_bin: Vec<usize> = (0..cols).map(|c| c * w / cols).collect();
    for y in 0..rows {
        if y % 64 == 0 {
            if p.cancel.load(Ordering::Relaxed) {
                return None;
            }
            p.done.store((y * 1000 / rows) as u32, Ordering::Relaxed);
        }
        if !r.read(first + y, &mut buf) {
            return None;
        }
        acc.add(&buf);
        bin(&buf, range, &mut hist);
        let py = (y as usize) * h / rows as usize;
        let base = py * w;
        let mut rsum = 0f64;
        for (c, &v) in buf.iter().enumerate() {
            if !v.is_finite() {
                continue;
            }
            let v2 = (v as f64) * (v as f64);
            let px = col_bin[c];
            sq[base + px] += v2;
            cnt[base + px] += 1;
            col_sq[px] += v2;
            rsum += v2;
        }
        row_sq[py] += rsum;
    }
    p.done.store(1000, Ordering::Relaxed);
    let rms = sq.iter().zip(&cnt).map(|(s, &n)| if n > 0 { (s / n as f64).sqrt() as f32 } else { 0.0 }).collect();
    let per_col = rows as f64 * (cols as f64 / w as f64);
    let per_row = cols as f64 * (rows as f64 / h as f64);
    let col_rms = col_sq.iter().map(|s| (s / per_col).sqrt() as f32).collect();
    let row_rms = row_sq.iter().map(|s| (s / per_row).sqrt() as f32).collect();
    Some(Overview {
        w,
        h,
        rms,
        col_rms,
        row_rms,
        stats: acc.finish(hist, range, rows, rows),
        rows,
        cols: cols as u64,
    })
}

/// Signed weights of a `rows` × `cols` window starting at (row0, col0) of one slice.
pub fn window(t: &TensorInfo, data: &[u8], slice: u64, row0: u64, col0: usize, rows: usize, cols: usize) -> Option<Vec<f32>> {
    let r = RowReader::new(t, data)?;
    let (n_rows, slices) = slice_rows(t);
    let slice = slice.min(slices - 1);
    let mut buf = vec![0f32; r.row_len];
    let mut out = Vec::with_capacity(rows * cols);
    for y in 0..rows as u64 {
        let row = row0 + y;
        if row >= n_rows {
            out.extend(std::iter::repeat_n(0.0, cols));
            continue;
        }
        r.read(slice * n_rows + row, &mut buf);
        for x in 0..cols {
            out.push(buf.get(col0 + x).copied().unwrap_or(0.0));
        }
    }
    Some(out)
}

/// A 1-D tensor's values (norm scales, biases), pooled to at most `max` points as (min, max) pairs.
pub fn vector(t: &TensorInfo, data: &[u8], max: usize) -> Option<Vec<(f32, f32)>> {
    let r = RowReader::new(t, data)?;
    let mut buf = vec![0f32; r.row_len];
    r.read(0, &mut buf);
    let n = buf.len();
    let bins = n.min(max).max(1);
    let mut out = vec![(f32::INFINITY, f32::NEG_INFINITY); bins];
    for (i, &v) in buf.iter().enumerate() {
        let b = i * bins / n;
        out[b].0 = out[b].0.min(v);
        out[b].1 = out[b].1.max(v);
    }
    Some(out)
}
