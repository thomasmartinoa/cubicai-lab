//! `rawdq <type-id> <in.bin> <out.f32>`: decodes raw blocks (decoder cross-check).
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let ty = cubic_gguf::GgmlType::from_id(a[1].parse().unwrap());
    let data = std::fs::read(&a[2]).unwrap();
    let (bs, ts) = ty.layout().unwrap();
    let n = data.len() as u64 / ts * bs;
    let mut out = vec![0f32; n as usize];
    assert!(cubic_gguf::dequant::dequantize(ty, &data, &mut out), "cannot decode");
    std::fs::write(&a[3], out.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
}
