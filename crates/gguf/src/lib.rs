//! Reading GGUF model files: header, tensor directory, weight decoding and statistics.
//! Pure Rust and dependency-light so it can be tested against real models on its own.

pub mod dequant;
pub mod layout;
pub mod parse;
pub mod stats;
pub mod types;
pub mod write;

pub use layout::{Anatomy, Part};
pub use parse::{Gguf, ParseError, TensorInfo, Value};
pub use types::GgmlType;
pub use write::Writer;
