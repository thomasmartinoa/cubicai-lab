# Regenerate: python -m venv v && v/bin/pip install gguf numpy && v/bin/python make.py .
# Writes <TYPE>.bin (random blocks) and <TYPE>.f32 (gguf-py's dequantisation) as regression fixtures.
import sys, gguf, numpy as np
out = sys.argv[1]
Q = gguf.GGMLQuantizationType
f16pos = {'Q4_0':[0],'Q4_1':[0,2],'Q5_0':[0],'Q5_1':[0,2],'Q8_0':[0],'Q2_K':[80,82],'Q3_K':[108],'Q4_K':[0,2],
          'Q5_K':[0,2],'Q6_K':[208],'IQ4_NL':[0],'IQ4_XS':[0],'TQ2_0':[64],'MXFP4':[],'BF16':None,'F16':None}
rng = np.random.default_rng(7)
for name, pos in f16pos.items():
    t = Q[name]; bs, ts = gguf.GGML_QUANT_SIZES[t]; nb = 8
    if name == 'F16':
        b = rng.normal(0, 0.05, size=256).astype(np.float16).view(np.uint8)
    elif name == 'BF16':
        b = (rng.normal(0, 0.05, size=256).astype(np.float32).view(np.uint32) >> 16).astype(np.uint16).view(np.uint8)
    else:
        b = rng.integers(0, 256, size=(nb, ts), dtype=np.uint8)
        for p in pos: b[:, p:p+2] = rng.uniform(-0.05, 0.05, size=nb).astype(np.float16).view(np.uint8).reshape(nb, 2)
        if name == 'MXFP4': b[:, 0] = rng.integers(118, 130, size=nb)  # sane power-of-two scales
    b = np.ascontiguousarray(b).reshape(-1)
    ref = gguf.quants.dequantize(b.reshape(1, -1), t).reshape(-1).astype(np.float32)
    b.tofile(f'{out}/{name}.bin'); ref.tofile(f'{out}/{name}.f32')
    print(name, b.size, 'bytes ->', ref.size, 'values')
