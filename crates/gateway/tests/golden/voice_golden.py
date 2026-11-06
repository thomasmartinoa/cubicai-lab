"""Golden outputs from the Python voice module, for the Rust port's tests.
Run with the CubicAI server's venv: voice_golden.py <server dir> <mode> (reads JSON on stdin)."""
import json
import sys

sys.path.insert(0, sys.argv[1])
import cubicai_voice as v  # noqa: E402

mode = sys.argv[2]
args = json.load(sys.stdin)
if mode == "speakable":
    print(json.dumps([v.speakable(t) for t in args]))
elif mode == "segments":
    # Each case: a list of streamed pieces → the parts cut after each piece, then the flush.
    out = []
    for pieces in args:
        s = v.Segmenter()
        out.append({"fed": [s.feed(p) for p in pieces], "flush": s.flush()})
    print(json.dumps(out))
elif mode == "phonemes":
    from kokoro_onnx.tokenizer import Tokenizer
    from kokoro_onnx.config import EspeakConfig
    t = Tokenizer(EspeakConfig(lib_path=v.ESPEAK_LIB, data_path=v.ESPEAK_DATA))
    print(json.dumps([t.phonemize(text, lang) for text, lang in args]))
elif mode == "raw_espeak":
    # What espeak itself gives phonemizer for a chunk (before punctuation handling).
    from phonemizer.backend import EspeakBackend
    from phonemizer.backend.espeak.wrapper import EspeakWrapper
    EspeakWrapper.set_library(v.ESPEAK_LIB)
    EspeakWrapper.set_data_path(v.ESPEAK_DATA)
    out = []
    for text, lang in args:
        b = EspeakBackend(lang, preserve_punctuation=True, with_stress=True)
        out.append(b._espeak.text_to_phonemes(text, None))
    print(json.dumps(out))
elif mode == "punct":
    from phonemizer.punctuation import Punctuation
    p = Punctuation()
    out = []
    for text in args:
        chunks, marks = p.preserve(text)
        out.append({"chunks": chunks, "marks": [[m.index, m.mark, m.position] for m in marks]})
    print(json.dumps(out))
elif mode == "split":
    from kokoro_onnx import Kokoro
    print(json.dumps([Kokoro._split_phonemes(None, p) for p in args]))
elif mode == "vocab":
    from kokoro_onnx.config import DEFAULT_VOCAB
    print(json.dumps(DEFAULT_VOCAB))
elif mode == "bank":
    import numpy as np
    from importlib.resources import files
    import onnx_asr.preprocessors
    d = np.load(files(onnx_asr.preprocessors).joinpath("data").joinpath("fbanks.npz"))
    d["nemo128"].astype("<f4").tofile(args["out"])
elif mode == "features":
    # onnx-asr's own preprocessor (the ONNX one it really uses, and its numpy twin).
    import numpy as np
    from onnx_asr.preprocessors.preprocessor import OnnxPreprocessor
    from onnx_asr.preprocessors.numpy_preprocessor import NemoPreprocessorNumpy
    audio = np.fromfile(args["audio"], dtype="<f4")[None, :]
    lens = np.array([audio.shape[1]], dtype=np.int64)
    f, fl = OnnxPreprocessor("nemo128", {})(audio, lens)
    f.astype("<f4").tofile(args["onnx"])
    g, gl = NemoPreprocessorNumpy("nemo128")(audio.copy(), lens)
    g.astype("<f4").tofile(args["numpy"])
    print(json.dumps({"shape": list(f.shape), "len": int(fl[0]), "numpy_len": int(gl[0])}))
elif mode == "speak":
    # Kokoro as the Python gateway runs it.
    import numpy as np
    v.ENGINES.load()
    out = []
    for i, (text, voice, speed) in enumerate(args["lines"]):
        samples = v.ENGINES.speak(text, voice, speed)
        samples.astype("<f4").tofile(f"{args['dir']}/py{i}.f32")
        out.append(len(samples))
    print(json.dumps(out))
elif mode == "transcribe":
    import numpy as np
    v.ENGINES.load()
    print(json.dumps([v.ENGINES.transcribe(np.fromfile(p, dtype="<f4")) for p in args]))
