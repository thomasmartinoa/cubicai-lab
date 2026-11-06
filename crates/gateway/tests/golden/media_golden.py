"""Prints what the Python gateway's media code makes of this PC's real workflows, as JSON, so the
Rust port can be compared with it. Run with the CubicAI server folder on sys.path:
    python3 media_golden.py <cubicai/server> <what>
<what>: to_api (every saved workflow), build (every generator with fixed params), generators, table."""
import json, sys
sys.path.insert(0, sys.argv[1])
import cubicai_media as m
from comfy_workflow import to_api

info = json.loads(m.INFO_CACHE.read_text())
m.COMFY._info = info  # no ComfyUI needed: the node list it last reported
what = sys.argv[2]
out = {}
if what == "to_api":
    for p in sorted(m.WORKFLOWS.glob("*.json")):
        try:
            prompt, named = to_api(json.loads(p.read_text()), info)
            out[p.name] = {"prompt": prompt, "named": named}
        except Exception as e:
            out[p.name] = {"error": f"{e.__class__.__name__}: {e}"}
elif what == "build":
    params = {"prompt": "a red fox in snow", "negative": "blurry", "lyrics": "[verse]\nla la", "seed": 123456789012,
              "steps": 25.7, "cfg": 4, "aspect": "9:16", "seconds": 3, "language": "fr"}
    for g in m.GENERATORS:
        for with_files in (False, True):
            key = f"{g['id']}{'+files' if with_files else ''}"
            try:
                out[key] = m.build(g, dict(params), "JOBID", "cubicai/JOBID.png" if with_files else None,
                                   "cubicai/JOBID.wav" if with_files else None)
            except Exception as e:
                out[key] = {"error": f"{e.__class__.__name__}: {e}"}
elif what == "generators":
    out = m.generators()
elif what == "table":
    out = {"generators": m.GENERATORS, "aspects": m.ASPECTS,
           "sizes": {f"{a}/{side}/{mult}": m._size(a, side, mult) for a in m.ASPECTS for side in (640, 704, 1024) for mult in (16, 32, 64)}}
print(json.dumps(out))
