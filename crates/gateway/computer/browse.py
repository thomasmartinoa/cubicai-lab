#!/usr/bin/env python3
"""One browser command from the gateway (JSON on stdin) → the browser service → JSON on stdout."""
import json, sys, urllib.request
req = urllib.request.Request("http://127.0.0.1:7111/", data=sys.stdin.buffer.read(), headers={"Content-Type": "application/json"})
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
try:
    with opener.open(req, timeout=90) as r:
        sys.stdout.write(r.read().decode())
except Exception as e:
    sys.stdout.write(json.dumps({"ok": False, "error": f"The browser didn't answer: {e}"}))
