#!/usr/bin/env python3
"""The Bots' browser: one Chromium (through the gateway's proxy) kept open between commands.

POST / with {"do": open|click|type|scroll|state, ...}; answers with the page as the model reads it
(title, address, text, numbered elements) and a screenshot in /shots. Clicking or pressing Enter in a
form that POSTs (sends something somewhere) answers {"needs_approval": true, "what": ...} unless the
command says "approved": true, so the gateway can ask the user first.
"""
import json, os, time
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse

TEXT_CHARS = 6000
MAX_ELEMENTS = 60

state = {"pw": None, "browser": None, "page": None, "shot": 0}

MARK = """
() => {
  document.querySelectorAll('[data-cubic-ref]').forEach(e => e.removeAttribute('data-cubic-ref'));
  const sel = 'a[href], button, input:not([type=hidden]), select, textarea, [role=button], [role=link], [role=tab], [role=menuitem], [onclick], [contenteditable=true]';
  const out = [];
  let n = 0;
  for (const e of document.querySelectorAll(sel)) {
    const r = e.getBoundingClientRect();
    const st = getComputedStyle(e);
    if (r.width < 2 || r.height < 2 || st.visibility === 'hidden' || st.display === 'none') continue;
    if (r.bottom < 0 || r.top > innerHeight * 2.5) continue;
    n += 1;
    e.setAttribute('data-cubic-ref', String(n));
    const tag = e.tagName.toLowerCase();
    let kind = tag === 'a' ? 'link' : tag === 'button' ? 'button' : tag;
    if (tag === 'input') kind = 'input ' + (e.type || 'text');
    const role = e.getAttribute('role'); if (role && tag !== 'a' && tag !== 'button') kind = role;
    let label = (e.innerText || e.getAttribute('aria-label') || e.getAttribute('title') || e.placeholder || e.value || e.name || e.getAttribute('href') || '').replace(/\\s+/g, ' ').trim();
    if (tag === 'input' && e.value && e.type !== 'password' && e.type !== 'submit') label = (e.placeholder || e.name || '') + ' = "' + e.value + '"';
    out.push([n, kind, label.slice(0, 80)]);
    if (n >= %d) break;
  }
  return out;
}
""" % MAX_ELEMENTS

SENDS = """
(e) => {
  const form = e.form || e.closest('form');
  if (!form) return null;
  const method = (form.getAttribute('method') || 'get').toLowerCase();
  if (method !== 'post') return null;
  const fields = [...form.elements].filter(f => f.name && !['hidden','submit','button'].includes(f.type)).map(f => f.name).filter((n, i, all) => all.indexOf(n) === i).slice(0, 8);
  let host = location.host;
  try { host = new URL(form.action || location.href).host; } catch (x) {}
  return {host, fields};
}
"""

IS_SUBMIT = """
(e) => {
  const t = (e.getAttribute('type') || '').toLowerCase();
  if (e.tagName === 'BUTTON') return (t === '' || t === 'submit') && !!(e.form || e.closest('form'));
  if (e.tagName === 'INPUT') return t === 'submit' || t === 'image';
  return false;
}
"""


def page():
    if state["page"] is None or state["page"].is_closed():
        if state["pw"] is None:
            from playwright.sync_api import sync_playwright
            state["pw"] = sync_playwright().start()
        if state["browser"] is None:
            state["browser"] = state["pw"].chromium.launch(proxy={"server": "http://127.0.0.1:3128"}, args=["--disable-dev-shm-usage"])
        ctx = state["browser"].new_context(viewport={"width": 1280, "height": 900}, locale="en-US", accept_downloads=False)
        state["page"] = ctx.new_page()
        state["page"].set_default_timeout(20000)
    return state["page"]


def settle(p):
    try:
        p.wait_for_load_state("domcontentloaded", timeout=15000)
    except Exception:
        pass
    time.sleep(0.8)


def view(p, note=None):
    elements = p.evaluate(MARK)
    try:
        text = p.inner_text("body")
    except Exception:
        text = ""
    text = "\n".join(l.strip() for l in text.splitlines() if l.strip())
    state["shot"] += 1
    shot = f"/shots/{os.getpid()}-{state['shot']}.jpg"
    try:
        p.screenshot(path=shot, type="jpeg", quality=55)
    except Exception:
        shot = None
    y = p.evaluate("() => [Math.round(scrollY), document.documentElement.scrollHeight, innerHeight]")
    return {
        "ok": True, "url": p.url, "title": p.title(), "text": text[:TEXT_CHARS], "more_text": len(text) > TEXT_CHARS,
        "elements": elements, "shot": shot, "scroll": y, "note": note,
    }


def find(p, ref):
    loc = p.locator(f'[data-cubic-ref="{int(ref)}"]')
    if loc.count() == 0:
        raise ValueError(f"There's no element {ref} on the page now; look at the page again for the current numbers.")
    return loc.first


def approval(el, what):
    sends = el.evaluate(SENDS)
    if not sends:
        return None
    fields = ", ".join(sends["fields"]) or "no named fields"
    return {"ok": True, "needs_approval": True, "what": f"{what} a form that sends data to {sends['host']} (fields: {fields})"}


def handle(cmd):
    do = cmd.get("do")
    if do == "ping":
        return {"ok": True}
    p = page()
    if do == "open":
        url = (cmd.get("url") or "").strip()
        if not urlparse(url).scheme:
            url = "https://" + url
        if urlparse(url).scheme not in ("http", "https"):
            return {"ok": False, "error": "Open http(s) addresses only."}
        p.goto(url, wait_until="domcontentloaded", timeout=30000)
        settle(p)
        return view(p)
    if do == "click":
        el = find(p, cmd.get("element"))
        if not cmd.get("approved") and el.evaluate(IS_SUBMIT):
            need = approval(el, "Submit")
            if need:
                return need
        el.scroll_into_view_if_needed(timeout=5000)
        el.click(timeout=10000)
        settle(p)
        return view(p)
    if do == "type":
        el = find(p, cmd.get("element"))
        submit = bool(cmd.get("submit"))
        if submit and not cmd.get("approved"):
            need = approval(el, "Press Enter in")
            if need:
                return need
        el.fill(str(cmd.get("text", "")), timeout=10000)
        if submit:
            el.press("Enter")
            settle(p)
        return view(p)
    if do == "scroll":
        down = cmd.get("direction", "down") != "up"
        p.mouse.wheel(0, 800 if down else -800)
        time.sleep(0.5)
        return view(p)
    if do == "back":
        p.go_back(wait_until="domcontentloaded")
        settle(p)
        return view(p)
    if do == "state":
        return view(p)
    return {"ok": False, "error": f"Unknown browser command {do!r}."}


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            cmd = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
            out = handle(cmd)
        except Exception as e:
            msg = str(e).split("\n")[0][:300]
            if "net::ERR" in msg or "ERR_TUNNEL" in msg:
                msg += " (the computer reaches public websites only)"
            out = {"ok": False, "error": msg}
        body = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    HTTPServer(("127.0.0.1", 7111), Handler).serve_forever()
