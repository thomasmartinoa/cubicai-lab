#!/usr/bin/env python3
"""The Bots' browser: one Chromium (through the gateway's proxy) kept open between commands.

POST / with {"do": open|click|type|scroll|state, ...}; answers with the page as the model reads it
(title, address, text, numbered elements) and a screenshot in /shots. Clicking or pressing Enter in a
form that POSTs (sends something somewhere) answers {"needs_approval": true, "what": ...} unless the
command says "approved": true, so the gateway can ask the user first.

The user teaching a task from the phone drives it by point instead: tap {x, y} (fractions of the
screenshot), text {text} (typed into what has focus), key {key}, and shot (just a fresh screenshot).
Those answer with only the address, title, screenshot and "did": what was tapped or typed into.
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


# What is at a point (or has focus): the thing a person meant, described as a step would say it.
DESCRIBE = """
(at) => {
  let e = at ? document.elementFromPoint(at[0], at[1]) : document.activeElement;
  if (!e || e === document.body || e === document.documentElement) return null;
  const sel = 'a[href], button, input, select, textarea, label, [role=button], [role=link], [role=tab], [role=menuitem], [role=checkbox], [onclick], [contenteditable=true]';
  const hit = e.closest(sel) || e;
  const tag = hit.tagName.toLowerCase();
  let kind = tag === 'a' ? 'link' : tag === 'button' ? 'button' : tag === 'select' ? 'list' : tag === 'textarea' ? 'text box' : tag === 'label' ? 'label' : null;
  if (tag === 'input') {
    const t = (hit.type || 'text').toLowerCase();
    kind = ['submit', 'button', 'image', 'reset'].includes(t) ? 'button' : t === 'checkbox' ? 'checkbox' : t === 'radio' ? 'option' : t === 'password' ? 'password field' : t === 'search' ? 'search box' : 'field';
  }
  const role = hit.getAttribute('role'); if (!kind && role) kind = role;
  if (!kind) kind = hit.isContentEditable ? 'text box' : 'text';
  let label = '';
  if (hit.labels && hit.labels.length) label = hit.labels[0].innerText;
  if (!label) label = hit.getAttribute('aria-label') || hit.getAttribute('placeholder') || hit.getAttribute('title') || '';
  if (!label && tag === 'input' && ['submit', 'button'].includes((hit.type || '').toLowerCase())) label = hit.value || '';
  if (!label && !['input', 'textarea', 'select'].includes(tag)) label = hit.innerText || hit.getAttribute('alt') || '';
  if (!label) label = hit.getAttribute('name') || '';
  label = label.replace(/\\s+/g, ' ').trim().slice(0, 80);
  return {kind, label, secret: tag === 'input' && (hit.type || '').toLowerCase() === 'password'};
}
"""

KEYS = {"Enter", "Tab", "Backspace", "Escape", "ArrowDown", "ArrowUp", "ArrowLeft", "ArrowRight", "PageDown", "PageUp", "Space"}


def light(p, did=None):
    """The page for a person watching: where it is and a fresh screenshot."""
    state["shot"] += 1
    shot = f"/shots/{os.getpid()}-{state['shot']}.jpg"
    try:
        p.screenshot(path=shot, type="jpeg", quality=60)
    except Exception:
        shot = None
    try:
        title = p.title()
    except Exception:
        title = ""
    return {"ok": True, "url": p.url, "title": title, "shot": shot, "did": did}


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
    if do == "shot":
        return light(p)
    if do == "tap":
        vp = p.viewport_size or {"width": 1280, "height": 900}
        x = min(max(float(cmd.get("x", 0)), 0.0), 1.0) * vp["width"]
        y = min(max(float(cmd.get("y", 0)), 0.0), 1.0) * vp["height"]
        did = p.evaluate(DESCRIBE, [x, y])
        p.mouse.click(x, y)
        settle(p)
        return light(p, did)
    if do == "text":
        did = p.evaluate(DESCRIBE, None)
        p.keyboard.type(str(cmd.get("text", ""))[:2000], delay=15)
        time.sleep(0.3)
        return light(p, did)
    if do == "key":
        key = str(cmd.get("key", ""))
        if key not in KEYS:
            return {"ok": False, "error": f"Unknown key {key!r}."}
        did = p.evaluate(DESCRIBE, None)
        p.keyboard.press(" " if key == "Space" else key)
        if key == "Enter":
            settle(p)
        else:
            time.sleep(0.3)
        return light(p, did)
    if do in ("open_light", "scroll_light", "back_light"):
        inner = dict(cmd, do=do[:-6])
        out = handle(inner)
        return light(p) if out.get("ok") else out
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
