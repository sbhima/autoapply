"""Playwright-side DOM operations: scan fields, probe dropdown options, fill, read back."""
from __future__ import annotations

import asyncio
import re
from pathlib import Path

from playwright.async_api import Frame, Page

from .fields import Field
from .matching import match_option, norm

JS_DIR = Path(__file__).parent / "js"
JS_EXTRACT = (JS_DIR / "extract.js").read_text()
JS_READBACK = (JS_DIR / "readback.js").read_text()

SKIP_FRAME_RE = re.compile(
    r"recaptcha|hcaptcha|challenges\.cloudflare|doubleclick|googletagmanager|google-analytics|youtube|vimeo|"
    r"facebook|twitter|intercom|drift|hotjar|onetrust|cookielaw|stripe\.network|about:blank|qualified\.com|"
    r"linkedin\.com/(?!in/)|bat\.bing|adsrvr|zendesk|hubspot",
    re.I,
)

# JS: tag currently visible dropdown options with data-aa-opt=<i> and return their texts.
JS_OPTIONS = r"""
() => {
  const raw = (s) => (s || '').replace(/[\u00a0\s]+/g, ' ').trim();
  document.querySelectorAll('[data-aa-opt]').forEach((e) => e.removeAttribute('data-aa-opt'));
  const roots = [document];
  for (let i = 0; i < roots.length; i++) for (const el of roots[i].querySelectorAll('*')) if (el.shadowRoot) roots.push(el.shadowRoot);
  const sel = '[role="option"], [data-automation-id="promptOption"], [data-automation-id="menuItem"], li[id*="-option-"], [class*="select__option"], [class*="-option"]:not([class*="-options"])';
  const seen = new Set(); const out = [];
  for (const r of roots) for (const el of r.querySelectorAll(sel)) {
    if (seen.has(el)) continue;
    // prefer the outermost option element
    if (el.parentElement && el.parentElement.closest && el.parentElement.closest('[role="option"]')) continue;
    const rect = el.getBoundingClientRect();
    if (rect.width < 1 || rect.height < 1) continue;
    const st = getComputedStyle(el); if (st.visibility === 'hidden' || st.display === 'none') continue;
    if (el.getAttribute('aria-disabled') === 'true') continue;
    const t = raw(el.textContent);
    if (!t || /^(no (options|results|matches)|loading|searching|type to search|start typing|no items)/i.test(t)) continue;
    seen.add(el);
    el.setAttribute('data-aa-opt', String(out.length));
    out.push(t);
  }
  return out;
}
"""


class FillError(Exception):
    pass


def _shorter_query(q: str) -> str:
    q = re.sub(r"[,(].*$", "", q).strip()
    words = q.split()
    if len(words) > 3:
        return " ".join(words[:3])
    if len(words) > 1:
        return words[0] if len(words[0]) > 3 else " ".join(words[:2])
    return q[:4] if len(q) > 6 else q


class Dom:
    def __init__(self, page: Page):
        self.page = page
        self._keys: dict[int, str] = {}
        self._frames: dict[str, Frame] = {}

    # ------------------------------------------------------------------ frames
    def _key(self, frame: Frame) -> str:
        k = self._keys.get(id(frame))
        if not k:
            k = "f0" if frame == self.page.main_frame else f"fr{len(self._keys) + 1}"
            self._keys[id(frame)] = k
        self._frames[k] = frame
        return k

    def frame(self, key: str) -> Frame:
        return self._frames.get(key, self.page.main_frame)

    def loc(self, f: Field, uid: str | None = None):
        return self.frame(f.frame_key).locator(f'[data-aa-id="{uid or f.uid}"]').first

    # ------------------------------------------------------------------ scan
    async def scan(self) -> list[Field]:
        out: list[Field] = []
        for frame in list(self.page.frames):
            if frame.is_detached():
                continue
            if frame != self.page.main_frame and (SKIP_FRAME_RE.search(frame.url or "") or not frame.url):
                continue
            key = self._key(frame)
            try:
                res = await frame.evaluate(JS_EXTRACT, {"prefix": key})
            except Exception:  # noqa: BLE001 - cross-origin hiccups, navigation races
                continue
            for d in res:
                out.append(Field.from_scan(d, key))
        return out

    async def probe_options(self, f: Field) -> None:
        """Open a custom dropdown to learn its options without choosing anything."""
        if f.kind not in ("combobox", "listbox_button") or f.options:
            return
        frame = self.frame(f.frame_key)
        loc = self.loc(f)
        try:
            await loc.scroll_into_view_if_needed(timeout=2000)
            await loc.click(timeout=2500)
            texts = await self._wait_options(frame, 1200, settle=True)
            f.options = [type(f.options).__args__[0](t) if False else _opt(t) for t in texts] if texts else []  # type: ignore[attr-defined]
            f.typeahead = not texts
        except Exception:  # noqa: BLE001
            f.typeahead = True
        finally:
            try:
                await self.page.keyboard.press("Escape")
            except Exception:  # noqa: BLE001
                pass
        # a "probe" can leave focus in the input with a partial value; blur it
        try:
            await frame.evaluate("() => document.activeElement && document.activeElement.blur && document.activeElement.blur()")
        except Exception:  # noqa: BLE001
            pass

    async def _wait_options(self, frame: Frame, timeout_ms: int, settle: bool = False) -> list[str]:
        loop = asyncio.get_event_loop()
        end = loop.time() + timeout_ms / 1000
        last: list[str] = []
        stable = 0
        while loop.time() < end:
            try:
                texts = await frame.evaluate(JS_OPTIONS)
            except Exception:  # noqa: BLE001
                texts = []
            if texts:
                if texts == last:
                    stable += 1
                    if stable >= (2 if settle else 1):
                        return texts
                else:
                    stable = 0
                last = texts
            await asyncio.sleep(0.15)
        return last

    # ------------------------------------------------------------------ fill
    async def fill(self, f: Field, value, resume_path: str = "") -> None:
        if value is None or value == "" or value == []:
            return
        frame = self.frame(f.frame_key)
        loc = self.loc(f)
        k = f.kind
        if k in ("text", "textarea", "date"):
            await loc.scroll_into_view_if_needed(timeout=3000)
            try:
                await loc.fill(str(value), timeout=5000)
            except Exception:  # noqa: BLE001 - masked inputs: type char by char
                await loc.click(timeout=3000)
                await loc.press_sequentially(str(value), delay=20)
            await loc.evaluate("el => { el.dispatchEvent(new Event('change', {bubbles:true})); el.dispatchEvent(new Event('blur', {bubbles:true})); }")
            return
        if k == "file":
            path = str(value)
            if not Path(path).exists():
                raise FillError(f"file not found: {path}")
            await loc.set_input_files(path, timeout=8000)
            await self.page.wait_for_timeout(1500)  # let the ATS upload/parse
            return
        if k == "select":
            vals = value if isinstance(value, list) else [value]
            texts = f.option_texts
            chosen = []
            for v in vals:
                i = match_option(v, texts)
                if i is None:
                    raise FillError(f"no option matching {v!r}")
                o = f.options[i]
                chosen.append(o.value if o.value != "" else o.text)
            await loc.select_option(chosen if f.multi else chosen[0], force=True, timeout=5000)
            # select2/chosen wrappers listen for jQuery change
            await loc.evaluate("el => { try { if (window.jQuery) window.jQuery(el).trigger('change'); } catch(e){} el.dispatchEvent(new Event('change', {bubbles:true})); }")
            return
        if k in ("radio", "checkbox_group", "button_group"):
            vals = value if isinstance(value, list) else [value]
            texts = f.option_texts
            for v in vals:
                i = match_option(v, texts)
                if i is None:
                    raise FillError(f"no option matching {v!r} in {texts[:8]}")
                await self._click_option(f, f.options[i].uid, button=(k == "button_group"))
            return
        if k == "checkbox":
            want = value is True or str(value).lower() in ("true", "yes", "y", "1", "checked")
            await self._set_checkbox(f, f.uid, want)
            return
        if k in ("combobox", "listbox_button"):
            vals = value if isinstance(value, list) else [value]
            for v in vals:
                await self._fill_dropdown(frame, f, str(v) if not isinstance(v, bool) else ("Yes" if v else "No"))
            return
        raise FillError(f"unsupported field kind {k}")

    async def _click_option(self, f: Field, uid: str, button: bool = False) -> None:
        frame = self.frame(f.frame_key)
        el = frame.locator(f'[data-aa-id="{uid}"]').first
        if button:
            await el.click(timeout=3000)
            return
        try:
            if await el.is_checked(timeout=1000):
                return
        except Exception:  # noqa: BLE001
            pass
        for attempt in (
            lambda: el.check(timeout=2000),
            lambda: frame.locator(f'[data-aa-label="{uid}"]').first.click(timeout=2000),
            lambda: el.click(timeout=2000, force=True),
            lambda: el.evaluate("e => e.click()"),
        ):
            try:
                await attempt()
                try:
                    if await el.is_checked(timeout=500):
                        return
                except Exception:  # noqa: BLE001 - role=radio divs: trust the click
                    return
            except Exception:  # noqa: BLE001
                continue
        raise FillError("could not select option")

    async def _set_checkbox(self, f: Field, uid: str, want: bool) -> None:
        frame = self.frame(f.frame_key)
        el = frame.locator(f'[data-aa-id="{uid}"]').first

        async def state() -> bool:
            return await el.evaluate("e => !!(e.checked || e.getAttribute('aria-checked') === 'true')")

        if await state() == want:
            return
        for attempt in (
            lambda: el.set_checked(want, timeout=2000),
            lambda: frame.locator(f'[data-aa-label="{uid}"]').first.click(timeout=2000),
            lambda: el.evaluate("e => e.click()"),
        ):
            try:
                await attempt()
            except Exception:  # noqa: BLE001
                continue
            if await state() == want:
                return
        raise FillError("checkbox would not toggle")

    async def _fill_dropdown(self, frame: Frame, f: Field, v: str) -> None:
        loc = self.loc(f)
        await loc.scroll_into_view_if_needed(timeout=3000)
        await loc.click(timeout=3000)
        editable = await loc.evaluate("e => (e.tagName === 'INPUT' || e.isContentEditable) && !e.readOnly && !e.disabled")
        texts: list[str] = []
        if editable:
            for q in dict.fromkeys([v[:80], _shorter_query(v)]):
                await loc.fill("")
                await loc.press_sequentially(q, delay=12)
                texts = await self._wait_options(frame, 4000 if f.typeahead else 1800, settle=True)
                if texts and match_option(v, texts) is not None:
                    break
        else:
            texts = await self._wait_options(frame, 2000, settle=True)
        idx = match_option(v, texts)
        if idx is None and editable and len(texts) == 1 and f.typeahead:
            idx = 0  # single typeahead suggestion for e.g. a city name
        if idx is None and editable and texts and f.typeahead:
            # typeahead (schools, cities): accept the closest suggestion if it contains our first word
            first = norm(v).split(" ")[0]
            for i, t in enumerate(texts):
                if first and first in norm(t):
                    idx = i
                    break
        if idx is None:
            # Workday popup searchBox fallback: type query + Enter inside [data-automation-id="searchBox"]
            sb = frame.locator('[data-automation-id="searchBox"]').first
            try:
                if await sb.is_visible(timeout=400):
                    await sb.fill( _shorter_query(v))
                    await sb.press("Enter")
                    texts = await self._wait_options(frame, 2500, settle=True)
                    idx = match_option(v, texts)
                    if idx is None and texts:
                        idx = 0
            except Exception:  # noqa: BLE001
                pass
        if idx is None:
            await self.page.keyboard.press("Escape")
            raise FillError(f"no option matching {v!r}; saw {texts[:6]}")
        await frame.locator(f'[data-aa-opt="{idx}"]').first.click(timeout=3000)
        await self.page.wait_for_timeout(300)
        # Workday hierarchical menus: if clicking a category opened sub-options, pick the best sub-option
        sub_texts = await self._wait_options(frame, 600, settle=False)
        if sub_texts and sub_texts != texts:
            sub_idx = match_option(v, sub_texts)
            if sub_idx is None:
                sub_idx = 0
            try:
                await frame.locator(f'[data-aa-opt="{sub_idx}"]').first.click(timeout=2000)
                await self.page.wait_for_timeout(200)
            except Exception:  # noqa: BLE001
                pass

    # ------------------------------------------------------------------ readback & verify
    async def readback(self, f: Field) -> str:
        frame = self.frame(f.frame_key)
        try:
            return await frame.evaluate(
                JS_READBACK, {"uid": f.uid, "kind": f.kind, "optionUids": [o.uid for o in f.options if o.uid]}
            )
        except Exception:  # noqa: BLE001
            return ""

    async def verify(self, f: Field) -> str:
        rb = await self.readback(f)
        if rb == "__GONE__":
            # Some ATSs re-render or replace the control with a chip after selection
            f.filled = str(f.value or "")
            f.status = "ok" if f.value not in (None, "", []) else "skipped"
            return f.status
        f.filled = rb
        if f.error:
            f.status = "error"
            return f.status
        if f.source == "prefilled" and rb:
            f.status = "ok"
            return f.status
        if f.value in (None, "", []):
            f.status = "missing" if f.required and not rb else ("ok" if rb else "skipped")
            return f.status
        if f.kind == "checkbox":
            want = f.value is True or str(f.value).lower() in ("true", "yes", "1", "checked")
            got = rb == "true"
            f.status = "ok" if want == got else "mismatch"
            return f.status
        if f.kind == "file":
            want_name = Path(str(f.value)).name.lower()
            f.status = "ok" if (rb and (want_name in rb.lower() or len(rb) > 2)) else "mismatch"
            return f.status
        if not rb:
            # Combobox chips sometimes render outside the input container; trust if no error was raised
            f.status = "ok" if f.kind == "combobox" else "mismatch"
            return f.status
        # Compare normalized values (or option match)
        want_s = ", ".join(str(x) for x in f.value) if isinstance(f.value, list) else str(f.value)
        n_want, n_got = norm(want_s), norm(rb)
        if n_want == n_got or n_want in n_got or n_got in n_want:
            f.status = "ok"
        elif f.is_choice and match_option(want_s, [rb]) is not None:
            f.status = "ok"
        elif f.kind == "text" and re.sub(r"\D", "", want_s) and re.sub(r"\D", "", want_s) in re.sub(r"\D", "", rb):
            f.status = "ok"  # formatted phone numbers
        else:
            f.status = "mismatch"
        return f.status


def _opt(t: str):
    from .fields import Option

    return Option(text=t)
