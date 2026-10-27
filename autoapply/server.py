"""Multi-user async web server & remote Playwright session controller for phone + laptop access."""
from __future__ import annotations

import asyncio
import base64
import datetime as dt
import hashlib
import json
import os
import re
import socket
import traceback
import urllib.parse
import uuid
from pathlib import Path
from typing import Any

from playwright.async_api import Browser, BrowserContext, Page, async_playwright

from .answers import AnswerEngine, Memory, _label_sig
from .ats import (
    click_nav_action,
    detect_ats,
    detect_step_name,
    dismiss_banners,
    enter_application,
    expand_workday_sections,
    extract_job_context,
    find_nav_action,
    handle_auth_gate,
    is_auth_gate,
    read_page_errors,
)
from .dom import Dom, FillError
from .fields import Field
from .llm import DEFAULT_MODELS, LLM
from .profile import Profile, _deep_merge
from .resume import _merge_nonempty, parse_resume
from .runner import TYPEAHEAD_HINT_RE

WEB_DIR = Path(__file__).parent / "web"


def _sanitize_user_id(raw: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_-]", "", (raw or "").strip().lower())
    return s[:40] or "default"


def _hash_passcode(passcode: str) -> str:
    if not passcode:
        return ""
    return hashlib.sha256(("autoapply-salt:" + passcode).encode()).hexdigest()


class UserWorkspace:
    """Isolated per-user storage on disk: profile, resume, answer memory, and run history."""

    def __init__(self, root: Path, user_id: str):
        self.user_id = _sanitize_user_id(user_id)
        self.dir = root / "users" / self.user_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.dir / "meta.json"
        self.profile_path = self.dir / "profile.yaml"
        self.memory_path = self.dir / "memory.json"
        self.history_path = self.dir / "history.json"

    def load_meta(self) -> dict[str, Any]:
        if self.meta_path.exists():
            try:
                return json.loads(self.meta_path.read_text())
            except Exception:  # noqa: BLE001
                pass
        return {}

    def save_meta(self, meta: dict[str, Any]) -> None:
        self.meta_path.write_text(json.dumps(meta, indent=2))

    def verify_or_init_passcode(self, passcode: str) -> bool:
        meta = self.load_meta()
        existing = meta.get("passcode_hash", "")
        given = _hash_passcode(passcode)
        if not existing:
            if passcode:
                meta["passcode_hash"] = given
                self.save_meta(meta)
            return True
        return existing == given

    def load_profile(self) -> Profile:
        meta = self.load_meta()
        prof = Profile.load(self.profile_path) if self.profile_path.exists() else Profile()
        resume_file = meta.get("resume_filename", "")
        if resume_file:
            rp = self.dir / resume_file
            if rp.exists():
                prof.resume_path = str(rp.resolve())
                prof.resume_text = meta.get("resume_text", "")
        return prof

    def save_profile(self, data: dict[str, Any]) -> Profile:
        prof = self.load_profile()
        prof.data = _deep_merge(prof.data, data)
        prof.save(self.profile_path)
        return prof

    def load_memory(self) -> Memory:
        return Memory(self.memory_path)

    def load_history(self) -> list[dict[str, Any]]:
        if self.history_path.exists():
            try:
                return json.loads(self.history_path.read_text())
            except Exception:  # noqa: BLE001
                pass
        return []

    def append_history(self, entry: dict[str, Any]) -> None:
        hist = self.load_history()
        hist.insert(0, entry)
        self.history_path.write_text(json.dumps(hist[:100], indent=2))

    def build_llm(self, override: dict[str, Any] | None = None) -> LLM | None:
        meta = self.load_meta()
        cfg = {**(meta.get("llm") or {}), **(override or {})}
        provider = str(cfg.get("provider") or "").lower().strip()
        if provider in ("none", "off", "rules"):
            return None
        api_key = str(cfg.get("api_key") or "").strip()
        max_in = int(cfg.get("max_input_tokens") or 6000)
        budget = int(cfg.get("token_budget") or 60000)
        if provider and api_key:
            model = str(cfg.get("model") or "").strip() or DEFAULT_MODELS.get(provider, "gemini-2.5-flash")
            return LLM(provider, api_key, model, max_input_tokens=max_in, token_budget=budget)
        return LLM.from_env(max_input_tokens=max_in, token_budget=budget)


class WebApplicationSession:
    """Interactive Playwright application session controllable from a phone or laptop browser."""

    def __init__(
        self,
        session_id: str,
        workspace: UserWorkspace,
        urls: list[str],
        llm: LLM | None,
        auto_advance: bool = False,
        auto_submit: bool = False,
        headless: bool = True,
    ):
        self.id = session_id
        self.workspace = workspace
        self.urls = urls
        self.llm = llm
        self.auto_advance = auto_advance
        self.auto_submit = auto_submit
        self.headless = headless

        self.profile = workspace.load_profile()
        self.memory = workspace.load_memory()
        self.engine = AnswerEngine(self.profile, llm=llm, memory=self.memory)

        self.status: str = "starting"  # starting | working | waiting_review | auth_gate | completed | stopped | error
        self.status_detail: str = "Launching browser..."
        self.url_index: int = 0
        self.current_url: str = urls[0] if urls else ""
        self.ats: str = ""
        self.job_ctx: dict[str, str] = {"title": "", "company": "", "description": ""}
        self.step_name: str = ""
        self.nav_kind: str = ""
        self.nav_text: str = ""
        self.fields: list[Field] = []
        self.errors: list[str] = []
        self.screenshot_bytes: bytes = b""
        self.results: list[dict[str, Any]] = []

        self._pw = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None
        self._dom: Dom | None = None
        self._nav_frame = None
        self._action_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._task: asyncio.Task | None = None
        self._lock = asyncio.Lock()

    def start(self) -> None:
        self._task = asyncio.create_task(self._run_loop())

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.workspace.user_id,
            "status": self.status,
            "status_detail": self.status_detail,
            "url_index": self.url_index,
            "total_urls": len(self.urls),
            "current_url": self.current_url,
            "ats": self.ats,
            "job": {
                "title": self.job_ctx.get("title", ""),
                "company": self.job_ctx.get("company", ""),
            },
            "step_name": self.step_name,
            "nav_kind": self.nav_kind,
            "nav_text": self.nav_text,
            "errors": self.errors,
            "fields": [
                {
                    "index": idx,
                    "uid": f.uid,
                    "kind": f.kind,
                    "label": f.label,
                    "section": f.section,
                    "required": f.required,
                    "multi": f.multi,
                    "options": [o.text for o in f.options if o.text][:60],
                    "value": f.value,
                    "filled": f.filled,
                    "display_value": f.display_value,
                    "status": f.status or "pending",
                    "source": f.source or "",
                    "error": f.error,
                    "description": f.description,
                }
                for idx, f in enumerate(self.fields)
            ],
            "llm": {
                "active": bool(self.llm),
                "provider": self.llm.provider if self.llm else "none",
                "model": self.llm.model if self.llm else "",
                "tokens_used": self.llm.tokens_used if self.llm else 0,
                "token_budget": self.llm.token_budget if self.llm else 0,
                "calls_made": self.llm.calls_made if self.llm else 0,
            },
            "results": self.results,
            "has_screenshot": bool(self.screenshot_bytes),
        }

    async def refresh_screenshot(self) -> None:
        if not self._page or self._page.is_closed():
            return
        try:
            self.screenshot_bytes = await self._page.screenshot(type="jpeg", quality=68, full_page=False)
        except Exception:  # noqa: BLE001
            pass

    async def update_field(self, index: int, value: Any) -> dict[str, Any]:
        async with self._lock:
            if not self._dom or not (0 <= index < len(self.fields)):
                raise ValueError("Invalid field index or inactive session")
            f = self.fields[index]
            if f.kind in ("combobox", "listbox_button") and not f.options and not f.typeahead:
                await self._dom.probe_options(f)
            if f.multi and isinstance(value, str) and "," in value:
                value = [x.strip() for x in value.split(",") if x.strip()]
            f.value = value
            f.source = "user"
            f.error = ""
            try:
                await self._dom.fill(f, value, resume_path=self.profile.resume_path)
            except Exception as e:  # noqa: BLE001
                f.error = str(e)[:140]
            await self._dom.verify(f)
            if f.status == "ok":
                self.memory.remember(f, f.filled or value)
            await self.refresh_screenshot()
            return self.to_dict()

    async def send_action(self, action: str, payload: dict[str, Any] | None = None) -> None:
        await self._action_queue.put({"action": action, **(payload or {})})

    async def stop(self) -> None:
        self.status = "stopped"
        self.status_detail = "Session stopped by user."
        await self._action_queue.put({"action": "stop"})
        await self._cleanup()

    async def _cleanup(self) -> None:
        try:
            if self._context:
                await self._context.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            if self._browser:
                await self._browser.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            if self._pw:
                await self._pw.stop()
        except Exception:  # noqa: BLE001
            pass
        self._context = None
        self._browser = None
        self._pw = None

    async def _run_loop(self) -> None:
        try:
            self._pw = await async_playwright().start()
            bdir = self.workspace.dir / "browser_profile"
            bdir.mkdir(parents=True, exist_ok=True)
            chromium_args = [
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-dev-shm-usage",
                "--disable-gpu",
                "--disable-software-rasterizer",
                "--disable-extensions",
                "--disable-background-networking",
                "--no-first-run",
                "--js-flags=--max-old-space-size=256",
            ]
            try:
                self._context = await self._pw.chromium.launch_persistent_context(
                    user_data_dir=str(bdir),
                    headless=self.headless,
                    viewport={"width": 1280, "height": 900},
                    args=chromium_args,
                )
            except Exception:  # noqa: BLE001
                self._browser = await self._pw.chromium.launch(
                    headless=self.headless,
                    args=chromium_args,
                )
                self._context = await self._browser.new_context(viewport={"width": 1280, "height": 900})

            for idx, url in enumerate(self.urls):
                if self.status == "stopped":
                    break
                self.url_index = idx
                self.current_url = url
                outcome = await self._process_single_url(url)
                rec = {
                    "url": url,
                    "ats": self.ats,
                    "title": self.job_ctx.get("title", ""),
                    "company": self.job_ctx.get("company", ""),
                    "outcome": outcome,
                    "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "tokens_used": self.llm.tokens_used if self.llm else 0,
                }
                self.results.append(rec)
                self.workspace.append_history(rec)

            if self.status != "stopped":
                self.status = "completed"
                self.status_detail = f"Processed {len(self.results)} job application(s)."
        except Exception as e:  # noqa: BLE001
            self.status = "error"
            self.status_detail = f"Error: {e}"
            self.errors.append(str(e))
        finally:
            await self._cleanup()

    async def _process_single_url(self, url: str) -> str:
        assert self._context is not None
        self.status = "working"
        self.status_detail = f"Opening {url}..."
        self.fields = []
        self.errors = []

        page = await self._context.new_page()
        self._page = page
        try:
            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=45000)
            except Exception as e:  # noqa: BLE001
                self.errors.append(f"Load warning: {e}")
            await page.wait_for_timeout(1200)
            await dismiss_banners(page)
            self.ats = await detect_ats(page)
            self.job_ctx = await extract_job_context(page)
            await self.refresh_screenshot()

            self.status_detail = "Locating application form..."
            dom = Dom(page)
            self._dom = dom
            page = await enter_application(page, dom, self.profile)
            self._page = page
            self.ats = await detect_ats(page)
            if not self.job_ctx.get("title"):
                self.job_ctx = await extract_job_context(page)
            await self.refresh_screenshot()

            for step_num in range(1, 13):
                if self.status == "stopped":
                    return "stopped"
                self.status = "working"
                self.status_detail = f"Scanning step {step_num}..."
                await page.wait_for_timeout(600)
                await dismiss_banners(page)

                if self.ats == "workday":
                    await expand_workday_sections(page, self.profile)

                fields = await dom.scan()
                if is_auth_gate(fields):
                    self.status_detail = "Handling account sign-in gate..."
                    handled = await handle_auth_gate(page, dom, fields, self.profile)
                    if handled:
                        await page.wait_for_timeout(2000)
                        fields = await dom.scan()

                if not fields:
                    page = await enter_application(page, dom, self.profile)
                    self._page = page
                    fields = await dom.scan()
                    if not fields:
                        self.status_detail = "No form fields found on page."
                        await self.refresh_screenshot()
                        return "no_fields"

                self.step_name = await detect_step_name(page) or (
                    "Sign In / Account Gate" if is_auth_gate(fields) else f"Step {step_num}"
                )
                for f in fields:
                    f.step = self.step_name

                self.status_detail = f"Filling {len(fields)} field(s) on {self.step_name}..."
                fields = await self._fill_and_verify(dom, fields)
                self.fields = fields
                self.nav_kind, self._nav_frame, self.nav_text = await find_nav_action(page)
                await self.refresh_screenshot()

                problems = [f for f in fields if f.required and f.status in ("missing", "error", "mismatch")]
                if not problems and self.nav_kind == "next" and self.auto_advance and not is_auth_gate(fields):
                    self.status_detail = f"Auto-advancing via '{self.nav_text or 'Next'}'..."
                    if self._nav_frame:
                        await click_nav_action(self._nav_frame, "next")
                        await page.wait_for_timeout(1800)
                        self.errors = await read_page_errors(page)
                        if not self.errors:
                            continue

                if not problems and self.nav_kind == "submit" and self.auto_submit and not is_auth_gate(fields):
                    if self._nav_frame:
                        await click_nav_action(self._nav_frame, "submit")
                        await page.wait_for_timeout(2200)
                        self.errors = await read_page_errors(page)
                        await self.refresh_screenshot()
                        if not self.errors:
                            return "submitted"

                # Wait for user action from phone / laptop UI
                self.status = "auth_gate" if is_auth_gate(fields) else "waiting_review"
                if problems:
                    self.status_detail = f"{len(problems)} required field(s) need review before continuing."
                elif self.nav_kind == "submit":
                    self.status_detail = "Ready to submit! Review fields below and tap Submit."
                else:
                    self.status_detail = f"Ready on {self.step_name}. Review or tap Next."

                while True:
                    msg = await self._action_queue.get()
                    act = msg.get("action", "")
                    if act == "stop":
                        return "stopped"
                    if act == "skip":
                        return "skipped"
                    if act == "rescan":
                        self.status = "working"
                        self.status_detail = "Re-scanning page..."
                        self.fields = await dom.scan()
                        for f in self.fields:
                            await dom.verify(f)
                        self.nav_kind, self._nav_frame, self.nav_text = await find_nav_action(page)
                        await self.refresh_screenshot()
                        self.status = "waiting_review"
                        self.status_detail = f"Re-scanned {len(self.fields)} field(s)."
                        continue
                    if act == "refill":
                        self.status = "working"
                        self.status_detail = "Re-running answer engine & filling..."
                        self.fields = await self._fill_and_verify(dom, await dom.scan())
                        self.nav_kind, self._nav_frame, self.nav_text = await find_nav_action(page)
                        await self.refresh_screenshot()
                        self.status = "waiting_review"
                        self.status_detail = "Fields updated."
                        continue
                    if act == "next":
                        self.status = "working"
                        self.status_detail = f"Clicking {self.nav_text or 'Next'}..."
                        if self._nav_frame:
                            await click_nav_action(self._nav_frame, "next")
                        await page.wait_for_timeout(1800)
                        self.errors = await read_page_errors(page)
                        await self.refresh_screenshot()
                        break
                    if act == "submit":
                        self.status = "working"
                        self.status_detail = f"Submitting application via {self.nav_text or 'Submit'}..."
                        if self._nav_frame:
                            await click_nav_action(self._nav_frame, "submit")
                            await page.wait_for_timeout(2500)
                            self.errors = await read_page_errors(page)
                            await self.refresh_screenshot()
                            if self.errors:
                                self.status = "waiting_review"
                                self.status_detail = "Submission reported validation errors — please review."
                                continue
                        return "submitted"
            return "max_steps"
        finally:
            try:
                await page.close()
            except Exception:  # noqa: BLE001
                pass

    async def _fill_and_verify(self, dom: Dom, fields: list[Field]) -> list[Field]:
        async with self._lock:
            for f in fields:
                if f.kind == "file" and not f.current and not re.search(r"cover\s*letter|transcript|portfolio", f.label, re.I):
                    if self.profile.resume_path and Path(self.profile.resume_path).exists():
                        try:
                            await dom.fill(f, self.profile.resume_path)
                            f.value = self.profile.resume_path
                            f.source = "profile"
                            f.rule = "file.resume"
                        except Exception as e:  # noqa: BLE001
                            f.error = str(e)[:120]

            if any(f.kind == "file" and re.search(r"cover\s*letter", f.label, re.I) for f in fields):
                self.engine.ensure_cover_letter(self.job_ctx, self.workspace.dir / "generated")

            for f in fields:
                if f.kind in ("combobox", "listbox_button") and not f.options and not f.current:
                    if TYPEAHEAD_HINT_RE.search(_label_sig(f)):
                        f.typeahead = True
                    else:
                        await dom.probe_options(f)

            self.engine.answer_fields(fields, self.job_ctx)
            await self._fill_list(dom, fields)

            # Check for conditional follow-up fields revealed by initial answers
            for _ in range(2):
                known = {f.uid for f in fields}
                rescan = await dom.scan()
                new_f = [nf for nf in rescan if nf.uid not in known]
                if not new_f:
                    break
                for nf in new_f:
                    if nf.kind in ("combobox", "listbox_button") and not nf.options:
                        if TYPEAHEAD_HINT_RE.search(_label_sig(nf)):
                            nf.typeahead = True
                        else:
                            await dom.probe_options(nf)
                self.engine.answer_fields(new_f, self.job_ctx)
                await self._fill_list(dom, new_f)
                fields.extend(new_f)

            for f in fields:
                await dom.verify(f)
            return fields

    async def _fill_list(self, dom: Dom, fields: list[Field]) -> None:
        for f in fields:
            if f.source in ("prefilled", "skip", "missing") or f.value in (None, "", []):
                continue
            if f.kind == "file" and f.filled:
                continue
            f.error = ""
            try:
                await dom.fill(f, f.value, resume_path=self.profile.resume_path)
            except FillError as e:
                f.error = str(e)
            except Exception as e:  # noqa: BLE001
                f.error = str(e)[:120]


# ---------------------------------------------------------------------- HTTP Server
def _default_data_dir() -> Path:
    env_dir = os.environ.get("AUTOAPPLY_DATA_DIR")
    if env_dir:
        return Path(env_dir)
    hf_data = Path("/data")
    if hf_data.exists() and os.access(hf_data, os.W_OK):
        return hf_data / "autoapply"
    return Path.home() / ".autoapply"


class AutoApplyServer:
    def __init__(self, data_dir: Path | None = None, default_headless: bool = True):
        self.data_dir = data_dir or _default_data_dir()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.default_headless = default_headless
        self.sessions: dict[str, WebApplicationSession] = {}

    def _auth_workspace(self, headers: dict[str, str]) -> tuple[UserWorkspace | None, str]:
        user_id = headers.get("x-user-id", "default")
        passcode = headers.get("x-passcode", "")
        ws = UserWorkspace(self.data_dir, user_id)
        if not ws.verify_or_init_passcode(passcode):
            return None, "Invalid passcode for this username."
        return ws, ""

    async def handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            req_line = await asyncio.wait_for(reader.readline(), timeout=15.0)
            if not req_line:
                writer.close()
                return
            parts = req_line.decode("latin-1", errors="ignore").strip().split(" ")
            if len(parts) < 2:
                writer.close()
                return
            method, raw_target = parts[0].upper(), parts[1]
            parsed_url = urllib.parse.urlsplit(raw_target)
            path = parsed_url.path

            headers: dict[str, str] = {}
            while True:
                hline = await asyncio.wait_for(reader.readline(), timeout=10.0)
                if not hline or hline in (b"\r\n", b"\n"):
                    break
                hs = hline.decode("latin-1", errors="ignore").strip()
                if ":" in hs:
                    k, v = hs.split(":", 1)
                    headers[k.strip().lower()] = v.strip()

            content_len = int(headers.get("content-length", "0"))
            if content_len > 25 * 1024 * 1024:
                await self._send_json(writer, 413, {"error": "Payload too large"})
                return
            body = await asyncio.wait_for(reader.readexactly(content_len), timeout=30.0) if content_len > 0 else b""

            await self.route(writer, method, path, headers, body)
        except Exception:  # noqa: BLE001
            pass
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:  # noqa: BLE001
                pass

    async def route(
        self,
        writer: asyncio.StreamWriter,
        method: str,
        path: str,
        headers: dict[str, str],
        body: bytes,
    ) -> None:
        # Static / PWA / Health routes
        if method == "GET" and path == "/healthz":
            await self._send_json(writer, 200, {"ok": True})
            return
        if method == "GET" and path in ("/", "/index.html"):
            html = (WEB_DIR / "index.html").read_bytes()
            await self._send_raw(writer, 200, "text/html; charset=utf-8", html)
            return
        if method == "GET" and path == "/manifest.json":
            manifest = {
                "name": "AutoApply",
                "short_name": "AutoApply",
                "start_url": "/",
                "display": "standalone",
                "background_color": "#0f172a",
                "theme_color": "#2563eb",
            }
            await self._send_json(writer, 200, manifest)
            return

        # Screenshot route (allows query param or open access by session id)
        m_shot = re.match(r"^/api/session/([a-zA-Z0-9_-]+)/screenshot$", path)
        if method == "GET" and m_shot:
            sid = m_shot.group(1)
            sess = self.sessions.get(sid)
            if not sess or not sess.screenshot_bytes:
                await self._send_raw(writer, 404, "text/plain", b"No screenshot")
                return
            await self._send_raw(writer, 200, "image/jpeg", sess.screenshot_bytes, cache="no-store")
            return

        # Authenticated API routes
        ws, err = self._auth_workspace(headers)
        if ws is None:
            await self._send_json(writer, 401, {"error": err})
            return

        if method == "GET" and path == "/api/me":
            prof = ws.load_profile()
            meta = ws.load_meta()
            mem = ws.load_memory()
            active_sids = [s.id for s in self.sessions.values() if s.workspace.user_id == ws.user_id and s.status not in ("completed", "stopped", "error")]
            await self._send_json(
                writer,
                200,
                {
                    "user_id": ws.user_id,
                    "profile": prof.data,
                    "resume_filename": meta.get("resume_filename", ""),
                    "has_resume": bool(prof.resume_path and Path(prof.resume_path).exists()),
                    "memory": mem.entries,
                    "history": ws.load_history()[:30],
                    "active_session_id": active_sids[-1] if active_sids else "",
                },
            )
            return

        if method == "POST" and path == "/api/profile":
            payload = json.loads(body.decode() or "{}")
            prof = ws.save_profile(payload.get("profile") or {})
            await self._send_json(writer, 200, {"ok": True, "profile": prof.data})
            return

        if method == "POST" and path == "/api/resume":
            payload = json.loads(body.decode() or "{}")
            filename = re.sub(r"[^a-zA-Z0-9._-]", "_", payload.get("filename") or "resume.pdf")
            if not filename.lower().endswith((".pdf", ".docx", ".txt")):
                filename += ".pdf"
            raw_b64 = payload.get("content_base64") or ""
            file_bytes = base64.b64decode(raw_b64)
            dest = ws.dir / f"resume_{filename}"
            dest.write_bytes(file_bytes)

            llm = ws.build_llm(payload.get("llm"))
            resume_text, parsed = parse_resume(dest, llm)
            existing = ws.load_profile().data
            merged = _merge_nonempty(parsed, existing)
            # Also let freshly parsed resume education/experience populate if existing was empty
            if parsed.get("education") and not existing.get("education"):
                merged["education"] = parsed["education"]
            if parsed.get("experience") and not existing.get("experience"):
                merged["experience"] = parsed["experience"]
            if parsed.get("skills") and not existing.get("skills"):
                merged["skills"] = parsed["skills"]
            # Override personal/links where resume found non-empty values and profile was blank
            for sec in ("personal", "links"):
                for k, v in (parsed.get(sec) or {}).items():
                    if v and not existing.get(sec, {}).get(k):
                        merged.setdefault(sec, {})[k] = v

            prof = Profile(merged, resume_text=resume_text, resume_path=str(dest.resolve()))
            prof.save(ws.profile_path)
            meta = ws.load_meta()
            meta["resume_filename"] = dest.name
            meta["resume_text"] = resume_text[:25000]
            ws.save_meta(meta)
            await self._send_json(
                writer,
                200,
                {
                    "ok": True,
                    "resume_filename": dest.name,
                    "profile": prof.data,
                    "tokens_used": llm.tokens_used if llm else 0,
                },
            )
            return

        if method == "DELETE" and path == "/api/memory":
            payload = json.loads(body.decode() or "{}")
            mem = ws.load_memory()
            key = payload.get("key")
            if key and key in mem.entries:
                del mem.entries[key]
            elif not key:
                mem.entries.clear()
            mem.save()
            await self._send_json(writer, 200, {"ok": True, "memory": mem.entries})
            return

        if method == "POST" and path == "/api/session/start":
            payload = json.loads(body.decode() or "{}")
            raw_urls = payload.get("urls") or []
            if isinstance(raw_urls, str):
                raw_urls = [u.strip() for u in raw_urls.splitlines() if u.strip()]
            urls = [u for u in raw_urls if u.startswith("http")]
            if not urls:
                await self._send_json(writer, 400, {"error": "Provide at least one valid http(s) job URL."})
                return

            # Stop any prior active session for this user
            for s in list(self.sessions.values()):
                if s.workspace.user_id == ws.user_id and s.status not in ("completed", "stopped", "error"):
                    await s.stop()

            llm = ws.build_llm(payload.get("llm"))
            sid = uuid.uuid4().hex[:12]
            headless = payload.get("headless", self.default_headless)
            sess = WebApplicationSession(
                session_id=sid,
                workspace=ws,
                urls=urls,
                llm=llm,
                auto_advance=bool(payload.get("auto_advance", False)),
                auto_submit=bool(payload.get("auto_submit", False)),
                headless=bool(headless),
            )
            self.sessions[sid] = sess
            sess.start()
            await self._send_json(writer, 200, sess.to_dict())
            return

        m_sess = re.match(r"^/api/session/([a-zA-Z0-9_-]+)$", path)
        if method == "GET" and m_sess:
            sid = m_sess.group(1)
            sess = self.sessions.get(sid)
            if not sess or sess.workspace.user_id != ws.user_id:
                await self._send_json(writer, 404, {"error": "Session not found"})
                return
            await self._send_json(writer, 200, sess.to_dict())
            return

        m_field = re.match(r"^/api/session/([a-zA-Z0-9_-]+)/field$", path)
        if method == "POST" and m_field:
            sid = m_field.group(1)
            sess = self.sessions.get(sid)
            if not sess or sess.workspace.user_id != ws.user_id:
                await self._send_json(writer, 404, {"error": "Session not found"})
                return
            payload = json.loads(body.decode() or "{}")
            try:
                state = await sess.update_field(int(payload["index"]), payload.get("value"))
                await self._send_json(writer, 200, state)
            except Exception as e:  # noqa: BLE001
                await self._send_json(writer, 400, {"error": str(e)})
            return

        m_act = re.match(r"^/api/session/([a-zA-Z0-9_-]+)/action$", path)
        if method == "POST" and m_act:
            sid = m_act.group(1)
            sess = self.sessions.get(sid)
            if not sess or sess.workspace.user_id != ws.user_id:
                await self._send_json(writer, 404, {"error": "Session not found"})
                return
            payload = json.loads(body.decode() or "{}")
            act = payload.get("action", "")
            if act == "stop":
                await sess.stop()
            else:
                await sess.send_action(act, payload)
                await asyncio.sleep(0.3)
            await self._send_json(writer, 200, sess.to_dict())
            return

        await self._send_json(writer, 404, {"error": f"Not found: {path}"})

    async def _send_json(self, writer: asyncio.StreamWriter, status: int, data: Any) -> None:
        raw = json.dumps(data, ensure_ascii=False).encode("utf-8")
        await self._send_raw(writer, status, "application/json; charset=utf-8", raw, cache="no-store")

    async def _send_raw(
        self,
        writer: asyncio.StreamWriter,
        status: int,
        content_type: str,
        body: bytes,
        cache: str = "public, max-age=60",
    ) -> None:
        reason = {200: "OK", 400: "Bad Request", 401: "Unauthorized", 404: "Not Found", 413: "Payload Too Large", 500: "Server Error"}.get(status, "OK")
        hdr = (
            f"HTTP/1.1 {status} {reason}\r\n"
            f"Content-Type: {content_type}\r\n"
            f"Content-Length: {len(body)}\r\n"
            f"Cache-Control: {cache}\r\n"
            f"Connection: close\r\n\r\n"
        ).encode("latin-1")
        writer.write(hdr + body)
        await writer.drain()


def get_lan_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:  # noqa: BLE001
        return "127.0.0.1"


async def run_server(host: str = "0.0.0.0", port: int = 8000, data_dir: Path | None = None, headless: bool = True) -> None:
    app = AutoApplyServer(data_dir=data_dir, default_headless=headless)
    server = await asyncio.start_server(app.handle_client, host, port)
    lan_ip = get_lan_ip()
    print(f"\n  AutoApply Web Service is running!")
    print(f"  • Laptop / Local:  http://localhost:{port}")
    print(f"  • Phone (Wi-Fi):   http://{lan_ip}:{port}")
    print(f"  • Data directory:  {app.data_dir}\n")
    async with server:
        await server.serve_forever()
