"""ATS detection, job-context extraction, entry into application forms, and multi-step navigation."""
from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

from playwright.async_api import Frame, Page

from .dom import Dom
from .fields import Field
from .llm import clip_text
from .profile import Profile

ATS_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("greenhouse", re.compile(r"greenhouse\.io|grnhse", re.I)),
    ("ashby", re.compile(r"ashbyhq\.com", re.I)),
    ("lever", re.compile(r"lever\.co", re.I)),
    ("workday", re.compile(r"myworkdayjobs\.com|workday\.com|wd\d+\.myworkday", re.I)),
    ("smartrecruiters", re.compile(r"smartrecruiters\.com", re.I)),
    ("icims", re.compile(r"icims\.com", re.I)),
    ("bamboohr", re.compile(r"bamboohr\.com", re.I)),
    ("jobvite", re.compile(r"jobvite\.com", re.I)),
    ("rippling", re.compile(r"rippling\.com", re.I)),
    ("dover", re.compile(r"dover\.com|dover\.io", re.I)),
    ("workable", re.compile(r"workable\.com", re.I)),
    ("breezy", re.compile(r"breezy\.hr", re.I)),
    ("recruitee", re.compile(r"recruitee\.com", re.I)),
    ("taleo", re.compile(r"taleo\.net", re.I)),
]


async def detect_ats(page: Page) -> str:
    urls = [page.url] + [f.url for f in page.frames if f.url]
    joined = " ".join(urls)
    for name, rx in ATS_PATTERNS:
        if rx.search(joined):
            return name
    try:
        sig = await page.evaluate(
            """() => {
                if (document.querySelector('#grnhse_iframe, #application_form[action*="greenhouse"]')) return 'greenhouse';
                if (document.querySelector('[name^="_systemfield_"], [id^="_systemfield_"]')) return 'ashby';
                if (document.querySelector('[data-automation-id="applyManually"], [data-automation-id="legalNameSection_firstName"]')) return 'workday';
                if (document.querySelector('form#application-form[action*="lever"]')) return 'lever';
                if (document.querySelector('oc-application-form, [class*="smartrecruiters"]')) return 'smartrecruiters';
                if (document.querySelector('#icims_content_iframe')) return 'icims';
                return 'generic';
            }"""
        )
        return sig or "generic"
    except Exception:  # noqa: BLE001
        return "generic"


async def extract_job_context(page: Page) -> dict[str, str]:
    """Extract compact {title, company, description} from the job page (token-bounded)."""
    try:
        raw = await page.evaluate(
            """() => {
                const clean = (s) => (s || '').replace(/\\s+/g, ' ').trim();
                let title = '', company = '', desc = '';
                // 1. Check JSON-LD JobPosting
                for (const s of document.querySelectorAll('script[type="application/ld+json"]')) {
                    try {
                        const j = JSON.parse(s.textContent || '{}');
                        const items = Array.isArray(j) ? j : (j['@graph'] || [j]);
                        for (const it of items) {
                            if (it && /JobPosting/i.test(it['@type'] || '')) {
                                title = title || clean(it.title || it.name);
                                if (it.hiringOrganization) {
                                    company = company || clean(it.hiringOrganization.name || '');
                                }
                                if (it.description) {
                                    const tmp = document.createElement('div');
                                    tmp.innerHTML = it.description;
                                    desc = desc || clean(tmp.textContent);
                                }
                            }
                        }
                    } catch (e) {}
                }
                // 2. Headings / meta fallback
                if (!title) {
                    const h1 = document.querySelector('h1, [data-automation-id="jobPostingHeader"], .app-title, .posting-headline h2');
                    if (h1) title = clean(h1.textContent);
                }
                if (!company) {
                    const ogSite = document.querySelector('meta[property="og:site_name"]');
                    if (ogSite) company = clean(ogSite.getAttribute('content'));
                    if (!company) {
                        const m = document.title.match(/(?:at|@|-|\\|)\\s*([^-|]+)$/);
                        if (m) company = clean(m[1]);
                    }
                }
                if (!desc) {
                    const main = document.querySelector('[data-automation-id="jobPostingDescription"], #content, main, article, .posting, .job-description, body');
                    if (main) desc = clean(main.innerText || main.textContent);
                }
                return { title: title.slice(0, 160), company: company.slice(0, 120), description: desc.slice(0, 3000) };
            }"""
        )
        return {
            "title": raw.get("title", ""),
            "company": raw.get("company", ""),
            "description": clip_text(raw.get("description", ""), 2200),
        }
    except Exception:  # noqa: BLE001
        return {"title": "", "company": "", "description": ""}


async def dismiss_banners(page: Page) -> None:
    """Best-effort dismissal of cookie/consent overlays that block clicks."""
    selectors = [
        "#onetrust-accept-btn-handler",
        "#CybotCookiebotDialogBodyLevelButtonLevelOptinAllowAll",
        'button[data-automation-id="legalNoticeAcceptButton"]',
        'button:has-text("Accept All Cookies")',
        'button:has-text("Accept Cookies")',
        'button:has-text("Accept all")',
    ]
    for sel in selectors:
        try:
            loc = page.locator(sel).first
            if await loc.is_visible(timeout=200):
                await loc.click(timeout=1000)
                await page.wait_for_timeout(200)
        except Exception:  # noqa: BLE001
            continue


def _is_application_form(fields: list[Field]) -> bool:
    """True when scanned fields look like an actual job application (or auth gate) rather than a search bar."""
    if len(fields) >= 3:
        return True
    for f in fields:
        if f.kind == "file":
            return True
        if re.search(r"first\s*name|last\s*name|full\s*name|email|resume|password", f.label, re.I):
            return True
    return False


async def enter_application(page: Page, dom: Dom, profile: Profile) -> Page:
    """Navigate from a job description landing page into the application form if needed."""
    await dismiss_banners(page)

    # Scroll slightly to trigger lazy-loaded embedded iframes (e.g. Greenhouse #grnhse_iframe)
    try:
        await page.evaluate("() => { const f = document.querySelector('#grnhse_iframe, iframe[src*=\"greenhouse\"], iframe[src*=\"icims\"]'); if (f) f.scrollIntoView({block:'center'}); }")
    except Exception:  # noqa: BLE001
        pass

    fields = await dom.scan()
    if _is_application_form(fields):
        return page

    # Try clicking Apply button / tab (potentially opening a new tab or Workday modal)
    apply_selectors = [
        'a[data-automation-id="adventureButton"]',
        'button[data-automation-id="adventureButton"]',
        'a.postings-btn[href*="/apply"]',
        'button:has-text("Apply for this Job")',
        'a:has-text("Apply for this Job")',
        'button:has-text("Apply for this job")',
        'a:has-text("Apply for this job")',
        'button:has-text("Apply Now")',
        'a:has-text("Apply Now")',
        'button:has-text("I\'m interested")',
        'a:has-text("I\'m interested")',
        '[role="tab"]:has-text("Application")',
        'a:has-text("Application")',
        'button:has-text("Apply")',
        'a:has-text("Apply")',
    ]

    ctx = page.context
    before_pages = list(ctx.pages)

    for sel in apply_selectors:
        try:
            loc = page.locator(sel).first
            if not await loc.is_visible(timeout=300):
                continue
            await loc.scroll_into_view_if_needed(timeout=1500)
            await loc.click(timeout=3000)
            await page.wait_for_timeout(1200)
            break
        except Exception:  # noqa: BLE001
            continue

    # If a new tab opened, switch to it
    after_pages = list(ctx.pages)
    if len(after_pages) > len(before_pages):
        new_page = after_pages[-1]
        try:
            await new_page.wait_for_load_state("domcontentloaded", timeout=10000)
        except Exception:  # noqa: BLE001
            pass
        page = new_page
        dom.page = page

    # Handle Workday's "Start Your Application" modal (Apply Manually / Autofill with Resume)
    await _handle_workday_entry_modal(page, profile)
    await page.wait_for_timeout(800)
    return page


async def _handle_workday_entry_modal(page: Page, profile: Profile) -> None:
    """Click through Workday's 'Apply Manually' or 'Autofill with Resume' modal if visible."""
    manual_sels = [
        '[data-automation-id="applyManually"]',
        'a:has-text("Apply Manually")',
        'button:has-text("Apply Manually")',
        '[data-automation-id="autofillWithResume"]',
        'button:has-text("Autofill with Resume")',
    ]
    for sel in manual_sels:
        try:
            loc = page.locator(sel).first
            if await loc.is_visible(timeout=500):
                await loc.click(timeout=2500)
                await page.wait_for_timeout(1500)
                return
        except Exception:  # noqa: BLE001
            continue


def is_auth_gate(fields: list[Field]) -> bool:
    """Return True if the current page is a Login / Create Account wall (e.g. Workday, iCIMS)."""
    has_pw = any(f.itype == "password" or "password" in f.label.lower() for f in fields)
    non_auth = [
        f for f in fields
        if f.itype != "password" and not re.search(r"email|user\s*name|password|confirm|verify|remember|terms|consent|create|sign", f.label, re.I)
    ]
    return has_pw and len(non_auth) <= 1


async def handle_auth_gate(page: Page, dom: Dom, fields: list[Field], profile: Profile) -> bool:
    """Attempt to sign in or create an account if credentials are in profile. Returns True if handled."""
    email = profile.get("accounts.email") or profile.get("personal.email")
    password = profile.get("accounts.password")
    if not email or not password:
        return False

    for f in fields:
        if f.itype == "password" or "password" in f.label.lower():
            await dom.fill(f, password)
        elif re.search(r"email|user\s*name", f.label, re.I):
            await dom.fill(f, email)
        elif f.kind == "checkbox" and f.required:
            await dom.fill(f, True)

    # Click Sign In or Create Account submit button
    btn_sels = [
        '[data-automation-id="click_filter"]',
        'button[data-automation-id="signInSubmitButton"]',
        'button[data-automation-id="createAccountSubmitButton"]',
        'button[type="submit"]',
        'button:has-text("Sign In")',
        'button:has-text("Create Account")',
        'button:has-text("Log In")',
    ]
    for sel in btn_sels:
        try:
            b = page.locator(sel).first
            if await b.is_visible(timeout=400):
                await b.click(timeout=3000)
                await page.wait_for_timeout(2000)
                return True
        except Exception:  # noqa: BLE001
            continue
    return False


async def expand_workday_sections(page: Page, profile: Profile) -> bool:
    """On Workday 'My Experience' steps, click 'Add' on empty Education / Work Experience sections."""
    try:
        clicked = await page.evaluate(
            """(opts) => {
                let didClick = false;
                const sections = document.querySelectorAll('[data-automation-id="workExperienceSection"], [data-automation-id="educationSection"]');
                for (const sec of sections) {
                    const id = sec.getAttribute('data-automation-id') || '';
                    if (id.includes('workExperience') && !opts.hasJob) continue;
                    if (id.includes('education') && !opts.hasEdu) continue;
                    const inputs = sec.querySelectorAll('input, textarea, button[aria-haspopup="listbox"]');
                    if (inputs.length === 0) {
                        const addBtn = sec.querySelector('[data-automation-id="add-button"], button');
                        if (addBtn) { addBtn.click(); didClick = true; }
                    }
                }
                return didClick;
            }""",
            {"hasJob": bool(profile.job), "hasEdu": bool(profile.edu)},
        )
        if clicked:
            await page.wait_for_timeout(800)
        return bool(clicked)
    except Exception:  # noqa: BLE001
        return False


async def detect_step_name(page: Page) -> str:
    try:
        return await page.evaluate(
            """() => {
                const clean = (s) => (s || '').replace(/\\s+/g, ' ').trim();
                const step = document.querySelector('[data-automation-id="progressBarActiveStep"], [aria-current="step"], .step.active, .wizard-step.active');
                if (step) return clean(step.textContent).slice(0, 60);
                const h = document.querySelector('form h2, form h3, [data-automation-id="formHeader"] h2, [data-automation-id="formHeader"]');
                if (h) return clean(h.textContent).slice(0, 60);
                return '';
            }"""
        )
    except Exception:  # noqa: BLE001
        return ""


# ---------------------------------------------------------------------- navigation & submission buttons
JS_FIND_NAV_BUTTONS = r"""
() => {
  const raw = (s) => (s || '').replace(/[\u00a0\s]+/g, ' ').trim();
  const roots = [document];
  for (let i = 0; i < roots.length; i++) for (const el of roots[i].querySelectorAll('*')) if (el.shadowRoot) roots.push(el.shadowRoot);
  const cands = [];
  for (const r of roots) {
    for (const el of r.querySelectorAll('button, input[type="submit"], input[type="button"], a[role="button"]')) {
      const rect = el.getBoundingClientRect();
      if (rect.width < 5 || rect.height < 5) continue;
      const st = getComputedStyle(el);
      if (st.display === 'none' || st.visibility === 'hidden' || el.disabled) continue;
      const txt = raw(el.value || el.textContent || el.getAttribute('aria-label') || '');
      const aid = el.getAttribute('data-automation-id') || '';
      if (!txt && !aid) continue;
      let kind = '';
      if (
        aid === 'bottom-navigation-next-button' && /^(submit|submit application)$/i.test(txt) ||
        /^(submit(\s+application)?|send\s+application|finish(\s+and\s+submit)?|apply\s+now|complete\s+application)$/i.test(txt)
      ) {
        kind = 'submit';
      } else if (
        aid === 'bottom-navigation-next-button' ||
        /^(next|continue|save\s*(&|and)\s*continue|proceed|review|next\s+step|continue\s+application)$/i.test(txt)
      ) {
        kind = 'next';
      } else if (el.type === 'submit' && /apply|submit/i.test(txt) && !/search|filter|sign|log/i.test(txt)) {
        kind = 'submit';
      }
      if (kind) {
        el.setAttribute('data-aa-nav', kind);
        cands.push({ kind, text: txt });
      }
    }
  }
  return cands;
}
"""


async def find_nav_action(page: Page) -> tuple[str, Frame | None, str]:
    """Return (kind, frame, button_text) where kind is 'next', 'submit', or ''."""
    frames = [page.main_frame] + [f for f in page.frames if f != page.main_frame and not f.is_detached()]
    # Prefer 'next' over 'submit' if both exist on a wizard step
    found_submit: tuple[Frame, str] | None = None
    for frame in frames:
        try:
            btns = await frame.evaluate(JS_FIND_NAV_BUTTONS)
        except Exception:  # noqa: BLE001
            continue
        for b in btns:
            if b["kind"] == "next":
                return "next", frame, b["text"]
            if b["kind"] == "submit" and not found_submit:
                found_submit = (frame, b["text"])
    if found_submit:
        return "submit", found_submit[0], found_submit[1]
    return "", None, ""


async def click_nav_action(frame: Frame, kind: str) -> bool:
    try:
        loc = frame.locator(f'[data-aa-nav="{kind}"]').first
        await loc.scroll_into_view_if_needed(timeout=2000)
        await loc.click(timeout=4000)
        return True
    except Exception:  # noqa: BLE001
        return False


async def read_page_errors(page: Page) -> list[str]:
    """Read visible validation/error messages after an attempted step transition or submit."""
    out: list[str] = []
    for frame in page.frames:
        if frame.is_detached():
            continue
        try:
            errs = await frame.evaluate(
                """() => {
                    const raw = (s) => (s || '').replace(/\\s+/g, ' ').trim();
                    const sels = '[role="alert"], [aria-invalid="true"], [data-automation-id="errorMessage"], .error-message, .field-error, [class*="errorMessage"], [class*="field-error"]';
                    const seen = new Set();
                    const res = [];
                    for (const el of document.querySelectorAll(sels)) {
                        const r = el.getBoundingClientRect();
                        if (r.width < 1 || r.height < 1) continue;
                        const t = raw(el.textContent);
                        if (t && t.length > 2 && t.length < 240 && !seen.has(t)) {
                            seen.add(t);
                            res.push(t);
                        }
                    }
                    return res.slice(0, 8);
                }"""
            )
            out.extend(errs)
        except Exception:  # noqa: BLE001
            continue
    return out[:10]
