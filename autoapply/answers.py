"""Answer engine: memory -> deterministic rules -> token-bounded LLM fallback."""
from __future__ import annotations

import datetime as dt
import json
import re
from pathlib import Path
from typing import Any

from rapidfuzz import fuzz, process

from .fields import CHOICE_KINDS, Field
from .llm import LLM, TokenBudgetExceeded, clip_text
from .matching import DECLINE_RE, US_STATES, best_options, is_placeholder, match_bool, match_option, norm
from .profile import Profile

MONTH_NAMES = [
    "", "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]


class Memory:
    """Persistent question -> answer store so user answers are reused across applications."""

    def __init__(self, path: Path | None = None):
        self.path = path or (Path.home() / ".autoapply" / "memory.json")
        self.entries: dict[str, Any] = {}
        self.load()

    def load(self) -> None:
        if self.path.exists():
            try:
                self.entries = json.loads(self.path.read_text())
            except Exception:  # noqa: BLE001
                self.entries = {}

    def save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.entries, indent=2, ensure_ascii=False))
        except Exception:  # noqa: BLE001
            pass

    @staticmethod
    def _key(label: str, kind: str) -> str:
        bucket = "choice" if kind in CHOICE_KINDS else ("check" if kind == "checkbox" else "text")
        return f"{bucket}::{norm(label)}"

    def lookup(self, f: Field) -> Any:
        if not f.label or f.kind == "file":
            return None
        exact = self._key(f.label, f.kind)
        if exact in self.entries:
            return self._validate(f, self.entries[exact])
        # Fuzzy match within same bucket for long question labels
        nl = norm(f.label)
        if len(nl) < 12:
            return None
        bucket_prefix = exact.split("::", 1)[0] + "::"
        candidates = {k[len(bucket_prefix):]: v for k, v in self.entries.items() if k.startswith(bucket_prefix)}
        if not candidates:
            return None
        hit = process.extractOne(nl, list(candidates.keys()), scorer=fuzz.token_sort_ratio)
        if hit and hit[1] >= 92:
            return self._validate(f, candidates[hit[0]])
        return None

    def remember(self, f: Field, value: Any) -> None:
        if not f.label or f.kind == "file" or value in (None, "", []):
            return
        self.entries[self._key(f.label, f.kind)] = value
        self.save()

    @staticmethod
    def _validate(f: Field, val: Any) -> Any:
        if f.is_choice and f.options:
            if isinstance(val, list):
                idxs = best_options([str(x) for x in val], f.option_texts)
                return [f.options[i].text for i in idxs] if idxs else None
            idx = match_option(val, f.option_texts)
            return f.options[idx].text if idx is not None else None
        return val


# ---------------------------------------------------------------------- helpers
def _sig(f: Field) -> str:
    """Combined normalized signature of label, section, name, id, autocomplete, placeholder."""
    parts = [f.label, f.section, f.name, f.elem_id, f.autocomplete, f.placeholder]
    return norm(" | ".join(p for p in parts if p))


def _label_sig(f: Field) -> str:
    return norm(f.label or f.placeholder or f.name or f.elem_id)


def _split_ym(ym: str) -> tuple[str, str]:
    """Split 'YYYY-MM' or 'YYYY' into (year, month_num_str)."""
    m = re.match(r"^\s*(\d{4})(?:[-/](\d{1,2}))?", str(ym or ""))
    if not m:
        return "", ""
    return m.group(1), (f"{int(m.group(2)):02d}" if m.group(2) else "")


def _format_date_for_field(ym_or_date: str, f: Field, default_month: str = "05") -> str:
    if not ym_or_date:
        return ""
    if str(ym_or_date).lower() in ("present", "current", "now"):
        today = dt.date.today()
        y, m, d = str(today.year), f"{today.month:02d}", f"{today.day:02d}"
    else:
        m_full = re.match(r"^\s*(\d{4})-(\d{1,2})-(\d{1,2})", str(ym_or_date))
        if m_full:
            y, m, d = m_full.group(1), f"{int(m_full.group(2)):02d}", f"{int(m_full.group(3)):02d}"
        else:
            y, m = _split_ym(ym_or_date)
            if not y:
                return str(ym_or_date)
            m = m or default_month
            d = "01"
    ph = (f.placeholder or "").upper()
    if f.itype == "date" or "YYYY-MM-DD" in ph:
        return f"{y}-{m}-{d}"
    if f.itype == "month" or "YYYY-MM" in ph:
        return f"{y}-{m}"
    if "MM/YYYY" in ph:
        return f"{m}/{y}"
    if "MM/DD/YYYY" in ph:
        return f"{m}/{d}/{y}"
    return f"{m}/{d}/{y}" if "date" in _label_sig(f) else f"{y}-{m}"


def _match_numeric_range(num: float, options: list[str]) -> str | None:
    """Match a number (e.g. years of experience = 1, GPA = 3.8) against range options like '0-1 years', '1-3', '5+'."""
    for opt in options:
        if is_placeholder(opt):
            continue
        low = opt.lower()
        if num == 0 and re.search(r"\b(none|0|no experience|less than 1|< ?1|entry|new grad)\b", low):
            return opt
        m_range = re.search(r"(\d+(?:\.\d+)?)\s*(?:[-–—]|to)\s*(\d+(?:\.\d+)?)", low)
        if m_range:
            lo, hi = float(m_range.group(1)), float(m_range.group(2))
            if lo <= num <= hi:
                return opt
        m_plus = re.search(r"(\d+(?:\.\d+)?)\s*\+|(?:more than|over|at least|>=?)\s*(\d+(?:\.\d+)?)", low)
        if m_plus:
            thresh = float(m_plus.group(1) or m_plus.group(2))
            if num >= thresh:
                return opt
        m_under = re.search(r"(?:less than|under|fewer than|<)\s*(\d+(?:\.\d+)?)", low)
        if m_under and num < float(m_under.group(1)):
            return opt
        m_exact = re.match(r"^\s*(\d+)\b", low)
        if m_exact and int(num) == int(m_exact.group(1)):
            return opt
    return None


def _resolve_for_field(f: Field, raw_val: Any) -> Any:
    """Coerce `raw_val` to fit `f` (matching dropdown/radio options when `f.options` is populated)."""
    if raw_val in (None, "", []):
        return None
    if f.kind == "checkbox":
        if isinstance(raw_val, bool):
            return raw_val
        return str(raw_val).lower() in ("true", "yes", "y", "1", "checked")
    if f.is_choice and f.options:
        texts = f.option_texts
        if isinstance(raw_val, list):
            idxs = best_options([str(x) for x in raw_val], texts)
            return [f.options[i].text for i in idxs] if idxs else None
        idx = match_option(raw_val, texts)
        if idx is not None:
            return f.options[idx].text
        # Try numeric range matching if raw_val looks numeric
        m_num = re.search(r"\d+(?:\.\d+)?", str(raw_val))
        if m_num:
            hit = _match_numeric_range(float(m_num.group(0)), texts)
            if hit is not None:
                return hit
        return None
    if isinstance(raw_val, bool):
        return "Yes" if raw_val else "No"
    return str(raw_val)


# ---------------------------------------------------------------------- rules
def rule_answer(f: Field, profile: Profile, job_ctx: dict | None = None) -> tuple[Any, str, str]:
    """Return (value, source, rule_name) or (None, '', '') if no deterministic rule matches."""
    job_ctx = job_ctx or {}
    lbl = _label_sig(f)
    sig = _sig(f)
    sec = norm(f.section)
    ac = (f.autocomplete or "").lower().strip()

    # 1. File uploads
    if f.kind == "file":
        if re.search(r"\b(cover\s*letter|motivational)\b", sig):
            if profile.cover_letter_path and Path(profile.cover_letter_path).exists():
                return profile.cover_letter_path, "profile", "file.cover_letter"
            return None, "skip", "file.cover_letter_optional"
        if re.search(r"\b(transcript|grade|portfolio|writing sample|work sample|additional)\b", sig) and not re.search(r"\b(resume|cv)\b", sig):
            return None, "skip", "file.other_optional"
        if profile.resume_path:
            return profile.resume_path, "profile", "file.resume"
        return None, "", ""

    # 2. HTML autocomplete attribute shortcuts
    if ac:
        ac_map = {
            "given-name": (profile.first_name, "personal.first_name"),
            "family-name": (profile.last_name, "personal.last_name"),
            "additional-name": (profile.get("personal.middle_name"), "personal.middle_name"),
            "name": (profile.full_name, "personal.full_name"),
            "email": (profile.get("personal.email"), "personal.email"),
            "tel": (profile.get("personal.phone"), "personal.phone"),
            "tel-national": (profile.phone_national(), "personal.phone_national"),
            "street-address": (profile.get("personal.address.street"), "address.street"),
            "address-line1": (profile.get("personal.address.street"), "address.street"),
            "address-line2": (profile.get("personal.address.street2"), "address.street2"),
            "address-level2": (profile.get("personal.address.city"), "address.city"),
            "address-level1": (profile.get("personal.address.state"), "address.state"),
            "postal-code": (profile.get("personal.address.zip"), "address.zip"),
            "country": (profile.get("personal.address.country"), "address.country"),
            "country-name": (profile.get("personal.address.country"), "address.country"),
            "organization": (profile.job.get("company", ""), "experience.company"),
            "organization-title": (profile.job.get("title", ""), "experience.title"),
        }
        if ac in ac_map and ac_map[ac][0]:
            v = _resolve_for_field(f, ac_map[ac][0])
            if v is not None:
                return v, "profile", ac_map[ac][1]

    # 3. EEO / Demographic questions (check before generic patterns)
    if re.search(r"\b(hispanic|latino|latina|latinx)\b", lbl):
        ans = profile.get("eeo.hispanic_latino", "decline")
        return _resolve_eeo(f, ans), "profile", "eeo.hispanic_latino"
    if re.search(r"\b(race|ethnic|ethnicity|racial)\b", lbl):
        ans = profile.get("eeo.race", "decline")
        return _resolve_eeo(f, ans), "profile", "eeo.race"
    if re.search(r"\b(transgender)\b", lbl):
        ans = profile.get("eeo.transgender", "decline")
        return _resolve_eeo(f, ans), "profile", "eeo.transgender"
    if re.search(r"\b(sexual orientation|lgbt|lgbtq)\b", lbl):
        ans = profile.get("eeo.sexual_orientation", "decline")
        return _resolve_eeo(f, ans), "profile", "eeo.sexual_orientation"
    if re.search(r"\b(gender|sex)\b", lbl) and not re.search(r"\b(offender)\b", lbl):
        ans = profile.get("eeo.gender", "decline")
        return _resolve_eeo(f, ans), "profile", "eeo.gender"
    if re.search(r"\b(veteran|military|armed forces|protected veteran)\b", lbl):
        ans = profile.get("eeo.veteran", "decline")
        return _resolve_eeo(f, ans), "profile", "eeo.veteran"
    if re.search(r"\b(disability|disabled|handicap)\b", lbl) and not re.search(r"\b(accommodat)\b", lbl):
        ans = profile.get("eeo.disability", "decline")
        return _resolve_eeo(f, ans), "profile", "eeo.disability"
    if re.search(r"\b(pronoun)\b", lbl):
        p = profile.get("personal.pronouns")
        if p:
            return _resolve_for_field(f, p), "profile", "personal.pronouns"
        if f.is_choice:
            return _resolve_eeo(f, "decline"), "default", "personal.pronouns_decline"

    # 4. Personal identity & contact
    if re.search(r"\b(preferred|chosen|nick)\s*name\b", lbl):
        val = profile.get("personal.preferred_name") or profile.first_name
        return _resolve_for_field(f, val), "profile", "personal.preferred_name"
    if re.search(r"\b(first|given|forename)\s*name\b|\bfname\b", sig):
        return _resolve_for_field(f, profile.first_name), "profile", "personal.first_name"
    if re.search(r"\b(last|family|sur)\s*name\b|\blname\b", sig):
        return _resolve_for_field(f, profile.last_name), "profile", "personal.last_name"
    if re.search(r"\b(middle)\s*(name|initial)\b", sig):
        m = profile.get("personal.middle_name", "")
        if "initial" in lbl and m:
            m = m[:1]
        return (_resolve_for_field(f, m), "profile", "personal.middle_name") if m else (None, "skip", "personal.middle_name")
    if re.search(r"\b(signature|electronic signature|type your (full )?name|sign here|printed name)\b", lbl) or (
        lbl == "name" and re.search(r"disability|voluntary|self.?identify|acknowledge|certify", sec)
    ):
        return _resolve_for_field(f, profile.full_name), "profile", "personal.signature"
    if re.match(r"^(full\s*name|your\s*name|legal\s*name|applicant\s*name|candidate\s*name|name)$", lbl) or (
        "_systemfield_name" in sig
    ):
        return _resolve_for_field(f, profile.full_name), "profile", "personal.full_name"
    if re.search(r"\b(e-?mail)\b", sig) and f.kind == "text":
        return _resolve_for_field(f, profile.get("personal.email")), "profile", "personal.email"
    if re.search(r"\b(country\s*code|phone\s*code|dial\s*code)\b", lbl) or (
        re.search(r"\bphone\b", lbl) and f.is_choice and any("+1" in o or "united states" in o.lower() for o in f.option_texts)
    ):
        cc = profile.get("personal.phone_country_code", "+1")
        v = _resolve_for_field(f, cc) or _resolve_for_field(f, profile.get("personal.address.country", "United States"))
        return v, "profile", "personal.phone_country_code"
    if re.search(r"\b(phone|device)\s*type\b", lbl):
        return _resolve_for_field(f, profile.get("personal.phone_type", "Mobile")), "profile", "personal.phone_type"
    if re.search(r"\b(phone|mobile|cell|telephone|contact\s*number)\b", sig) and f.kind == "text" and not re.search(r"\b(ext|extension)\b", lbl):
        ph = profile.phone_national() if ("national" in sig or f.itype == "tel") else profile.get("personal.phone")
        return _resolve_for_field(f, ph or profile.get("personal.phone")), "profile", "personal.phone"
    if re.search(r"\b(phone\s*extension|extension)\b", lbl):
        return None, "skip", "personal.phone_ext"

    # 5. Links & URLs
    if re.search(r"\blinkedin\b", sig):
        return _resolve_for_field(f, profile.get("links.linkedin")), "profile", "links.linkedin"
    if re.search(r"\bgithub\b", sig):
        return _resolve_for_field(f, profile.get("links.github")), "profile", "links.github"
    if re.search(r"\b(twitter|x\.com)\b", sig):
        return _resolve_for_field(f, profile.get("links.twitter")), "profile", "links.twitter"
    if re.search(r"\b(portfolio|personal\s*(web)?site|website|blog|url|homepage|other\s*website)\b", sig):
        url = (
            profile.get("links.portfolio")
            or profile.get("links.github")
            or profile.get("links.linkedin")
            or profile.get("links.other")
        )
        return (_resolve_for_field(f, url), "profile", "links.portfolio") if url else (None, "skip", "links.portfolio")

    # 6. Address & Location
    if re.search(r"\b(street|address\s*line\s*1|address\s*1|mailing\s*address|home\s*address|permanent\s*address)\b", lbl) or lbl == "address":
        st = profile.get("personal.address.street") or profile.location
        return _resolve_for_field(f, st), "profile", "address.street"
    if re.search(r"\b(address\s*line\s*2|address\s*2|apt|apartment|suite|unit)\b", lbl):
        st2 = profile.get("personal.address.street2")
        return (_resolve_for_field(f, st2), "profile", "address.street2") if st2 else (None, "skip", "address.street2")
    if re.match(r"^(city|town|municipality)$", lbl) or re.search(r"\b(current\s*city|city\s*name)\b", lbl):
        city = profile.get("personal.address.city")
        if not city and profile.location:
            city = profile.location.split(",")[0].strip()
        if f.kind in ("combobox", "listbox_button") and f.typeahead and profile.location:
            return profile.location, "profile", "address.location"
        return _resolve_for_field(f, city), "profile", "address.city"
    if re.match(r"^(state|province|state\s*/\s*province|region|state/region)$", lbl):
        st = profile.get("personal.address.state")
        if not st and "," in profile.location:
            st = profile.location.split(",")[1].strip()
        return _resolve_for_field(f, st), "profile", "address.state"
    if re.search(r"\b(zip|postal\s*code|postcode|zip\s*code)\b", lbl):
        return _resolve_for_field(f, profile.get("personal.address.zip")), "profile", "address.zip"
    if re.match(r"^(country|country/region|country\s*of\s*residence)$", lbl):
        return _resolve_for_field(f, profile.get("personal.address.country", "United States")), "profile", "address.country"
    if re.search(r"\b(location|where are you (currently )?(based|located)|current\s*location|city\s*and\s*state)\b", lbl):
        loc = profile.location or profile.get("personal.address.city")
        return _resolve_for_field(f, loc), "profile", "personal.location"

    # 7. Work Authorization & Sponsorship
    if re.search(r"\b(sponsor|sponsorship|visa|h-?1b|f-?1|opt|cpt|work\s*permit|immigration\s*filing)\b", lbl) and re.search(
        r"\b(require|need|will you|do you|future|now)\b", lbl
    ):
        ans = bool(profile.get("work_authorization.require_sponsorship", False))
        return _resolve_for_field(f, ans), "profile", "work_auth.require_sponsorship"
    if re.search(r"\b(legally\s*authorized|authorized\s*to\s*work|eligible\s*to\s*work|right\s*to\s*work|work\s*authorization|legal\s*right\s*to\s*work)\b", lbl):
        ans = bool(profile.get("work_authorization.authorized_us", True))
        return _resolve_for_field(f, ans), "profile", "work_auth.authorized"
    if re.search(r"\b(citizenship|citizen\s*status|visa\s*status|work\s*authorization\s*status|immigration\s*status)\b", lbl):
        cit = profile.get("work_authorization.citizenship")
        if not cit and profile.get("work_authorization.authorized_us", True) and not profile.get("work_authorization.require_sponsorship", False):
            cit = "US Citizen"
        return _resolve_for_field(f, cit), "profile", "work_auth.citizenship"
    if re.search(r"\b(security\s*clearance|clearance\s*level|active\s*clearance|government\s*clearance)\b", lbl):
        clr = profile.get("work_authorization.security_clearance", "None")
        if f.is_choice and match_bool(False, f.option_texts) is not None and clr.lower() in ("none", "no", "false", ""):
            return _resolve_for_field(f, False), "profile", "work_auth.clearance"
        return _resolve_for_field(f, clr), "profile", "work_auth.clearance"

    # 8. Education
    edu = profile.edu
    if edu:
        if re.search(r"\b(school|university|college|institution|alma\s*mater)\b", lbl) and not re.search(r"\bhigh\s*school\b", lbl):
            return _resolve_for_field(f, edu.get("school", "")), "profile", "education.school"
        if re.search(r"\b(degree|highest\s*level\s*of\s*education|education\s*level|level\s*of\s*education)\b", lbl):
            return _resolve_for_field(f, edu.get("degree", "")), "profile", "education.degree"
        if re.search(r"\b(major|field\s*of\s*study|discipline|area\s*of\s*study|concentration)\b", lbl):
            return _resolve_for_field(f, edu.get("major", "")), "profile", "education.major"
        if re.search(r"\b(minor)\b", lbl) and "age" not in lbl and "18" not in lbl:
            m = edu.get("minor", "")
            return (_resolve_for_field(f, m), "profile", "education.minor") if m else (None, "skip", "education.minor")
        if re.search(r"\b(gpa|grade\s*point\s*average|cumulative\s*gpa)\b", lbl):
            gpa = edu.get("gpa", "")
            return (_resolve_for_field(f, gpa), "profile", "education.gpa") if gpa else (None, "skip", "education.gpa")
        if re.search(r"\b(graduation|grad\s*date|expected\s*graduation|completion\s*date|degree\s*end|end\s*date)\b", lbl) and (
            "education" in sec or "school" in sec or "grad" in lbl or "degree" in lbl
        ):
            y, m = _split_ym(edu.get("end", ""))
            if "year" in lbl and "month" not in lbl:
                return _resolve_for_field(f, y), "profile", "education.end_year"
            if "month" in lbl and "year" not in lbl and m:
                return _resolve_for_field(f, MONTH_NAMES[int(m)] if f.is_choice else m), "profile", "education.end_month"
            return _resolve_for_field(f, _format_date_for_field(edu.get("end", ""), f)), "profile", "education.end"
        if re.search(r"\b(start\s*date|start\s*year|start\s*month|from\s*date)\b", lbl) and ("education" in sec or "school" in sec):
            y, m = _split_ym(edu.get("start", ""))
            if "year" in lbl and "month" not in lbl:
                return _resolve_for_field(f, y), "profile", "education.start_year"
            if "month" in lbl and "year" not in lbl and m:
                return _resolve_for_field(f, MONTH_NAMES[int(m)] if f.is_choice else m), "profile", "education.start_month"
            return _resolve_for_field(f, _format_date_for_field(edu.get("start", ""), f, default_month="09")), "profile", "education.start"
        if re.search(r"\b(currently\s*enrolled|current\s*student|are\s*you\s*a\s*student|attending\s*school)\b", lbl):
            y, _ = _split_ym(edu.get("end", ""))
            enrolled = bool(y and int(y) >= dt.date.today().year)
            return _resolve_for_field(f, enrolled), "profile", "education.currently_enrolled"

    # 9. Experience
    job = profile.job
    if re.search(r"\b(current\s*company|current\s*employer|most\s*recent\s*(company|employer)|company\s*name|employer\s*name|organization)\b", lbl):
        comp = job.get("company") or (profile.edu.get("school") if profile.edu else "N/A")
        return _resolve_for_field(f, comp), "profile", "experience.company"
    if re.search(r"\b(current\s*(job\s*)?title|most\s*recent\s*(job\s*)?title|job\s*title|position\s*title|role\s*title)\b", lbl):
        title = job.get("title") or ("Student" if profile.edu else "")
        return _resolve_for_field(f, title), "profile", "experience.title"
    if re.search(r"\b(years?\s*of\s*(.*?\s*)?experience|total\s*experience|how\s*many\s*years)\b", lbl):
        yoe = str(profile.get("preferences.years_experience") or "0")
        return _resolve_for_field(f, yoe), "profile", "preferences.years_experience"
    if re.search(r"\b(currently\s*work\s*here|current\s*job|i\s*currently\s*work)\b", lbl):
        cur = str(job.get("end", "")).lower() in ("present", "current", "now")
        return _resolve_for_field(f, cur), "profile", "experience.current"

    # 10. Preferences, Logistics & Legal
    if re.search(r"\b(relocat|willing\s*to\s*move|open\s*to\s*relocation)\b", lbl):
        return _resolve_for_field(f, bool(profile.get("preferences.willing_to_relocate", True))), "profile", "preferences.relocate"
    if re.search(r"\b(commute|work\s*on-?site|in-?office|hybrid|days?\s*in\s*office|able\s*to\s*work\s*from)\b", lbl) and (
        f.kind == "checkbox" or match_bool(True, f.option_texts) is not None
    ):
        return _resolve_for_field(f, bool(profile.get("preferences.onsite_ok", True))), "profile", "preferences.onsite_ok"
    if re.search(r"\b(remote|work\s*remotely)\b", lbl) and (f.kind == "checkbox" or match_bool(True, f.option_texts) is not None):
        return _resolve_for_field(f, bool(profile.get("preferences.remote_ok", True))), "profile", "preferences.remote_ok"
    if re.search(r"\b(start\s*date|available\s*to\s*start|when\s*can\s*you\s*start|earliest\s*start|availability\s*date|date\s*available)\b", lbl):
        sd = profile.get("preferences.start_date") or (
            "Immediately" if f.kind in ("text", "textarea") and f.itype != "date" else (dt.date.today() + dt.timedelta(days=14)).isoformat()
        )
        if f.itype == "date" and not re.match(r"^\d{4}-\d{2}-\d{2}$", str(sd)):
            sd = (dt.date.today() + dt.timedelta(days=14)).isoformat()
        return _resolve_for_field(f, sd), "profile", "preferences.start_date"
    if re.search(r"\b(notice\s*period|how\s*much\s*notice)\b", lbl):
        return _resolve_for_field(f, profile.get("preferences.notice_period", "2 weeks")), "profile", "preferences.notice_period"
    if re.search(r"\b(salary|compensation|pay\s*expectation|desired\s*pay|expected\s*compensation|base\s*salary|hourly\s*rate)\b", lbl):
        sal = profile.get("preferences.salary_expectation", "Open / negotiable")
        if f.itype == "number":
            m_num = re.search(r"\d+", str(sal).replace(",", ""))
            sal = m_num.group(0) if m_num else "0"
        return _resolve_for_field(f, sal), "profile", "preferences.salary"
    if re.search(r"\b(sms|text\s*messag)\b", lbl):
        return _resolve_for_field(f, bool(profile.get("preferences.sms_opt_in", False))), "profile", "preferences.sms_opt_in"
    if re.search(r"\b(18\s*years|over\s*18|at\s*least\s*18|legal\s*working\s*age|of\s*legal\s*age)\b", lbl):
        return _resolve_for_field(f, bool(profile.get("misc.over_18", True))), "profile", "misc.over_18"
    if re.search(r"\b(background\s*check|criminal\s*history\s*check|pre-?employment\s*screening)\b", lbl):
        return _resolve_for_field(f, bool(profile.get("misc.background_check_ok", True))), "profile", "misc.background_check"
    if re.search(r"\b(drug\s*(test|screen))\b", lbl):
        return _resolve_for_field(f, bool(profile.get("misc.drug_test_ok", True))), "profile", "misc.drug_test"
    if re.search(r"\b(previously\s*(been\s*)?(employed|worked)|worked\s*for\s*us\s*before|former\s*employee|ever\s*worked\s*(at|for))\b", lbl):
        return _resolve_for_field(f, bool(profile.get("misc.previously_employed_here", False))), "profile", "misc.previously_employed"
    if re.search(r"\b(relative|family\s*member|related\s*to\s*anyone|know\s*anyone\s*working)\b", lbl):
        return _resolve_for_field(f, bool(profile.get("misc.relatives_at_company", False))), "profile", "misc.relatives"
    if re.search(r"\b(non-?compete|restrictive\s*covenant|non-?solicitation|bound\s*by\s*any\s*agreement)\b", lbl):
        return _resolve_for_field(f, bool(profile.get("misc.non_compete", False))), "profile", "misc.non_compete"
    if re.search(r"\b(how\s*did\s*you\s*hear|where\s*did\s*you\s*hear|source|referral\s*source|learn\s*about\s*(this|the)\s*(role|job|position|opening))\b", lbl):
        src = profile.get("misc.how_did_you_hear", "Company website")
        v = _resolve_for_field(f, src)
        if v is None and f.options:
            for fallback in ("Job Board", "LinkedIn", "Company Website", "Career Site", "Online", "Other"):
                v = _resolve_for_field(f, fallback)
                if v is not None:
                    break
        return v, "profile", "misc.how_did_you_hear"
    if re.search(r"\b(referr(ed|al)\s*by|who\s*referred\s*you|employee\s*referral)\b", lbl):
        ref = profile.get("misc.referral_name", "")
        return (_resolve_for_field(f, ref), "profile", "misc.referral") if ref else (None, "skip", "misc.referral")
    if re.match(r"^(today.?s\s*date|date|date\s*signed|signature\s*date)$", lbl):
        today = dt.date.today().isoformat()
        return _resolve_for_field(f, _format_date_for_field(today, f)), "default", "misc.today_date"

    # 11. Consent / Privacy / Acknowledgement checkboxes
    if re.search(
        r"\b(privacy\s*(policy|notice|statement)|terms\s*(of\s*use|and\s*conditions|of\s*service)|consent\s*to|acknowledge|certify\s*that|accurate\s*and\s*complete|data\s*protection|gdpr|i\s*agree|i\s*understand)\b",
        lbl,
    ):
        if f.kind == "checkbox" or (f.is_choice and match_bool(True, f.option_texts) is not None):
            return _resolve_for_field(f, True), "default", "misc.consent"

    # 12. Cover letter text field
    if f.kind == "textarea" and re.search(r"\b(cover\s*letter)\b", lbl):
        if not f.required and profile.get("documents.cover_letter") == "none":
            return None, "skip", "documents.cover_letter_none"

    return None, "", ""


def _resolve_eeo(f: Field, desired: str) -> Any:
    opts = f.option_texts
    if not opts:
        return desired if desired != "decline" else "Prefer not to say"
    if desired and desired.lower() != "decline":
        v = _resolve_for_field(f, desired)
        if v is not None:
            return v
    for i, o in enumerate(opts):
        if DECLINE_RE.search(o):
            return [f.options[i].text] if f.multi else f.options[i].text
    return _resolve_for_field(f, desired)


# ---------------------------------------------------------------------- LLM batching & engine
def _compact_profile(profile: Profile) -> dict:
    """Return a compact profile dict with empty strings/lists removed to save prompt tokens."""
    def prune(obj: Any) -> Any:
        if isinstance(obj, dict):
            out = {k: prune(v) for k, v in obj.items() if k not in ("accounts", "eeo")}
            return {k: v for k, v in out.items() if v not in ("", None, [], {})}
        if isinstance(obj, list):
            out_l = [prune(x) for x in obj]
            return [x for x in out_l if x not in ("", None, [], {})]
        return obj

    return prune(profile.summary_for_llm())


class AnswerEngine:
    """Populates `f.value`, `f.source`, `f.rule` for a list of form fields."""

    MAX_FIELDS_PER_BATCH = 8
    MAX_OPTIONS_PER_FIELD = 25
    MAX_JOB_CHARS = 2000
    MAX_RESUME_CHARS = 2400

    def __init__(self, profile: Profile, llm: LLM | None = None, memory: Memory | None = None):
        self.profile = profile
        self.llm = llm
        self.memory = memory or Memory()

    def answer_fields(self, fields: list[Field], job_ctx: dict | None = None) -> None:
        job_ctx = job_ctx or {}
        need_llm: list[Field] = []

        for f in fields:
            # 1) Already prefilled on the page with a non-placeholder value
            if f.current and not is_placeholder(f.current):
                if f.kind != "checkbox" or f.current == "true":
                    f.value = f.current
                    f.source = "prefilled"
                    f.confidence = 1.0
                    continue

            # 2) Memory lookup (exact or high-confidence fuzzy match)
            mem_val = self.memory.lookup(f)
            if mem_val not in (None, "", []):
                f.value = mem_val
                f.source = "memory"
                f.confidence = 0.95
                continue

            # 3) Deterministic rule engine
            val, src, rule_name = rule_answer(f, self.profile, job_ctx)
            if src == "skip":
                f.source = "skip"
                f.rule = rule_name
                continue
            if val not in (None, "", []):
                f.value = val
                f.source = src
                f.rule = rule_name
                f.confidence = 0.95
                continue

            # 4) Queue for LLM if required or meaningful question
            if f.kind != "file" and (f.required or len(_label_sig(f)) > 3):
                need_llm.append(f)
            else:
                f.source = "skip"

        # 5) Batch LLM calls within token limits
        if need_llm and self.llm:
            self._answer_with_llm(need_llm, job_ctx)

        for f in fields:
            if f.value in (None, "", []) and f.source not in ("prefilled", "skip"):
                f.source = "missing" if f.required else "skip"

    def _answer_with_llm(self, fields: list[Field], job_ctx: dict) -> None:
        if not self.llm or not self.llm.can_spend(300):
            return

        compact_prof = json.dumps(_compact_profile(self.profile), ensure_ascii=False)
        resume_snip = clip_text(self.profile.resume_text, self.MAX_RESUME_CHARS)
        job_title = job_ctx.get("title", "")
        job_company = job_ctx.get("company", "")
        job_desc = clip_text(job_ctx.get("description", ""), self.MAX_JOB_CHARS)

        sys = (
            "You fill job application form fields accurately on behalf of the applicant.\n"
            "Rules:\n"
            "1. Never fabricate experience, degrees, employers, or credentials not in the profile/resume.\n"
            "2. For choice fields with options, return the EXACT option text from the provided options list.\n"
            "3. For multi-select fields (`multi: true`), return a JSON array of option strings.\n"
            "4. For checkboxes (`kind: checkbox`), return true or false.\n"
            "5. For open-ended questions ('Why do you want to work here?', 'Tell us about a project'), write a concise, natural, specific 2-4 sentence answer grounded in the applicant's actual resume and the job context.\n"
            "6. If an optional field cannot be answered truthfully from the profile/resume, return null.\n"
            "Respond with a single JSON object mapping each field `uid` to its answer."
        )

        # Split into small batches so neither input nor output exceeds token limits
        for start in range(0, len(fields), self.MAX_FIELDS_PER_BATCH):
            if not self.llm.can_spend(400):
                break
            batch = fields[start : start + self.MAX_FIELDS_PER_BATCH]
            specs = []
            for f in batch:
                spec: dict[str, Any] = {
                    "uid": f.uid,
                    "label": clip_text(f.label, 220),
                    "kind": f.kind,
                    "required": f.required,
                }
                if f.section:
                    spec["section"] = clip_text(f.section, 80)
                if f.description:
                    spec["hint"] = clip_text(f.description, 160)
                if f.multi:
                    spec["multi"] = True
                if f.options:
                    spec["options"] = [clip_text(o.text, 80) for o in f.options[: self.MAX_OPTIONS_PER_FIELD] if o.text]
                specs.append(spec)

            prompt_parts = [f"APPLICANT_PROFILE:\n{compact_prof}"]
            if resume_snip:
                prompt_parts.append(f"RESUME_EXCERPT:\n{resume_snip}")
            if job_title or job_company or job_desc:
                prompt_parts.append(
                    f"JOB_CONTEXT:\nCompany: {job_company}\nTitle: {job_title}\nDescription:\n{job_desc}"
                )
            prompt_parts.append(f"FIELDS_TO_ANSWER:\n{json.dumps(specs, ensure_ascii=False)}")
            prompt = "\n\n".join(prompt_parts)

            # Scale max output tokens with number of fields (essay fields need ~180 tokens each)
            has_essay = any(f.kind == "textarea" for f in batch)
            max_out = min(2048, max(512, len(batch) * (220 if has_essay else 90)))

            try:
                ans_map = self.llm.complete_json(sys, prompt, max_tokens=max_out)
            except TokenBudgetExceeded:
                break
            except Exception:  # noqa: BLE001
                continue

            if not isinstance(ans_map, dict):
                continue

            for f in batch:
                raw_ans = ans_map.get(f.uid)
                if raw_ans in (None, "", []):
                    continue
                resolved = _resolve_for_field(f, raw_ans)
                if resolved not in (None, "", []):
                    f.value = resolved
                    f.source = "llm"
                    f.rule = "llm.batch"
                    f.confidence = 0.8

    def ensure_cover_letter(self, job_ctx: dict, out_dir: Path) -> str:
        """Generate or resolve a cover letter file path if configured."""
        pref = str(self.profile.get("documents.cover_letter", "auto")).strip()
        if not pref or pref.lower() == "none":
            return ""
        if pref.lower() != "auto":
            p = Path(pref).expanduser()
            if p.exists():
                self.profile.cover_letter_path = str(p.resolve())
                return self.profile.cover_letter_path
        if self.profile.cover_letter_path and Path(self.profile.cover_letter_path).exists():
            return self.profile.cover_letter_path
        if not self.llm or not self.llm.can_spend(600):
            return ""

        company = job_ctx.get("company") or "Hiring Team"
        title = job_ctx.get("title") or "the open role"
        desc = clip_text(job_ctx.get("description", ""), 1800)
        resume_snip = clip_text(self.profile.resume_text, 2200)
        sys = (
            "Write a concise, genuine, 3-paragraph plain-text cover letter (under 250 words). "
            "Use only facts from the applicant's profile and resume. No placeholders like [Company]."
        )
        prompt = (
            f"Applicant: {self.profile.full_name} ({self.profile.get('personal.email')}, {self.profile.get('personal.phone')})\n"
            f"Company: {company}\nRole: {title}\nJob Description:\n{desc}\n\nResume:\n{resume_snip}"
        )
        try:
            body = self.llm.complete(sys, prompt, json_mode=False, max_tokens=600).strip()
            if body:
                out_dir.mkdir(parents=True, exist_ok=True)
                slug = re.sub(r"[^a-z0-9]+", "_", f"{company}_{title}".lower()).strip("_")[:40] or "job"
                cl_path = out_dir / f"cover_letter_{slug}.txt"
                cl_path.write_text(body)
                self.profile.cover_letter_path = str(cl_path.resolve())
                return self.profile.cover_letter_path
        except Exception:  # noqa: BLE001
            pass
        return ""
