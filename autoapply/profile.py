"""Applicant profile: everything we know about the applicant.

The profile is a YAML file (default: ~/.autoapply/profile.yaml or ./profile.yaml).
It is bootstrapped from the resume (see resume.py) and then hand-edited by the user
for information a resume never contains (work authorization, EEO answers, address...).
"""
from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any

import yaml

DEFAULT_PROFILE: dict[str, Any] = {
    "personal": {
        "first_name": "",
        "last_name": "",
        "middle_name": "",
        "preferred_name": "",
        "email": "",
        "phone": "",
        "phone_country_code": "+1",
        "phone_type": "Mobile",
        "pronouns": "",
        "address": {"street": "", "street2": "", "city": "", "state": "", "zip": "", "country": "United States"},
        # Free-form "City, ST" used for "Location" style questions
        "location": "",
    },
    "links": {"linkedin": "", "github": "", "portfolio": "", "twitter": "", "other": ""},
    "education": [
        # {school, degree, major, minor, gpa, start (YYYY-MM), end (YYYY-MM, expected grad)}
    ],
    "experience": [
        # {company, title, location, start, end ('present'), description}
    ],
    "skills": [],
    "work_authorization": {
        "authorized_us": True,  # legally authorized to work in the US
        "require_sponsorship": False,  # now or in the future
        "citizenship": "",  # e.g. "US Citizen", "Permanent Resident", "F-1 (OPT eligible)"
        "security_clearance": "None",
    },
    "preferences": {
        "willing_to_relocate": True,
        "remote_ok": True,
        "onsite_ok": True,
        "start_date": "",  # e.g. "2027-06-01" or "Immediately"
        "salary_expectation": "Open / negotiable",
        "notice_period": "2 weeks",
        "years_experience": "",
        "sms_opt_in": False,
    },
    "eeo": {
        # Use "decline" to pick the "I don't wish to answer" style option.
        "gender": "decline",
        "transgender": "decline",
        "sexual_orientation": "decline",
        "hispanic_latino": "decline",
        "race": "decline",
        "veteran": "decline",
        "disability": "decline",
    },
    "misc": {
        "how_did_you_hear": "Company website",
        "over_18": True,
        "background_check_ok": True,
        "drug_test_ok": True,
        "previously_employed_here": False,
        "relatives_at_company": False,
        "non_compete": False,
        "referral_name": "",
    },
    "documents": {
        "cover_letter": "auto",  # auto (LLM-generated if available) | none | path/to/file.pdf
    },
    "accounts": {
        # Used for ATSs that require an account (Workday, iCIMS, Taleo...).
        # Leave password empty to be prompted at runtime.
        "email": "",
        "password": "",
    },
}


def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


class Profile:
    def __init__(self, data: dict | None = None, resume_text: str = "", resume_path: str = ""):
        self.data = _deep_merge(DEFAULT_PROFILE, data or {})
        self.resume_text = resume_text
        self.resume_path = resume_path
        self.cover_letter_path: str = ""

    # --------------------------------------------------------------- access
    def get(self, dotted: str, default: Any = "") -> Any:
        cur: Any = self.data
        for part in dotted.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                return default
        return default if cur is None else cur

    @property
    def first_name(self) -> str:
        return self.get("personal.first_name")

    @property
    def last_name(self) -> str:
        return self.get("personal.last_name")

    @property
    def full_name(self) -> str:
        parts = [self.first_name, self.get("personal.middle_name"), self.last_name]
        return " ".join(p for p in parts if p)

    @property
    def edu(self) -> dict:
        e = self.get("education", [])
        return e[0] if e else {}

    @property
    def job(self) -> dict:
        e = self.get("experience", [])
        return e[0] if e else {}

    @property
    def location(self) -> str:
        loc = self.get("personal.location")
        if loc:
            return loc
        a = self.get("personal.address", {})
        return ", ".join(x for x in [a.get("city"), a.get("state")] if x)

    def phone_digits(self) -> str:
        return re.sub(r"\D", "", self.get("personal.phone"))

    def phone_national(self) -> str:
        d = self.phone_digits()
        cc = re.sub(r"\D", "", self.get("personal.phone_country_code", "+1"))
        if cc and d.startswith(cc) and len(d) > 10:
            d = d[len(cc):]
        return d

    def summary_for_llm(self) -> dict:
        d = copy.deepcopy(self.data)
        d.pop("accounts", None)
        return d

    # --------------------------------------------------------------- io
    @classmethod
    def load(cls, path: Path) -> "Profile":
        data = yaml.safe_load(path.read_text()) or {}
        return cls(data)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        header = (
            "# autoapply profile — edit freely. Values here override anything parsed from your resume.\n"
            "# EEO answers: use 'decline' to choose the 'I don't wish to answer' option, or write your answer.\n"
        )
        path.write_text(header + yaml.safe_dump(self.data, sort_keys=False, allow_unicode=True))

    def missing_essentials(self) -> list[str]:
        need = ["personal.first_name", "personal.last_name", "personal.email", "personal.phone"]
        return [k for k in need if not self.get(k)]
