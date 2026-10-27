"""Fuzzy matching of a desired answer onto the options a form actually offers."""
from __future__ import annotations

import re
from typing import Iterable

from rapidfuzz import fuzz, process

US_STATES = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California", "CO": "Colorado",
    "CT": "Connecticut", "DE": "Delaware", "DC": "District of Columbia", "FL": "Florida", "GA": "Georgia",
    "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas",
    "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland", "MA": "Massachusetts",
    "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi", "MO": "Missouri", "MT": "Montana",
    "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico",
    "NY": "New York", "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma",
    "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina", "SD": "South Dakota",
    "TN": "Tennessee", "TX": "Texas", "UT": "Utah", "VT": "Vermont", "VA": "Virginia", "WA": "Washington",
    "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming", "PR": "Puerto Rico",
}

SYNONYMS = [
    {"united states", "united states of america", "usa", "us", "u.s.", "u.s.a.", "america"},
    {"united kingdom", "uk", "great britain", "england"},
    {"bachelor's degree", "bachelors", "bachelor", "bachelor of science", "bachelor of arts", "bs", "ba", "b.s.", "b.a.", "bsc", "undergraduate"},
    {"master's degree", "masters", "master", "master of science", "ms", "m.s.", "msc", "meng", "master of engineering"},
    {"doctorate", "phd", "ph.d.", "doctor of philosophy", "doctoral"},
    {"computer science", "cs", "comp sci", "computer sciences"},
    {"computer engineering", "ce", "compe"},
    {"male", "man"},
    {"female", "woman"},
    {"mobile", "cell", "cell phone", "mobile phone"},
]

DECLINE_RE = re.compile(
    r"decline|prefer not|not (wish|want|like) to|don.?t (wish|want)|do not (wish|want)|choose not|rather not|"
    r"not to (answer|say|disclose|self|identify|respond)|no answer|undisclosed|not specified|not declared|i don.?t know",
    re.I,
)
YES_RE = re.compile(r"^\s*(yes|y|true|i am\b(?! not)|i do\b(?! not)|i have\b(?! not)|i will\b(?! not)|i can\b(?! ?not)|i agree|i acknowledge|i consent|i accept|i understand|i confirm|i certify|agree|accept|acknowledge)", re.I)
NO_RE = re.compile(r"^\s*(no|n|false|none|i am not|i do not|i don.?t|i have not|i haven.?t|i will not|i won.?t|i cannot|i can.?t|not )", re.I)


def norm(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"[’`]", "'", s)
    s = re.sub(r"[^\w'+.#/ -]", " ", s)
    return re.sub(r"\s+", " ", s).strip(" .")


def expand(desired: str) -> list[str]:
    d = norm(desired)
    out = [d]
    up = desired.strip().upper()
    if up in US_STATES:
        out.append(norm(US_STATES[up]))
    for k, v in US_STATES.items():
        if d == v.lower():
            out.append(k.lower())
    for group in SYNONYMS:
        if d in group:
            out.extend(sorted(group - {d}, key=len, reverse=True))
    return list(dict.fromkeys(out))


def is_placeholder(opt: str) -> bool:
    return bool(re.match(r"^\s*(select|choose|please select|--|—|none selected)\b", opt or "", re.I)) or not (opt or "").strip()


def match_bool(value: bool, options: list[str]) -> int | None:
    rx = YES_RE if value else NO_RE
    for i, o in enumerate(options):
        if rx.search(o) and not (value and NO_RE.search(o)):
            return i
    return None


def match_option(desired, options: list[str], min_score: int = 78) -> int | None:
    """Return index of the option that best represents `desired`, or None."""
    if not options:
        return None
    opts = [o for o in options]
    if isinstance(desired, bool):
        return match_bool(desired, opts)
    if desired is None:
        return None
    ds = str(desired).strip()
    if not ds:
        return None
    if ds.lower() in ("decline", "prefer not to say", "decline to self-identify", "i don't wish to answer"):
        for i, o in enumerate(opts):
            if DECLINE_RE.search(o):
                return i
    if ds.lower() in ("yes", "true"):
        r = match_bool(True, opts)
        if r is not None:
            return r
    if ds.lower() in ("no", "false"):
        r = match_bool(False, opts)
        if r is not None:
            return r
    normed = [norm(o) for o in opts]
    cands = expand(ds)
    for c in cands:  # exact
        for i, o in enumerate(normed):
            if o == c:
                return i
    for c in cands:  # option starts with desired / desired starts with option
        for i, o in enumerate(normed):
            if not o or is_placeholder(opts[i]):
                continue
            if o.startswith(c + " ") or o.startswith(c + ",") or o.startswith(c + "(") or (len(o) > 3 and c.startswith(o + " ")):
                return i
    for c in cands:  # whole-word containment
        if len(c) < 3:
            continue
        for i, o in enumerate(normed):
            if re.search(rf"(^|\W){re.escape(c)}(\W|$)", o):
                return i
    best = process.extractOne(norm(ds), normed, scorer=fuzz.token_set_ratio)
    if best and best[1] >= min_score:
        return best[2]
    best = process.extractOne(norm(ds), normed, scorer=fuzz.WRatio)
    if best and best[1] >= max(min_score, 86):
        return best[2]
    return None


def best_options(desired: Iterable[str], options: list[str]) -> list[int]:
    out = []
    for d in desired:
        i = match_option(d, options)
        if i is not None and i not in out:
            out.append(i)
    return out
