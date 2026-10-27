"""Data model for a single form field discovered on an application page."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

# Kinds produced by the DOM scanner (extract.js)
TEXT_KINDS = {"text", "textarea", "date"}
CHOICE_KINDS = {"select", "combobox", "radio", "checkbox_group", "button_group", "listbox_button"}


@dataclass
class Option:
    text: str
    value: str = ""
    uid: str = ""  # element uid for clickable options (radio inputs, buttons)


@dataclass
class Field:
    uid: str
    kind: str  # text | textarea | date | select | combobox | radio | checkbox | checkbox_group | file | button_group | listbox_button
    label: str
    frame_key: str = "main"
    itype: str = ""  # html input type (email, tel, url, number, ...)
    name: str = ""
    elem_id: str = ""
    required: bool = False
    options: list[Option] = field(default_factory=list)
    multi: bool = False
    placeholder: str = ""
    description: str = ""
    section: str = ""
    autocomplete: str = ""
    current: str = ""  # value already present when scanned
    typeahead: bool = False  # combobox whose options only appear after typing
    step: str = ""  # wizard step name (multi-page ATS like Workday)

    # --- populated by the answer engine ---
    value: Any = None
    source: str = ""  # memory | rule | profile | llm | default | user | prefilled | missing | skip
    rule: str = ""  # which rule produced the value
    confidence: float = 0.0

    # --- populated after filling ---
    filled: str = ""  # value read back from the DOM after filling
    status: str = ""  # ok | mismatch | missing | error | skipped | pending
    error: str = ""

    @property
    def option_texts(self) -> list[str]:
        return [o.text for o in self.options if o.text]

    @property
    def is_choice(self) -> bool:
        return self.kind in CHOICE_KINDS

    @property
    def display_value(self) -> str:
        v = self.filled if self.filled not in ("", None) else self.value
        if v is None:
            return ""
        if isinstance(v, bool):
            return "Yes (checked)" if v else "No (unchecked)"
        if isinstance(v, (list, tuple)):
            return ", ".join(str(x) for x in v)
        return str(v)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_scan(cls, d: dict, frame_key: str) -> "Field":
        opts = [Option(text=o.get("text", ""), value=o.get("value", ""), uid=o.get("uid", "")) for o in d.get("options", [])]
        return cls(
            uid=d["uid"],
            kind=d["kind"],
            label=d.get("label", "") or d.get("name", "") or d.get("placeholder", ""),
            frame_key=frame_key,
            itype=d.get("itype", ""),
            name=d.get("name", ""),
            elem_id=d.get("id", ""),
            required=bool(d.get("required")),
            options=opts,
            multi=bool(d.get("multi")),
            placeholder=d.get("placeholder", ""),
            description=d.get("description", ""),
            section=d.get("section", ""),
            autocomplete=d.get("autocomplete", ""),
            current=d.get("current", "") or "",
        )
