"""Tiny provider-agnostic LLM client (stdlib only) with token-budget management.

Picks the first provider with an API key in the environment:
  GEMINI_API_KEY / GOOGLE_API_KEY  -> Google Gemini
  OPENAI_API_KEY                   -> OpenAI
  ANTHROPIC_API_KEY                -> Anthropic
Override the model with AUTOAPPLY_LLM_MODEL, force a provider with AUTOAPPLY_LLM_PROVIDER,
or disable with AUTOAPPLY_LLM_PROVIDER=none.

Token limits are enforced on both input prompts and total session usage:
  AUTOAPPLY_MAX_INPUT_TOKENS  -> max input tokens per request (default: 6000)
  AUTOAPPLY_TOKEN_BUDGET      -> max total tokens across a run (default: 60000)

The LLM is optional: without it, autoapply still fills everything a rule can answer and
asks you for the rest during review.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request

DEFAULT_MODELS = {
    "gemini": "gemini-2.5-flash",
    "openai": "gpt-4.1-mini",
    "anthropic": "claude-sonnet-4-5",
}

CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    """Fast heuristic token estimate (~4 chars/token for English/JSON)."""
    if not text:
        return 0
    return (len(text) + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN


def clip_text(text: str, max_chars: int, marker: str = "\n...[truncated]...\n") -> str:
    """Clip text to at most `max_chars`, keeping both head and tail for context."""
    if not text or len(text) <= max_chars:
        return text or ""
    if max_chars <= len(marker) + 40:
        return text[:max_chars]
    budget = max_chars - len(marker)
    head = int(budget * 0.75)
    tail = budget - head
    return text[:head].rstrip() + marker + text[-tail:].lstrip()


def clip_to_tokens(text: str, max_tokens: int) -> str:
    return clip_text(text, max_tokens * CHARS_PER_TOKEN)


class TokenBudgetExceeded(RuntimeError):
    """Raised when the configured session token budget has been exhausted."""


class LLM:
    def __init__(
        self,
        provider: str,
        key: str,
        model: str,
        max_input_tokens: int | None = None,
        token_budget: int | None = None,
    ):
        self.provider, self.key, self.model = provider, key, model
        self.max_input_tokens = max_input_tokens or int(os.environ.get("AUTOAPPLY_MAX_INPUT_TOKENS", "6000"))
        self.token_budget = token_budget or int(os.environ.get("AUTOAPPLY_TOKEN_BUDGET", "60000"))
        self.tokens_used: int = 0
        self.calls_made: int = 0

    @classmethod
    def from_env(cls, max_input_tokens: int | None = None, token_budget: int | None = None) -> "LLM | None":
        forced = os.environ.get("AUTOAPPLY_LLM_PROVIDER", "").lower()
        if forced == "none":
            return None
        candidates = [
            ("gemini", os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")),
            ("openai", os.environ.get("OPENAI_API_KEY")),
            ("anthropic", os.environ.get("ANTHROPIC_API_KEY")),
        ]
        for name, key in candidates:
            if key and (not forced or forced == name):
                model = os.environ.get("AUTOAPPLY_LLM_MODEL") or DEFAULT_MODELS[name]
                return cls(name, key, model, max_input_tokens=max_input_tokens, token_budget=token_budget)
        return None

    @property
    def remaining_tokens(self) -> int:
        return max(0, self.token_budget - self.tokens_used)

    def can_spend(self, estimated_tokens: int = 200) -> bool:
        return (self.tokens_used + estimated_tokens) <= self.token_budget

    def __repr__(self) -> str:
        return f"{self.provider}:{self.model} (used={self.tokens_used}/{self.token_budget} tok)"

    # ------------------------------------------------------------------
    def _post(self, url: str, body: dict, headers: dict) -> dict:
        payload = json.dumps(body).encode()
        last_err: Exception | None = None
        for attempt in range(3):
            req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json", **headers})
            try:
                with urllib.request.urlopen(req, timeout=90) as r:
                    return json.loads(r.read().decode())
            except urllib.error.HTTPError as e:
                err_body = ""
                try:
                    err_body = e.read().decode()[:500]
                except Exception:  # noqa: BLE001
                    pass
                last_err = RuntimeError(f"LLM HTTP {e.code}: {err_body}")
                if e.code in (429, 500, 502, 503, 504) and attempt < 2:
                    time.sleep(1.5 * (2 ** attempt))
                    continue
                raise last_err from e
            except urllib.error.URLError as e:
                last_err = RuntimeError(f"LLM network error: {e}")
                if attempt < 2:
                    time.sleep(1.0 * (attempt + 1))
                    continue
                raise last_err from e
        raise last_err or RuntimeError("LLM request failed")

    def complete(self, system: str, prompt: str, json_mode: bool = False, max_tokens: int = 2048) -> str:
        sys_tokens = estimate_tokens(system)
        prompt_budget = max(256, self.max_input_tokens - sys_tokens)
        if estimate_tokens(prompt) > prompt_budget:
            prompt = clip_to_tokens(prompt, prompt_budget)

        in_est = sys_tokens + estimate_tokens(prompt)
        if self.tokens_used + in_est > self.token_budget:
            raise TokenBudgetExceeded(
                f"Token budget exceeded ({self.tokens_used + in_est} > {self.token_budget})"
            )
        # Clamp max_tokens to remaining budget (with a sensible minimum so responses don't truncate mid-JSON)
        effective_max_out = min(max_tokens, max(256, self.token_budget - self.tokens_used - in_est))

        self.calls_made += 1
        if self.provider == "gemini":
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.key}"
            body: dict = {
                "systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.2, "maxOutputTokens": effective_max_out},
            }
            if json_mode:
                body["generationConfig"]["responseMimeType"] = "application/json"
            data = self._post(url, body, {})
            usage = data.get("usageMetadata", {})
            parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
            out = "".join(p.get("text", "") for p in parts)
            self.tokens_used += usage.get("totalTokenCount") or (in_est + estimate_tokens(out))
            return out

        if self.provider == "openai":
            body = {
                "model": self.model,
                "temperature": 0.2,
                "max_tokens": effective_max_out,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            }
            if json_mode:
                body["response_format"] = {"type": "json_object"}
            data = self._post("https://api.openai.com/v1/chat/completions", body, {"Authorization": f"Bearer {self.key}"})
            usage = data.get("usage", {})
            out = data["choices"][0]["message"]["content"]
            self.tokens_used += usage.get("total_tokens") or (in_est + estimate_tokens(out))
            return out

        if self.provider == "anthropic":
            body = {
                "model": self.model,
                "max_tokens": effective_max_out,
                "temperature": 0.2,
                "system": system + ("\nRespond with a single JSON object only." if json_mode else ""),
                "messages": [{"role": "user", "content": prompt}],
            }
            data = self._post(
                "https://api.anthropic.com/v1/messages",
                body,
                {"x-api-key": self.key, "anthropic-version": "2023-06-01"},
            )
            usage = data.get("usage", {})
            out = "".join(b.get("text", "") for b in data.get("content", []))
            used = (usage.get("input_tokens", 0) + usage.get("output_tokens", 0)) or (in_est + estimate_tokens(out))
            self.tokens_used += used
            return out

        raise ValueError(self.provider)

    def complete_json(self, system: str, prompt: str, max_tokens: int = 2048) -> dict:
        txt = self.complete(system, prompt, json_mode=True, max_tokens=max_tokens)
        return parse_json_loose(txt)


def parse_json_loose(txt: str) -> dict:
    txt = (txt or "").strip()
    txt = re.sub(r"^```(?:json)?\s*|\s*```$", "", txt)
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", txt, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
        # Attempt to salvage truncated JSON object by closing open strings/braces
        salvaged = _salvage_truncated_json(txt)
        if salvaged is not None:
            return salvaged
        raise


def _salvage_truncated_json(txt: str) -> dict | None:
    """Recover completed key-value pairs if an LLM JSON response was cut off at max_tokens."""
    start = txt.find("{")
    if start < 0:
        return None
    frag = txt[start:]
    # Trim back to the last comma or complete value boundary
    for cut in [frag.rfind(","), frag.rfind("}"), frag.rfind("]")]:
        if cut <= 0:
            continue
        candidate = frag[:cut].rstrip().rstrip(",")
        # Balance brackets/braces
        open_sq = candidate.count("[") - candidate.count("]")
        open_br = candidate.count("{") - candidate.count("}")
        if open_sq >= 0 and open_br >= 0:
            attempt = candidate + ("]" * open_sq) + ("}" * open_br)
            try:
                res = json.loads(attempt)
                if isinstance(res, dict):
                    return res
            except json.JSONDecodeError:
                continue
    return None
