"""Command-line interface for autoapply (CLI mode + Web App server for phone & laptop)."""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from playwright.async_api import async_playwright
from rich.console import Console
from rich.prompt import Prompt

from .answers import Memory
from .llm import LLM
from .profile import Profile
from .resume import _merge_nonempty, parse_resume
from .runner import ApplicationRunner

console = Console()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="autoapply",
        description=(
            "Automated job application filler supporting Greenhouse, Ashby, Lever, Workday, SmartRecruiters, iCIMS, and more.\n"
            "Run `python -m autoapply serve` to launch the mobile + desktop Web App."
        ),
    )
    p.add_argument(
        "resume",
        nargs="?",
        help="Path to resume file (.pdf, .docx, or .txt), OR 'serve' to launch the Web App",
    )
    p.add_argument("urls", nargs="*", help="One or more job posting URLs to apply to")
    p.add_argument("--urls-file", "-f", type=Path, help="Text file containing job URLs (one per line)")
    p.add_argument(
        "--profile",
        "-p",
        type=Path,
        default=None,
        help="Path to profile YAML (default: ./profile.yaml or ~/.autoapply/profile.yaml)",
    )
    p.add_argument("--init-profile", action="store_true", help="Parse resume into profile YAML and exit")
    p.add_argument("--memory", type=Path, default=None, help="Path to answer memory JSON (default: ~/.autoapply/memory.json)")
    p.add_argument("--headless", action="store_true", help="Run browser in headless mode")
    p.add_argument("--headed", action="store_true", help="Show browser window when running web server locally")
    p.add_argument("--host", default="0.0.0.0", help="Web server bind host (default: 0.0.0.0)")
    p.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")), help="Web server port (default: 8000)")
    p.add_argument(
        "--user-data-dir",
        type=Path,
        default=Path.home() / ".autoapply" / "browser_profile",
        help="Persistent Chromium user data dir (keeps ATS logins/sessions)",
    )
    p.add_argument(
        "--auto-advance",
        "-y",
        action="store_true",
        help="Auto-advance intermediate wizard steps when all required fields are OK",
    )
    p.add_argument(
        "--auto-submit",
        action="store_true",
        help="Auto-click final Submit when all required fields are OK (use with caution)",
    )
    p.add_argument("--no-llm", action="store_true", help="Disable LLM calls (use rules + heuristics + memory only)")
    p.add_argument("--token-budget", type=int, default=None, help="Max total LLM tokens for this session (default: 60000)")
    p.add_argument("--max-input-tokens", type=int, default=None, help="Max input tokens per LLM call (default: 6000)")
    return p


def load_or_create_profile(resume_path: Path, profile_path: Path, llm: LLM | None) -> Profile:
    if not resume_path.exists():
        raise FileNotFoundError(f"Resume not found: {resume_path}")

    console.print(f"[cyan]Parsing resume:[/cyan] {resume_path}")
    resume_text, parsed = parse_resume(resume_path, llm)

    if profile_path.exists():
        existing = Profile.load(profile_path).data
        merged = _merge_nonempty(parsed, existing)
        prof = Profile(merged, resume_text=resume_text, resume_path=str(resume_path.resolve()))
    else:
        prof = Profile(parsed, resume_text=resume_text, resume_path=str(resume_path.resolve()))
        prof.save(profile_path)
        console.print(f"[green]Bootstrapped profile saved to[/green] {profile_path}")

    missing = prof.missing_essentials()
    if missing and sys.stdin.isatty():
        console.print("[yellow]A few essential fields were not found in your resume/profile:[/yellow]")
        for dotted in missing:
            sec, key = dotted.split(".", 1)
            val = Prompt.ask(f"  {dotted}").strip()
            if val:
                prof.data[sec][key] = val
        prof.save(profile_path)

    return prof


async def run_async(args: argparse.Namespace) -> int:
    # Web Server mode: `python -m autoapply serve` (or no positional args)
    if not args.resume or str(args.resume).lower() in ("serve", "server", "web"):
        from .server import run_server

        await run_server(host=args.host, port=args.port, headless=not args.headed)
        return 0

    resume_path = Path(args.resume)
    llm = None if args.no_llm else LLM.from_env(max_input_tokens=args.max_input_tokens, token_budget=args.token_budget)
    if llm:
        console.print(f"[dim]LLM active: {llm}[/dim]")
    else:
        console.print("[dim]LLM disabled or no API key found — running with deterministic rules + heuristics.[/dim]")

    profile_path = args.profile
    if profile_path is None:
        local = Path("profile.yaml")
        profile_path = local if local.exists() else (Path.home() / ".autoapply" / "profile.yaml")

    prof = load_or_create_profile(resume_path, profile_path, llm)
    if args.init_profile:
        prof.save(profile_path)
        console.print(f"[bold green]Profile initialized at {profile_path}[/bold green]")
        return 0

    urls: list[str] = list(args.urls or [])
    if args.urls_file and args.urls_file.exists():
        for line in args.urls_file.read_text().splitlines():
            u = line.strip()
            if u and not u.startswith("#"):
                urls.append(u)

    if not urls:
        console.print("[red]No job URLs provided. Pass URLs as arguments, via --urls-file, or run `python -m autoapply serve`.[/red]")
        return 1

    memory = Memory(args.memory)
    runner = ApplicationRunner(
        profile=prof,
        llm=llm,
        memory=memory,
        auto_advance=args.auto_advance or args.auto_submit,
        auto_submit=args.auto_submit,
    )

    async with async_playwright() as pw:
        context = None
        try:
            args.user_data_dir.mkdir(parents=True, exist_ok=True)
            context = await pw.chromium.launch_persistent_context(
                user_data_dir=str(args.user_data_dir),
                headless=args.headless,
                viewport={"width": 1280, "height": 900},
                args=["--disable-blink-features=AutomationControlled"],
            )
        except Exception:  # noqa: BLE001
            browser = await pw.chromium.launch(
                headless=args.headless,
                args=["--disable-blink-features=AutomationControlled"],
            )
            context = await browser.new_context(viewport={"width": 1280, "height": 900})

        try:
            for i, url in enumerate(urls, 1):
                if len(urls) > 1:
                    console.print(f"\n[bold magenta]=== Job {i}/{len(urls)} ===[/bold magenta]")
                await runner.apply_to_url(context, url)
        finally:
            if llm:
                console.print(
                    f"[dim]Session LLM usage: {llm.calls_made} call(s), ~{llm.tokens_used}/{llm.token_budget} tokens.[/dim]"
                )
            await context.close()

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return asyncio.run(run_async(args))
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user.[/yellow]")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
