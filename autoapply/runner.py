"""End-to-end application runner and interactive Rich review UI."""
from __future__ import annotations

import asyncio
import re
from pathlib import Path

from playwright.async_api import BrowserContext, Page
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich.text import Text

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
from .llm import LLM
from .profile import Profile

console = Console()

TYPEAHEAD_HINT_RE = re.compile(r"\b(school|university|college|institution|city|location|employer|company)\b", re.I)

STATUS_STYLE = {
    "ok": ("OK", "bold green"),
    "mismatch": ("MISMATCH", "bold yellow"),
    "missing": ("MISSING", "bold red"),
    "error": ("ERROR", "bold red"),
    "skipped": ("SKIP", "dim"),
    "pending": ("PENDING", "cyan"),
}


def render_fields_table(fields: list[Field], title: str) -> None:
    table = Table(title=title, show_lines=False, header_style="bold cyan", expand=True)
    table.add_column("#", style="dim", width=3, justify="right")
    table.add_column("Req", width=3, justify="center")
    table.add_column("Field", ratio=3)
    table.add_column("Kind", style="dim", width=10)
    table.add_column("Value on Page", ratio=3)
    table.add_column("Status", width=10)
    table.add_column("Source", style="dim", width=10)

    for idx, f in enumerate(fields, 1):
        req_mark = Text("*", style="bold red") if f.required else Text("")
        lbl = f.label or f.name or f.uid
        if f.section and f.section.lower() not in lbl.lower():
            lbl = f"[{f.section}] {lbl}"
        val_str = f.display_value
        if len(val_str) > 90:
            val_str = val_str[:87] + "..."
        if f.error:
            val_str = f"{val_str} ({f.error})" if val_str else f"ERR: {f.error}"
        st_label, st_style = STATUS_STYLE.get(f.status or "pending", (f.status or "?", "white"))
        table.add_row(
            str(idx),
            req_mark,
            lbl[:85],
            f.kind,
            val_str,
            Text(st_label, style=st_style),
            f.source or "-",
        )
    console.print(table)


class ApplicationRunner:
    def __init__(
        self,
        profile: Profile,
        llm: LLM | None = None,
        memory: Memory | None = None,
        auto_advance: bool = False,
        auto_submit: bool = False,
        work_dir: Path | None = None,
    ):
        self.profile = profile
        self.llm = llm
        self.memory = memory or Memory()
        self.engine = AnswerEngine(profile, llm=llm, memory=self.memory)
        self.auto_advance = auto_advance
        self.auto_submit = auto_submit
        self.work_dir = work_dir or (Path.home() / ".autoapply" / "generated")

    async def apply_to_url(self, context: BrowserContext, url: str) -> bool:
        page = await context.new_page()
        console.print(Panel(f"[bold]Opening[/bold] {url}", border_style="blue"))
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=45000)
        except Exception as e:  # noqa: BLE001
            console.print(f"[yellow]Warning during initial page load:[/yellow] {e}")
        await page.wait_for_timeout(1200)

        await dismiss_banners(page)
        ats = await detect_ats(page)
        job_ctx = await extract_job_context(page)

        dom = Dom(page)
        page = await enter_application(page, dom, self.profile)
        ats = await detect_ats(page)
        if not job_ctx.get("title"):
            job_ctx = await extract_job_context(page)

        header_info = f"ATS: [bold cyan]{ats}[/bold cyan]"
        if job_ctx.get("title") or job_ctx.get("company"):
            header_info += f"  |  Role: [bold]{job_ctx.get('title', '?')}[/bold] @ [bold]{job_ctx.get('company', '?')}[/bold]"
        if self.llm:
            header_info += f"  |  LLM: [dim]{self.llm}[/dim]"
        console.print(header_info)

        max_steps = 12
        for step_num in range(1, max_steps + 1):
            await page.wait_for_timeout(600)
            await dismiss_banners(page)

            if ats == "workday":
                await expand_workday_sections(page, self.profile)

            fields = await dom.scan()

            # Check if we hit an account Sign-In / Create-Account gate
            if is_auth_gate(fields):
                console.print("[yellow]Authentication gate detected (Sign In / Create Account).[/yellow]")
                handled = await handle_auth_gate(page, dom, fields, self.profile)
                if handled:
                    await page.wait_for_timeout(2000)
                    fields = await dom.scan()
                if is_auth_gate(fields):
                    Prompt.ask(
                        "[bold yellow]Please sign in or create an account in the browser window, then press Enter to continue[/bold yellow]",
                        default="",
                    )
                    fields = await dom.scan()

            if not fields:
                # Try entering application again in case a modal/tab appeared late
                page = await enter_application(page, dom, self.profile)
                fields = await dom.scan()
                if not fields:
                    console.print("[yellow]No form fields detected on this page.[/yellow]")
                    return False

            step_name = await detect_step_name(page) or f"Step {step_num}"
            for f in fields:
                f.step = step_name

            # Fill + verify current step (including conditional follow-up fields)
            fields = await self._fill_and_verify_step(dom, fields, job_ctx)

            nav_kind, nav_frame, nav_text = await find_nav_action(page)

            # Interactive review / auto-advance decision
            action = await self._review_step(dom, fields, step_name, nav_kind, nav_text, job_ctx)
            if action == "quit":
                console.print("[yellow]Stopped before submitting.[/yellow]")
                return False
            if action == "done":
                return True
            if action == "next" and nav_frame:
                before_sig = (page.url, step_name, tuple(f.uid for f in fields))
                await click_nav_action(nav_frame, "next")
                await page.wait_for_timeout(1800)
                errs = await read_page_errors(page)
                after_step = await detect_step_name(page) or f"Step {step_num + 1}"
                after_fields = await dom.scan()
                after_sig = (page.url, after_step, tuple(f.uid for f in after_fields))
                if errs and after_sig == before_sig:
                    console.print("[bold red]Page reported validation errors:[/bold red]")
                    for err in errs:
                        console.print(f"  - {err}")
                    # Let the user fix them in review on the next iteration
                continue
            if action == "submit" and nav_frame:
                await click_nav_action(nav_frame, "submit")
                await page.wait_for_timeout(2500)
                errs = await read_page_errors(page)
                if errs:
                    console.print("[bold red]Errors reported after clicking submit:[/bold red]")
                    for err in errs:
                        console.print(f"  - {err}")
                    continue
                console.print("[bold green]Application submitted![/bold green]")
                return True

        return False

    async def _fill_and_verify_step(self, dom: Dom, fields: list[Field], job_ctx: dict) -> list[Field]:
        # 1) Upload resume first if an empty resume file field exists (triggers ATS autofill/conditional fields)
        for f in fields:
            if f.kind == "file" and not f.current and not re.search(r"cover\s*letter|transcript|portfolio", f.label, re.I):
                if self.profile.resume_path:
                    try:
                        await dom.fill(f, self.profile.resume_path)
                        f.value = self.profile.resume_path
                        f.source = "profile"
                        f.rule = "file.resume"
                    except Exception as e:  # noqa: BLE001
                        f.error = str(e)

        # 2) Ensure cover letter file exists if a cover letter upload field is on the page
        if any(f.kind == "file" and re.search(r"cover\s*letter", f.label, re.I) for f in fields):
            self.engine.ensure_cover_letter(job_ctx, self.work_dir)

        # 3) Probe dropdown options for custom comboboxes / listbox buttons (skip obvious school/city typeaheads)
        for f in fields:
            if f.kind in ("combobox", "listbox_button") and not f.options and not f.current:
                if TYPEAHEAD_HINT_RE.search(_label_sig(f)):
                    f.typeahead = True
                else:
                    await dom.probe_options(f)

        # 4) Compute answers (memory -> rules -> token-bounded LLM)
        self.engine.answer_fields(fields, job_ctx)

        # 5) Fill fields in DOM
        await self._fill_fields_list(dom, fields)

        # 6) Check for newly revealed conditional fields (up to 2 passes)
        for _ in range(2):
            known_uids = {f.uid for f in fields}
            rescan = await dom.scan()
            new_fields = [nf for nf in rescan if nf.uid not in known_uids]
            if not new_fields:
                break
            for nf in new_fields:
                if nf.kind in ("combobox", "listbox_button") and not nf.options:
                    if TYPEAHEAD_HINT_RE.search(_label_sig(nf)):
                        nf.typeahead = True
                    else:
                        await dom.probe_options(nf)
            self.engine.answer_fields(new_fields, job_ctx)
            await self._fill_fields_list(dom, new_fields)
            fields.extend(new_fields)

        # 7) Read back & verify every field
        for f in fields:
            await dom.verify(f)

        return fields

    async def _fill_fields_list(self, dom: Dom, fields: list[Field]) -> None:
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

    async def _review_step(
        self,
        dom: Dom,
        fields: list[Field],
        step_name: str,
        nav_kind: str,
        nav_text: str,
        job_ctx: dict,
    ) -> str:
        while True:
            title = f"{step_name}  ({len(fields)} fields)"
            if self.llm:
                title += f"  [tokens: {self.llm.tokens_used}/{self.llm.token_budget}]"
            render_fields_table(fields, title)

            problems = [f for f in fields if f.required and f.status in ("missing", "error", "mismatch")]
            if problems:
                console.print(
                    f"[bold yellow]{len(problems)} required field(s) need attention before proceeding.[/bold yellow]"
                )

            # Auto-advance on intermediate wizard steps if all required fields are OK
            if not problems and nav_kind == "next" and self.auto_advance:
                console.print(f"[green]All required fields verified — advancing via '{nav_text}'...[/green]")
                return "next"

            if not problems and nav_kind == "submit" and self.auto_submit:
                console.print(f"[green]All required fields verified — auto-submitting via '{nav_text}'...[/green]")
                return "submit"

            # Interactive prompt
            hints = [
                "[bold]#[/bold] or [bold]#=val[/bold] edit field",
                "[bold]a[/bold] answer missing required",
                "[bold]r[/bold] rescan page",
            ]
            if nav_kind == "next":
                hints.append(f"[bold]c[/bold]/Enter -> {nav_text or 'Next'}")
            elif nav_kind == "submit":
                hints.append(f"[bold green]submit[/bold green] -> {nav_text or 'Submit Application'}")
            else:
                hints.append("[bold]d[/bold] mark done")
            hints.append("[bold]q[/bold] quit")

            cmd = Prompt.ask("Action (" + " | ".join(hints) + ")", default="c" if nav_kind == "next" and not problems else "").strip()

            if cmd.lower() in ("q", "quit", "exit"):
                return "quit"
            if cmd.lower() in ("d", "done"):
                return "done"
            if cmd.lower() in ("r", "rescan"):
                fields[:] = await dom.scan()
                for f in fields:
                    await dom.verify(f)
                continue
            if cmd.lower() == "a":
                targets = [f for f in fields if f.required and f.status in ("missing", "error", "mismatch")]
                if not targets:
                    targets = [f for f in fields if f.status in ("missing", "error", "mismatch")]
                for f in targets:
                    await self._prompt_and_fill_one(dom, f)
                continue
            if cmd.lower() in ("c", "next", "") and nav_kind == "next":
                return "next"
            if cmd.lower() == "submit":
                if nav_kind == "submit":
                    return "submit"
                if Confirm.ask("No standard submit button detected; mark done anyway?", default=False):
                    return "done"
                continue

            # Field edit: `<num>` or `<num>=<value>`
            m = re.match(r"^(\d+)(?:\s*=\s*(.*))?$", cmd)
            if m:
                idx = int(m.group(1)) - 1
                if 0 <= idx < len(fields):
                    await self._prompt_and_fill_one(dom, fields[idx], preset_val=m.group(2))
                else:
                    console.print(f"[red]Field #{idx + 1} out of range (1..{len(fields)}).[/red]")
                continue

            console.print("[yellow]Unrecognized command. Type a field number (e.g. 3 or 3=Yes), 'a', 'r', 'c', 'submit', or 'q'.[/yellow]")

    async def _prompt_and_fill_one(self, dom: Dom, f: Field, preset_val: str | None = None) -> None:
        if f.kind in ("combobox", "listbox_button") and not f.options and not f.typeahead:
            await dom.probe_options(f)
        if preset_val is None:
            console.print(f"\n[bold cyan]{f.label}[/bold cyan] ({f.kind}{', required' if f.required else ''})")
            if f.description:
                console.print(f"[dim]{f.description}[/dim]")
            if f.options:
                opts_preview = ", ".join(f"{i+1}:{o.text}" for i, o in enumerate(f.options[:25]))
                console.print(f"Options: {opts_preview}")
            raw_in = Prompt.ask("Value (or option #)", default=str(f.display_value or "")).strip()
        else:
            raw_in = preset_val.strip()

        if not raw_in:
            return

        # Allow picking an option by its 1-based index
        val: object = raw_in
        if f.options and raw_in.isdigit():
            opt_idx = int(raw_in) - 1
            if 0 <= opt_idx < len(f.options):
                val = f.options[opt_idx].text

        if f.multi and isinstance(val, str) and "," in val:
            val = [x.strip() for x in val.split(",") if x.strip()]

        f.value = val
        f.source = "user"
        f.error = ""
        try:
            await dom.fill(f, val, resume_path=self.profile.resume_path)
        except Exception as e:  # noqa: BLE001
            f.error = str(e)
        await dom.verify(f)
        if f.status == "ok":
            self.memory.remember(f, f.filled or val)
