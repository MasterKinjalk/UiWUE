"""Build docs/HOW_IT_WORKS.pdf from docs/HOW_IT_WORKS.md with headless Microsoft Edge.

WeasyPrint is not installed here, so the Markdown is rendered to a standalone HTML page (Python-Markdown,
plus a print stylesheet in the app's fonts and colours, read from design/tokens.css) and Edge prints that
page to an A4 PDF with Chromium's own layout engine (--print-to-pdf).

Run from any folder; relative paths are taken from the repo root:

    .venv/Scripts/python.exe scripts/build_how_it_works_pdf.py
    .venv/Scripts/python.exe scripts/build_how_it_works_pdf.py --out C:/tmp/how.pdf --html C:/tmp/how.html

Needs the `markdown` package (`pygments` for muted comments in code blocks) and Microsoft Edge at its
default install path (or pass --edge).

Python-Markdown reads a few things differently from GitHub, so the source is bridged first: a list may
follow a paragraph line directly and nests by 2 spaces; an escaped pipe inside code in a table cell
shows as a plain pipe; a mermaid flowchart becomes a text tree (no JavaScript runs in the print); bare
URLs become links; heading ids follow GitHub's rule so the Contents links work.
"""

import argparse
import html
import os
import re
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as etree
from pathlib import Path

import markdown
from markdown.inlinepatterns import InlineProcessor
from markdown.util import AtomicString

ROOT = Path(__file__).resolve().parents[1]
TOKENS = ROOT / "design" / "tokens.css"
EDGE_DIR = Path("Microsoft", "Edge", "Application", "msedge.exe")

# Colours and faces are design tokens (var(--uw-...)), replaced by their values from design/tokens.css.
CSS = """
@page {
  size: A4;
  margin: 16mm 16mm 18mm;
  @bottom-left { content: "__FOOTER__"; font: 9pt var(--uw-font-sans); color: var(--uw-ink-muted); }
  @bottom-right {
    content: counter(page) " / " counter(pages);
    font: 9pt var(--uw-font-sans); color: var(--uw-ink-muted);
  }
}
html { color-scheme: light; -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body {
  margin: 0; background: var(--uw-surface); color: var(--uw-ink);
  font-family: var(--uw-font-sans); font-size: 10.5pt; line-height: 1.42;
}
h1, h2, h3, h4, h5, h6 {
  font-family: var(--uw-font-display); color: var(--uw-ink-display); font-weight: 700; line-height: 1.22;
  break-after: avoid; break-inside: avoid;
}
h1 { font-size: 22pt; margin: 0 0 10pt; }
h2 { font-size: 15pt; margin: 18pt 0 7pt; padding-bottom: 3pt; border-bottom: 1px solid var(--uw-rule); }
h3 { font-size: 12.5pt; margin: 14pt 0 5pt; }
h4, h5, h6 { font-size: 11pt; margin: 11pt 0 4pt; }
p { margin: 0 0 7pt; orphans: 3; widows: 3; }
a { color: var(--uw-teal-700); text-decoration: none; }
hr { border: 0; border-top: 1px solid var(--uw-rule); margin: 16pt 0; }
hr:has(+ h2) { display: none; }
ul, ol { margin: 0 0 7pt; padding-left: 16pt; }
li { margin: 1.5pt 0; }
li > p { margin: 0 0 3pt; }
li > ul, li > ol, li > table, li > div, li > pre { margin-top: 2pt; margin-bottom: 4pt; }
li::marker { color: var(--uw-ink-muted); }
code, pre { font-family: Consolas, "Cascadia Mono", "Courier New", monospace; }
code { font-size: 0.9em; background: var(--uw-gray); border-radius: 3px; padding: 0 0.2em; }
pre {
  font-size: 8.5pt; line-height: 1.4; margin: 4pt 0 10pt; padding: 6pt 8pt;
  background: var(--uw-surface-2); border: 1px solid var(--uw-rule); border-radius: var(--uw-radius-sm);
  white-space: pre-wrap; overflow-wrap: anywhere; break-inside: avoid;
}
pre code { font-size: inherit; background: none; padding: 0; border-radius: 0; }
.codehilite :is(.c, .ch, .cm, .cp, .cpf, .c1, .cs) { color: var(--uw-ink-muted); }
table {
  width: 100%; border-collapse: collapse; margin: 4pt 0 10pt;
  font-size: 9pt; line-height: 1.3; font-variant-numeric: tabular-nums;
}
thead { display: table-header-group; }
tr { break-inside: avoid; }
th, td { overflow-wrap: break-word; padding: 3.5pt 6pt; vertical-align: top; text-align: left; }
th { background: var(--uw-navy); color: var(--uw-surface); font-weight: 700; vertical-align: bottom; }
td { border-bottom: 1px solid var(--uw-rule); }
tbody tr:nth-child(even) td { background: var(--uw-surface-2); }
th code { background: none; color: inherit; padding: 0; }
img { max-width: 100%; }
"""

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="color-scheme" content="light">
<title>{title}</title>
<style>{css}</style>
</head>
<body>
{body}
</body>
</html>
"""

ITEM_RE = re.compile(r"( *)([-*+]|\d{1,9}[.)])( +|$)")
FENCE_RE = re.compile(r"( *)(`{3,}|~{3,})\s*([\w+-]*)")
HEADING_RE = re.compile(r" {0,3}#{1,6}(\s|$)")
MERMAID_EDGE = re.compile(r"\s*(-\.+->?|={2,}>|-{2,}>|-{3,})\s*(?:\|[^|]*\|)?\s*")
MERMAID_NODE = re.compile(r'([A-Za-z_][\w-]*)\s*(?:[\[({>]+"?(.*?)"?[\])}]+)?')
URL_RE = r"\bhttps?://[^\s<>()\[\]`]*[^\s<>()\[\]`.,;:!?'\"*_]"


def github_slug(value: str, separator: str) -> str:
    """Heading id as GitHub makes it, so links written for GitHub ("#1-the-experiment") work."""
    return re.sub(r"[^\w\- ]", "", value.strip().lower()).replace(" ", separator)


def mermaid_tree(src: list[str]) -> list[str] | None:
    """A simple Mermaid flowchart as a text tree, or None when it is anything else."""
    lines = [ln.split("%%")[0].strip() for ln in src]
    lines = [ln for ln in lines if ln]
    if not lines or not re.match(r"(flowchart|graph)\b", lines[0]):
        return None
    labels: dict[str, str] = {}
    kids: dict[str, list[tuple[str, bool]]] = {}
    has_parent: set[str] = set()
    for line in lines[1:]:
        if re.match(r"(classDef|class|style|linkStyle|click|subgraph|end|direction)\b", line):
            continue
        parts = MERMAID_EDGE.split(line)
        ids = []
        for part in parts[::2]:
            m = MERMAID_NODE.fullmatch(part.strip())
            if not m:
                return None
            kids.setdefault(m[1], [])
            if m[2]:
                first, *rest = re.split(r"\s*(?:\\n|<br\s*/?>)\s*", m[2])
                labels[m[1]] = first + (": " + ", ".join(rest) if rest else "")
            ids.append(m[1])
        for a, op, b in zip(ids, parts[1::2], ids[1:]):
            kids[a].append((b, "." in op))
            has_parent.add(b)
    out: list[str] = []
    shown: set[str] = set()

    def walk(node: str, lead: str, branch: str) -> None:
        label = labels.get(node, node)
        if node in shown:  # drawn already: point back up to it
            out.append(f"{lead}{branch}{label} ↑")
            return
        shown.add(node)
        out.append(lead + branch + label)
        lead += "   " if branch.startswith("└") else "│  " if branch else ""
        for i, (kid, dotted) in enumerate(kids[node]):
            walk(kid, lead, ("└" if i == len(kids[node]) - 1 else "├") + ("┄ " if dotted else "─ "))

    for node in [n for n in kids if n not in has_parent] + list(kids):
        if node not in shown:
            walk(node, "", "")
    return out


def bridge(text: str) -> str:
    """Rewrite the GitHub Markdown details that Python-Markdown reads differently (module docstring)."""
    lines = text.splitlines()
    out: list[str] = []
    stack: list[int] = []  # content column of each open list item, outermost first
    blank = True
    i = 0
    while i < len(lines):
        line = lines[i]
        i += 1
        fence = FENCE_RE.match(line)
        if fence:  # copy a fenced block at column 0 (Python-Markdown needs that); it ends any list
            indent, mark, lang = fence.groups()
            body = []
            while i < len(lines) and not re.fullmatch(rf" *{mark[0]}{{{len(mark)},}}\s*", lines[i]):
                body.append(re.sub(rf"^ {{0,{len(indent)}}}", "", lines[i]))
                i += 1
            i += 1
            tree = mermaid_tree(body) if lang.lower() == "mermaid" else None
            out += ["", f"{mark}text", *tree, mark, ""] if tree else ["", line.strip(), *body, mark, ""]
            stack, blank = [], True
            continue
        stripped = line.lstrip(" ")
        if not stripped:
            out.append("")
            blank = True
            continue
        indent = len(line) - len(stripped)
        item = ITEM_RE.match(line)
        if item and not stack and not blank and item[2][0].isdigit() and int(item[2][:-1]) != 1:
            item = None  # only a bullet or "1." can interrupt a paragraph
        if item:
            in_list = bool(stack)
            while stack and indent < stack[-1]:
                stack.pop()
            if not in_list and not blank:
                out.append("")  # Python-Markdown needs a blank line before a list
            marker = item[2][:-1] + "." if item[2].endswith(")") else item[2]
            out.append(" " * 4 * len(stack) + marker + " " + line[item.end() :])
            stack.append(indent + len(item[2]) + max(1, len(item[3])))
        else:
            if stack and (blank or HEADING_RE.match(line)):
                while stack and indent < stack[-1]:
                    stack.pop()
                if not stack and not blank:
                    out.append("")
            # inside a list: 4 spaces per level for Python-Markdown, keeping any extra indent
            out.append(" " * (4 * len(stack) + max(0, indent - stack[-1])) + stripped if stack else line)
        blank = False
    return "\n".join(out) + "\n"


class BareUrl(InlineProcessor):
    """Link a bare https:// URL, as GitHub does."""

    def handleMatch(self, m, data):
        link = etree.Element("a", href=m[0])
        link.text = AtomicString(m[0])
        return link, m.start(0), m.end(0)


def build_html(md_path: Path) -> str:
    """The Markdown file as one standalone, print-styled HTML page."""
    md = markdown.Markdown(
        extensions=["tables", "fenced_code", "codehilite", "toc", "sane_lists"],
        extension_configs={"codehilite": {"guess_lang": False}, "toc": {"slugify": github_slug}},
        output_format="html",
    )
    md.inlinePatterns.register(BareUrl(URL_RE, md), "bare_url", 105)
    body = md.convert(bridge(md_path.read_text(encoding="utf-8-sig")))
    # an escaped pipe inside code in a table cell keeps its backslash in Python-Markdown, not on GitHub
    body = re.sub(r"<t([dh])\b.*?</t\1>", lambda m: m[0].replace("\\|", "|"), body, flags=re.DOTALL)
    # images resolve against the Markdown's folder, wherever the HTML is written
    body = re.sub(
        r'(<img\b[^>]*\bsrc=")(?![a-zA-Z][\w+.-]*:|/|#)([^"]+)"',
        lambda m: m[1] + (md_path.parent / html.unescape(m[2])).resolve().as_uri() + '"',
        body,
    )
    ids = set(re.findall(r'\bid="([^"]+)"', body))
    missing = sorted({a for a in re.findall(r'href="#([^"]+)"', body) if a not in ids})
    if missing:
        print("warning: links to missing headings: " + ", ".join(missing), file=sys.stderr)

    h1 = re.search(r"<h1\b[^>]*>(.*?)</h1>", body, re.DOTALL)
    title = html.unescape(re.sub(r"<[^>]+>", "", h1[1])).strip() if h1 else md_path.stem
    tokens = dict(re.findall(r"--uw-([\w-]+):\s*([^;]+);", TOKENS.read_text(encoding="utf-8")))
    footer = title.split(":")[0].replace("\\", "\\\\").replace('"', '\\"')
    css = re.sub(r"var\(--uw-([\w-]+)\)", lambda m: tokens[m[1]].strip(), CSS.replace("__FOOTER__", footer))
    return PAGE.format(title=html.escape(title), css=css, body=body)


def find_edge(given: str | None) -> Path:
    """msedge.exe: the given path, else the standard install folders, else PATH."""
    if given:
        candidates = [Path(given)]
    else:
        program_dirs = [os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"), os.environ.get("ProgramFiles")]
        candidates = [Path(d) / EDGE_DIR for d in program_dirs if d]
        candidates += [Path(p) for p in [shutil.which("msedge")] if p]
    for path in candidates:
        if path.is_file():
            return path
    raise SystemExit("Microsoft Edge not found; pass its msedge.exe with --edge")


def print_pdf(edge: Path, page: Path, workdir: Path) -> Path:
    """Print the HTML page to PDF with headless Edge in a throwaway profile (a running Edge is not touched)."""
    pdf = workdir / "out.pdf"
    cmd = [
        str(edge),
        "--headless=new",
        "--disable-gpu",
        "--no-pdf-header-footer",
        "--generate-pdf-document-outline",
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-extensions",
        f"--user-data-dir={workdir / 'edge-profile'}",
        f"--print-to-pdf={pdf}",
        page.resolve().as_uri(),
    ]
    try:
        run = subprocess.run(cmd, check=False, capture_output=True, text=True, errors="replace", timeout=180)
    except subprocess.TimeoutExpired:
        raise SystemExit("Edge did not finish printing within 180 s") from None
    if not pdf.is_file() or pdf.stat().st_size == 0 or pdf.read_bytes()[:5] != b"%PDF-":
        raise SystemExit(f"Edge wrote no PDF (exit code {run.returncode}):\n{run.stderr[-2000:]}")
    return pdf


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Render docs/HOW_IT_WORKS.md to PDF with headless Edge.")
    ap.add_argument("--md", default="docs/HOW_IT_WORKS.md", help="Markdown source (default: %(default)s)")
    ap.add_argument("--out", default="docs/HOW_IT_WORKS.pdf", help="PDF to write (default: %(default)s)")
    ap.add_argument("--html", help="keep the intermediate HTML at this path (default: a temporary folder)")
    ap.add_argument("--edge", help="path to msedge.exe (default: the standard install folders)")
    args = ap.parse_args(argv)
    md_path, out_path = ROOT / args.md, ROOT / args.out  # an absolute path stays as given

    edge = find_edge(args.edge)
    page = build_html(md_path)
    with tempfile.TemporaryDirectory(prefix="how-it-works-pdf-", ignore_cleanup_errors=True) as tmp:
        html_path = ROOT / args.html if args.html else Path(tmp) / "page.html"
        html_path.parent.mkdir(parents=True, exist_ok=True)
        html_path.write_text(page, encoding="utf-8")
        pdf = print_pdf(edge, html_path, Path(tmp))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(pdf, out_path)

    data = out_path.read_bytes()
    pages = len(re.findall(rb"/Type\s*/Page\b", data))
    print(f"wrote {out_path} ({pages} pages, {len(data) / 1024:.0f} KB)")
    if args.html:
        print(f"kept {html_path}")


if __name__ == "__main__":
    main()
