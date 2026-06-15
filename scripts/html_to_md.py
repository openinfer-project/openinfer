#!/usr/bin/env python3
"""Convert HTML pages to Markdown.

Designed for Sphinx / NVIDIA docs, but works on any HTML with optional CSS selector.

Examples:
    # Local HTML -> Markdown
    uv run --with beautifulsoup4 --with lxml --with markdownify python scripts/html_to_md.py -i page.html -o page.md

    # Fetch URL (Sphinx article extraction)
    uv run --with beautifulsoup4 --with lxml --with markdownify python scripts/html_to_md.py --url https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/green-contexts.html -o green-contexts.md

    # Custom main-content selector
    uv run --with beautifulsoup4 --with lxml --with markdownify python scripts/html_to_md.py -i page.html --selector 'div#main-content' -o page.md
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup
from markdownify import markdownify as html_to_markdown

DEFAULT_SPHINX_SELECTOR = "article.bd-article"
DEFAULT_STRIP_SELECTORS = (
    ".prev-next-area",
    ".page-info",
    "script",
    "style",
    "noscript",
)

# Sphinx cross-ref anchors appended to headings/tables, e.g. [#](#foo "Link to this heading")
_SPHINX_ANCHOR_RE = re.compile(
    r'\[#\]\(#[^)]*(?:\s+"[^"]*")?\)'
)
# Trailing lone # from Sphinx headings, e.g. "## Title#"
_TRAILING_HASH_RE = re.compile(r"^(#{1,6}\s+.+?)#+\s*$", re.MULTILINE)
# Collapse 3+ blank lines
_BLANK_LINES_RE = re.compile(r"\n{3,}")


def fetch_url(url: str, timeout: float = 60.0) -> str:
    req = Request(
        url,
        headers={
            "User-Agent": "pegainfer-html-to-md/1.0 (+https://github.com/)",
            "Accept": "text/html,application/xhtml+xml",
        },
    )
    with urlopen(req, timeout=timeout) as resp:
        charset = resp.headers.get_content_charset() or "utf-8"
        return resp.read().decode(charset, errors="replace")


def load_html(*, input_path: Path | None, url: str | None) -> tuple[str, str | None]:
    if input_path and url:
        raise SystemExit("error: use either --input or --url, not both")
    if input_path:
        return input_path.read_text(encoding="utf-8", errors="replace"), None
    if url:
        return fetch_url(url), url
    raise SystemExit("error: one of --input or --url is required")


def extract_fragment(html: str, selector: str, strip_selectors: tuple[str, ...]) -> tuple[str, str | None]:
    soup = BeautifulSoup(html, "lxml")
    root = soup.select_one(selector)
    if root is None:
        raise SystemExit(f"error: selector not found: {selector!r}")

    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else None

    for sel in strip_selectors:
        for node in root.select(sel):
            node.decompose()

    return str(root), title


def postprocess_markdown(text: str) -> str:
    text = _SPHINX_ANCHOR_RE.sub("", text)
    text = _TRAILING_HASH_RE.sub(r"\1", text)
    text = _BLANK_LINES_RE.sub("\n\n", text)
    return text.strip() + "\n"


def build_frontmatter(*, source: str | None, title: str | None) -> str:
    lines = ["---"]
    if title:
        lines.append(f"title: {title}")
    if source:
        lines.append(f"source: {source}")
    lines.append(f"fetched: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}")
    lines.append("---\n")
    return "\n".join(lines)


def convert(
    html: str,
    *,
    selector: str,
    strip_selectors: tuple[str, ...],
    code_language: str,
    include_frontmatter: bool,
    source: str | None,
) -> str:
    fragment, title = extract_fragment(html, selector, strip_selectors)
    body = html_to_markdown(
        fragment,
        heading_style="ATX",
        bullets="-",
        code_language=code_language,
        strip=["script", "style"],
    )
    body = postprocess_markdown(body)
    if include_frontmatter:
        return build_frontmatter(source=source, title=title) + "\n" + body
    return body


def default_output_path(*, input_path: Path | None, url: str | None) -> Path:
    if input_path:
        return input_path.with_suffix(".md")
    assert url is not None
    name = Path(urlparse(url).path).name or "page.html"
    return Path(name).with_suffix(".md")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("-i", "--input", type=Path, help="Local HTML file")
    src.add_argument("--url", help="Fetch HTML from URL")
    parser.add_argument("-o", "--output", type=Path, help="Output Markdown path (default: derived from input/url)")
    parser.add_argument(
        "--selector",
        default=DEFAULT_SPHINX_SELECTOR,
        help=f"CSS selector for main content (default: {DEFAULT_SPHINX_SELECTOR!r})",
    )
    parser.add_argument(
        "--strip",
        action="append",
        default=[],
        help="Extra CSS selectors to remove before conversion (repeatable)",
    )
    parser.add_argument(
        "--code-language",
        default="cpp",
        help="Default fenced-code language tag (default: cpp)",
    )
    parser.add_argument(
        "--no-frontmatter",
        action="store_true",
        help="Omit YAML frontmatter with title/source/fetched timestamp",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    html, source = load_html(input_path=args.input, url=args.url)
    strip_selectors = DEFAULT_STRIP_SELECTORS + tuple(args.strip)
    md = convert(
        html,
        selector=args.selector,
        strip_selectors=strip_selectors,
        code_language=args.code_language,
        include_frontmatter=not args.no_frontmatter,
        source=source or (str(args.input.resolve()) if args.input else None),
    )
    out = args.output or default_output_path(input_path=args.input, url=args.url)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    print(f"wrote {out} ({len(md)} bytes)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
