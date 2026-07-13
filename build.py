"""Static-site generator for *Foundations of Large Language Models*.

Reads the structure from `toc.py` and the prose from `content/*.md`, then
writes a folder of linked, self-contained HTML files to `docs/`. Any chapter
without a matching markdown file is rendered as a stub built from its `outline`,
so the whole book stays navigable while it is being written.

Run with `python build.py`. There is no configuration and no partial build:
the whole site is regenerated every time, which keeps navigation consistent.
"""

from __future__ import annotations

import html
import re
import shutil
from pathlib import Path

import markdown

import toc

ROOT = Path(__file__).resolve().parent
CONTENT_DIR = ROOT / "content"
ASSETS_DIR = ROOT / "assets"
OUTPUT_DIR = ROOT / "docs"

BOOK_TITLE = "Foundations of Large Language Models"
BOOK_SUBTITLE = "Training, serving, and shipping LLMs — everything for a 2026 engineering interview."

# MathJax is loaded from a CDN, so equations require an internet connection to
# render. Everything else (fonts, layout, navigation) works fully offline.
# Shared <head> markup for every page. `viewport-fit=cover` and the status-bar
# tag make the book behave when saved to an iPhone home screen; `theme-color`
# tints Safari's chrome to match the paper background.
HEAD_META = """\
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="theme-color" content="#f4f3ee">
<meta name="color-scheme" content="light">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="default">
<meta name="apple-mobile-web-app-title" content="LLM Book">
<meta name="description" content="A textbook on how large language models are trained, aligned, served, and shipped.">
"""

MATHJAX_HEAD = """\
<script>
  window.MathJax = {
    tex: { inlineMath: [["\\\\(", "\\\\)"]], displayMath: [["\\\\[", "\\\\]"]] },
    options: { ignoreHtmlClass: ".*\\\\|", processHtmlClass: "arithmatex" }
  };
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js" async></script>
"""

MENU_SCRIPT = """\
<script>
  document.querySelector(".menu-toggle").addEventListener("click", function () {
    document.body.classList.toggle("nav-open");
  });
  document.addEventListener("keydown", function (event) {
    if (event.key === "Escape") {
      document.body.classList.remove("nav-open");
    }
  });
</script>
"""


def make_markdown() -> markdown.Markdown:
    """Return a configured Markdown converter.

    Extensions give fenced code, tables, callout boxes (`admonition`), heading
    ids (`toc`), and MathJax-ready math (`pymdownx.arithmatex`). The converter
    is reset between documents by the caller.
    """
    return markdown.Markdown(
        extensions=[
            "extra",
            "admonition",
            "sane_lists",
            "toc",
            "pymdownx.arithmatex",
            "pymdownx.superfences",
        ],
        extension_configs={
            "pymdownx.arithmatex": {"generic": True},
            "toc": {"permalink": False},
        },
    )


HEADING_PATTERN = re.compile(
    r"<h(?P<level>[23])(?P<attrs>[^>]*)>(?P<inner>.*?)</h(?P=level)>", re.DOTALL
)
ID_PATTERN = re.compile(r'id="(?P<id>[^"]+)"')


def number_headings(
    body_html: str, label: str
) -> tuple[str, list[tuple[str, str, str]]]:
    """Prefix section headings with numbers and collect the page's h2 outline.

    `label` is the chapter's display number (e.g. "4" or "A"). h2 headings
    become "<label>.<n>" and h3 headings become "<label>.<n>.<m>". When `label`
    is empty (front matter) no numbers are added. Returns the rewritten HTML and
    a list of (anchor_id, number_or_empty, text) tuples for the on-this-page
    rail, one per h2.
    """
    state = {"section": 0, "subsection": 0}
    rail: list[tuple[str, str, str]] = []

    def replace(match: re.Match[str]) -> str:
        level = match.group("level")
        attrs = match.group("attrs")
        inner = match.group("inner")

        id_match = ID_PATTERN.search(attrs)
        anchor = id_match.group("id") if id_match else ""

        if level == "2":
            state["section"] += 1
            state["subsection"] = 0
            number = f"{label}.{state['section']}" if label else ""
            plain_text = re.sub(r"<[^>]+>", "", inner).strip()
            rail.append((anchor, number, plain_text))
        else:
            assert state["section"] > 0, (
                f"An h3 appeared before any h2 in chapter '{label}'; chapters "
                "must open each subsection under a numbered h2 section."
            )
            state["subsection"] += 1
            number = (
                f"{label}.{state['section']}.{state['subsection']}" if label else ""
            )

        prefix = f'<span class="heading-number">{number}</span>' if number else ""
        return f"<h{level}{attrs}>{prefix}{inner}</h{level}>"

    return HEADING_PATTERN.sub(replace, body_html), rail


FIGCAPTION_PATTERN = re.compile(r"<figcaption>(?P<caption>.*?)</figcaption>", re.DOTALL)
IMG_SRC_PATTERN = re.compile(r'<img[^>]*\ssrc="(?P<src>[^"]+)"')


def number_figures(body_html: str, label: str, slug: str) -> str:
    """Prefix every figcaption with an auto-incremented figure number.

    Figures are numbered `<label>.<n>` in document order, matching the section
    numbering scheme, so reordering chapters in `toc.py` renumbers the figures
    with no edits to the prose. Also asserts that every referenced image exists,
    which turns a silently broken image into a loud build failure.
    """
    for match in IMG_SRC_PATTERN.finditer(body_html):
        src = match.group("src")
        if src.startswith(("http://", "https://", "data:")):
            continue
        asset_path = ASSETS_DIR / Path(src).relative_to("assets")
        assert asset_path.exists(), (
            f"Chapter '{slug}' references a missing image: {src}. "
            "Run `python figures/make_figures.py` to regenerate the figures."
        )

    counter = {"n": 0}

    def replace(match: re.Match[str]) -> str:
        counter["n"] += 1
        number = f"{label}.{counter['n']}" if label else str(counter["n"])
        caption = match.group("caption").strip()
        return (
            f'<figcaption><span class="fig-label">Figure {number}</span>{caption}'
            f"</figcaption>"
        )

    return FIGCAPTION_PATTERN.sub(replace, body_html)


def wrap_tables(body_html: str) -> str:
    """Wrap every table in a horizontally scrollable container.

    A table whose content is wider than the reading column would otherwise push
    the whole page sideways on a phone. Wrapping it lets the table scroll on its
    own while the page stays put.
    """
    body_html = body_html.replace("<table>", '<div class="table-scroll"><table>')
    return body_html.replace("</table>", "</table></div>")


def synthesize_stub(chapter: toc.Chapter) -> str:
    """Build markdown for a chapter that has no drafted content file yet.

    The stub renders the planned section headings from the outline so the page
    is a real, numbered, navigable placeholder rather than an empty file.
    """
    assert chapter.outline, (
        f"Chapter '{chapter.slug}' has neither a content file nor an outline; "
        "add one of them in toc.py or content/."
    )
    lines = [
        '<div class="stub-banner">Draft pending — this page is an outline of '
        "planned content. Write it by creating "
        f"<code>content/{chapter.slug}.md</code>.</div>",
        "",
    ]
    for section_title, note in chapter.outline:
        lines.append(f"## {section_title}")
        lines.append("")
        lines.append(f'<span class="stub-note">Planned: {html.escape(note)}</span>')
        lines.append("")
    return "\n".join(lines)


def render_body(
    chapter: toc.Chapter, md: markdown.Markdown
) -> tuple[str, list[tuple[str, str, str]]]:
    """Render one chapter's body HTML and its rail entries."""
    content_path = CONTENT_DIR / f"{chapter.slug}.md"
    if content_path.exists():
        source = content_path.read_text(encoding="utf-8")
        assert source.strip(), f"Content file '{content_path}' is empty."
    else:
        source = synthesize_stub(chapter)

    md.reset()
    body_html = md.convert(source)
    body_html = number_figures(body_html, chapter.label, chapter.slug)
    body_html = wrap_tables(body_html)
    return number_headings(body_html, chapter.label)


def sidebar_html(active_slug: str) -> str:
    """Build the sidebar table of contents, marking the active page."""
    parts = [
        '<a class="book-brand" href="index.html">Foundations of<br>Large Language Models'
        f"<span>{html.escape(BOOK_SUBTITLE)}</span></a>"
    ]

    def link(chapter: toc.Chapter) -> str:
        active = " active" if chapter.slug == active_slug else ""
        num = chapter.label if chapter.label else "•"
        return (
            f'<li><a class="{active.strip()}" href="{chapter.slug}.html">'
            f'<span class="toc-num">{num}</span>'
            f"<span>{html.escape(chapter.title)}</span></a></li>"
        )

    parts.append('<ul class="toc-list">')
    parts.append(link(toc.PREFACE))
    parts.append("</ul>")

    for part in toc.BOOK:
        parts.append(f'<div class="toc-part">{html.escape(part.title)}</div>')
        parts.append('<ul class="toc-list">')
        for chapter in part.chapters:
            parts.append(link(chapter))
        parts.append("</ul>")

    parts.append('<div class="toc-part">Appendices</div>')
    parts.append('<ul class="toc-list">')
    for chapter in toc.APPENDICES:
        parts.append(link(chapter))
    parts.append("</ul>")

    return "\n".join(parts)


def rail_html(entries: list[tuple[str, str, str]]) -> str:
    """Build the on-this-page rail from collected h2 entries."""
    if not entries:
        return '<nav class="rail"></nav>'
    items = ['<nav class="rail"><div class="rail-title">On this page</div>']
    for anchor, number, text in entries:
        assert (
            anchor
        ), "A section heading is missing an id; the toc extension should add one."
        prefix = f"{number}  " if number else ""
        items.append(
            f'<a href="#{anchor}">{html.escape(prefix)}{html.escape(text)}</a>'
        )
    items.append("</nav>")
    return "\n".join(items)


def pager_html(index: int, pages: tuple[toc.Chapter, ...]) -> str:
    """Build the previous/next footer for the page at `index` in reading order."""
    blocks = ['<nav class="pager">']

    if index == 0:
        blocks.append(
            '<a class="pager-prev" href="index.html">'
            '<span class="pager-dir">Previous</span>'
            '<span class="pager-title">Cover</span></a>'
        )
    else:
        prev_page = pages[index - 1]
        blocks.append(
            f'<a class="pager-prev" href="{prev_page.slug}.html">'
            '<span class="pager-dir">Previous</span>'
            f'<span class="pager-title">{html.escape(prev_page.title)}</span></a>'
        )

    if index + 1 < len(pages):
        next_page = pages[index + 1]
        blocks.append(
            f'<a class="pager-next" href="{next_page.slug}.html">'
            '<span class="pager-dir">Next</span>'
            f'<span class="pager-title">{html.escape(next_page.title)}</span></a>'
        )
    else:
        blocks.append('<span class="spacer"></span>')

    blocks.append("</nav>")
    return "\n".join(blocks)


def chapter_eyebrow(chapter: toc.Chapter) -> str:
    """Return the small label shown above a chapter title."""
    if chapter.slug == toc.PREFACE.slug:
        return "Front matter"
    if chapter.label.isdigit():
        return f"Chapter {chapter.label}"
    return f"Appendix {chapter.label}"


def page_html(
    chapter: toc.Chapter,
    index: int,
    pages: tuple[toc.Chapter, ...],
    md: markdown.Markdown,
) -> str:
    """Assemble the full HTML document for one chapter."""
    body, rail_entries = render_body(chapter, md)

    number_span = (
        f'<span class="chapter-number">{chapter.label}</span>' if chapter.label else ""
    )
    title_line = (
        f'<h1 class="chapter-title">{number_span}{html.escape(chapter.title)}</h1>'
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
{HEAD_META}<title>{html.escape(chapter.title)} · {html.escape(BOOK_TITLE)}</title>
<link rel="stylesheet" href="assets/style.css">
{MATHJAX_HEAD}</head>
<body>
<button class="menu-toggle" aria-label="Open contents">☰ Contents</button>
<div class="nav-scrim" onclick="document.body.classList.remove('nav-open')"></div>
<div class="layout">
<aside class="sidebar">
{sidebar_html(chapter.slug)}
</aside>
<main class="main">
<div class="content-wrap">
<article>
<div class="chapter-eyebrow">{html.escape(chapter_eyebrow(chapter))}</div>
{title_line}
{body}
{pager_html(index, pages)}
</article>
{rail_html(rail_entries)}
</div>
</main>
</div>
{MENU_SCRIPT}</body>
</html>
"""


def landing_html() -> str:
    """Assemble the cover / table-of-contents landing page."""
    rows = []

    def row(chapter: toc.Chapter) -> str:
        num = chapter.label if chapter.label else "•"
        return (
            f'<a class="chapter-row" href="{chapter.slug}.html">'
            f'<span class="chapter-row-num">{num}</span>'
            f'<span class="chapter-row-title">{html.escape(chapter.title)}</span></a>'
        )

    rows.append('<div class="part-block">')
    rows.append(row(toc.PREFACE))
    rows.append("</div>")

    for part in toc.BOOK:
        rows.append('<div class="part-block">')
        rows.append(f"<h2>{html.escape(part.title)}</h2>")
        for chapter in part.chapters:
            rows.append(row(chapter))
        rows.append("</div>")

    rows.append('<div class="part-block">')
    rows.append("<h2>Appendices</h2>")
    for chapter in toc.APPENDICES:
        rows.append(row(chapter))
    rows.append("</div>")

    body = "\n".join(rows)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
{HEAD_META}<title>{html.escape(BOOK_TITLE)}</title>
<link rel="stylesheet" href="assets/style.css">
</head>
<body>
<button class="menu-toggle" aria-label="Open contents">☰ Contents</button>
<div class="nav-scrim" onclick="document.body.classList.remove('nav-open')"></div>
<div class="layout">
<aside class="sidebar">
{sidebar_html("index")}
</aside>
<main class="main">
<div class="content-wrap">
<article class="landing">
<h1>{html.escape(BOOK_TITLE)}</h1>
<p class="subtitle">{html.escape(BOOK_SUBTITLE)}</p>
{body}
</article>
</div>
</main>
</div>
{MENU_SCRIPT}</body>
</html>
"""


def build() -> None:
    """Regenerate the entire site into `docs/`."""
    assert CONTENT_DIR.exists(), f"Missing content directory: {CONTENT_DIR}"
    assert (ASSETS_DIR / "style.css").exists(), "Missing assets/style.css."

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True)
    shutil.copytree(ASSETS_DIR, OUTPUT_DIR / "assets")

    pages = toc.all_pages()
    md = make_markdown()

    for index, chapter in enumerate(pages):
        document = page_html(chapter, index, pages, md)
        (OUTPUT_DIR / f"{chapter.slug}.html").write_text(document, encoding="utf-8")

    (OUTPUT_DIR / "index.html").write_text(landing_html(), encoding="utf-8")

    # GitHub Pages runs Jekyll by default, which ignores paths beginning with an
    # underscore and can silently drop assets. This opts out and serves the
    # folder verbatim. It is harmless when serving locally.
    (OUTPUT_DIR / ".nojekyll").write_text("", encoding="utf-8")

    drafted = sum(1 for c in pages if (CONTENT_DIR / f"{c.slug}.md").exists())
    print(f"Built {len(pages) + 1} pages into {OUTPUT_DIR.relative_to(ROOT)}/")
    print(
        f"  {drafted}/{len(pages)} chapters have drafted content; the rest are stubs."
    )


if __name__ == "__main__":
    build()
