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

import quizzes
import references
import toc

ROOT = Path(__file__).resolve().parent
CONTENT_DIR = ROOT / "content"
ASSETS_DIR = ROOT / "assets"
OUTPUT_DIR = ROOT / "docs"

BOOK_TITLE = "Foundations of Large Language Models"
BOOK_SUBTITLE = "Training, serving, and shipping LLMs — everything for a 2026 engineering interview."

# ---------------------------------------------------------------------------
# Integration with a personal site. Everything needed to fold this book into a
# wider site lives here; today the book is deliberately standalone, so the
# optional fields are empty and nothing extra is rendered.
# ---------------------------------------------------------------------------

# Set to your site to add a "back to <SITE_NAME>" link at the top of the
# sidebar. Leave both empty to keep the book standalone (no link is rendered).
SITE_NAME = ""
SITE_URL = ""

# The book's canonical address, e.g. "https://zorian15.github.io/llm-textbook".
# Set this to emit canonical and Open Graph tags so shared links preview nicely.
# Leave empty while the address is not settled.
CANONICAL_BASE = ""

# While True, every page asks search engines not to index it. The book stays
# reachable by URL; it just will not show up in search results. Flip to False
# when the draft is ready to be found.
DRAFT = True

# MathJax is loaded from a CDN, so equations require an internet connection to
# render. Everything else (fonts, layout, navigation) works fully offline.
# Shared <head> markup for every page. `viewport-fit=cover` and the status-bar
# tag make the book behave when saved to an iPhone home screen; `theme-color`
# tints Safari's chrome to match the paper background.
HEAD_META_BASE = """\
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="theme-color" content="#f4f3ee">
<meta name="color-scheme" content="light">
<link rel="icon" href="assets/icon.svg" type="image/svg+xml">
<link rel="apple-touch-icon" href="assets/apple-touch-icon.png">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="default">
<meta name="apple-mobile-web-app-title" content="LLM Book">
<meta name="description" content="A textbook on how large language models are trained, aligned, served, and shipped.">
"""


def head_meta(page_title: str, page_file: str) -> str:
    """Return the <head> metadata for one page.

    Emits the shared tags always, a `noindex` directive while `DRAFT` is True,
    and canonical plus Open Graph tags once `CANONICAL_BASE` is set. Keeping
    this in one function means the landing page and the chapters can never drift
    apart.
    """
    tags = [HEAD_META_BASE]

    if DRAFT:
        tags.append('<meta name="robots" content="noindex, nofollow">\n')

    if CANONICAL_BASE:
        base = CANONICAL_BASE.rstrip("/")
        url = f"{base}/{page_file}"
        tags.append(f'<link rel="canonical" href="{url}">\n')
        tags.append('<meta property="og:type" content="article">\n')
        tags.append(f'<meta property="og:title" content="{html.escape(page_title)}">\n')
        tags.append(
            f'<meta property="og:description" content="{html.escape(BOOK_SUBTITLE)}">\n'
        )
        tags.append(f'<meta property="og:url" content="{url}">\n')
        tags.append(
            f'<meta property="og:site_name" content="{html.escape(BOOK_TITLE)}">\n'
        )

    return "".join(tags)


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

# Makes the end-of-chapter quizzes interactive. First it shuffles each
# question's options on load, so the correct answer's position varies every
# visit and can never be guessed from where it sits (the correct option is
# flagged by a data attribute, not by its order, so reordering is safe).
# Clicking an option then locks the question, marks the chosen option right or
# wrong, reveals the correct one if the reader missed it, and unhides the
# explanation. It is a no-op on pages with no quiz. The reveal is an instant
# hidden-toggle with no animation, so it needs no reduced-motion handling.
QUIZ_SCRIPT = """\
<script>
  // Shuffle each question's options so their order is randomized per visit.
  // A Fisher-Yates shuffle gives a uniform permutation; appending the buttons
  // in the shuffled order reorders them in place because they already share a
  // parent. Correctness rides on data-correct, so order does not affect it.
  document.querySelectorAll(".quiz-options").forEach(function (group) {
    var options = Array.prototype.slice.call(group.children);
    for (var i = options.length - 1; i > 0; i--) {
      var j = Math.floor(Math.random() * (i + 1));
      var tmp = options[i];
      options[i] = options[j];
      options[j] = tmp;
    }
    options.forEach(function (option) {
      group.appendChild(option);
    });
  });

  document.querySelectorAll(".quiz-option").forEach(function (button) {
    button.addEventListener("click", function () {
      var question = button.closest(".quiz-q");
      if (question.classList.contains("answered")) return;
      question.classList.add("answered");
      button.classList.add(
        button.dataset.correct === "true" ? "is-correct" : "is-incorrect"
      );
      question.querySelectorAll(".quiz-option").forEach(function (option) {
        if (option.dataset.correct === "true") option.classList.add("is-correct");
        option.disabled = true;
      });
      var explanation = question.querySelector(".quiz-explain");
      if (explanation) explanation.hidden = false;
    });
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


CITATION_PATTERN = re.compile(r"\[(?P<body>@[^\]]+)\]")
FENCE_PATTERN = re.compile(r"^\s*(```|~~~)")


def citation_html(keys: list[str]) -> str:
    """Render one parenthetical author-year citation for the given keys."""
    links = []
    for key in keys:
        ref = references.REFERENCES[key]
        label = html.escape(ref.in_text_label())
        links.append(f'<a href="#ref-{key}">{label}</a>')
    return f'<span class="citation">({"; ".join(links)})</span>'


def process_citations(source: str, slug: str) -> tuple[str, list[str]]:
    """Replace every `[@key]` (or `[@a; @b]`) citation with linked HTML.

    Fenced code blocks are left untouched. Returns the rewritten source and
    the cited keys in order of appearance. An unknown key is a build failure,
    not a silently unlinked citation.
    """
    cited: list[str] = []

    def replace(match: re.Match[str]) -> str:
        keys: list[str] = []
        for segment in match.group("body").split(";"):
            segment = segment.strip()
            assert segment.startswith("@"), (
                f"Malformed citation {match.group(0)!r} in chapter '{slug}'; "
                "every key in a citation must start with '@'."
            )
            key = segment[1:]
            assert key in references.REFERENCES, (
                f"Chapter '{slug}' cites unknown key '@{key}'; "
                "add the entry to references.py."
            )
            keys.append(key)
        cited.extend(keys)
        return citation_html(keys)

    lines = []
    in_fence = False
    for line in source.split("\n"):
        if FENCE_PATTERN.match(line):
            in_fence = not in_fence
            lines.append(line)
            continue
        lines.append(line if in_fence else CITATION_PATTERN.sub(replace, line))
    assert not in_fence, f"Unclosed code fence in chapter '{slug}'."
    return "\n".join(lines), cited


def format_reference_authors(ref: references.Reference) -> str:
    """Render an entry's author list in the reference-list style."""
    authors = [html.escape(author) for author in ref.authors]
    if ref.truncated:
        return ", ".join(authors) + ", et al."
    if len(authors) == 1:
        return authors[0]
    return ", ".join(authors[:-1]) + ", &amp; " + authors[-1]


def references_section_html(cited_keys: list[str]) -> str:
    """Build a chapter's References section from the keys it cites.

    Entries are ordered alphabetically by first author, then year, matching
    the author-year in-text style.
    """
    assert cited_keys, "references_section_html called with no citations."
    unique = sorted(
        set(cited_keys),
        key=lambda k: (
            references.REFERENCES[k].first_author_family().lower(),
            references.REFERENCES[k].year,
            k,
        ),
    )
    items = []
    for key in unique:
        ref = references.REFERENCES[key]
        title = html.escape(ref.title)
        link = ref.link()
        if link:
            title = f'<a href="{link}">{title}</a>'
        tail = f" arXiv:{ref.arxiv}." if ref.arxiv else ""
        items.append(
            f'<li id="ref-{key}">{format_reference_authors(ref)} ({ref.year}). '
            f"{title}. <em>{html.escape(ref.venue)}</em>.{tail}</li>"
        )
    return (
        '<section class="references">\n<h2 id="references">References</h2>\n'
        '<ul class="reference-list">\n' + "\n".join(items) + "\n</ul>\n</section>"
    )


def quiz_section_html(slug: str) -> str:
    """Render a chapter's "Check yourself" quiz as static, interactive HTML.

    Each question renders as a list item with a prompt, a group of option
    buttons (the correct one flagged via a data attribute for the inline
    script), and a hidden explanation revealed after answering. Options are
    emitted in source order but the inline script shuffles them on load, so the
    rendered position of the correct answer carries no information. The build
    asserts exactly one correct option per question, so a malformed quiz fails
    loudly rather than shipping a question with no answer or two.
    """
    questions = quizzes.QUIZZES[slug]
    items = []
    for index, question in enumerate(questions, start=1):
        qid = f"quiz-{slug}-q{index}"
        correct_count = 0
        option_html = []
        for option_index, option in enumerate(question.options):
            is_correct = option_index == question.answer
            correct_count += is_correct
            option_html.append(
                f'<button type="button" class="quiz-option" '
                f'data-correct="{"true" if is_correct else "false"}">'
                f"{html.escape(option)}</button>"
            )
        assert (
            correct_count == 1
        ), f"Question {qid} must have exactly one correct option, has {correct_count}."
        items.append(
            f'<li class="quiz-q" id="{qid}">\n'
            f'<p class="quiz-prompt">{html.escape(question.prompt)}</p>\n'
            f'<div class="quiz-options" role="group" aria-label="Answer options">\n'
            + "\n".join(option_html)
            + "\n</div>\n"
            f'<div class="quiz-explain" hidden aria-live="polite">'
            f"{html.escape(question.explanation)}</div>\n</li>"
        )
    return (
        '<section class="quiz" aria-labelledby="check-yourself">\n'
        '<h2 id="check-yourself">Check yourself</h2>\n'
        '<p class="quiz-intro">Interview-style questions on this chapter. '
        "Pick an answer to see whether it holds up.</p>\n"
        '<ol class="quiz-list">\n' + "\n".join(items) + "\n</ol>\n</section>"
    )


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
) -> tuple[str, list[tuple[str, str, str]], list[str]]:
    """Render one chapter's body HTML, its rail entries, and its cited keys."""
    content_path = CONTENT_DIR / f"{chapter.slug}.md"
    if content_path.exists():
        source = content_path.read_text(encoding="utf-8")
        assert source.strip(), f"Content file '{content_path}' is empty."
    else:
        source = synthesize_stub(chapter)

    source, cited_keys = process_citations(source, chapter.slug)
    md.reset()
    body_html = md.convert(source)
    body_html = number_figures(body_html, chapter.label, chapter.slug)
    body_html = wrap_tables(body_html)
    body_html, rail = number_headings(body_html, chapter.label)

    # The References section is appended after heading numbering on purpose:
    # it is back matter and stays unnumbered, like the numbered-section rule
    # in reverse.
    if cited_keys:
        body_html += references_section_html(cited_keys)
        rail.append(("references", "", "References"))

    # The quiz is independent of citations: a chapter can have one without the
    # other. Like References, it is appended after heading numbering so it stays
    # an unnumbered back-matter section.
    if chapter.slug in quizzes.QUIZZES:
        body_html += quiz_section_html(chapter.slug)
        rail.append(("check-yourself", "", "Check yourself"))
    return body_html, rail, cited_keys


def sidebar_html(active_slug: str) -> str:
    """Build the sidebar table of contents, marking the active page."""
    parts: list[str] = []

    # A way home, rendered only once the book is part of a wider site.
    if SITE_URL:
        assert SITE_NAME, "SITE_URL is set but SITE_NAME is empty; set both or neither."
        parts.append(
            f'<a class="site-backlink" href="{SITE_URL}">&#8592; '
            f"{html.escape(SITE_NAME)}</a>"
        )

    parts.append(
        '<a class="book-brand" href="index.html">Foundations of<br>Large Language Models'
        f"<span>{html.escape(BOOK_SUBTITLE)}</span></a>"
    )

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
) -> tuple[str, list[str]]:
    """Assemble the full HTML document for one chapter.

    Returns the document and the chapter's cited reference keys so the build
    can report which entries in references.py are never cited.
    """
    body, rail_entries, cited_keys = render_body(chapter, md)

    number_span = (
        f'<span class="chapter-number">{chapter.label}</span>' if chapter.label else ""
    )
    title_line = (
        f'<h1 class="chapter-title">{number_span}{html.escape(chapter.title)}</h1>'
    )

    document = f"""<!DOCTYPE html>
<html lang="en">
<head>
{head_meta(chapter.title, f'{chapter.slug}.html')}<title>{html.escape(chapter.title)} · {html.escape(BOOK_TITLE)}</title>
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
{MENU_SCRIPT}{QUIZ_SCRIPT}</body>
</html>
"""
    return document, cited_keys


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
{head_meta(BOOK_TITLE, 'index.html')}<title>{html.escape(BOOK_TITLE)}</title>
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
<div class="landing-hero">
<img class="landing-cover" src="assets/cover.svg" width="640" height="960"
  alt="The book cover: a row of tokens with attention arcs converging on the empty next position.">
<div>
<h1>{html.escape(BOOK_TITLE)}</h1>
<p class="subtitle">{html.escape(BOOK_SUBTITLE)}</p>
</div>
</div>
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
    for asset in ("cover.svg", "icon.svg", "apple-touch-icon.png"):
        assert (ASSETS_DIR / asset).exists(), (
            f"Missing assets/{asset}; run `python figures/make_figures.py` "
            "to generate the cover and icons."
        )

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True)
    shutil.copytree(ASSETS_DIR, OUTPUT_DIR / "assets")

    pages = toc.all_pages()

    # Every quiz must target a real chapter, so a typo in quizzes.py fails the
    # build instead of silently rendering nowhere.
    page_slugs = {chapter.slug for chapter in pages}
    unknown_quiz_slugs = set(quizzes.QUIZZES) - page_slugs
    assert not unknown_quiz_slugs, (
        f"quizzes.py targets unknown chapter slug(s): {sorted(unknown_quiz_slugs)}. "
        "Fix the key in quizzes.py to match a slug in toc.py."
    )

    md = make_markdown()

    cited_anywhere: set[str] = set()
    for index, chapter in enumerate(pages):
        document, cited_keys = page_html(chapter, index, pages, md)
        cited_anywhere.update(cited_keys)
        (OUTPUT_DIR / f"{chapter.slug}.html").write_text(document, encoding="utf-8")

    (OUTPUT_DIR / "index.html").write_text(landing_html(), encoding="utf-8")

    # GitHub Pages runs Jekyll by default, which ignores paths beginning with an
    # underscore and can silently drop assets. This opts out and serves the
    # folder verbatim. It is harmless when serving locally.
    (OUTPUT_DIR / ".nojekyll").write_text("", encoding="utf-8")

    # Keep robots.txt consistent with the DRAFT flag, so a draft stays out of
    # search results even if something links to it.
    robots = "User-agent: *\nDisallow: /\n" if DRAFT else "User-agent: *\nAllow: /\n"
    (OUTPUT_DIR / "robots.txt").write_text(robots, encoding="utf-8")

    drafted = sum(1 for c in pages if (CONTENT_DIR / f"{c.slug}.md").exists())
    print(f"Built {len(pages) + 1} pages into {OUTPUT_DIR.relative_to(ROOT)}/")
    print(
        f"  {drafted}/{len(pages)} chapters have drafted content; the rest are stubs."
    )

    uncited = sorted(set(references.REFERENCES) - cited_anywhere)
    if uncited:
        print(
            f"  Note: {len(uncited)} reference(s) in references.py are not yet "
            f"cited anywhere: {', '.join(uncited)}."
        )

    quizless = sorted(
        c.slug
        for c in pages
        if (CONTENT_DIR / f"{c.slug}.md").exists() and c.slug not in quizzes.QUIZZES
    )
    if quizless:
        print(
            f"  Note: {len(quizless)} drafted chapter(s) have no quiz in "
            f"quizzes.py: {', '.join(quizless)}."
        )

    # A gentle regression guard for the "correct answer is always the longest,
    # most detailed option" tell. Flag any chapter where the correct choice is
    # the strictly longest option in most of its questions. This is a note, not
    # an assertion: an occasional longest-correct answer is fine, only a
    # systematic pattern telegraphs the answer by shape.
    telling = []
    for slug, questions in quizzes.QUIZZES.items():
        longest_correct = sum(
            1
            for q in questions
            if len(q.options[q.answer])
            > max(len(o) for i, o in enumerate(q.options) if i != q.answer)
        )
        if longest_correct >= 0.8 * len(questions):
            telling.append(f"{slug} ({longest_correct}/{len(questions)})")
    if telling:
        print(
            f"  Note: in {len(telling)} chapter(s) the correct option is the "
            f"longest in most questions, a length tell: {', '.join(telling)}."
        )


if __name__ == "__main__":
    build()
