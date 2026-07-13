# Foundations of Large Language Models

A static, bookdown-style textbook on how LLMs are trained, aligned, served, and
wrapped into products — aimed at an engineer who wants everything needed to ace
an LLM interview in 2026. It is a folder of linked HTML files generated from
Markdown.

## Quickstart

```bash
pip install markdown pymdown-extensions pygments matplotlib
python figures/make_figures.py   # regenerate figures (SVG)
python build.py                  # regenerate the site
open docs/index.html        # or: python -m http.server -d docs
```

`python build.py` regenerates the entire `docs/` site from `toc.py` and
`content/`. Math renders via MathJax (needs internet); everything else is
offline.

## Structure

| Path | Purpose |
|------|---------|
| `toc.py` | The table of contents and per-chapter outlines — the source of truth. |
| `content/*.md` | Chapter prose. Missing chapters render as outline stubs. |
| `figures/make_figures.py` | Generates all figures as SVG into `assets/figures/`. |
| `assets/style.css` | The full visual design (macOS-native fonts, no CDN). |
| `build.py` | The static-site generator. |
| `docs/` | Build output. Do not edit by hand; not committed to git. |
| `.github/workflows/deploy.yml` | Builds and publishes to GitHub Pages on push. |
| `CLAUDE.md` | Authoring guide and style reference for editing sessions. |

## Status

4 of 30 chapters are drafted (`00-preface`, `01-introduction`, `04-transformer`,
`A-local-llms`); the rest are navigable stubs. See `CLAUDE.md` for how to fill
them in and the style to match.

## Publishing

`git push` to a public GitHub repo with Pages set to **GitHub Actions** builds
and publishes the book automatically. See `DEPLOY.md` for the two-minute setup.

Reads well on a phone: the contents collapse into a slide-in drawer, and Safari's
**Add to Home Screen** gives it a standalone icon.
