# Publishing the book to GitHub Pages

The site is static HTML, so hosting is free and takes about two minutes. Two
options; the first is recommended.

## Option 1 — GitHub Actions (recommended)

The workflow in `.github/workflows/deploy.yml` rebuilds the book from source on
every push to `main` and publishes it. Build output (`docs/`) stays out of git,
so what is served always matches the source. There is nothing to remember.

1. **Create a public repo** and push this folder.

   ```bash
   git init
   git add .
   git commit -m "The book, scaffolded."
   git branch -M main
   git remote add origin https://github.com/<you>/llm-textbook.git
   git push -u origin main
   ```

   The repo must be **public** for free Pages hosting. Pages on a private repo
   requires a paid GitHub plan.

2. **Turn Pages on.** In the repo: **Settings → Pages → Build and deployment →
   Source**, choose **GitHub Actions**. (Not "Deploy from a branch".)

3. **Wait for the build.** The Actions tab shows the run. On success your book
   is live at:

   ```
   https://<you>.github.io/llm-textbook/
   ```

From then on, `git push` publishes. To redeploy without a code change, use
**Actions → Build and deploy the book → Run workflow**.

## Option 2 — no CI, commit the built site

Simpler to reason about, but you must remember to rebuild before pushing, and
the build output lives in git.

1. Delete the `docs/` line from `.gitignore`.
2. Build and commit:

   ```bash
   python figures/make_figures.py
   python build.py
   git add -A && git commit -m "Rebuild." && git push
   ```

3. **Settings → Pages → Source → Deploy from a branch**, then choose branch
   `main` and folder **`/docs`**.

If you take this route, delete `.github/workflows/deploy.yml` so the two
mechanisms do not fight.

## Reading it on a phone

The layout is responsive: below roughly 48rem the contents collapse into a
slide-in drawer behind the **☰ Contents** button, tables scroll on their own
rather than pushing the page sideways, and figures scale to the column. Verified
clean down to a 320 px viewport.

In Safari on iOS, **Share → Add to Home Screen** gives you a standalone icon
that opens the book without browser chrome (the meta tags for this are already
in every page).

Two caveats worth knowing:

- **Math needs a network.** Equations render via MathJax loaded from a CDN, so
  they need a connection. Everything else — text, layout, figures, navigation —
  is fully self-contained. If you want true offline math, vendor MathJax into
  `assets/` and point `MATHJAX_HEAD` in `build.py` at the local copy.
- **Fonts degrade gracefully.** The book asks for Charter and the system sans,
  which iOS and macOS both have. Elsewhere it falls back to Georgia and a system
  sans without breaking the layout.

## Keeping it private

If you would rather not publish a draft to the open web:

- Push to a private repo and just read it locally with
  `python -m http.server -d docs`, or
- Deploy to a host with free private/password-protected static sites (Netlify
  and Cloudflare Pages both offer this on free tiers). Both can build with the
  same two commands: `pip install markdown pymdown-extensions pygments
  matplotlib && python figures/make_figures.py && python build.py`, publishing
  the `docs/` directory.
