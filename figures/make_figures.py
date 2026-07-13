"""Generate every figure in the book as an SVG into `assets/figures/`.

Two kinds of figures live here:

* **Diagrams** are hand-authored SVG emitted from Python string templates. They
  are conceptual (boxes, arrows, grids), so a plotting library would fight us.
* **Plots** are matplotlib figures saved as SVG. They are quantitative, so the
  numbers should come from an explicit model rather than from an artist's hand.

Both share the book's palette so the figures sit on the page as if drawn by the
same pen as the text. Run `python figures/make_figures.py` from the repo root,
then rerun `build.py`. Every function returns the path it wrote so the module
can assert that the manifest and the filesystem agree.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "assets" / "figures"
ASSETS_DIR = ROOT / "assets"

# The book's palette, mirrored from assets/style.css. Keep these in sync.
PAPER = "#f4f3ee"
INK = "#17181b"
INK_SOFT = "#3b3d42"
MUTED = "#6a6d73"
RULE = "#e4e3dd"
RULE_STRONG = "#cfcdc4"
ACCENT = "#274b6d"
ACCENT_SOFT = "#eaf0f6"
AMBER = "#9c6b12"
VIOLET = "#6b4f9c"
BRICK = "#b04a3f"

SANS = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"
SERIF = "'Charter', 'Iowan Old Style', 'Palatino Linotype', Palatino, Georgia, serif"
MONO = "'SF Mono', 'SFMono-Regular', ui-monospace, Menlo, Consolas, monospace"


def write_svg(name: str, svg: str) -> Path:
    """Write a raw SVG string to the figures directory and return its path."""
    assert name.endswith(".svg"), f"Figure name must end in .svg, got '{name}'."
    assert svg.lstrip().startswith("<svg"), f"Figure '{name}' is not an SVG document."
    path = OUTPUT_DIR / name
    path.write_text(svg, encoding="utf-8")
    return path


def write_root_asset(name: str, svg: str) -> Path:
    """Write a raw SVG string to the assets root (cover, icon) and return its path."""
    assert name.endswith(".svg"), f"Asset name must end in .svg, got '{name}'."
    assert svg.lstrip().startswith("<svg"), f"Asset '{name}' is not an SVG document."
    path = ASSETS_DIR / name
    path.write_text(svg, encoding="utf-8")
    return path


def svg_doc(width: float, height: float, label: str, body: list[str]) -> str:
    """Wrap SVG body elements in a document with the book's default font.

    `label` becomes the accessible description; keep it plain ASCII so it needs
    no escaping. `body` is the list of element strings, in draw order.
    """
    head = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'font-family="{SANS}" role="img" aria-label="{label}">'
    )
    return "\n".join([head, *body, "</svg>"])


def arrow_marker(color: str, name: str) -> str:
    """Return a `<defs>` block holding one triangular arrowhead marker."""
    return (
        f'<defs><marker id="{name}" viewBox="0 0 10 10" refX="9" refY="5" '
        f'markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
        f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{color}"/></marker></defs>'
    )


def token_box(
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    *,
    fill: str = "#ffffff",
    stroke: str = RULE_STRONG,
    text_fill: str = INK,
    font_size: float = 12,
    weight: int = 400,
) -> list[str]:
    """Return a rounded rectangle with centered text: the book's token chip."""
    stroke_attr = "none" if stroke == "none" else stroke
    return [
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="6" '
        f'fill="{fill}" stroke="{stroke_attr}"/>',
        f'<text x="{x + w / 2:.1f}" y="{y + h / 2 + font_size * 0.35:.1f}" '
        f'font-size="{font_size}" font-weight="{weight}" text-anchor="middle" '
        f'fill="{text_fill}">{text}</text>',
    ]


def eyebrow(x: float, y: float, text: str, fill: str = MUTED) -> str:
    """Return a small uppercase section label, as used across the diagrams."""
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="11" font-weight="700" '
        f'fill="{fill}" letter-spacing="1">{text}</text>'
    )


def style_plot() -> None:
    """Apply the book's typographic style to matplotlib's global state."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
            "font.size": 9,
            "text.color": INK,
            "axes.edgecolor": RULE_STRONG,
            "axes.labelcolor": INK_SOFT,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "grid.color": RULE,
            "grid.linewidth": 0.8,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "svg.fonttype": "none",  # Keep text as text so it inherits page fonts.
        }
    )


def save_plot(fig: plt.Figure, name: str) -> Path:
    """Save a matplotlib figure as a transparent SVG and close it."""
    assert name.endswith(".svg"), f"Figure name must end in .svg, got '{name}'."
    path = OUTPUT_DIR / name
    fig.savefig(path, format="svg", transparent=True, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Figure 1.1 — the lifecycle of a model, from raw text to a shipped product.
# ---------------------------------------------------------------------------


def fig_lifecycle() -> Path:
    """Diagram: pretraining to post-training to the harness, as one pipeline."""
    stages = [
        (
            "Pretraining",
            "next-token prediction\non trillions of tokens",
            "base model",
            ACCENT,
        ),
        (
            "Supervised\nfine-tuning",
            "imitate demonstrations\nof good answers",
            "instruct model",
            VIOLET,
        ),
        (
            "Preference\noptimization",
            "learn from rankings\n(RLHF, DPO)",
            "aligned model",
            AMBER,
        ),
        (
            "The harness",
            "system prompt, tools,\nretrieval, guardrails",
            "the product",
            BRICK,
        ),
    ]

    box_w, box_h, gap = 148, 96, 42
    left, top = 16, 54
    width = left * 2 + len(stages) * box_w + (len(stages) - 1) * gap
    height = 224

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'font-family="{SANS}" role="img" '
        f'aria-label="The four stages that turn raw text into a shipped assistant.">',
        f'<text x="{left}" y="22" font-size="11" font-weight="700" fill="{MUTED}" '
        f'letter-spacing="1.1">WHAT YOU HAVE AFTER EACH STAGE</text>',
    ]

    for i, (title, detail, artifact, color) in enumerate(stages):
        x = left + i * (box_w + gap)

        # The artifact label sits above the box, naming what the stage produces.
        parts.append(
            f'<text x="{x + box_w / 2}" y="44" font-size="10.5" font-style="italic" '
            f'text-anchor="middle" fill="{color}">{artifact}</text>'
        )

        parts.append(
            f'<rect x="{x}" y="{top}" width="{box_w}" height="{box_h}" rx="8" '
            f'fill="#ffffff" stroke="{RULE_STRONG}" stroke-width="1"/>'
        )
        parts.append(
            f'<rect x="{x}" y="{top}" width="4" height="{box_h}" rx="2" fill="{color}"/>'
        )

        title_lines = title.split("\n")
        for j, line in enumerate(title_lines):
            parts.append(
                f'<text x="{x + 16}" y="{top + 26 + j * 15}" font-size="12.5" '
                f'font-weight="700" fill="{INK}">{line}</text>'
            )

        detail_top = top + 26 + len(title_lines) * 15 + 8
        for j, line in enumerate(detail.split("\n")):
            parts.append(
                f'<text x="{x + 16}" y="{detail_top + j * 13}" font-size="10.5" '
                f'fill="{MUTED}">{line}</text>'
            )

        if i < len(stages) - 1:
            ax = x + box_w + 8
            ay = top + box_h / 2
            parts.append(
                f'<path d="M {ax} {ay} L {ax + gap - 18} {ay}" stroke="{RULE_STRONG}" '
                f'stroke-width="1.5" marker-end="url(#arrow)"/>'
            )

    parts.append(
        f'<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" '
        f'markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
        f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{RULE_STRONG}"/></marker></defs>'
    )

    # The bracket underneath separates what changes weights from what does not.
    weights_end = left + 3 * box_w + 2 * gap
    bar_y = top + box_h + 22
    parts.append(
        f'<path d="M {left} {bar_y} L {left} {bar_y + 6} L {weights_end} {bar_y + 6} '
        f'L {weights_end} {bar_y}" fill="none" stroke="{RULE_STRONG}" stroke-width="1"/>'
    )
    parts.append(
        f'<text x="{(left + weights_end) / 2}" y="{bar_y + 22}" font-size="10.5" '
        f'text-anchor="middle" fill="{MUTED}">these change the weights</text>'
    )

    harness_x = left + 3 * (box_w + gap)
    parts.append(
        f'<path d="M {harness_x} {bar_y} L {harness_x} {bar_y + 6} '
        f'L {harness_x + box_w} {bar_y + 6} L {harness_x + box_w} {bar_y}" '
        f'fill="none" stroke="{RULE_STRONG}" stroke-width="1"/>'
    )
    parts.append(
        f'<text x="{harness_x + box_w / 2}" y="{bar_y + 22}" font-size="10.5" '
        f'text-anchor="middle" fill="{MUTED}">this does not</text>'
    )

    parts.append("</svg>")
    return write_svg("lifecycle.svg", "\n".join(parts))


# ---------------------------------------------------------------------------
# Figure 4.1 — self-attention as a soft dictionary lookup.
# ---------------------------------------------------------------------------


def fig_attention_lookup() -> Path:
    """Diagram: one query reading a weighted blend of values from all keys."""
    tokens = ["The", "cat", "sat", "on", "it"]
    # Illustrative softmax weights for the query token "it" attending backwards.
    weights = [0.05, 0.62, 0.14, 0.07, 0.12]
    assert (
        abs(sum(weights) - 1.0) < 1e-9
    ), "Illustrative attention weights must sum to 1."

    width, height = 640, 300
    row_y = 92
    row_h = 34
    key_x = 232
    key_w = 150
    query_x = 30
    out_x = 512

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'font-family="{SANS}" role="img" aria-label="The query token \'it\' attends '
        f"most strongly to the key and value for 'cat'.\">"
    ]

    parts.append(
        f'<text x="{query_x}" y="30" font-size="11" font-weight="700" fill="{MUTED}" '
        f'letter-spacing="1">QUERY</text>'
    )
    parts.append(
        f'<text x="{key_x}" y="30" font-size="11" font-weight="700" fill="{MUTED}" '
        f'letter-spacing="1">KEYS &#183; VALUES</text>'
    )
    parts.append(
        f'<text x="{out_x}" y="30" font-size="11" font-weight="700" fill="{MUTED}" '
        f'letter-spacing="1">OUTPUT</text>'
    )

    # The query box, vertically centred against the stack of keys.
    q_center = row_y + (len(tokens) * row_h) / 2 - row_h / 2
    parts.append(
        f'<rect x="{query_x}" y="{q_center}" width="86" height="{row_h - 6}" rx="6" '
        f'fill="{ACCENT}" stroke="none"/>'
    )
    parts.append(
        f'<text x="{query_x + 43}" y="{q_center + 19}" font-size="13" font-weight="700" '
        f'text-anchor="middle" fill="#ffffff">&#8220;it&#8221;</text>'
    )
    parts.append(
        f'<text x="{query_x + 43}" y="{q_center + row_h + 12}" font-size="10.5" '
        f'text-anchor="middle" fill="{MUTED}">what am I</text>'
    )
    parts.append(
        f'<text x="{query_x + 43}" y="{q_center + row_h + 25}" font-size="10.5" '
        f'text-anchor="middle" fill="{MUTED}">looking for?</text>'
    )

    for i, (token, weight) in enumerate(zip(tokens, weights)):
        y = row_y + i * row_h

        parts.append(
            f'<rect x="{key_x}" y="{y}" width="{key_w}" height="{row_h - 6}" rx="6" '
            f'fill="#ffffff" stroke="{RULE_STRONG}"/>'
        )
        parts.append(
            f'<text x="{key_x + 12}" y="{y + 19}" font-size="12" fill="{INK}">{token}</text>'
        )
        parts.append(
            f'<text x="{key_x + key_w - 12}" y="{y + 19}" font-size="10.5" '
            f'text-anchor="end" fill="{MUTED}" font-variant="tabular-nums">'
            f"{weight:.2f}</text>"
        )

        # Arrow width and opacity both encode the softmax weight.
        stroke = 0.8 + weight * 9
        opacity = 0.22 + weight * 0.78
        parts.append(
            f'<path d="M {query_x + 92} {q_center + 14} C 170 {q_center + 14}, '
            f'190 {y + 14}, {key_x - 6} {y + 14}" fill="none" stroke="{ACCENT}" '
            f'stroke-width="{stroke:.2f}" opacity="{opacity:.2f}"/>'
        )
        parts.append(
            f'<path d="M {key_x + key_w + 6} {y + 14} C 440 {y + 14}, '
            f'460 {q_center + 14}, {out_x - 6} {q_center + 14}" fill="none" '
            f'stroke="{VIOLET}" stroke-width="{stroke:.2f}" opacity="{opacity:.2f}"/>'
        )

    parts.append(
        f'<rect x="{out_x}" y="{q_center}" width="98" height="{row_h - 6}" rx="6" '
        f'fill="{ACCENT_SOFT}" stroke="{ACCENT}"/>'
    )
    parts.append(
        f'<text x="{out_x + 49}" y="{q_center + 19}" font-size="11.5" font-weight="600" '
        f'text-anchor="middle" fill="{ACCENT}">weighted mix</text>'
    )
    parts.append(
        f'<text x="{out_x + 49}" y="{q_center + row_h + 12}" font-size="10.5" '
        f'text-anchor="middle" fill="{MUTED}">mostly the value</text>'
    )
    parts.append(
        f'<text x="{out_x + 49}" y="{q_center + row_h + 25}" font-size="10.5" '
        f'text-anchor="middle" fill="{MUTED}">for &#8220;cat&#8221;</text>'
    )

    parts.append(
        f'<text x="{width / 2}" y="{height - 10}" font-size="10.5" text-anchor="middle" '
        f'fill="{MUTED}" font-style="italic">Thicker arrow = larger softmax weight. '
        f"No entry is ever missed; each is returned in proportion to its match.</text>"
    )

    parts.append("</svg>")
    return write_svg("attention-lookup.svg", "\n".join(parts))


# ---------------------------------------------------------------------------
# Figure 4.2 — the causal mask.
# ---------------------------------------------------------------------------


def fig_causal_mask() -> Path:
    """Diagram: the lower-triangular attention matrix that forbids peeking ahead."""
    tokens = ["The", "cat", "sat", "on", "it"]
    n = len(tokens)
    cell = 42
    grid_x, grid_y = 118, 74
    width = grid_x + n * cell + 190
    height = grid_y + n * cell + 54

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'font-family="{SANS}" role="img" aria-label="A lower-triangular attention '
        f'matrix: each row may attend only to itself and earlier tokens.">'
    ]

    parts.append(
        f'<text x="{grid_x}" y="26" font-size="11" font-weight="700" fill="{MUTED}" '
        f'letter-spacing="1">ATTENDING TO (key position)</text>'
    )

    for j, token in enumerate(tokens):
        parts.append(
            f'<text x="{grid_x + j * cell + cell / 2}" y="{grid_y - 10}" font-size="11" '
            f'text-anchor="middle" fill="{INK_SOFT}">{token}</text>'
        )

    for i, token in enumerate(tokens):
        parts.append(
            f'<text x="{grid_x - 12}" y="{grid_y + i * cell + cell / 2 + 4}" '
            f'font-size="11" text-anchor="end" fill="{INK_SOFT}">{token}</text>'
        )
        for j in range(n):
            x = grid_x + j * cell
            y = grid_y + i * cell
            if j <= i:
                # Allowed. Shade by an illustrative weight that favours recency.
                weight = 0.35 + 0.65 * (j + 1) / (i + 1)
                parts.append(
                    f'<rect x="{x}" y="{y}" width="{cell - 3}" height="{cell - 3}" rx="4" '
                    f'fill="{ACCENT}" opacity="{weight * 0.85:.2f}"/>'
                )
            else:
                # Masked to negative infinity before the softmax, so weight is zero.
                parts.append(
                    f'<rect x="{x}" y="{y}" width="{cell - 3}" height="{cell - 3}" rx="4" '
                    f'fill="#f0efe9" stroke="{RULE}"/>'
                )
                parts.append(
                    f'<text x="{x + (cell - 3) / 2}" y="{y + (cell - 3) / 2 + 5}" '
                    f'font-size="13" text-anchor="middle" fill="{RULE_STRONG}">&#215;</text>'
                )

    parts.append(
        f'<text x="14" y="{grid_y + n * cell / 2}" font-size="11" font-weight="700" '
        f'fill="{MUTED}" letter-spacing="1" transform="rotate(-90 14 '
        f'{grid_y + n * cell / 2})" text-anchor="middle">QUERY POSITION</text>'
    )

    legend_x = grid_x + n * cell + 28
    parts.append(
        f'<rect x="{legend_x}" y="{grid_y + 4}" width="16" height="16" rx="4" '
        f'fill="{ACCENT}" opacity="0.8"/>'
    )
    parts.append(
        f'<text x="{legend_x + 24}" y="{grid_y + 17}" font-size="11" fill="{INK_SOFT}">'
        f"allowed</text>"
    )
    parts.append(
        f'<rect x="{legend_x}" y="{grid_y + 32}" width="16" height="16" rx="4" '
        f'fill="#f0efe9" stroke="{RULE}"/>'
    )
    parts.append(
        f'<text x="{legend_x + 24}" y="{grid_y + 45}" font-size="11" fill="{INK_SOFT}">'
        f"masked to &#8722;&#8734;</text>"
    )
    parts.append(
        f'<text x="{legend_x}" y="{grid_y + 78}" font-size="10.5" fill="{MUTED}">'
        f"The future is</text>"
    )
    parts.append(
        f'<text x="{legend_x}" y="{grid_y + 92}" font-size="10.5" fill="{MUTED}">'
        f"unreadable, so we</text>"
    )
    parts.append(
        f'<text x="{legend_x}" y="{grid_y + 106}" font-size="10.5" fill="{MUTED}">'
        f"can train on every</text>"
    )
    parts.append(
        f'<text x="{legend_x}" y="{grid_y + 120}" font-size="10.5" fill="{MUTED}">'
        f"position at once.</text>"
    )

    parts.append("</svg>")
    return write_svg("causal-mask.svg", "\n".join(parts))


# ---------------------------------------------------------------------------
# Figure A.1 — will it fit? Weight memory against machine capacity.
# ---------------------------------------------------------------------------

BYTES_PER_PARAM = {"bf16": 2.0, "int8": 1.0, "4-bit": 0.5}
MODEL_SIZES_B = [8, 32, 70, 405]


def fig_capacity() -> Path:
    """Plot: weight memory by model size and precision, against machine capacity."""
    style_plot()
    fig, ax = plt.subplots(figsize=(7.0, 3.6))

    colors = {"bf16": BRICK, "int8": AMBER, "4-bit": ACCENT}
    bar_width = 0.26
    positions = range(len(MODEL_SIZES_B))

    for k, (precision, bytes_each) in enumerate(BYTES_PER_PARAM.items()):
        memory = [size * 1e9 * bytes_each / 1e9 for size in MODEL_SIZES_B]
        offsets = [p + (k - 1) * bar_width for p in positions]
        bars = ax.bar(
            offsets,
            memory,
            width=bar_width,
            label=precision,
            color=colors[precision],
            edgecolor="none",
        )
        for bar, value in zip(bars, memory):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value * 1.08,
                f"{value:.0f}",
                ha="center",
                va="bottom",
                fontsize=7,
                color=MUTED,
            )

    # Reference lines for representative machines the reader might actually own.
    machines = [
        (24, "24 GB discrete GPU"),
        (128, "128 GB unified memory"),
        (512, "512 GB unified memory"),
    ]
    for capacity, label in machines:
        ax.axhline(capacity, color=INK_SOFT, linewidth=0.9, linestyle=(0, (4, 3)))
        ax.text(
            len(MODEL_SIZES_B) - 0.42,
            capacity * 1.07,
            label,
            fontsize=7.5,
            color=INK_SOFT,
            ha="right",
        )

    ax.set_yscale("log")
    ax.set_xticks(list(positions))
    ax.set_xticklabels([f"{size}B" for size in MODEL_SIZES_B])
    ax.set_xlabel("model size (parameters)")
    ax.set_ylabel("weight memory (GB, log scale)")
    ax.set_title(
        "Whether a model fits is just parameters × bytes per parameter", loc="left"
    )
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.grid(axis="y", alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend(title="precision", loc="upper left", title_fontsize=8)

    return save_plot(fig, "capacity.svg")


# ---------------------------------------------------------------------------
# Figure A.2 — how fast? Bandwidth divided by model bytes.
# ---------------------------------------------------------------------------


def fig_bandwidth() -> Path:
    """Plot: the tokens-per-second ceiling implied by memory bandwidth."""
    style_plot()
    fig, ax = plt.subplots(figsize=(7.0, 3.6))

    # Bandwidths spanning a laptop-class chip up to a high-end workstation part.
    bandwidths = [200, 400, 800]
    efficiency = 0.6  # Real stacks reach roughly 50-70% of the theoretical ceiling.

    sizes_gb = [4 + i * 2 for i in range(39)]  # 4 GB to 80 GB of resident weights.

    shades = [RULE_STRONG, VIOLET, ACCENT]
    for bandwidth, color in zip(bandwidths, shades):
        ceiling = [bandwidth / size for size in sizes_gb]
        realistic = [rate * efficiency for rate in ceiling]
        ax.plot(sizes_gb, ceiling, color=color, linewidth=1.0, linestyle=(0, (3, 2)))
        ax.plot(
            sizes_gb,
            realistic,
            color=color,
            linewidth=2.0,
            label=f"{bandwidth} GB/s",
        )

    # Anchor the reader with the worked example from the text.
    example_size = 35
    example_rate = 400 / example_size * efficiency
    ax.scatter([example_size], [example_rate], s=34, color=AMBER, zorder=5)
    ax.annotate(
        "70B at 4-bit (~35 GB)\non 400 GB/s: ~7 tok/s",
        xy=(example_size, example_rate),
        xytext=(example_size + 9, example_rate + 13),
        fontsize=8,
        color=AMBER,
        arrowprops={"arrowstyle": "-", "color": AMBER, "linewidth": 0.8},
    )

    ax.axhspan(0, 5, color=BRICK, alpha=0.06)
    ax.text(78, 2.0, "painfully slow", fontsize=7.5, color=BRICK, ha="right")

    ax.set_xlabel("resident model size (GB) — set by parameters and quantization")
    ax.set_ylabel("generation speed (tokens/sec)")
    ax.set_title("Decode speed is memory bandwidth ÷ model size, not FLOPs", loc="left")
    ax.set_ylim(0, 105)
    ax.set_xlim(4, 80)
    ax.grid(alpha=0.6)
    ax.set_axisbelow(True)
    legend = ax.legend(title="memory bandwidth", loc="upper right", title_fontsize=8)
    ax.text(
        0.985,
        0.60,
        "solid = realistic (~60% of ceiling)\ndashed = theoretical ceiling",
        transform=ax.transAxes,
        fontsize=7.5,
        color=MUTED,
        ha="right",
    )
    assert (
        legend is not None
    ), "The legend should exist so the bandwidth curves are labelled."

    return save_plot(fig, "bandwidth.svg")


# ---------------------------------------------------------------------------
# Chapter 1 — What Is a Large Language Model?
# ---------------------------------------------------------------------------


def fig_autoregression() -> Path:
    """Diagram: context to next-token distribution to sample, looped back."""
    width, height = 680, 300
    ctx = ["The", "cat", "sat", "on"]
    candidates = [("mat", 0.41), ("rug", 0.17), ("floor", 0.16), ("sofa", 0.15)]
    row_y, bw, bh, gap = 176, 50, 32, 8

    body: list[str] = [eyebrow(24, 34, "CONTEXT SO FAR")]
    x0 = 24
    for i, tok in enumerate(ctx):
        body += token_box(x0 + i * (bw + gap), row_y, bw, bh, tok)
    ctx_end = x0 + len(ctx) * (bw + gap) - gap

    # The model.
    mx, my, mw, mh = 300, 158, 96, 68
    body.append(
        f'<path d="M {ctx_end + 6} {row_y + bh / 2} L {mx - 6} {my + mh / 2}" '
        f'stroke="{RULE_STRONG}" stroke-width="1.5" marker-end="url(#a1)"/>'
    )
    body += [
        f'<rect x="{mx}" y="{my}" width="{mw}" height="{mh}" rx="10" fill="{ACCENT}"/>',
        f'<text x="{mx + mw / 2}" y="{my + 30}" font-size="22" font-weight="700" '
        f'text-anchor="middle" fill="#ffffff">p<tspan font-size="14" dy="4">θ</tspan></text>',
        f'<text x="{mx + mw / 2}" y="{my + 54}" font-size="10.5" text-anchor="middle" '
        f'fill="{ACCENT_SOFT}">next-token model</text>',
    ]

    # The distribution, as horizontal bars. The eyebrow sits just above the bars
    # (not at the top) so the loop-back arrow can pass overhead without crossing it.
    body.append(eyebrow(444, 48, "P(NEXT TOKEN)"))
    lab_x, bar_x, bar_max, dist_top, rh = 444, 486, 150, 58, 30
    for i, (tok, p) in enumerate(candidates):
        y = dist_top + i * rh
        top = i == 0
        bar_fill = AMBER if top else ACCENT
        bar_op = 1.0 if top else 0.28
        body += [
            f'<text x="{lab_x}" y="{y + 15}" font-size="11.5" fill="{INK if top else INK_SOFT}" '
            f'font-weight="{700 if top else 400}">{tok}</text>',
            f'<rect x="{bar_x}" y="{y + 3}" width="{bar_max * p:.1f}" height="14" rx="3" '
            f'fill="{bar_fill}" opacity="{bar_op}"/>',
            f'<text x="{bar_x + bar_max * p + 6:.1f}" y="{y + 15}" font-size="10" '
            f'fill="{MUTED}" font-variant="tabular-nums">{p:.2f}</text>',
        ]
    body.append(
        f'<path d="M {mx + mw + 6} {my + mh / 2} L {lab_x - 12} {dist_top + 12}" '
        f'stroke="{RULE_STRONG}" stroke-width="1.5" marker-end="url(#a1)"/>'
    )

    # Sample the top token and loop it back onto the context.
    sample_y = dist_top + 10
    body.append(
        f'<text x="640" y="{sample_y + 6}" font-size="10.5" font-style="italic" '
        f'text-anchor="end" fill="{AMBER}">sample</text>'
    )
    body.append(
        f'<path d="M 636 {sample_y} C 660 8, 300 8, {x0 + bw / 2} {row_y - 6}" '
        f'fill="none" stroke="{AMBER}" stroke-width="1.6" stroke-dasharray="5 4" '
        f'marker-end="url(#a1amber)"/>'
    )
    body.append(
        f'<text x="330" y="20" font-size="11" text-anchor="middle" fill="{AMBER}">'
        f"append the sampled token, then repeat</text>"
    )
    body.append(arrow_marker(RULE_STRONG, "a1"))
    body.append(arrow_marker(AMBER, "a1amber"))
    return write_svg(
        "autoregression.svg", svg_doc(width, height, "autoregression loop", body)
    )


def fig_scaling_emergence() -> Path:
    """Plot: loss falls smoothly with compute while capabilities switch on."""
    style_plot()
    fig, ax = plt.subplots(figsize=(7.0, 3.7))

    # A clean power law in compute (arbitrary illustrative units).
    compute = [10**e for e in [x / 40 for x in range(0, 241)]]
    loss = [2.2 * c**-0.05 for c in compute]
    ax.plot(compute, loss, color=ACCENT, linewidth=2.2)
    ax.set_xscale("log")

    # Capabilities appear at points along the smooth curve.
    marks = [
        (10**1.4, "grammar"),
        (10**2.8, "3-digit\naddition"),
        (10**4.2, "instruction\nfollowing"),
        (10**5.4, "multi-step\ncode"),
    ]
    for c, label in marks:
        loss_at = 2.2 * c**-0.05
        ax.scatter([c], [loss_at], s=40, color=AMBER, zorder=5)
        ax.annotate(
            label,
            xy=(c, loss_at),
            xytext=(c, loss_at + 0.16),
            fontsize=8,
            color=AMBER,
            ha="center",
        )

    ax.set_xlabel("training compute (log scale, arbitrary units)")
    ax.set_ylabel("pretraining loss")
    ax.set_title(
        "The loss curve is smooth; the capabilities it buys arrive in jumps", loc="left"
    )
    ax.set_ylim(1.0, 2.4)
    ax.grid(alpha=0.5)
    ax.set_axisbelow(True)
    ax.get_xaxis().set_ticks([])
    return save_plot(fig, "scaling-emergence.svg")


def fig_base_vs_assistant() -> Path:
    """Diagram: the same prompt, continued by a base model vs an assistant."""
    width, height = 660, 300
    prompt = "“How do I reverse a list in Python?”"

    body: list[str] = [
        f'<text x="{width / 2}" y="30" font-size="13" text-anchor="middle" '
        f'fill="{INK}">Same prompt: <tspan font-style="italic" fill="{ACCENT}">'
        f"{prompt}</tspan></text>",
    ]

    panels = [
        (
            24,
            "BASE MODEL",
            "continues the document",
            [
                "How do I sort a list in Python?",
                "How do I remove duplicates from a",
                "list in Python? How do I reverse a",
                "string in Python? ...",
            ],
            BRICK,
        ),
        (
            width / 2 + 12,
            "AFTER POST-TRAINING",
            "answers the request",
            [
                "Use the .reverse() method to reverse",
                "in place, or the [::-1] slice to get a",
                "reversed copy:",
                "    nums[::-1]",
            ],
            ACCENT,
        ),
    ]
    pw = width / 2 - 36
    for px, tag, sub, lines, color in panels:
        body.append(eyebrow(px, 66, tag, color))
        body.append(
            f'<text x="{px}" y="82" font-size="10.5" font-style="italic" '
            f'fill="{MUTED}">{sub}</text>'
        )
        body.append(
            f'<rect x="{px}" y="96" width="{pw}" height="150" rx="8" fill="#ffffff" '
            f'stroke="{RULE_STRONG}"/>'
        )
        body.append(
            f'<rect x="{px}" y="96" width="4" height="150" rx="2" fill="{color}"/>'
        )
        for j, line in enumerate(lines):
            style = ' font-family="' + MONO + '"' if line.startswith("    ") else ""
            body.append(
                f'<text x="{px + 18}" y="{124 + j * 24}" font-size="11"{style} '
                f'fill="{INK_SOFT}">{line.strip()}</text>'
            )
    body.append(
        f'<text x="{width / 2}" y="284" font-size="10.5" text-anchor="middle" '
        f'fill="{MUTED}" font-style="italic">Same weights would continue the pattern; '
        f"post-training teaches it to treat the prompt as a request.</text>"
    )
    return write_svg(
        "base-vs-assistant.svg",
        svg_doc(width, height, "base model versus assistant", body),
    )


def fig_test_time_compute() -> Path:
    """Diagram: a direct answer versus a long chain of thought before answering."""
    width, height = 660, 260
    body: list[str] = [eyebrow(24, 30, "TWO WAYS TO SPEND COMPUTE")]

    def token_strip(x, y, n, color, opacity):
        cells = []
        cw, cg = 9, 3
        for i in range(n):
            cells.append(
                f'<rect x="{x + i * (cw + cg):.1f}" y="{y}" width="{cw}" height="14" '
                f'rx="2" fill="{color}" opacity="{opacity}"/>'
            )
        return cells, x + n * (cw + cg) - cg

    # Row 1: standard model — prompt straight to a short answer.
    y1 = 74
    body += token_box(
        24, y1 - 8, 78, 30, "prompt", fill=ACCENT_SOFT, stroke=ACCENT, text_fill=ACCENT
    )
    strip, end = token_strip(120, y1, 4, MUTED, 0.5)
    body += strip
    body.append(
        f'<path d="M 106 {y1 + 7} L 116 {y1 + 7}" stroke="{RULE_STRONG}" '
        f'stroke-width="1.4" marker-end="url(#a2)"/>'
    )
    body += token_box(
        end + 14, y1 - 8, 92, 30, "answer", fill=BRICK, stroke="none", text_fill="#fff"
    )
    body.append(
        f'<text x="24" y="{y1 - 20}" font-size="11" font-weight="700" fill="{INK_SOFT}">'
        f"Standard model</text>"
    )

    # Row 2: reasoning model — a long hidden chain before the answer.
    y2 = 168
    body += token_box(
        24, y2 - 8, 78, 30, "prompt", fill=ACCENT_SOFT, stroke=ACCENT, text_fill=ACCENT
    )
    strip, end = token_strip(120, y2, 30, VIOLET, 0.55)
    body += strip
    body.append(
        f'<path d="M 106 {y2 + 7} L 116 {y2 + 7}" stroke="{RULE_STRONG}" '
        f'stroke-width="1.4" marker-end="url(#a2)"/>'
    )
    body += token_box(
        end + 14, y2 - 8, 92, 30, "answer", fill=ACCENT, stroke="none", text_fill="#fff"
    )
    body.append(
        f'<text x="24" y="{y2 - 20}" font-size="11" font-weight="700" fill="{INK_SOFT}">'
        f"Reasoning model</text>"
    )
    body.append(
        f'<text x="{(120 + end) / 2}" y="{y2 + 34}" font-size="10" text-anchor="middle" '
        f'fill="{VIOLET}">hidden chain of thought (many tokens)</text>'
    )
    body.append(
        f'<text x="{width / 2}" y="{height - 12}" font-size="10.5" text-anchor="middle" '
        f'fill="{MUTED}" font-style="italic">Test-time compute is a second axis to scale: '
        f"think for longer, not just train larger.</text>"
    )
    body.append(arrow_marker(RULE_STRONG, "a2"))
    return write_svg(
        "test-time-compute.svg",
        svg_doc(width, height, "direct answer versus long reasoning", body),
    )


def fig_reading_map() -> Path:
    """Diagram: the seven parts, with the build-and-ship arc highlighted."""
    width, height = 680, 205
    parts = [
        ("I", "Foundations"),
        ("II", "Pretraining"),
        ("III", "Alignment"),
        ("IV", "Serving"),
        ("V", "Harness"),
        ("VI", "Evaluation"),
        ("VII", "Frontier"),
    ]
    # Parts I-V are the through-line: how a model is made (I-III) and shipped
    # (IV-V). VI-VII step back to ask how we check it and where it is going.
    on_path = {"I", "II", "III", "IV", "V"}

    body: list[str] = [
        eyebrow(24, 30, "THE ARGUMENT, PART BY PART"),
    ]
    n = len(parts)
    bw, bh = 78, 54
    total = n * bw + (n - 1) * 14
    x0 = (width - total) / 2
    y = 78
    for i, (num, name) in enumerate(parts):
        x = x0 + i * (bw + 14)
        highlighted = num in on_path
        fill = ACCENT if highlighted else "#ffffff"
        stroke = ACCENT if highlighted else RULE_STRONG
        text_fill = "#ffffff" if highlighted else INK_SOFT
        body += [
            f'<rect x="{x:.1f}" y="{y}" width="{bw}" height="{bh}" rx="8" fill="{fill}" '
            f'stroke="{stroke}"/>',
            f'<text x="{x + bw / 2:.1f}" y="{y + 24}" font-size="15" font-weight="700" '
            f'text-anchor="middle" fill="{text_fill}">{num}</text>',
            f'<text x="{x + bw / 2:.1f}" y="{y + 42}" font-size="10" text-anchor="middle" '
            f'fill="{ACCENT_SOFT if highlighted else MUTED}">{name}</text>',
        ]
        if i < n - 1:
            ax_ = x + bw + 2
            body.append(
                f'<path d="M {ax_} {y + bh / 2} L {ax_ + 10} {y + bh / 2}" '
                f'stroke="{RULE_STRONG}" stroke-width="1.4" marker-end="url(#a3)"/>'
            )
    body.append(
        f'<text x="{width / 2}" y="{y - 20}" font-size="11" text-anchor="middle" '
        f'fill="{ACCENT}" font-weight="600">how a model is made (I–III), then shipped (IV–V)</text>'
    )
    body.append(
        f'<text x="{width / 2}" y="{y + bh + 30}" font-size="10.5" text-anchor="middle" '
        f'fill="{MUTED}" font-style="italic">Read straight through, or follow the highlighted '
        f"spine and branch out as questions arise.</text>"
    )
    body.append(arrow_marker(RULE_STRONG, "a3"))
    return write_svg(
        "reading-map.svg", svg_doc(width, height, "map of the book's parts", body)
    )


# ---------------------------------------------------------------------------
# Chapter 2 — Deep Learning, Just Enough
# ---------------------------------------------------------------------------


def fig_manifold() -> Path:
    """Plot: two interleaved spirals a network untangles into separable blobs."""
    import math

    style_plot()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.4))

    # Two interleaved spirals: the classic not-linearly-separable input.
    n = 90
    for k, color in ((0, ACCENT), (1, AMBER)):
        t = [0.3 + 3.2 * i / n for i in range(n)]
        sign = 1 if k == 0 else -1
        xs = [sign * ti * math.cos(ti) for ti in t]
        ys = [sign * ti * math.sin(ti) for ti in t]
        ax1.scatter(xs, ys, s=10, color=color, alpha=0.8, edgecolors="none")
    ax1.set_title("input space", loc="center", fontsize=9, color=INK_SOFT)

    # After the learned transformation: two rounded clusters split by a line.
    import random

    rng = random.Random(0)
    for k, color, cx in ((0, ACCENT, -1.15), (1, AMBER, 1.15)):
        xs = [cx + rng.gauss(0, 0.3) for _ in range(46)]
        ys = [rng.gauss(0, 0.7) for _ in range(46)]
        ax2.scatter(xs, ys, s=10, color=color, alpha=0.8, edgecolors="none")
    ax2.axvline(0, color=INK_SOFT, linewidth=1.0, linestyle=(0, (4, 3)))
    ax2.set_xlim(-2.4, 2.4)
    ax2.set_title("last-layer representation", loc="center", fontsize=9, color=INK_SOFT)

    for ax in (ax1, ax2):
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.suptitle(
        "Depth untangles the data until a straight cut separates the classes",
        x=0.5,
        y=1.02,
        fontsize=10,
        fontweight="bold",
        color=INK,
    )
    return save_plot(fig, "manifold.svg")


def fig_training_loop() -> Path:
    """Diagram: the forward pass down, the gradient pass back, as a loop."""
    width, height = 660, 250
    stages = [
        ("batch", "inputs + targets", ACCENT_SOFT, ACCENT),
        ("model", "predict logits", "#ffffff", INK),
        ("loss", "cross-entropy", "#ffffff", INK),
    ]
    bw, bh, gap = 128, 58, 60
    y = 74
    x0 = 40
    body: list[str] = [eyebrow(40, 34, "ONE STEP OF TRAINING")]
    centers = []
    for i, (title, sub, fill, tf) in enumerate(stages):
        x = x0 + i * (bw + gap)
        centers.append(x + bw / 2)
        stroke = ACCENT if fill == ACCENT_SOFT else RULE_STRONG
        body += [
            f'<rect x="{x}" y="{y}" width="{bw}" height="{bh}" rx="9" fill="{fill}" '
            f'stroke="{stroke}"/>',
            f'<text x="{x + bw / 2}" y="{y + 26}" font-size="14" font-weight="700" '
            f'text-anchor="middle" fill="{tf}">{title}</text>',
            f'<text x="{x + bw / 2}" y="{y + 44}" font-size="10.5" text-anchor="middle" '
            f'fill="{MUTED}">{sub}</text>',
        ]
        if i < len(stages) - 1:
            body.append(
                f'<path d="M {x + bw + 6} {y + bh / 2} L {x + bw + gap - 6} {y + bh / 2}" '
                f'stroke="{ACCENT}" stroke-width="1.8" marker-end="url(#fwd)"/>'
            )
    body.append(
        f'<text x="{(centers[0] + centers[-1]) / 2}" y="{y - 8}" font-size="10.5" '
        f'text-anchor="middle" fill="{ACCENT}">forward pass</text>'
    )

    # The backward pass: gradients flow from the loss back to the parameters.
    by = y + bh + 46
    lx, rx = centers[0], centers[-1]
    body.append(
        f'<path d="M {rx} {y + bh + 4} L {rx} {by} L {lx} {by} L {lx} {y + bh + 4}" '
        f'fill="none" stroke="{BRICK}" stroke-width="1.8" stroke-dasharray="6 4" '
        f'marker-end="url(#bwd)"/>'
    )
    body.append(
        f'<text x="{(lx + rx) / 2}" y="{by + 18}" font-size="11" text-anchor="middle" '
        f'fill="{BRICK}">backward pass: gradient of the loss w.r.t. every parameter</text>'
    )
    body.append(
        f'<text x="{lx}" y="{by - 8}" font-size="10" text-anchor="middle" fill="{BRICK}">'
        f"update</text>"
    )
    body.append(
        f'<text x="{width / 2}" y="{height - 8}" font-size="10.5" text-anchor="middle" '
        f'fill="{MUTED}" font-style="italic">Every model in this book is trained by running '
        f"this loop millions of times.</text>"
    )
    body.append(arrow_marker(ACCENT, "fwd"))
    body.append(arrow_marker(BRICK, "bwd"))
    return write_svg(
        "training-loop.svg", svg_doc(width, height, "the training loop", body)
    )


def fig_lr_curves() -> Path:
    """Plot: loss over steps for a learning rate too high, too low, and right."""
    import math

    style_plot()
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    steps = list(range(0, 100))

    too_low = [2.4 - 0.6 * (s / 100) for s in steps]
    just_right = [0.5 + 1.9 * math.exp(-s / 16) for s in steps]
    # A gentle wobble that turns into a monotonic blow-up, so the line leaves the
    # top of the chart once and never re-enters (no square-wave clipping).
    too_high = [
        1.42 + 0.08 * math.sin(s / 1.5) + 0.9 * math.exp((s - 58) / 11) for s in steps
    ]

    ax.plot(steps, just_right, color=ACCENT, linewidth=2.2, label="well tuned")
    ax.plot(
        steps,
        too_low,
        color=MUTED,
        linewidth=1.8,
        linestyle=(0, (5, 2)),
        label="too low",
    )
    ax.plot(steps, too_high, color=BRICK, linewidth=1.8, label="too high")

    ax.annotate(
        "converges",
        xy=(70, just_right[70]),
        xytext=(58, 1.0),
        fontsize=8,
        color=ACCENT,
    )
    ax.annotate(
        "barely moves",
        xy=(80, too_low[80]),
        xytext=(40, 2.15),
        fontsize=8,
        color=MUTED,
    )
    ax.annotate(
        "diverges",
        xy=(64, 3.25),
        xytext=(40, 3.15),
        fontsize=8,
        color=BRICK,
    )

    ax.set_xlabel("training step")
    ax.set_ylabel("loss")
    ax.set_title(
        "The learning rate is the knob everything else works around", loc="left"
    )
    ax.set_ylim(0, 3.5)
    ax.set_xlim(0, 99)
    ax.grid(alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="center right", title_fontsize=8)
    return save_plot(fig, "lr-curves.svg")


def fig_generalization() -> Path:
    """Plot: an overparameterized net fits real and random labels, but only
    generalizes on real ones."""
    style_plot()
    fig, ax = plt.subplots(figsize=(7.0, 3.4))

    groups = ["Real labels", "Random labels"]
    train = [100, 100]
    test = [91, 10]
    x = range(len(groups))
    bw = 0.34

    ax.bar(
        [i - bw / 2 for i in x],
        train,
        width=bw,
        color=RULE_STRONG,
        label="train accuracy",
    )
    ax.bar([i + bw / 2 for i in x], test, width=bw, color=ACCENT, label="test accuracy")
    for i, v in enumerate(train):
        ax.text(i - bw / 2, v + 2, f"{v}%", ha="center", fontsize=8, color=MUTED)
    for i, v in enumerate(test):
        ax.text(i + bw / 2, v + 2, f"{v}%", ha="center", fontsize=8, color=ACCENT)
    ax.axhline(10, color=BRICK, linewidth=0.9, linestyle=(0, (3, 3)))
    ax.text(1.42, 13, "chance", fontsize=7.5, color=BRICK, ha="right")

    ax.set_xticks(list(x))
    ax.set_xticklabels(groups)
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 112)
    ax.set_title(
        "The same network memorizes noise yet generalizes on real data", loc="left"
    )
    ax.grid(axis="y", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="center right")
    return save_plot(fig, "generalization.svg")


def fig_precision() -> Path:
    """Diagram: the bit layout of float32, float16, and bfloat16 aligned."""
    width, height = 660, 280
    formats = [
        ("float32", 1, 8, 23),
        ("float16", 1, 5, 10),
        ("bfloat16", 1, 8, 7),
    ]
    unit = 15.0  # Pixels per bit, so the three rows share a scale.
    x0, y0, rh, rgap = 118, 60, 40, 26
    body: list[str] = [eyebrow(24, 34, "WHERE THE 16 (OR 32) BITS GO")]

    fields = [("sign", INK_SOFT), ("exponent", ACCENT), ("mantissa", AMBER)]
    for r, (name, s, e, m) in enumerate(formats):
        y = y0 + r * (rh + rgap)
        body.append(
            f'<text x="{x0 - 12}" y="{y + rh / 2 + 4}" font-size="12" text-anchor="end" '
            f'font-family="{MONO}" fill="{INK}">{name}</text>'
        )
        x = float(x0)
        for (label, color), bits in zip(fields, (s, e, m)):
            w = bits * unit
            fill_op = {"sign": 0.55, "exponent": 0.85, "mantissa": 0.85}[label]
            body.append(
                f'<rect x="{x:.1f}" y="{y}" width="{w:.1f}" height="{rh}" '
                f'fill="{color}" opacity="{fill_op}" stroke="#ffffff" stroke-width="1"/>'
            )
            body.append(
                f'<text x="{x + w / 2:.1f}" y="{y + rh / 2 + 4:.1f}" font-size="10" '
                f'text-anchor="middle" fill="#ffffff" font-weight="600">{bits}</text>'
            )
            x += w

    # Guide line showing the exponent widths of fp32 and bf16 match.
    exp_end = x0 + 1 * unit + 8 * unit
    body.append(
        f'<path d="M {exp_end:.1f} {y0 - 6} L {exp_end:.1f} '
        f'{y0 + 2 * (rh + rgap) + rh + 6}" stroke="{ACCENT}" stroke-width="1" '
        f'stroke-dasharray="3 3" opacity="0.7"/>'
    )
    legend_y = y0 + 3 * (rh + rgap) - rgap + 22
    lx = float(x0)
    for label, color in fields:
        body.append(
            f'<rect x="{lx}" y="{legend_y - 11}" width="13" height="13" rx="2" '
            f'fill="{color}" opacity="0.85"/>'
        )
        body.append(
            f'<text x="{lx + 19}" y="{legend_y}" font-size="10.5" fill="{INK_SOFT}">{label}</text>'
        )
        lx += 40 + len(label) * 6.2
    body.append(
        f'<text x="{width / 2}" y="{height - 12}" font-size="10.5" text-anchor="middle" '
        f'fill="{MUTED}" font-style="italic">bfloat16 keeps float32’s exponent (its range) '
        f"and spends the savings out of the mantissa (its precision).</text>"
    )
    return write_svg(
        "precision.svg",
        svg_doc(width, height, "float32, float16, and bfloat16 bit layouts", body),
    )


# ---------------------------------------------------------------------------
# Chapter 3 — Tokenization
# ---------------------------------------------------------------------------


def fig_tokenizer_tradeoff() -> Path:
    """Plot: sequence length falls with vocabulary size, with a subword sweet spot."""
    style_plot()
    fig, ax = plt.subplots(figsize=(7.0, 3.6))

    vocab = [200, 1000, 5000, 32000, 100000, 500000]
    tokens = [520, 340, 205, 138, 120, 110]  # Tokens for a fixed passage.
    ax.plot(vocab, tokens, color=ACCENT, linewidth=2.2, marker="o", markersize=5)
    ax.set_xscale("log")

    ax.axvspan(30000, 128000, color=ACCENT_SOFT, alpha=0.7)
    ax.text(62000, 490, "subword\nsweet spot", fontsize=8.5, color=ACCENT, ha="center")
    ax.annotate(
        "characters:\ntiny vocab, long sequences",
        xy=(200, 520),
        xytext=(300, 250),
        fontsize=8,
        color=MUTED,
        arrowprops={"arrowstyle": "-", "color": RULE_STRONG, "linewidth": 0.8},
    )
    ax.annotate(
        "words:\nhuge vocab, out-of-vocabulary misses",
        xy=(500000, 110),
        xytext=(11000, 330),
        fontsize=8,
        color=BRICK,
        arrowprops={"arrowstyle": "-", "color": BRICK, "linewidth": 0.8},
    )

    ax.set_xlabel("vocabulary size (log scale)")
    ax.set_ylabel("tokens per passage")
    ax.set_title(
        "Bigger vocabulary buys shorter sequences, with diminishing returns", loc="left"
    )
    ax.set_ylim(80, 560)
    ax.grid(alpha=0.5)
    ax.set_axisbelow(True)
    return save_plot(fig, "tokenizer-tradeoff.svg")


def fig_bpe_merges() -> Path:
    """Diagram: 'lowest' as a merge tree; its tokens are the root nodes."""
    width, height = 470, 250
    body: list[str] = [eyebrow(24, 30, "TOKENIZING “lowest”")]

    cw, ch = 40, 30
    leaf_y, lo_y, low_y = 176, 118, 58
    leaves = [("l", 54), ("o", 104), ("w", 154), ("e", 236), ("s", 286), ("t", 336)]
    lo_x, low_x = 79, 116

    def line(x1, y1, x2, y2):
        return (
            f'<path d="M {x1:.1f} {y1:.1f} L {x2:.1f} {y2:.1f}" stroke="{RULE_STRONG}" '
            f'stroke-width="1.4"/>'
        )

    # Connectors for the merged subtree: l,o -> lo ; lo,w -> low.
    body.append(line(54 + cw / 2, leaf_y, lo_x + cw / 2, lo_y + ch))
    body.append(line(104 + cw / 2, leaf_y, lo_x + cw / 2, lo_y + ch))
    body.append(line(lo_x + cw / 2, lo_y, low_x + (cw + 6) / 2, low_y + ch))
    body.append(line(154 + cw / 2, leaf_y, low_x + (cw + 6) / 2, low_y + ch))

    # Leaf characters; e, s, t are their own roots, so they are marked final.
    for label, x in leaves[:3]:
        body += token_box(x, leaf_y, cw, ch, label, font_size=13)
    for label, x in leaves[3:]:
        body += token_box(
            x,
            leaf_y,
            cw,
            ch,
            label,
            fill="#ffffff",
            stroke=AMBER,
            text_fill=AMBER,
            weight=700,
            font_size=13,
        )
    # The one learned intermediate, and the root token it forms.
    body += token_box(
        lo_x,
        lo_y,
        cw,
        ch,
        "lo",
        fill=ACCENT_SOFT,
        stroke=ACCENT,
        text_fill=ACCENT,
        weight=700,
    )
    body += token_box(
        low_x,
        low_y,
        cw + 6,
        ch,
        "low",
        fill=ACCENT,
        stroke="none",
        text_fill="#fff",
        weight=700,
    )

    # Name the final tokenization and the two kinds of root.
    body.append(
        f'<text x="366" y="{low_y + 20}" font-size="12" fill="{INK}">→ '
        f'<tspan fill="{ACCENT}" font-weight="700">low</tspan> · '
        f'<tspan fill="{AMBER}" font-weight="700">e·s·t</tspan></text>'
    )
    body.append(
        f'<text x="366" y="{low_y + 38}" font-size="10" fill="{MUTED}">4 tokens</text>'
    )
    body.append(
        f'<text x="{width / 2}" y="{height - 12}" font-size="10.5" text-anchor="middle" '
        f'fill="{MUTED}" font-style="italic">A word’s tokens are the roots: a learned merge '
        f"(low) plus whatever never merged (e, s, t).</text>"
    )
    return write_svg(
        "bpe-merges.svg",
        svg_doc(width, height, "byte-pair merges building the word lowest", body),
    )


def fig_byte_fallback() -> Path:
    """Diagram: byte-level tokenizers spell anything from 256 bytes, no OOV."""
    width, height = 620, 250
    body: list[str] = [eyebrow(24, 32, "NO UNKNOWN TOKEN, EVER")]

    # A common word: one token.
    body += token_box(40, 78, 74, 34, "“ the ”", font_size=13)
    body.append(
        f'<path d="M 120 95 L 156 95" stroke="{RULE_STRONG}" stroke-width="1.5" '
        f'marker-end="url(#b1)"/>'
    )
    body += token_box(
        162, 78, 70, 34, "1 token", fill=ACCENT, stroke="none", text_fill="#fff"
    )
    body.append(
        f'<text x="40" y="70" font-size="10.5" fill="{MUTED}">a frequent word</text>'
    )

    # A rare glyph: several byte tokens.
    gy = 166
    body += token_box(40, gy, 74, 34, "“ 🌍 ”", font_size=15)
    body.append(
        f'<path d="M 120 {gy + 17} L 156 {gy + 17}" stroke="{RULE_STRONG}" '
        f'stroke-width="1.5" marker-end="url(#b1)"/>'
    )
    byte_labels = ["F0", "9F", "8C", "8D"]
    bx = 162
    for i, b in enumerate(byte_labels):
        body += token_box(
            bx + i * 52,
            gy,
            44,
            34,
            b,
            fill=AMBER,
            stroke="none",
            text_fill="#fff",
            font_size=11,
            weight=700,
        )
    body.append(
        f'<text x="40" y="{gy - 8}" font-size="10.5" fill="{MUTED}">a rare glyph → its UTF-8 bytes</text>'
    )
    body.append(
        f'<text x="{162 + 4 * 52 + 6}" y="{gy + 22}" font-size="10.5" fill="{MUTED}">'
        f"4 byte-tokens</text>"
    )

    body.append(
        f'<text x="{width / 2}" y="{height - 12}" font-size="10.5" text-anchor="middle" '
        f'fill="{MUTED}" font-style="italic">Falling back to 256 bytes means nothing is '
        f"unrepresentable — but rare characters cost several tokens.</text>"
    )
    body.append(arrow_marker(RULE_STRONG, "b1"))
    return write_svg(
        "byte-fallback.svg",
        svg_doc(width, height, "byte-level fallback for a rare glyph", body),
    )


def fig_language_tax() -> Path:
    """Plot: the same sentence costs very different token counts by language."""
    style_plot()
    fig, ax = plt.subplots(figsize=(7.0, 3.4))

    langs = ["English", "Spanish", "German", "Hindi", "Thai", "Burmese"]
    counts = [9, 12, 14, 24, 33, 41]  # Illustrative tokens for one sentence.
    colors = [ACCENT if i == 0 else AMBER for i in range(len(langs))]
    opacities = [1.0] + [0.45 + 0.1 * i for i in range(len(langs) - 1)]

    bars = ax.barh(list(range(len(langs))), counts, color=colors, edgecolor="none")
    for bar, op in zip(bars, opacities):
        bar.set_alpha(op)
    for i, c in enumerate(counts):
        label = f"{c}  (baseline)" if i == 0 else f"{c}  ({c / counts[0]:.1f}×)"
        color = ACCENT if i == 0 else MUTED
        ax.text(c + 0.7, i, label, va="center", fontsize=8, color=color)

    ax.set_yticks(list(range(len(langs))))
    ax.set_yticklabels(langs)
    ax.invert_yaxis()
    ax.set_xlabel("tokens for the same sentence (illustrative)")
    ax.set_title(
        "A vocabulary learned on English taxes every other language", loc="left"
    )
    ax.set_xlim(0, 46)
    ax.grid(axis="x", alpha=0.5)
    ax.set_axisbelow(True)
    return save_plot(fig, "language-tax.svg")


# ---------------------------------------------------------------------------
# Chapter 4 — The Transformer
# ---------------------------------------------------------------------------


def fig_rnn_vs_attention() -> Path:
    """Diagram: an RNN's relayed hidden state versus attention's direct reads."""
    width, height = 660, 300
    tokens = ["The", "cat", "that", "ran", "hid"]
    bw, bh, gap = 60, 32, 44
    x0 = 96

    def xs():
        return [x0 + i * (bw + gap) for i in range(len(tokens))]

    body: list[str] = []

    # RNN row: a hidden state relayed token to token.
    ry = 92
    body.append(eyebrow(24, 40, "RNN: A RELAYED SUMMARY", BRICK))
    for i, (tok, x) in enumerate(zip(tokens, xs())):
        body += token_box(x, ry, bw, bh, tok)
        if i < len(tokens) - 1:
            body.append(
                f'<path d="M {x + bw + 4} {ry + bh / 2} L {x + bw + gap - 4} {ry + bh / 2}" '
                f'stroke="{BRICK}" stroke-width="1.6" marker-end="url(#c1)"/>'
            )
    body.append(
        f'<path d="M {x0 + bw / 2} {ry - 8} C {x0 + bw / 2} {ry - 42}, '
        f'{xs()[-1] + bw / 2} {ry - 42}, {xs()[-1] + bw / 2} {ry - 8}" fill="none" '
        f'stroke="{BRICK}" stroke-width="1.4" stroke-dasharray="5 4" marker-end="url(#c1)"/>'
    )
    body.append(
        f'<text x="{(x0 + xs()[-1] + bw) / 2}" y="{ry - 48}" font-size="10.5" '
        f'text-anchor="middle" fill="{BRICK}">token 1’s signal must survive every step</text>'
    )

    # Attention row: the last token reads every earlier token directly.
    ay = 232
    body.append(eyebrow(24, 180, "ATTENTION: DIRECT ACCESS", ACCENT))
    last = xs()[-1] + bw / 2
    for i, (tok, x) in enumerate(zip(tokens, xs())):
        body += token_box(x, ay, bw, bh, tok)
        if i < len(tokens) - 1:
            src = x + bw / 2
            peak = ay - 20 - (len(tokens) - 1 - i) * 14
            body.append(
                f'<path d="M {last} {ay - 4} Q {(src + last) / 2} {peak}, {src} {ay - 4}" '
                f'fill="none" stroke="{ACCENT}" stroke-width="1.4" opacity="0.7"/>'
            )
    body.append(
        f'<text x="{(x0 + xs()[-1] + bw) / 2}" y="{ay + bh + 24}" font-size="10.5" '
        f'text-anchor="middle" fill="{MUTED}" font-style="italic">No bottleneck: every '
        f"token is one hop away, and all reads happen in parallel.</text>"
    )
    body.append(arrow_marker(BRICK, "c1"))
    return write_svg(
        "rnn-vs-attention.svg",
        svg_doc(width, height, "RNN hidden state versus attention", body),
    )


def fig_multi_head() -> Path:
    """Diagram: several heads reading one sentence for different relations."""
    width, height = 520, 300
    tokens = ["The", "cat", "sat", "because", "it", "purred"]
    bw, bh, gap = 50, 28, 8
    x0 = 132

    def token_x(i):
        return x0 + i * (bw + gap)

    heads = [
        ("syntax", ACCENT, (2, 1), "subject of the verb"),
        ("coreference", VIOLET, (4, 1), "what “it” refers to"),
        ("position", AMBER, (5, 4), "the token just before"),
    ]
    row_h = 82
    body: list[str] = [eyebrow(24, 30, "ONE SENTENCE, THREE HEADS")]

    for r, (name, color, (src, dst), gloss) in enumerate(heads):
        base = 78 + r * row_h
        body.append(
            f'<text x="24" y="{base - 4}" font-size="12" font-weight="700" fill="{color}">'
            f"{name}</text>"
        )
        body.append(
            f'<text x="24" y="{base + 12}" font-size="9.5" fill="{MUTED}">{gloss}</text>'
        )
        for i, tok in enumerate(tokens):
            hot = i in (src, dst)
            body += token_box(
                token_x(i),
                base,
                bw,
                bh,
                tok,
                fill=color if i == src else "#ffffff",
                stroke=color if hot else RULE_STRONG,
                text_fill="#ffffff" if i == src else (color if hot else INK_SOFT),
                font_size=11,
                weight=700 if hot else 400,
            )
        sx, dx = token_x(src) + bw / 2, token_x(dst) + bw / 2
        peak = base - 20
        body.append(
            f'<path d="M {sx} {base - 3} Q {(sx + dx) / 2} {peak}, {dx} {base - 3}" '
            f'fill="none" stroke="{color}" stroke-width="1.8" marker-end="url(#mh{r})"/>'
        )
        body.append(arrow_marker(color, f"mh{r}"))

    return write_svg(
        "multi-head.svg",
        svg_doc(width, height, "multi-head attention reading one sentence", body),
    )


def fig_transformer_block() -> Path:
    """Diagram: a pre-norm block, two sublayers hanging off a residual stream."""
    width, height = 660, 260
    body: list[str] = [eyebrow(24, 30, "ONE PRE-NORM BLOCK")]

    stream_y = 74
    body.append(
        f'<text x="24" y="{stream_y - 12}" font-size="10.5" fill="{ACCENT}" '
        f'font-weight="600">residual stream (an identity path the signal always keeps)</text>'
    )
    # The residual highway.
    body.append(
        f'<path d="M 40 {stream_y} L 620 {stream_y}" stroke="{ACCENT}" stroke-width="2.5" '
        f'marker-end="url(#d1)"/>'
    )
    body += token_box(
        24,
        stream_y - 15,
        30,
        30,
        "x",
        fill=ACCENT,
        stroke="none",
        text_fill="#fff",
        weight=700,
    )

    sublayers = [
        (150, "RMSNorm", "Attention", "mix across positions"),
        (400, "RMSNorm", "FFN", "transform each position"),
    ]
    for cx, norm, op, note in sublayers:
        # The add node on the stream.
        plus_x = cx + 120
        body.append(
            f'<circle cx="{plus_x}" cy="{stream_y}" r="12" fill="#ffffff" stroke="{ACCENT}" '
            f'stroke-width="2"/>'
        )
        body.append(
            f'<text x="{plus_x}" y="{stream_y + 5}" font-size="15" text-anchor="middle" '
            f'fill="{ACCENT}">+</text>'
        )
        # The branch down into norm -> op and back up.
        by = stream_y + 70
        body.append(
            f'<path d="M {cx - 20} {stream_y + 4} L {cx - 20} {by} L {cx} {by}" '
            f'fill="none" stroke="{RULE_STRONG}" stroke-width="1.4"/>'
        )
        body += token_box(
            cx,
            by - 16,
            92,
            32,
            norm,
            fill=ACCENT_SOFT,
            stroke=ACCENT,
            text_fill=ACCENT,
            font_size=11,
        )
        body.append(
            f'<path d="M {cx + 92} {by} L {cx + 108} {by}" stroke="{RULE_STRONG}" '
            f'stroke-width="1.4" marker-end="url(#d2)"/>'
        )
        body += token_box(
            cx + 108,
            by - 16,
            92,
            32,
            op,
            fill="#ffffff",
            stroke=RULE_STRONG,
            font_size=12,
            weight=700,
        )
        body.append(
            f'<path d="M {cx + 154} {by - 16} L {cx + 154} {stream_y + 14} L {plus_x} {stream_y + 14}" '
            f'fill="none" stroke="{RULE_STRONG}" stroke-width="1.4" marker-end="url(#d3)"/>'
        )
        body.append(
            f'<text x="{cx + 100}" y="{by + 34}" font-size="10" text-anchor="middle" '
            f'fill="{MUTED}">{note}</text>'
        )

    body += token_box(
        624,
        stream_y - 15,
        30,
        30,
        "→",
        fill="#ffffff",
        stroke="none",
        text_fill=ACCENT,
        font_size=16,
    )
    body.append(
        f'<text x="{width / 2}" y="{height - 12}" font-size="10.5" text-anchor="middle" '
        f'fill="{MUTED}" font-style="italic">Normalize, apply a sublayer, add the result back — '
        f"twice. The stream is never overwritten, only added to.</text>"
    )
    for name in ("d1", "d2", "d3"):
        color = ACCENT if name == "d1" else RULE_STRONG
        body.append(arrow_marker(color, name))
    return write_svg(
        "transformer-block.svg",
        svg_doc(width, height, "a pre-norm transformer block", body),
    )


def fig_decoder_stack() -> Path:
    """Diagram: the full decoder-only stack from token ids to a distribution."""
    width, height = 440, 340
    cx = width / 2
    bw = 240
    body: list[str] = [eyebrow(24, 28, "DECODER-ONLY TRANSFORMER")]

    def stage(y, label, sub, fill, stroke, tf):
        out = [
            f'<rect x="{cx - bw / 2}" y="{y}" width="{bw}" height="40" rx="8" fill="{fill}" '
            f'stroke="{stroke}"/>',
            f'<text x="{cx}" y="{y + 18}" font-size="12.5" font-weight="700" '
            f'text-anchor="middle" fill="{tf}">{label}</text>',
        ]
        if sub:
            out.append(
                f'<text x="{cx}" y="{y + 33}" font-size="9.5" text-anchor="middle" '
                f'fill="{MUTED if tf != "#fff" else ACCENT_SOFT}">{sub}</text>'
            )
        return out

    def up_arrow(y):
        return (
            f'<path d="M {cx} {y} L {cx} {y - 14}" stroke="{RULE_STRONG}" '
            f'stroke-width="1.6" marker-end="url(#e1)"/>'
        )

    # Bottom to top: embed, blocks, unembed, softmax, distribution.
    body += stage(280, "token embeddings", "ids → vectors", ACCENT_SOFT, ACCENT, ACCENT)
    body.append(up_arrow(280))
    # The repeated block, drawn as a shadowed stack.
    for off in (10, 5, 0):
        shade = "#ffffff" if off == 0 else "#f0efe9"
        body.append(
            f'<rect x="{cx - bw / 2 + off}" y="{224 - off}" width="{bw}" height="40" rx="8" '
            f'fill="{shade}" stroke="{RULE_STRONG}"/>'
        )
    body += [
        f'<text x="{cx}" y="{242}" font-size="12.5" font-weight="700" text-anchor="middle" '
        f'fill="{INK}">transformer block</text>',
        f'<text x="{cx + bw / 2 + 14}" y="{246}" font-size="12" fill="{ACCENT}" '
        f'font-weight="700">× L</text>',
        f'<text x="{cx}" y="{257}" font-size="9.5" text-anchor="middle" fill="{MUTED}">'
        f"attention, then FFN</text>",
    ]
    body.append(up_arrow(214))
    body += stage(
        162,
        "final norm + unembed",
        "vector → vocab scores",
        "#ffffff",
        RULE_STRONG,
        INK,
    )
    body.append(up_arrow(162))
    body += stage(110, "softmax", "scores → probabilities", "#ffffff", RULE_STRONG, INK)
    body.append(up_arrow(110))

    # The output distribution as little bars.
    probs = [0.08, 0.14, 0.44, 0.2, 0.1]
    n = len(probs)
    bwidth, bgap = 26, 12
    total = n * bwidth + (n - 1) * bgap
    bx0 = cx - total / 2
    for i, p in enumerate(probs):
        h = 8 + p * 66
        x = bx0 + i * (bwidth + bgap)
        top = p == max(probs)
        body.append(
            f'<rect x="{x}" y="{92 - h}" width="{bwidth}" height="{h}" rx="3" '
            f'fill="{AMBER if top else ACCENT}" opacity="{1.0 if top else 0.3}"/>'
        )
    body.append(
        f'<text x="{cx}" y="{46}" font-size="10.5" text-anchor="middle" fill="{MUTED}">'
        f"next-token distribution</text>"
    )
    body.append(arrow_marker(RULE_STRONG, "e1"))
    return write_svg(
        "decoder-stack.svg",
        svg_doc(width, height, "the decoder-only transformer stack", body),
    )


# ---------------------------------------------------------------------------
# Chapter 5 — How Modern Architectures Differ
# ---------------------------------------------------------------------------


def fig_rope() -> Path:
    """Diagram: position as rotation; attention depends on the angle between."""
    import math

    width, height = 620, 300
    body: list[str] = [eyebrow(24, 30, "POSITION AS ROTATION")]

    def clock(cx, cy, r, angle_deg, color, label, pos_label):
        a = math.radians(angle_deg)
        hx, hy = cx + r * math.cos(a), cy - r * math.sin(a)
        return [
            f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="#ffffff" stroke="{RULE_STRONG}" stroke-width="1.5"/>',
            f'<line x1="{cx}" y1="{cy}" x2="{hx:.1f}" y2="{hy:.1f}" stroke="{color}" '
            f'stroke-width="2.5" stroke-linecap="round"/>',
            f'<circle cx="{cx}" cy="{cy}" r="3" fill="{color}"/>',
            f'<text x="{cx}" y="{cy + r + 20}" font-size="11" text-anchor="middle" '
            f'fill="{INK}" font-weight="600">{label}</text>',
            f'<text x="{cx}" y="{cy + r + 34}" font-size="9.5" text-anchor="middle" fill="{MUTED}">{pos_label}</text>',
        ]

    r = 40
    cy = 120
    # The same token at later positions: the hand rotates further along the sequence.
    body += clock(110, cy, r, 25, ACCENT, "“cat”", "position 1")
    body += clock(300, cy, r, 95, ACCENT, "“cat”", "position 3")
    body += clock(490, cy, r, 165, ACCENT, "“cat”", "position 6")

    body.append(
        f'<text x="{width / 2}" y="{height - 44}" font-size="11" text-anchor="middle" '
        f'fill="{ACCENT}">each position turns the hand further; the score depends on the '
        f'<tspan font-weight="700">angle between</tspan> two hands</text>'
    )
    body.append(
        f'<text x="{width / 2}" y="{height - 26}" font-size="10.5" text-anchor="middle" '
        f'fill="{MUTED}" font-style="italic">so it encodes how far apart two tokens are '
        f"(relative position), not where each sits.</text>"
    )
    return write_svg(
        "rope.svg",
        svg_doc(width, height, "rotary position embeddings as clock hands", body),
    )


def fig_prenorm() -> Path:
    """Diagram: post-norm in the residual path versus pre-norm beside it."""
    width, height = 620, 280
    body: list[str] = [eyebrow(24, 30, "WHERE NORMALIZATION SITS")]

    def column(x0, title, order, clean_stream):
        out = [
            f'<text x="{x0 + 90}" y="60" font-size="12" font-weight="700" text-anchor="middle" '
            f'fill="{INK}">{title}</text>'
        ]
        # The residual stream, drawn solid+clean for pre-norm and broken for post-norm.
        stream_color = ACCENT if clean_stream else RULE_STRONG
        dash = "" if clean_stream else ' stroke-dasharray="4 4"'
        out.append(
            f'<path d="M {x0 + 90} 76 L {x0 + 90} 232" stroke="{stream_color}" '
            f'stroke-width="2.5"{dash}/>'
        )
        y = 92
        for kind, name in order:
            if kind == "norm":
                out += token_box(
                    x0 + 46,
                    y,
                    88,
                    26,
                    name,
                    fill=ACCENT_SOFT,
                    stroke=ACCENT,
                    text_fill=ACCENT,
                    font_size=10,
                )
            elif kind == "op":
                out += token_box(
                    x0 + 46,
                    y,
                    88,
                    26,
                    name,
                    fill="#ffffff",
                    stroke=RULE_STRONG,
                    font_size=10.5,
                    weight=700,
                )
            else:  # add
                out.append(
                    f'<circle cx="{x0 + 90}" cy="{y + 13}" r="11" fill="#ffffff" '
                    f'stroke="{ACCENT}" stroke-width="1.8"/>'
                )
                out.append(
                    f'<text x="{x0 + 90}" y="{y + 18}" font-size="13" text-anchor="middle" fill="{ACCENT}">+</text>'
                )
            y += 40
        return out

    # Post-norm: normalization sits ON the stream, after the add.
    body += column(
        70,
        "post-norm (2017)",
        [("op", "Attention"), ("add", ""), ("norm", "LayerNorm")],
        clean_stream=False,
    )
    # Pre-norm: normalization sits on the branch; the stream stays an identity path.
    body += column(
        360,
        "pre-norm (modern)",
        [("norm", "RMSNorm"), ("op", "Attention"), ("add", "")],
        clean_stream=True,
    )

    body.append(
        f'<text x="{width / 2}" y="{height - 10}" font-size="10.5" text-anchor="middle" '
        f'fill="{MUTED}" font-style="italic">Pre-norm keeps the residual stream an unbroken '
        f"identity path, so gradients survive great depth.</text>"
    )
    return write_svg(
        "prenorm.svg",
        svg_doc(width, height, "post-norm versus pre-norm placement", body),
    )


def fig_swiglu() -> Path:
    """Diagram: a gated FFN — one projection modulates the other."""
    width, height = 620, 260
    body: list[str] = [eyebrow(24, 30, "A GATED FEED-FORWARD LAYER")]

    body += token_box(
        30, 100, 44, 34, "x", fill=ACCENT, stroke="none", text_fill="#fff", weight=700
    )
    # Two up-projections.
    body.append(
        f'<path d="M 78 112 L 150 74" stroke="{RULE_STRONG}" stroke-width="1.4" marker-end="url(#s1)"/>'
    )
    body.append(
        f'<path d="M 78 122 L 150 160" stroke="{RULE_STRONG}" stroke-width="1.4" marker-end="url(#s1)"/>'
    )
    body += token_box(
        152,
        58,
        118,
        34,
        "Swish(x·W₁)",
        fill=ACCENT_SOFT,
        stroke=ACCENT,
        text_fill=ACCENT,
        font_size=11,
    )
    body += token_box(
        152, 146, 118, 34, "x·W₂", fill="#ffffff", stroke=RULE_STRONG, font_size=11
    )
    body.append(
        f'<text x="211" y="50" font-size="9.5" text-anchor="middle" fill="{MUTED}">the gate</text>'
    )
    body.append(
        f'<text x="211" y="198" font-size="9.5" text-anchor="middle" fill="{MUTED}">the value</text>'
    )

    # The elementwise gate.
    gx = 320
    body.append(
        f'<circle cx="{gx}" cy="119" r="16" fill="#ffffff" stroke="{AMBER}" stroke-width="2"/>'
    )
    body.append(
        f'<text x="{gx}" y="124" font-size="16" text-anchor="middle" fill="{AMBER}">×</text>'
    )
    body.append(
        f'<path d="M 270 75 C 296 90, 300 104, {gx - 14} 112" fill="none" stroke="{ACCENT}" stroke-width="1.6" marker-end="url(#s2)"/>'
    )
    body.append(
        f'<path d="M 270 163 C 296 148, 300 134, {gx - 14} 126" fill="none" stroke="{RULE_STRONG}" stroke-width="1.6" marker-end="url(#s1)"/>'
    )
    body.append(
        f'<path d="M {gx + 16} 119 L 420 119" stroke="{RULE_STRONG}" stroke-width="1.6" marker-end="url(#s1)"/>'
    )
    body += token_box(
        422,
        102,
        96,
        34,
        "down·W₃",
        fill="#ffffff",
        stroke=RULE_STRONG,
        font_size=11,
        weight=700,
    )
    body.append(
        f'<path d="M 518 119 L 560 119" stroke="{RULE_STRONG}" stroke-width="1.6" marker-end="url(#s1)"/>'
    )
    body += token_box(
        562,
        102,
        40,
        34,
        "→",
        fill="#ffffff",
        stroke="none",
        text_fill=ACCENT,
        font_size=16,
    )

    body.append(
        f'<text x="{width / 2}" y="{height - 12}" font-size="10.5" text-anchor="middle" '
        f'fill="{MUTED}" font-style="italic">The gate scales each feature of the value — '
        f"multiplicative control a single ReLU cannot give.</text>"
    )
    body.append(arrow_marker(RULE_STRONG, "s1"))
    body.append(arrow_marker(ACCENT, "s2"))
    return write_svg(
        "swiglu.svg",
        svg_doc(width, height, "the SwiGLU gated feed-forward layer", body),
    )


def fig_attention_sharing() -> Path:
    """Diagram: MHA, GQA, and MQA and the KV cache each implies."""
    width, height = 620, 300
    variants = [
        ("MHA", 8, 8, "one KV head each", BRICK),
        ("GQA", 8, 2, "shared in groups", ACCENT),
        ("MQA", 8, 1, "one shared KV head", AMBER),
    ]
    body: list[str] = [eyebrow(24, 30, "SHARING KEY / VALUE HEADS")]

    col_w = width / 3
    for c, (name, q, kv, note, color) in enumerate(variants):
        cx = col_w * c + col_w / 2
        body.append(
            f'<text x="{cx}" y="70" font-size="14" font-weight="700" text-anchor="middle" fill="{color}">{name}</text>'
        )
        # Query heads (top row).
        qw, qg = 15, 6
        qtotal = q * qw + (q - 1) * qg
        qx0 = cx - qtotal / 2
        qy = 92
        for i in range(q):
            body.append(
                f'<rect x="{qx0 + i * (qw + qg):.1f}" y="{qy}" width="{qw}" height="15" rx="2" '
                f'fill="{INK_SOFT}" opacity="0.7"/>'
            )
        # KV heads (bottom row).
        kvy = 170
        kw = 15
        # Center each KV head under its group of query heads.
        group = q // kv
        for j in range(kv):
            # Group's query span.
            gstart = qx0 + j * group * (qw + qg)
            gend = qx0 + ((j + 1) * group - 1) * (qw + qg) + qw
            kcx = (gstart + gend) / 2
            body.append(
                f'<rect x="{kcx - kw / 2:.1f}" y="{kvy}" width="{kw}" height="15" rx="2" fill="{color}"/>'
            )
            for i in range(group):
                qxc = qx0 + (j * group + i) * (qw + qg) + qw / 2
                body.append(
                    f'<path d="M {qxc:.1f} {qy + 15} L {kcx:.1f} {kvy}" stroke="{color}" '
                    f'stroke-width="0.9" opacity="0.5"/>'
                )
        body.append(
            f'<text x="{cx}" y="{qy - 8}" font-size="9" text-anchor="middle" fill="{MUTED}">{q} query heads</text>'
        )
        body.append(
            f'<text x="{cx}" y="{kvy + 30}" font-size="10.5" text-anchor="middle" fill="{color}" font-weight="600">{kv} KV head{"s" if kv > 1 else ""}</text>'
        )
        body.append(
            f'<text x="{cx}" y="{kvy + 45}" font-size="9.5" text-anchor="middle" fill="{MUTED}">{note}</text>'
        )

    body.append(
        f'<text x="{width / 2}" y="{height - 12}" font-size="10.5" text-anchor="middle" '
        f'fill="{MUTED}" font-style="italic">Fewer KV heads → a smaller KV cache → longer context '
        f"and bigger batches, at a little quality.</text>"
    )
    return write_svg(
        "attention-sharing.svg",
        svg_doc(
            width, height, "multi-head, grouped-query, and multi-query attention", body
        ),
    )


def fig_moe() -> Path:
    """Diagram: a router sends each token to its top experts; total vs active."""
    width, height = 620, 300
    body: list[str] = [eyebrow(24, 30, "MIXTURE OF EXPERTS")]

    body += token_box(
        30, 130, 56, 34, "token", fill=ACCENT_SOFT, stroke=ACCENT, text_fill=ACCENT
    )
    body += token_box(
        120,
        130,
        66,
        34,
        "router",
        fill=ACCENT,
        stroke="none",
        text_fill="#fff",
        weight=700,
    )
    body.append(
        f'<path d="M 86 147 L 116 147" stroke="{RULE_STRONG}" stroke-width="1.5" marker-end="url(#m1)"/>'
    )

    experts_x = 300
    n = 6
    active = {1, 4}
    ew, eh, egap = 150, 26, 12
    ey0 = 60
    for i in range(n):
        y = ey0 + i * (eh + egap)
        is_active = i in active
        color = AMBER if is_active else "#ffffff"
        stroke = AMBER if is_active else RULE_STRONG
        tf = "#ffffff" if is_active else MUTED
        body += token_box(
            experts_x,
            y,
            ew,
            eh,
            f"expert {i + 1}",
            fill=color,
            stroke=stroke,
            text_fill=tf,
            font_size=11,
            weight=700 if is_active else 400,
        )
        edge_color = AMBER if is_active else RULE
        edge_w = 2.0 if is_active else 0.8
        edge_op = 0.9 if is_active else 0.5
        dash = "" if is_active else ' stroke-dasharray="3 3"'
        body.append(
            f'<path d="M 186 147 C 240 147, 250 {y + eh / 2}, {experts_x - 4} {y + eh / 2}" '
            f'fill="none" stroke="{edge_color}" stroke-width="{edge_w}" opacity="{edge_op}"{dash}/>'
        )

    # Total vs active callout.
    body.append(
        f'<rect x="480" y="96" width="118" height="108" rx="8" fill="#ffffff" stroke="{RULE_STRONG}"/>'
    )
    body.append(
        f'<text x="539" y="120" font-size="10" text-anchor="middle" fill="{MUTED}">held in memory</text>'
    )
    body.append(
        f'<text x="539" y="140" font-size="15" text-anchor="middle" fill="{INK}" font-weight="700">6 experts</text>'
    )
    body.append(f'<line x1="496" y1="152" x2="582" y2="152" stroke="{RULE}"/>')
    body.append(
        f'<text x="539" y="172" font-size="10" text-anchor="middle" fill="{MUTED}">run per token</text>'
    )
    body.append(
        f'<text x="539" y="192" font-size="15" text-anchor="middle" fill="{AMBER}" font-weight="700">2 experts</text>'
    )

    body.append(
        f'<text x="{width / 2}" y="{height - 10}" font-size="10.5" text-anchor="middle" '
        f'fill="{MUTED}" font-style="italic">Capacity grows with total experts; compute grows '
        f"only with the few that fire.</text>"
    )
    body.append(arrow_marker(RULE_STRONG, "m1"))
    return write_svg(
        "moe.svg", svg_doc(width, height, "mixture-of-experts routing", body)
    )


def fig_config_tour() -> Path:
    """Diagram: a real config, each field annotated with its component."""
    width, height = 640, 340
    rows = [
        ('"num_attention_heads": 32,', "32 query heads", INK_SOFT),
        ('"num_key_value_heads": 8,', "GQA — 8 KV heads, a 4× smaller cache", ACCENT),
        ('"intermediate_size": 14336,', "SwiGLU width ≈ ⅔·4d, not 4d", AMBER),
        ('"hidden_act": "silu",', "the Swish gate of SwiGLU", AMBER),
        ('"rms_norm_eps": 1e-5,', "RMSNorm, not LayerNorm", VIOLET),
        ('"rope_theta": 500000.0,', "rotary base, raised for long context", ACCENT),
        ('"vocab_size": 128256,', "a large BPE vocabulary (Ch. 3)", MUTED),
    ]
    body: list[str] = [eyebrow(24, 30, "READING A REAL CONFIG (LLAMA-3-CLASS 8B)")]

    y0, rh = 60, 38
    code_x = 40
    ann_x = 330
    for i, (code, note, color) in enumerate(rows):
        y = y0 + i * rh
        body.append(
            f'<rect x="{code_x - 12}" y="{y}" width="290" height="{rh - 8}" rx="5" '
            f'fill="#faf9f5" stroke="{RULE}"/>'
        )
        body.append(
            f'<text x="{code_x}" y="{y + 20}" font-size="12" font-family="{MONO}" fill="{INK}">{code}</text>'
        )
        body.append(
            f'<path d="M {code_x + 284} {y + 15} L {ann_x - 8} {y + 15}" stroke="{color}" '
            f'stroke-width="1.2" opacity="0.5"/>'
        )
        body.append(f'<circle cx="{ann_x - 2}" cy="{y + 15}" r="3" fill="{color}"/>')
        body.append(
            f'<text x="{ann_x + 8}" y="{y + 20}" font-size="11" fill="{color}" font-weight="600">{note}</text>'
        )

    return write_svg(
        "config-tour.svg", svg_doc(width, height, "an annotated model config", body)
    )


# ---------------------------------------------------------------------------
# The cover and the icons.
#
# One motif carries the book's identity: a row of context tokens fading with
# distance, attention arcs converging on a dashed *empty* next position, and
# an amber token settling into it. Autoregression, attention, and the frontier
# (the token not yet written) in a single image. The cover sets it under the
# title; the icons reduce it to four cells so it survives at favicon size.
# ---------------------------------------------------------------------------


def next_token_motif_svg(
    x0: float, row_y: float, size: float, gap: float, n_context: int
) -> str:
    """Emit the next-token motif as SVG elements.

    Draws `n_context` filled tokens, a trailing dashed empty slot holding a
    smaller amber token, and attention arcs from every context token to the
    slot. `row_y` is the top edge of the token row.
    """
    step = size + gap
    centers = [x0 + i * step + size / 2 for i in range(n_context + 1)]
    slot_cx = centers[-1]
    parts = []

    # Arcs first, so the tokens sit on top of their endpoints.
    for i in range(n_context):
        distance = n_context - i
        peak = row_y - (size * 0.55 + distance * size * 0.5)
        width = 1.0 + (i + 1) * (2.2 / n_context)
        opacity = 0.22 + (i + 1) * (0.55 / n_context)
        mid_x = (centers[i] + slot_cx) / 2
        parts.append(
            f'<path d="M {centers[i]:.1f} {row_y} Q {mid_x:.1f} {peak:.1f}, '
            f'{slot_cx:.1f} {row_y}" fill="none" stroke="{ACCENT}" '
            f'stroke-width="{width:.2f}" opacity="{opacity:.2f}" '
            f'stroke-linecap="round"/>'
        )

    for i in range(n_context):
        opacity = 0.34 + 0.66 * (i + 1) / n_context
        parts.append(
            f'<rect x="{x0 + i * step:.1f}" y="{row_y}" width="{size}" '
            f'height="{size}" rx="{size * 0.16:.1f}" fill="{ACCENT}" '
            f'opacity="{opacity:.2f}"/>'
        )

    slot_x = x0 + n_context * step
    parts.append(
        f'<rect x="{slot_x:.1f}" y="{row_y}" width="{size}" height="{size}" '
        f'rx="{size * 0.16:.1f}" fill="none" stroke="{MUTED}" stroke-width="1.6" '
        f'stroke-dasharray="5 4"/>'
    )
    inset = size * 0.22
    parts.append(
        f'<rect x="{slot_x + inset:.1f}" y="{row_y + inset:.1f}" '
        f'width="{size - 2 * inset:.1f}" height="{size - 2 * inset:.1f}" '
        f'rx="{size * 0.1:.1f}" fill="{AMBER}"/>'
    )
    return "\n".join(parts)


def fig_cover() -> Path:
    """The book cover: title over the next-token motif, framed like a monograph."""
    width, height = 640, 960
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'role="img" aria-label="Book cover: Foundations of Large Language Models. '
        f"A row of tokens with attention arcs converging on the empty next "
        f'position, where an amber token is settling in.">',
        f'<rect width="{width}" height="{height}" fill="{PAPER}"/>',
        # A hairline inset frame makes the page read as a physical cover.
        f'<rect x="26" y="26" width="{width - 52}" height="{height - 52}" '
        f'fill="none" stroke="{RULE_STRONG}" stroke-width="1.5"/>',
        # Eyebrow and title, paired the way every chapter opens.
        f'<text x="72" y="152" font-family="{SANS}" font-size="16" '
        f'font-weight="650" letter-spacing="5" fill="{ACCENT}">FOUNDATIONS OF</text>',
    ]
    for i, line in enumerate(("Large", "Language", "Models")):
        parts.append(
            f'<text x="68" y="{232 + i * 74}" font-family="{SERIF}" font-size="68" '
            f'font-weight="700" fill="{INK}">{line}</text>'
        )

    parts.append(next_token_motif_svg(x0=72, row_y=596, size=46, gap=15, n_context=6))

    parts.append(
        f'<path d="M 72 830 L 148 830" stroke="{RULE_STRONG}" stroke-width="1.5"/>'
    )
    for i, line in enumerate(
        ("Training, serving, and shipping", "large language models.")
    ):
        parts.append(
            f'<text x="72" y="{862 + i * 23}" font-family="{SANS}" font-size="15.5" '
            f'fill="{MUTED}">{line}</text>'
        )
    parts.append("</svg>")
    return write_root_asset("cover.svg", "\n".join(parts))


def fig_icon() -> Path:
    """The favicon: the next-token motif alone on a rounded paper tile."""
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 180 180" role="img" '
        'aria-label="Site icon: three tokens with arcs converging on the empty '
        'next position.">',
        f'<rect width="180" height="180" rx="36" fill="{PAPER}"/>',
        next_token_motif_svg(x0=19, row_y=104, size=28, gap=10, n_context=3),
        "</svg>",
    ]
    return write_root_asset("icon.svg", "\n".join(parts))


def fig_touch_icon() -> Path:
    """The apple-touch-icon: the favicon motif, full-bleed PNG (iOS rounds it).

    iOS does not accept SVG here, so matplotlib re-draws the same geometry as
    `fig_icon` at exactly 180 by 180 pixels.
    """
    from matplotlib.patches import FancyBboxPatch, PathPatch, Rectangle
    from matplotlib.path import Path as MplPath

    dpi = 100
    fig = plt.figure(figsize=(1.8, 1.8), dpi=dpi)
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.set_xlim(0, 180)
    ax.set_ylim(180, 0)  # Flip y so the geometry matches the SVG coordinates.
    ax.axis("off")
    ax.add_patch(Rectangle((0, 0), 180, 180, facecolor=PAPER, edgecolor="none"))

    # Mirror the geometry of fig_icon exactly; a lw point is dpi/72 pixels.
    x0, row_y, size, gap, n_context = 19.0, 104.0, 28.0, 10.0, 3
    px = 72 / dpi
    step = size + gap
    centers = [x0 + i * step + size / 2 for i in range(n_context + 1)]
    slot_cx = centers[-1]

    for i in range(n_context):
        distance = n_context - i
        peak = row_y - (size * 0.55 + distance * size * 0.5)
        width = 1.0 + (i + 1) * (2.2 / n_context)
        opacity = 0.22 + (i + 1) * (0.55 / n_context)
        mid_x = (centers[i] + slot_cx) / 2
        arc = MplPath(
            [(centers[i], row_y), (mid_x, peak), (slot_cx, row_y)],
            [MplPath.MOVETO, MplPath.CURVE3, MplPath.CURVE3],
        )
        ax.add_patch(
            PathPatch(
                arc,
                fill=False,
                edgecolor=ACCENT,
                linewidth=width * px,
                alpha=opacity,
                capstyle="round",
            )
        )

    rounding = size * 0.16
    for i in range(n_context):
        opacity = 0.34 + 0.66 * (i + 1) / n_context
        ax.add_patch(
            FancyBboxPatch(
                (x0 + i * step + rounding, row_y + rounding),
                size - 2 * rounding,
                size - 2 * rounding,
                boxstyle=f"round,pad={rounding}",
                facecolor=ACCENT,
                edgecolor="none",
                alpha=opacity,
                mutation_aspect=1,
            )
        )

    slot_x = x0 + n_context * step
    ax.add_patch(
        FancyBboxPatch(
            (slot_x + rounding, row_y + rounding),
            size - 2 * rounding,
            size - 2 * rounding,
            boxstyle=f"round,pad={rounding}",
            facecolor="none",
            edgecolor=MUTED,
            linewidth=1.6 * px,
            linestyle=(0, (3, 2.4)),
            mutation_aspect=1,
        )
    )
    inset = size * 0.22
    ax.add_patch(
        FancyBboxPatch(
            (slot_x + inset + size * 0.1, row_y + inset + size * 0.1),
            size - 2 * inset - 2 * size * 0.1,
            size - 2 * inset - 2 * size * 0.1,
            boxstyle=f"round,pad={size * 0.1}",
            facecolor=AMBER,
            edgecolor="none",
            mutation_aspect=1,
        )
    )

    path = ASSETS_DIR / "apple-touch-icon.png"
    fig.savefig(path, dpi=dpi, facecolor=PAPER)
    plt.close(fig)
    return path


FIGURES = (
    fig_lifecycle,
    fig_attention_lookup,
    fig_causal_mask,
    fig_capacity,
    fig_bandwidth,
    fig_autoregression,
    fig_scaling_emergence,
    fig_base_vs_assistant,
    fig_test_time_compute,
    fig_reading_map,
    fig_manifold,
    fig_training_loop,
    fig_lr_curves,
    fig_generalization,
    fig_precision,
    fig_tokenizer_tradeoff,
    fig_bpe_merges,
    fig_byte_fallback,
    fig_language_tax,
    fig_rnn_vs_attention,
    fig_multi_head,
    fig_transformer_block,
    fig_decoder_stack,
    fig_rope,
    fig_prenorm,
    fig_swiglu,
    fig_attention_sharing,
    fig_moe,
    fig_config_tour,
    fig_cover,
    fig_icon,
    fig_touch_icon,
)


def main() -> None:
    """Regenerate every figure and report where it went."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for make in FIGURES:
        path = make()
        assert (
            path.exists()
        ), f"Figure function '{make.__name__}' did not write its file."
        print(f"  wrote {path.relative_to(ROOT)}")
    print(f"Generated {len(FIGURES)} figures.")


if __name__ == "__main__":
    main()
