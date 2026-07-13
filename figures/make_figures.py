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
