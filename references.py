"""Single source of truth for the book's references.

Chapters cite a work with `[@key]` in their markdown; `build.py` resolves each
key against `REFERENCES`, renders an author-year citation linked to the entry,
and appends a per-chapter References section listing exactly the works that
chapter cites. A citation whose key is missing here fails the build loudly.

Every entry should be verified against the actual paper (arXiv listing, venue
page) before it lands — a wrong citation is worse than no citation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

KEY_PATTERN = re.compile(r"^[a-z][a-z0-9]*$")
ARXIV_ID_PATTERN = re.compile(r"^\d{4}\.\d{4,5}$")


@dataclass(frozen=True)
class Reference:
    """One cited work.

    `authors` holds one "Lastname, F. M." string per author, in the paper's
    order; a corporate author (e.g. "DeepSeek-AI") is a single comma-free
    string. `truncated` marks an author list that is deliberately cut short
    (papers with dozens or hundreds of authors) and renders as "et al.".
    Exactly one of `arxiv` (a bare id like "1706.03762") and `url` may be set;
    both empty means the entry renders with no link.
    """

    key: str
    authors: tuple[str, ...]
    truncated: bool
    year: int
    title: str
    venue: str
    arxiv: str
    url: str

    def __post_init__(self) -> None:
        assert KEY_PATTERN.match(self.key), f"Bad reference key: {self.key!r}."
        assert self.authors, f"Reference '{self.key}' has no authors."
        assert all(a.strip() for a in self.authors), f"Empty author in '{self.key}'."
        assert 1900 < self.year <= 2100, f"Implausible year in '{self.key}'."
        assert self.title and not self.title.endswith(
            "."
        ), f"Title of '{self.key}' must be non-empty and carry no trailing period."
        assert self.venue, f"Reference '{self.key}' has no venue."
        assert not (
            self.arxiv and self.url
        ), f"Reference '{self.key}' sets both arxiv and url; pick one."
        if self.arxiv:
            assert ARXIV_ID_PATTERN.match(
                self.arxiv
            ), f"Bad arXiv id in '{self.key}': {self.arxiv!r}."

    def first_author_family(self) -> str:
        """Return the first author's family name (the part before the comma)."""
        first = self.authors[0]
        return first.split(",")[0].strip() if "," in first else first

    def in_text_label(self) -> str:
        """Return the author-year label, e.g. "Vaswani et al., 2017"."""
        family = self.first_author_family()
        if self.truncated or len(self.authors) >= 3:
            return f"{family} et al., {self.year}"
        if len(self.authors) == 2:
            second = self.authors[1]
            second_family = second.split(",")[0].strip() if "," in second else second
            return f"{family} & {second_family}, {self.year}"
        return f"{family}, {self.year}"

    def link(self) -> str:
        """Return the entry's URL, or "" when it has none."""
        if self.arxiv:
            return f"https://arxiv.org/abs/{self.arxiv}"
        return self.url


_ENTRIES: tuple[Reference, ...] = (
    # ---- Architecture ----------------------------------------------------
    Reference(
        key="vaswani2017",
        authors=(
            "Vaswani, A.",
            "Shazeer, N.",
            "Parmar, N.",
            "Uszkoreit, J.",
            "Jones, L.",
            "Gomez, A. N.",
            "Kaiser, Ł.",
            "Polosukhin, I.",
        ),
        truncated=False,
        year=2017,
        title="Attention is all you need",
        venue="Advances in Neural Information Processing Systems",
        arxiv="1706.03762",
        url="",
    ),
    Reference(
        key="bahdanau2015",
        authors=("Bahdanau, D.", "Cho, K.", "Bengio, Y."),
        truncated=False,
        year=2015,
        title="Neural machine translation by jointly learning to align and translate",
        venue="International Conference on Learning Representations",
        arxiv="1409.0473",
        url="",
    ),
    Reference(
        key="hochreiter1997",
        authors=("Hochreiter, S.", "Schmidhuber, J."),
        truncated=False,
        year=1997,
        title="Long short-term memory",
        venue="Neural Computation, 9(8)",
        arxiv="",
        url="https://doi.org/10.1162/neco.1997.9.8.1735",
    ),
    Reference(
        key="he2016",
        authors=("He, K.", "Zhang, X.", "Ren, S.", "Sun, J."),
        truncated=False,
        year=2016,
        title="Deep residual learning for image recognition",
        venue="IEEE Conference on Computer Vision and Pattern Recognition",
        arxiv="1512.03385",
        url="",
    ),
    Reference(
        key="ba2016",
        authors=("Ba, J. L.", "Kiros, J. R.", "Hinton, G. E."),
        truncated=False,
        year=2016,
        title="Layer normalization",
        venue="arXiv preprint",
        arxiv="1607.06450",
        url="",
    ),
    Reference(
        key="xiong2020",
        authors=(
            "Xiong, R.",
            "Yang, Y.",
            "He, D.",
            "Zheng, K.",
            "Zheng, S.",
            "Xing, C.",
            "Zhang, H.",
            "Lan, Y.",
            "Wang, L.",
            "Liu, T.-Y.",
        ),
        truncated=False,
        year=2020,
        title="On layer normalization in the transformer architecture",
        venue="International Conference on Machine Learning",
        arxiv="2002.04745",
        url="",
    ),
    Reference(
        key="zhang2019",
        authors=("Zhang, B.", "Sennrich, R."),
        truncated=False,
        year=2019,
        title="Root mean square layer normalization",
        venue="Advances in Neural Information Processing Systems",
        arxiv="1910.07467",
        url="",
    ),
    Reference(
        key="su2024",
        authors=(
            "Su, J.",
            "Ahmed, M.",
            "Lu, Y.",
            "Pan, S.",
            "Bo, W.",
            "Liu, Y.",
        ),
        truncated=False,
        year=2024,
        title="RoFormer: Enhanced transformer with rotary position embedding",
        venue="Neurocomputing, 568",
        arxiv="2104.09864",
        url="",
    ),
    Reference(
        key="shazeer2020",
        authors=("Shazeer, N.",),
        truncated=False,
        year=2020,
        title="GLU variants improve transformer",
        venue="arXiv preprint",
        arxiv="2002.05202",
        url="",
    ),
    Reference(
        key="shazeer2019",
        authors=("Shazeer, N.",),
        truncated=False,
        year=2019,
        title="Fast transformer decoding: One write-head is all you need",
        venue="arXiv preprint",
        arxiv="1911.02150",
        url="",
    ),
    Reference(
        key="ainslie2023",
        authors=(
            "Ainslie, J.",
            "Lee-Thorp, J.",
            "de Jong, M.",
            "Zemlyanskiy, Y.",
            "Lebrón, F.",
            "Sanghai, S.",
        ),
        truncated=False,
        year=2023,
        title=(
            "GQA: Training generalized multi-query transformer models "
            "from multi-head checkpoints"
        ),
        venue="Empirical Methods in Natural Language Processing",
        arxiv="2305.13245",
        url="",
    ),
    Reference(
        key="shazeer2017",
        authors=(
            "Shazeer, N.",
            "Mirhoseini, A.",
            "Maziarz, K.",
            "Davis, A.",
            "Le, Q.",
            "Hinton, G.",
            "Dean, J.",
        ),
        truncated=False,
        year=2017,
        title=(
            "Outrageously large neural networks: "
            "The sparsely-gated mixture-of-experts layer"
        ),
        venue="International Conference on Learning Representations",
        arxiv="1701.06538",
        url="",
    ),
    Reference(
        key="fedus2022",
        authors=("Fedus, W.", "Zoph, B.", "Shazeer, N."),
        truncated=False,
        year=2022,
        title=(
            "Switch transformers: Scaling to trillion parameter models "
            "with simple and efficient sparsity"
        ),
        venue="Journal of Machine Learning Research, 23(120)",
        arxiv="2101.03961",
        url="",
    ),
    Reference(
        key="touvron2023",
        authors=(
            "Touvron, H.",
            "Lavril, T.",
            "Izacard, G.",
            "Martinet, X.",
        ),
        truncated=True,
        year=2023,
        title="LLaMA: Open and efficient foundation language models",
        venue="arXiv preprint",
        arxiv="2302.13971",
        url="",
    ),
    Reference(
        key="grattafiori2024",
        authors=("Grattafiori, A.", "Dubey, A.", "Jauhri, A.", "Pandey, A."),
        truncated=True,
        year=2024,
        title="The Llama 3 herd of models",
        venue="arXiv preprint",
        arxiv="2407.21783",
        url="",
    ),
    # ---- Tokenization ----------------------------------------------------
    Reference(
        key="gage1994",
        authors=("Gage, P.",),
        truncated=False,
        year=1994,
        title="A new algorithm for data compression",
        venue="The C Users Journal, 12(2)",
        arxiv="",
        url="",
    ),
    Reference(
        key="sennrich2016",
        authors=("Sennrich, R.", "Haddow, B.", "Birch, A."),
        truncated=False,
        year=2016,
        title="Neural machine translation of rare words with subword units",
        venue="Association for Computational Linguistics",
        arxiv="1508.07909",
        url="",
    ),
    Reference(
        key="schuster2012",
        authors=("Schuster, M.", "Nakajima, K."),
        truncated=False,
        year=2012,
        title="Japanese and Korean voice search",
        venue=(
            "IEEE International Conference on Acoustics, Speech and "
            "Signal Processing"
        ),
        arxiv="",
        url="https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/37842.pdf",
    ),
    Reference(
        key="kudo2018",
        authors=("Kudo, T.",),
        truncated=False,
        year=2018,
        title=(
            "Subword regularization: Improving neural network translation "
            "models with multiple subword candidates"
        ),
        venue="Association for Computational Linguistics",
        arxiv="1804.10959",
        url="",
    ),
    Reference(
        key="kudo2018sp",
        authors=("Kudo, T.", "Richardson, J."),
        truncated=False,
        year=2018,
        title=(
            "SentencePiece: A simple and language independent subword "
            "tokenizer and detokenizer for neural text processing"
        ),
        venue="Empirical Methods in Natural Language Processing (System Demonstrations)",
        arxiv="1808.06226",
        url="",
    ),
    Reference(
        key="devlin2019",
        authors=("Devlin, J.", "Chang, M.-W.", "Lee, K.", "Toutanova, K."),
        truncated=False,
        year=2019,
        title=(
            "BERT: Pre-training of deep bidirectional transformers for "
            "language understanding"
        ),
        venue=(
            "North American Chapter of the Association for " "Computational Linguistics"
        ),
        arxiv="1810.04805",
        url="",
    ),
    Reference(
        key="petrov2023",
        authors=("Petrov, A.", "La Malfa, E.", "Torr, P. H. S.", "Bibi, A."),
        truncated=False,
        year=2023,
        title="Language model tokenizers introduce unfairness between languages",
        venue="Advances in Neural Information Processing Systems",
        arxiv="2305.15425",
        url="",
    ),
    # ---- Models and scale ------------------------------------------------
    Reference(
        key="radford2019",
        authors=(
            "Radford, A.",
            "Wu, J.",
            "Child, R.",
            "Luan, D.",
            "Amodei, D.",
            "Sutskever, I.",
        ),
        truncated=False,
        year=2019,
        title="Language models are unsupervised multitask learners",
        venue="OpenAI technical report",
        arxiv="",
        url="https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf",
    ),
    Reference(
        key="brown2020",
        authors=("Brown, T.", "Mann, B.", "Ryder, N.", "Subbiah, M."),
        truncated=True,
        year=2020,
        title="Language models are few-shot learners",
        venue="Advances in Neural Information Processing Systems",
        arxiv="2005.14165",
        url="",
    ),
    Reference(
        key="kaplan2020",
        authors=(
            "Kaplan, J.",
            "McCandlish, S.",
            "Henighan, T.",
            "Brown, T. B.",
        ),
        truncated=True,
        year=2020,
        title="Scaling laws for neural language models",
        venue="arXiv preprint",
        arxiv="2001.08361",
        url="",
    ),
    Reference(
        key="hoffmann2022",
        authors=(
            "Hoffmann, J.",
            "Borgeaud, S.",
            "Mensch, A.",
            "Buchatskaya, E.",
        ),
        truncated=True,
        year=2022,
        title="Training compute-optimal large language models",
        venue="Advances in Neural Information Processing Systems",
        arxiv="2203.15556",
        url="",
    ),
    Reference(
        key="wei2022",
        authors=("Wei, J.", "Tay, Y.", "Bommasani, R.", "Raffel, C."),
        truncated=True,
        year=2022,
        title="Emergent abilities of large language models",
        venue="Transactions on Machine Learning Research",
        arxiv="2206.07682",
        url="",
    ),
    # ---- Alignment -------------------------------------------------------
    Reference(
        key="christiano2017",
        authors=(
            "Christiano, P. F.",
            "Leike, J.",
            "Brown, T. B.",
            "Martic, M.",
            "Legg, S.",
            "Amodei, D.",
        ),
        truncated=False,
        year=2017,
        title="Deep reinforcement learning from human preferences",
        venue="Advances in Neural Information Processing Systems",
        arxiv="1706.03741",
        url="",
    ),
    Reference(
        key="ouyang2022",
        authors=("Ouyang, L.", "Wu, J.", "Jiang, X.", "Almeida, D."),
        truncated=True,
        year=2022,
        title="Training language models to follow instructions with human feedback",
        venue="Advances in Neural Information Processing Systems",
        arxiv="2203.02155",
        url="",
    ),
    Reference(
        key="rafailov2023",
        authors=(
            "Rafailov, R.",
            "Sharma, A.",
            "Mitchell, E.",
            "Manning, C. D.",
            "Ermon, S.",
            "Finn, C.",
        ),
        truncated=False,
        year=2023,
        title=(
            "Direct preference optimization: Your language model is "
            "secretly a reward model"
        ),
        venue="Advances in Neural Information Processing Systems",
        arxiv="2305.18290",
        url="",
    ),
    # ---- Deep learning fundamentals ---------------------------------------
    Reference(
        key="rumelhart1986",
        authors=("Rumelhart, D. E.", "Hinton, G. E.", "Williams, R. J."),
        truncated=False,
        year=1986,
        title="Learning representations by back-propagating errors",
        venue="Nature, 323",
        arxiv="",
        url="https://doi.org/10.1038/323533a0",
    ),
    Reference(
        key="kingma2015",
        authors=("Kingma, D. P.", "Ba, J."),
        truncated=False,
        year=2015,
        title="Adam: A method for stochastic optimization",
        venue="International Conference on Learning Representations",
        arxiv="1412.6980",
        url="",
    ),
    Reference(
        key="loshchilov2019",
        authors=("Loshchilov, I.", "Hutter, F."),
        truncated=False,
        year=2019,
        title="Decoupled weight decay regularization",
        venue="International Conference on Learning Representations",
        arxiv="1711.05101",
        url="",
    ),
    Reference(
        key="srivastava2014",
        authors=(
            "Srivastava, N.",
            "Hinton, G.",
            "Krizhevsky, A.",
            "Sutskever, I.",
            "Salakhutdinov, R.",
        ),
        truncated=False,
        year=2014,
        title="Dropout: A simple way to prevent neural networks from overfitting",
        venue="Journal of Machine Learning Research, 15(56)",
        arxiv="",
        url="https://jmlr.org/papers/v15/srivastava14a.html",
    ),
    Reference(
        key="zhang2017",
        authors=(
            "Zhang, C.",
            "Bengio, S.",
            "Hardt, M.",
            "Recht, B.",
            "Vinyals, O.",
        ),
        truncated=False,
        year=2017,
        title="Understanding deep learning requires rethinking generalization",
        venue="International Conference on Learning Representations",
        arxiv="1611.03530",
        url="",
    ),
    Reference(
        key="micikevicius2018",
        authors=(
            "Micikevicius, P.",
            "Narang, S.",
            "Alben, J.",
            "Diamos, G.",
        ),
        truncated=True,
        year=2018,
        title="Mixed precision training",
        venue="International Conference on Learning Representations",
        arxiv="1710.03740",
        url="",
    ),
    Reference(
        key="kalamkar2019",
        authors=(
            "Kalamkar, D.",
            "Mudigere, D.",
            "Mellempudi, N.",
            "Das, D.",
        ),
        truncated=True,
        year=2019,
        title="A study of BFLOAT16 for deep learning training",
        venue="arXiv preprint",
        arxiv="1905.12322",
        url="",
    ),
    Reference(
        key="goodfellow2016",
        authors=("Goodfellow, I.", "Bengio, Y.", "Courville, A."),
        truncated=False,
        year=2016,
        title="Deep learning",
        venue="MIT Press",
        arxiv="",
        url="https://www.deeplearningbook.org/",
    ),
    # ---- The frontier ------------------------------------------------------
    Reference(
        key="deepseek2025",
        authors=("DeepSeek-AI",),
        truncated=False,
        year=2025,
        title=(
            "DeepSeek-R1: Incentivizing reasoning capability in LLMs "
            "via reinforcement learning"
        ),
        venue="arXiv preprint",
        arxiv="2501.12948",
        url="",
    ),
    Reference(
        key="openai2024",
        authors=("OpenAI",),
        truncated=False,
        year=2024,
        title="Learning to reason with LLMs",
        venue="OpenAI blog",
        arxiv="",
        url="https://openai.com/index/learning-to-reason-with-llms/",
    ),
)


def _build_index(entries: tuple[Reference, ...]) -> dict[str, Reference]:
    """Index entries by key, failing loudly on duplicates."""
    index: dict[str, Reference] = {}
    for entry in entries:
        assert entry.key not in index, f"Duplicate reference key: {entry.key}."
        index[entry.key] = entry
    return index


REFERENCES: dict[str, Reference] = _build_index(_ENTRIES)
