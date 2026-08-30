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
    # ---- Part I second-layer depth ----------------------------------------
    Reference(
        key="schaeffer2023",
        authors=("Schaeffer, R.", "Miranda, B.", "Koyejo, S."),
        truncated=False,
        year=2023,
        title="Are emergent abilities of large language models a mirage?",
        venue="Advances in Neural Information Processing Systems",
        arxiv="2304.15004",
        url="",
    ),
    Reference(
        key="provilkov2020",
        authors=("Provilkov, I.", "Emelianenko, D.", "Voita, E."),
        truncated=False,
        year=2020,
        title="BPE-dropout: Simple and effective subword regularization",
        venue="Association for Computational Linguistics",
        arxiv="1910.13267",
        url="",
    ),
    Reference(
        key="peng2023",
        authors=("Peng, B.", "Quesnelle, J.", "Fan, H.", "Shippole, E."),
        truncated=False,
        year=2023,
        title="YaRN: Efficient context window extension of large language models",
        venue="International Conference on Learning Representations",
        arxiv="2309.00071",
        url="",
    ),
    Reference(
        key="deepseekv2",
        authors=("DeepSeek-AI",),
        truncated=False,
        year=2024,
        title=(
            "DeepSeek-V2: A strong, economical, and efficient "
            "mixture-of-experts language model"
        ),
        venue="arXiv preprint",
        arxiv="2405.04434",
        url="",
    ),
    Reference(
        key="deepseekv3",
        authors=("DeepSeek-AI",),
        truncated=False,
        year=2024,
        title="DeepSeek-V3 technical report",
        venue="arXiv preprint",
        arxiv="2412.19437",
        url="",
    ),
    Reference(
        key="jiang2023",
        authors=(
            "Jiang, A. Q.",
            "Sablayrolles, A.",
            "Mensch, A.",
            "Bamford, C.",
        ),
        truncated=True,
        year=2023,
        title="Mistral 7B",
        venue="arXiv preprint",
        arxiv="2310.06825",
        url="",
    ),
    Reference(
        key="raffel2020",
        authors=(
            "Raffel, C.",
            "Shazeer, N.",
            "Roberts, A.",
            "Lee, K.",
        ),
        truncated=True,
        year=2020,
        title=(
            "Exploring the limits of transfer learning with a unified "
            "text-to-text transformer"
        ),
        venue="Journal of Machine Learning Research, 21(140)",
        arxiv="1910.10683",
        url="",
    ),
    Reference(
        key="dehghani2023",
        authors=(
            "Dehghani, M.",
            "Djolonga, J.",
            "Mustafa, B.",
            "Padlewski, P.",
        ),
        truncated=True,
        year=2023,
        title="Scaling vision transformers to 22 billion parameters",
        venue="International Conference on Machine Learning",
        arxiv="2302.05442",
        url="",
    ),
    # ---- Pretraining data and objective (Part II) --------------------------
    Reference(
        key="shannon1951",
        authors=("Shannon, C. E.",),
        truncated=False,
        year=1951,
        title="Prediction and entropy of printed English",
        venue="Bell System Technical Journal, 30(1)",
        arxiv="",
        url="https://doi.org/10.1002/j.1538-7305.1951.tb01366.x",
    ),
    Reference(
        key="deletang2024",
        authors=(
            "Delétang, G.",
            "Ruoss, A.",
            "Duquenne, P.-A.",
            "Catt, E.",
        ),
        truncated=True,
        year=2024,
        title="Language modeling is compression",
        venue="International Conference on Learning Representations",
        arxiv="2309.10668",
        url="",
    ),
    Reference(
        key="gao2020",
        authors=("Gao, L.", "Biderman, S.", "Black, S.", "Golding, L."),
        truncated=True,
        year=2020,
        title="The Pile: An 800GB dataset of diverse text for language modeling",
        venue="arXiv preprint",
        arxiv="2101.00027",
        url="",
    ),
    Reference(
        key="penedo2024",
        authors=("Penedo, G.", "Kydlíček, H.", "Ben Allal, L.", "Lozhkov, A."),
        truncated=True,
        year=2024,
        title=(
            "The FineWeb datasets: Decanting the web for the finest "
            "text data at scale"
        ),
        venue="Advances in Neural Information Processing Systems (Datasets and Benchmarks)",
        arxiv="2406.17557",
        url="",
    ),
    Reference(
        key="lee2022",
        authors=("Lee, K.", "Ippolito, D.", "Nystrom, A.", "Zhang, C."),
        truncated=True,
        year=2022,
        title="Deduplicating training data makes language models better",
        venue="Association for Computational Linguistics",
        arxiv="2107.06499",
        url="",
    ),
    Reference(
        key="xie2023",
        authors=("Xie, S. M.", "Pham, H.", "Dong, X.", "Du, N."),
        truncated=True,
        year=2023,
        title="DoReMi: Optimizing data mixtures speeds up language model pretraining",
        venue="Advances in Neural Information Processing Systems",
        arxiv="2305.10429",
        url="",
    ),
    Reference(
        key="muennighoff2023",
        authors=("Muennighoff, N.", "Rush, A. M.", "Barak, B.", "Le Scao, T."),
        truncated=True,
        year=2023,
        title="Scaling data-constrained language models",
        venue="Advances in Neural Information Processing Systems",
        arxiv="2305.16264",
        url="",
    ),
    # ---- Training dynamics (Part II) ----------------------------------------
    Reference(
        key="goyal2017",
        authors=("Goyal, P.", "Dollár, P.", "Girshick, R.", "Noordhuis, P."),
        truncated=True,
        year=2017,
        title="Accurate, large minibatch SGD: Training ImageNet in 1 hour",
        venue="arXiv preprint",
        arxiv="1706.02677",
        url="",
    ),
    Reference(
        key="mccandlish2018",
        authors=("McCandlish, S.", "Kaplan, J.", "Amodei, D.", "OpenAI Dota Team"),
        truncated=False,
        year=2018,
        title="An empirical model of large-batch training",
        venue="arXiv preprint",
        arxiv="1812.06162",
        url="",
    ),
    Reference(
        key="loshchilov2017",
        authors=("Loshchilov, I.", "Hutter, F."),
        truncated=False,
        year=2017,
        title="SGDR: Stochastic gradient descent with warm restarts",
        venue="International Conference on Learning Representations",
        arxiv="1608.03983",
        url="",
    ),
    Reference(
        key="chowdhery2023",
        authors=("Chowdhery, A.", "Narang, S.", "Devlin, J.", "Bosma, M."),
        truncated=True,
        year=2023,
        title="PaLM: Scaling language modeling with Pathways",
        venue="Journal of Machine Learning Research, 24(240)",
        arxiv="2204.02311",
        url="",
    ),
    Reference(
        key="zoph2022",
        authors=("Zoph, B.", "Bello, I.", "Kumar, S.", "Du, N."),
        truncated=True,
        year=2022,
        title="ST-MoE: Designing stable and transferable sparse expert models",
        venue="arXiv preprint",
        arxiv="2202.08906",
        url="",
    ),
    Reference(
        key="zhang2022",
        authors=("Zhang, S.", "Roller, S.", "Goyal, N.", "Artetxe, M."),
        truncated=True,
        year=2022,
        title="OPT: Open pre-trained transformer language models",
        venue="arXiv preprint",
        arxiv="2205.01068",
        url="",
    ),
    Reference(
        key="hu2024",
        authors=("Hu, S.", "Tu, Y.", "Han, X.", "He, C."),
        truncated=True,
        year=2024,
        title=(
            "MiniCPM: Unveiling the potential of small language models "
            "with scalable training strategies"
        ),
        venue="arXiv preprint",
        arxiv="2404.06395",
        url="",
    ),
    # ---- Distributed training (Part II) --------------------------------------
    Reference(
        key="rajbhandari2020",
        authors=("Rajbhandari, S.", "Rasley, J.", "Ruwase, O.", "He, Y."),
        truncated=False,
        year=2020,
        title="ZeRO: Memory optimizations toward training trillion parameter models",
        venue=(
            "International Conference for High Performance Computing, "
            "Networking, Storage and Analysis (SC20)"
        ),
        arxiv="1910.02054",
        url="",
    ),
    Reference(
        key="zhao2023",
        authors=("Zhao, Y.", "Gu, A.", "Varma, R.", "Luo, L."),
        truncated=True,
        year=2023,
        title="PyTorch FSDP: Experiences on scaling fully sharded data parallel",
        venue="Proceedings of the VLDB Endowment, 16(12)",
        arxiv="2304.11277",
        url="",
    ),
    Reference(
        key="shoeybi2019",
        authors=(
            "Shoeybi, M.",
            "Patwary, M.",
            "Puri, R.",
            "LeGresley, P.",
            "Casper, J.",
            "Catanzaro, B.",
        ),
        truncated=False,
        year=2019,
        title=(
            "Megatron-LM: Training multi-billion parameter language models "
            "using model parallelism"
        ),
        venue="arXiv preprint",
        arxiv="1909.08053",
        url="",
    ),
    Reference(
        key="huang2019",
        authors=("Huang, Y.", "Cheng, Y.", "Bapna, A.", "Firat, O."),
        truncated=True,
        year=2019,
        title=(
            "GPipe: Efficient training of giant neural networks "
            "using pipeline parallelism"
        ),
        venue="Advances in Neural Information Processing Systems",
        arxiv="1811.06965",
        url="",
    ),
    Reference(
        key="narayanan2021",
        authors=("Narayanan, D.", "Shoeybi, M.", "Casper, J.", "LeGresley, P."),
        truncated=True,
        year=2021,
        title=(
            "Efficient large-scale language model training on GPU clusters "
            "using Megatron-LM"
        ),
        venue=(
            "International Conference for High Performance Computing, "
            "Networking, Storage and Analysis (SC21)"
        ),
        arxiv="2104.04473",
        url="",
    ),
    # ---- Scaling laws (Part II) ----------------------------------------------
    Reference(
        key="sardana2024",
        authors=("Sardana, N.", "Portes, J.", "Doubov, S.", "Frankle, J."),
        truncated=False,
        year=2024,
        title=(
            "Beyond Chinchilla-optimal: Accounting for inference in "
            "language model scaling laws"
        ),
        venue="International Conference on Machine Learning",
        arxiv="2401.00448",
        url="",
    ),
    Reference(
        key="wei2021",
        authors=(
            "Wei, J.",
            "Bosma, M.",
            "Zhao, V. Y.",
            "Guu, K.",
        ),
        truncated=True,
        year=2021,
        title="Finetuned language models are zero-shot learners",
        venue="arXiv preprint",
        arxiv="2109.01652",
        url="",
    ),
    Reference(
        key="zhou2023",
        authors=(
            "Zhou, C.",
            "Liu, P.",
            "Xu, P.",
            "Iyer, S.",
        ),
        truncated=True,
        year=2023,
        title="LIMA: Less is more for alignment",
        venue="Advances in Neural Information Processing Systems",
        arxiv="2305.11206",
        url="",
    ),
    Reference(
        key="wang2023",
        authors=(
            "Wang, Y.",
            "Kordi, Y.",
            "Mishra, S.",
            "Liu, A.",
        ),
        truncated=True,
        year=2023,
        title="Self-Instruct: Aligning language models with self-generated instructions",
        venue="Annual Meeting of the Association for Computational Linguistics",
        arxiv="2212.10560",
        url="",
    ),
    Reference(
        key="sharma2023",
        authors=(
            "Sharma, M.",
            "Tong, M.",
            "Korbak, T.",
            "Duvenaud, D.",
        ),
        truncated=True,
        year=2023,
        title="Towards understanding sycophancy in language models",
        venue="arXiv preprint",
        arxiv="2310.13548",
        url="",
    ),
    Reference(
        key="gekhman2024",
        authors=(
            "Gekhman, Z.",
            "Yona, G.",
            "Aharoni, R.",
            "Eyal, M.",
        ),
        truncated=True,
        year=2024,
        title="Does fine-tuning LLMs on new knowledge encourage hallucinations?",
        venue="Conference on Empirical Methods in Natural Language Processing",
        arxiv="2405.05904",
        url="",
    ),

    Reference(
        key="schulman2017",
        authors=(
            "Schulman, J.",
            "Wolski, F.",
            "Dhariwal, P.",
            "Radford, A.",
            "Klimov, O.",
        ),
        truncated=False,
        year=2017,
        title="Proximal policy optimization algorithms",
        venue="arXiv preprint",
        arxiv="1707.06347",
        url="",
    ),
    Reference(
        key="bai2022hh",
        authors=("Bai, Y.", "Jones, A.", "Ndousse, K.", "Askell, A."),
        truncated=True,
        year=2022,
        title=(
            "Training a helpful and harmless assistant with reinforcement "
            "learning from human feedback"
        ),
        venue="arXiv preprint",
        arxiv="2204.05862",
        url="",
    ),
    Reference(
        key="gao2023",
        authors=("Gao, L.", "Schulman, J.", "Hilton, J."),
        truncated=False,
        year=2023,
        title="Scaling laws for reward model overoptimization",
        venue="International Conference on Machine Learning",
        arxiv="2210.10760",
        url="",
    ),
    Reference(
        key="singhal2023",
        authors=("Singhal, P.", "Goyal, T.", "Xu, J.", "Durrett, G."),
        truncated=False,
        year=2023,
        title="A long way to go: Investigating length correlations in RLHF",
        venue="arXiv preprint",
        arxiv="2310.03716",
        url="",
    ),

    # ---- Preference optimization (Chapter 12) ----------------------------
    Reference(
        key="azar2024",
        authors=(
            "Gheshlaghi Azar, M.",
            "Guo, Z. D.",
            "Piot, B.",
            "Munos, R.",
            "Rowland, M.",
            "Valko, M.",
            "Calandriello, D.",
        ),
        truncated=False,
        year=2024,
        title=(
            "A general theoretical paradigm to understand learning from "
            "human preferences"
        ),
        venue="International Conference on Artificial Intelligence and Statistics",
        arxiv="2310.12036",
        url="",
    ),
    Reference(
        key="ethayarajh2024",
        authors=(
            "Ethayarajh, K.",
            "Xu, W.",
            "Muennighoff, N.",
            "Jurafsky, D.",
            "Kiela, D.",
        ),
        truncated=False,
        year=2024,
        title="KTO: Model alignment as prospect theoretic optimization",
        venue="International Conference on Machine Learning",
        arxiv="2402.01306",
        url="",
    ),
    Reference(
        key="hong2024",
        authors=("Hong, J.", "Lee, N.", "Thorne, J."),
        truncated=False,
        year=2024,
        title="ORPO: Monolithic preference optimization without reference model",
        venue="Conference on Empirical Methods in Natural Language Processing",
        arxiv="2403.07691",
        url="",
    ),
    Reference(
        key="meng2024",
        authors=("Meng, Y.", "Xia, M.", "Chen, D."),
        truncated=False,
        year=2024,
        title="SimPO: Simple preference optimization with a reference-free reward",
        venue="Advances in Neural Information Processing Systems",
        arxiv="2405.14734",
        url="",
    ),
    Reference(
        key="xu2024",
        authors=(
            "Xu, S.",
            "Fu, W.",
            "Gao, J.",
            "Ye, W.",
            "Liu, W.",
            "Mei, Z.",
            "Wang, G.",
            "Yu, C.",
            "Wu, Y.",
        ),
        truncated=False,
        year=2024,
        title="Is DPO superior to PPO for LLM alignment? A comprehensive study",
        venue="International Conference on Machine Learning",
        arxiv="2404.10719",
        url="",
    ),
    Reference(
        key="ivison2024",
        authors=(
            "Ivison, H.",
            "Wang, Y.",
            "Liu, J.",
            "Wu, Z.",
            "Pyatkin, V.",
            "Lambert, N.",
            "Smith, N. A.",
            "Choi, Y.",
            "Hajishirzi, H.",
        ),
        truncated=False,
        year=2024,
        title=(
            "Unpacking DPO and PPO: Disentangling best practices for "
            "learning from preference feedback"
        ),
        venue="Advances in Neural Information Processing Systems",
        arxiv="2406.09279",
        url="",
    ),
    Reference(
        key="bai2022",
        authors=(
            "Bai, Y.",
            "Kadavath, S.",
            "Kundu, S.",
            "Askell, A.",
        ),
        truncated=True,
        year=2022,
        title="Constitutional AI: Harmlessness from AI feedback",
        venue="arXiv preprint",
        arxiv="2212.08073",
        url="",
    ),
    Reference(
        key="lee2023",
        authors=(
            "Lee, H.",
            "Phatale, S.",
            "Mansoor, H.",
            "Mesnard, T.",
            "Ferret, J.",
            "Lu, K.",
            "Bishop, C.",
            "Hall, E.",
            "Carbune, V.",
            "Rastogi, A.",
            "Prakash, S.",
        ),
        truncated=False,
        year=2023,
        title=(
            "RLAIF: Scaling reinforcement learning from human feedback "
            "with AI feedback"
        ),
        venue="arXiv preprint",
        arxiv="2309.00267",
        url="",
    ),

    # ---- Parameter-efficient fine-tuning (Part III) --------------------------
    Reference(
        key="hu2021",
        authors=(
            "Hu, E. J.",
            "Shen, Y.",
            "Wallis, P.",
            "Allen-Zhu, Z.",
            "Li, Y.",
            "Wang, S.",
            "Wang, L.",
            "Chen, W.",
        ),
        truncated=False,
        year=2021,
        title="LoRA: Low-rank adaptation of large language models",
        venue="arXiv preprint",
        arxiv="2106.09685",
        url="",
    ),
    Reference(
        key="aghajanyan2021",
        authors=("Aghajanyan, A.", "Gupta, S.", "Zettlemoyer, L."),
        truncated=False,
        year=2021,
        title=(
            "Intrinsic dimensionality explains the effectiveness of "
            "language model fine-tuning"
        ),
        venue="Association for Computational Linguistics",
        arxiv="2012.13255",
        url="",
    ),
    Reference(
        key="dettmers2023",
        authors=(
            "Dettmers, T.",
            "Pagnoni, A.",
            "Holtzman, A.",
            "Zettlemoyer, L.",
        ),
        truncated=False,
        year=2023,
        title="QLoRA: Efficient finetuning of quantized LLMs",
        venue="Advances in Neural Information Processing Systems",
        arxiv="2305.14314",
        url="",
    ),
    Reference(
        key="liu2024",
        authors=(
            "Liu, S.-Y.",
            "Wang, C.-Y.",
            "Yin, H.",
            "Molchanov, P.",
            "Wang, Y.-C. F.",
            "Cheng, K.-T.",
            "Chen, M.-H.",
        ),
        truncated=False,
        year=2024,
        title="DoRA: Weight-decomposed low-rank adaptation",
        venue="International Conference on Machine Learning",
        arxiv="2402.09353",
        url="",
    ),
    Reference(
        key="sheng2024",
        authors=(
            "Sheng, Y.",
            "Cao, S.",
            "Li, D.",
            "Hooper, C.",
        ),
        truncated=True,
        year=2024,
        title="S-LoRA: Serving thousands of concurrent LoRA adapters",
        venue="Proceedings of Machine Learning and Systems (MLSys)",
        arxiv="2311.03285",
        url="",
    ),
    Reference(
        key="fan2018",
        authors=("Fan, A.", "Lewis, M.", "Dauphin, Y."),
        truncated=False,
        year=2018,
        title="Hierarchical neural story generation",
        venue="Annual Meeting of the Association for Computational Linguistics",
        arxiv="1805.04833",
        url="",
    ),

    Reference(
        key="holtzman2020",
        authors=("Holtzman, A.", "Buys, J.", "Du, L.", "Forbes, M.", "Choi, Y."),
        truncated=False,
        year=2020,
        title="The curious case of neural text degeneration",
        venue="International Conference on Learning Representations",
        arxiv="1904.09751",
        url="",
    ),

    Reference(
        key="keskar2019",
        authors=(
            "Keskar, N. S.",
            "McCann, B.",
            "Varshney, L. R.",
            "Xiong, C.",
            "Socher, R.",
        ),
        truncated=False,
        year=2019,
        title="CTRL: A conditional transformer language model for controllable generation",
        venue="arXiv preprint",
        arxiv="1909.05858",
        url="",
    ),

    Reference(
        key="meister2023",
        authors=("Meister, C.", "Pimentel, T.", "Wiher, G.", "Cotterell, R."),
        truncated=False,
        year=2023,
        title="Locally typical sampling",
        venue="Transactions of the Association for Computational Linguistics, 11",
        arxiv="2202.00666",
        url="",
    ),

    Reference(
        key="nguyen2024",
        authors=(
            "Nguyen, M.",
            "Baker, A.",
            "Neo, C.",
            "Roush, A.",
            "Kirsch, A.",
            "Shwartz-Ziv, R.",
        ),
        truncated=False,
        year=2024,
        title="Turning up the heat: Min-p sampling for creative and coherent LLM outputs",
        venue="International Conference on Learning Representations",
        arxiv="2407.01082",
        url="",
    ),

    # ---- Inference optimization (Chapter 15) --------------------------------
    Reference(
        key="dao2022",
        authors=(
            "Dao, T.",
            "Fu, D. Y.",
            "Ermon, S.",
            "Rudra, A.",
            "Ré, C.",
        ),
        truncated=False,
        year=2022,
        title=(
            "FlashAttention: Fast and memory-efficient exact attention "
            "with IO-awareness"
        ),
        venue="Advances in Neural Information Processing Systems",
        arxiv="2205.14135",
        url="",
    ),

    Reference(
        key="dao2023",
        authors=("Dao, T.",),
        truncated=False,
        year=2023,
        title=(
            "FlashAttention-2: Faster attention with better parallelism "
            "and work partitioning"
        ),
        venue="arXiv preprint",
        arxiv="2307.08691",
        url="",
    ),

    Reference(
        key="kwon2023",
        authors=(
            "Kwon, W.",
            "Li, Z.",
            "Zhuang, S.",
            "Sheng, Y.",
        ),
        truncated=True,
        year=2023,
        title=(
            "Efficient memory management for large language model serving "
            "with PagedAttention"
        ),
        venue="ACM Symposium on Operating Systems Principles (SOSP)",
        arxiv="2309.06180",
        url="",
    ),

    Reference(
        key="leviathan2023",
        authors=("Leviathan, Y.", "Kalman, M.", "Matias, Y."),
        truncated=False,
        year=2023,
        title="Fast inference from transformers via speculative decoding",
        venue="International Conference on Machine Learning",
        arxiv="2211.17192",
        url="",
    ),

    Reference(
        key="chen2023",
        authors=(
            "Chen, C.",
            "Borgeaud, S.",
            "Irving, G.",
            "Lespiau, J.-B.",
            "Sifre, L.",
            "Jumper, J.",
        ),
        truncated=False,
        year=2023,
        title=(
            "Accelerating large language model decoding with speculative sampling"
        ),
        venue="arXiv preprint",
        arxiv="2302.01318",
        url="",
    ),

    Reference(
        key="cai2024",
        authors=(
            "Cai, T.",
            "Li, Y.",
            "Geng, Z.",
            "Peng, H.",
        ),
        truncated=True,
        year=2024,
        title=(
            "Medusa: Simple LLM inference acceleration framework "
            "with multiple decoding heads"
        ),
        venue="International Conference on Machine Learning",
        arxiv="2401.10774",
        url="",
    ),

    # ---- Quantization (Chapter 16) ----------------------------------------
    Reference(
        key="dettmers2022",
        authors=(
            "Dettmers, T.",
            "Lewis, M.",
            "Belkada, Y.",
            "Zettlemoyer, L.",
        ),
        truncated=False,
        year=2022,
        title="LLM.int8(): 8-bit matrix multiplication for transformers at scale",
        venue="Advances in Neural Information Processing Systems",
        arxiv="2208.07339",
        url="",
    ),

    Reference(
        key="frantar2023",
        authors=(
            "Frantar, E.",
            "Ashkboos, S.",
            "Hoefler, T.",
            "Alistarh, D.",
        ),
        truncated=False,
        year=2023,
        title=(
            "GPTQ: Accurate post-training quantization for generative "
            "pre-trained transformers"
        ),
        venue="International Conference on Learning Representations",
        arxiv="2210.17323",
        url="",
    ),

    Reference(
        key="xiao2023",
        authors=(
            "Xiao, G.",
            "Lin, J.",
            "Seznec, M.",
            "Wu, H.",
            "Demouth, J.",
            "Han, S.",
        ),
        truncated=False,
        year=2023,
        title=(
            "SmoothQuant: Accurate and efficient post-training quantization "
            "for large language models"
        ),
        venue="International Conference on Machine Learning",
        arxiv="2211.10438",
        url="",
    ),

    Reference(
        key="lin2024",
        authors=(
            "Lin, J.",
            "Tang, J.",
            "Tang, H.",
            "Yang, S.",
        ),
        truncated=True,
        year=2024,
        title=(
            "AWQ: Activation-aware weight quantization for LLM compression "
            "and acceleration"
        ),
        venue="Proceedings of Machine Learning and Systems (MLSys)",
        arxiv="2306.00978",
        url="",
    ),

    # ---- Serving systems (Chapter 17) --------------------------------------
    Reference(
        key="yu2022",
        authors=(
            "Yu, G.-I.",
            "Jeong, J. S.",
            "Kim, G.-W.",
            "Kim, S.",
            "Chun, B.-G.",
        ),
        truncated=False,
        year=2022,
        title=(
            "Orca: A distributed serving system for transformer-based "
            "generative models"
        ),
        venue=(
            "USENIX Symposium on Operating Systems Design and "
            "Implementation (OSDI 22)"
        ),
        arxiv="",
        url="https://www.usenix.org/conference/osdi22/presentation/yu",
    ),

    Reference(
        key="zheng2024",
        authors=(
            "Zheng, L.",
            "Yin, L.",
            "Xie, Z.",
            "Sun, C.",
        ),
        truncated=True,
        year=2024,
        title="SGLang: Efficient execution of structured language model programs",
        venue="Advances in Neural Information Processing Systems",
        arxiv="2312.07104",
        url="",
    ),

    Reference(
        key="agrawal2024",
        authors=(
            "Agrawal, A.",
            "Kedia, N.",
            "Panwar, A.",
            "Mohan, J.",
        ),
        truncated=True,
        year=2024,
        title="Taming throughput-latency tradeoff in LLM inference with Sarathi-Serve",
        venue=(
            "USENIX Symposium on Operating Systems Design and "
            "Implementation (OSDI 24)"
        ),
        arxiv="2403.02310",
        url="",
    ),

    Reference(
        key="zhong2024",
        authors=(
            "Zhong, Y.",
            "Liu, S.",
            "Chen, J.",
            "Hu, J.",
        ),
        truncated=True,
        year=2024,
        title=(
            "DistServe: Disaggregating prefill and decoding for "
            "goodput-optimized large language model serving"
        ),
        venue=(
            "USENIX Symposium on Operating Systems Design and "
            "Implementation (OSDI 24)"
        ),
        arxiv="2401.09670",
        url="",
    ),
    Reference(
        key="wei2022cot",
        authors=("Wei, J.", "Wang, X.", "Schuurmans, D.", "Bosma, M."),
        truncated=True,
        year=2022,
        title="Chain-of-thought prompting elicits reasoning in large language models",
        venue="Advances in Neural Information Processing Systems",
        arxiv="2201.11903",
        url="",
    ),

    Reference(
        key="kojima2022",
        authors=(
            "Kojima, T.",
            "Gu, S. S.",
            "Reid, M.",
            "Matsuo, Y.",
            "Iwasawa, Y.",
        ),
        truncated=False,
        year=2022,
        title="Large language models are zero-shot reasoners",
        venue="Advances in Neural Information Processing Systems",
        arxiv="2205.11916",
        url="",
    ),

    Reference(
        key="wangsc2022",
        authors=("Wang, X.", "Wei, J.", "Schuurmans, D.", "Le, Q."),
        truncated=True,
        year=2023,
        title="Self-consistency improves chain of thought reasoning in language models",
        venue="International Conference on Learning Representations",
        arxiv="2203.11171",
        url="",
    ),

    Reference(
        key="greshake2023",
        authors=(
            "Greshake, K.",
            "Abdelnabi, S.",
            "Mishra, S.",
            "Endres, C.",
            "Holz, T.",
            "Fritz, M.",
        ),
        truncated=False,
        year=2023,
        title="Not what you've signed up for: compromising real-world LLM-integrated applications with indirect prompt injection",
        venue="ACM Workshop on Artificial Intelligence and Security (AISec)",
        arxiv="2302.12173",
        url="",
    ),

    Reference(
        key="schick2023",
        authors=(
            "Schick, T.",
            "Dwivedi-Yu, J.",
            "Dessì, R.",
            "Raileanu, R.",
            "Lomeli, M.",
            "Zettlemoyer, L.",
            "Cancedda, N.",
            "Scialom, T.",
        ),
        truncated=False,
        year=2023,
        title="Toolformer: Language Models Can Teach Themselves to Use Tools",
        venue="NeurIPS",
        arxiv="2302.04761",
        url="",
    ),

    Reference(
        key="patil2023",
        authors=(
            "Patil, S. G.",
            "Zhang, T.",
            "Wang, X.",
            "Gonzalez, J. E.",
        ),
        truncated=False,
        year=2023,
        title="Gorilla: Large Language Model Connected with Massive APIs",
        venue="NeurIPS",
        arxiv="2305.15334",
        url="",
    ),

    Reference(
        key="anthropic2024",
        authors=("Anthropic",),
        truncated=False,
        year=2024,
        title="Introducing the Model Context Protocol",
        venue="Anthropic",
        arxiv="",
        url="https://www.anthropic.com/news/model-context-protocol",
    ),

    Reference(
        key="willard2023",
        authors=("Willard, B. T.", "Louf, R."),
        truncated=False,
        year=2023,
        title="Efficient guided generation for large language models",
        venue="arXiv preprint",
        arxiv="2307.09702",
        url="",
    ),

    Reference(
        key="dong2024",
        authors=(
            "Dong, Y.",
            "Ruan, C. F.",
            "Cai, Y.",
            "Lai, R.",
            "Xu, Z.",
            "Zhao, Y.",
            "Chen, T.",
        ),
        truncated=False,
        year=2024,
        title=(
            "XGrammar: Flexible and efficient structured generation engine for "
            "large language models"
        ),
        venue="arXiv preprint",
        arxiv="2411.15100",
        url="",
    ),

    Reference(
        key="tam2024",
        authors=(
            "Tam, Z. R.",
            "Wu, C.-K.",
            "Tsai, Y.-L.",
            "Lin, C.-Y.",
            "Lee, H.-Y.",
            "Chen, Y.-N.",
        ),
        truncated=False,
        year=2024,
        title=(
            "Let me speak freely? A study on the impact of format restrictions on "
            "performance of large language models"
        ),
        venue="Proceedings of EMNLP 2024: Industry Track",
        arxiv="2408.02442",
        url="",
    ),

    Reference(
        key="lewis2020",
        authors=(
            "Lewis, P.",
            "Perez, E.",
            "Piktus, A.",
            "Petroni, F.",
            "Karpukhin, V.",
            "Goyal, N.",
            "Küttler, H.",
            "Lewis, M.",
            "Yih, W.",
            "Rocktäschel, T.",
            "Riedel, S.",
            "Kiela, D.",
        ),
        truncated=False,
        year=2020,
        title="Retrieval-augmented generation for knowledge-intensive NLP tasks",
        venue="Advances in Neural Information Processing Systems",
        arxiv="2005.11401",
        url="",
    ),

    Reference(
        key="karpukhin2020",
        authors=(
            "Karpukhin, V.",
            "Oğuz, B.",
            "Min, S.",
            "Lewis, P.",
            "Wu, L.",
            "Edunov, S.",
            "Chen, D.",
            "Yih, W.",
        ),
        truncated=False,
        year=2020,
        title="Dense passage retrieval for open-domain question answering",
        venue="Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
        arxiv="2004.04906",
        url="",
    ),

    Reference(
        key="guu2020",
        authors=(
            "Guu, K.",
            "Lee, K.",
            "Tung, Z.",
            "Pasupat, P.",
            "Chang, M.-W.",
        ),
        truncated=False,
        year=2020,
        title="REALM: Retrieval-augmented language model pre-training",
        venue="Proceedings of the 37th International Conference on Machine Learning (ICML)",
        arxiv="2002.08909",
        url="",
    ),

    Reference(
        key="robertson2009",
        authors=(
            "Robertson, S.",
            "Zaragoza, H.",
        ),
        truncated=False,
        year=2009,
        title="The probabilistic relevance framework: BM25 and beyond",
        venue="Foundations and Trends in Information Retrieval",
        arxiv="",
        url="https://doi.org/10.1561/1500000019",
    ),

    Reference(
        key="malkov2018",
        authors=(
            "Malkov, Y. A.",
            "Yashunin, D. A.",
        ),
        truncated=False,
        year=2018,
        title="Efficient and robust approximate nearest neighbor search using hierarchical navigable small world graphs",
        venue="IEEE Transactions on Pattern Analysis and Machine Intelligence",
        arxiv="1603.09320",
        url="",
    ),

    Reference(
        key="liu2023lost",
        authors=(
            "Liu, N. F.",
            "Lin, K.",
            "Hewitt, J.",
            "Paranjape, A.",
            "Bevilacqua, M.",
            "Petroni, F.",
            "Liang, P.",
        ),
        truncated=False,
        year=2024,
        title="Lost in the middle: How language models use long contexts",
        venue="Transactions of the Association for Computational Linguistics",
        arxiv="2307.03172",
        url="",
    ),

    Reference(
        key="yao2023",
        authors=(
            "Yao, S.",
            "Zhao, J.",
            "Yu, D.",
            "Du, N.",
            "Shafran, I.",
            "Narasimhan, K.",
            "Cao, Y.",
        ),
        truncated=False,
        year=2023,
        title="ReAct: Synergizing reasoning and acting in language models",
        venue="International Conference on Learning Representations",
        arxiv="2210.03629",
        url="",
    ),

    Reference(
        key="shinn2023",
        authors=(
            "Shinn, N.",
            "Cassano, F.",
            "Berman, E.",
            "Gopinath, A.",
            "Narasimhan, K.",
            "Yao, S.",
        ),
        truncated=False,
        year=2023,
        title="Reflexion: Language agents with verbal reinforcement learning",
        venue="Advances in Neural Information Processing Systems",
        arxiv="2303.11366",
        url="",
    ),

    Reference(
        key="park2023",
        authors=(
            "Park, J. S.",
            "O'Brien, J. C.",
            "Cai, C. J.",
            "Morris, M. R.",
            "Liang, P.",
            "Bernstein, M. S.",
        ),
        truncated=False,
        year=2023,
        title="Generative agents: Interactive simulacra of human behavior",
        venue="ACM Symposium on User Interface Software and Technology",
        arxiv="2304.03442",
        url="",
    ),

    Reference(
        key="anthropic2024agents",
        authors=("Anthropic",),
        truncated=False,
        year=2024,
        title="Building effective agents",
        venue="Anthropic engineering blog",
        arxiv="",
        url="https://www.anthropic.com/engineering/building-effective-agents",
    ),

    Reference(
        key="anthropic2025multiagent",
        authors=("Anthropic",),
        truncated=False,
        year=2025,
        title="How we built our multi-agent research system",
        venue="Anthropic engineering blog",
        arxiv="",
        url="https://www.anthropic.com/engineering/multi-agent-research-system",
    ),

    Reference(
        key="inan2023",
        authors=(
            "Inan, H.",
            "Upasani, K.",
            "Chi, J.",
            "Rungta, R.",
            "Iyer, K.",
            "Mao, Y.",
        ),
        truncated=True,
        year=2023,
        title="Llama Guard: LLM-based input-output safeguard for human-AI conversations",
        venue="arXiv preprint",
        arxiv="2312.06674",
        url="",
    ),

    Reference(
        key="zou2023",
        authors=(
            "Zou, A.",
            "Wang, Z.",
            "Carlini, N.",
            "Nasr, M.",
            "Kolter, J. Z.",
            "Fredrikson, M.",
        ),
        truncated=False,
        year=2023,
        title="Universal and transferable adversarial attacks on aligned language models",
        venue="arXiv preprint",
        arxiv="2307.15043",
        url="",
    ),

    Reference(
        key="wei2023",
        authors=("Wei, A.", "Haghtalab, N.", "Steinhardt, J."),
        truncated=False,
        year=2023,
        title="Jailbroken: How does LLM safety training fail?",
        venue="Advances in Neural Information Processing Systems",
        arxiv="2307.02483",
        url="",
    ),

    Reference(
        key="ganguli2022",
        authors=(
            "Ganguli, D.",
            "Lovitt, L.",
            "Kernion, J.",
            "Askell, A.",
            "Bai, Y.",
            "Kadavath, S.",
        ),
        truncated=True,
        year=2022,
        title="Red teaming language models to reduce harms: Methods, scaling behaviors, and lessons learned",
        venue="arXiv preprint",
        arxiv="2209.07858",
        url="",
    ),

    Reference(
        key="perez2022",
        authors=(
            "Perez, E.",
            "Huang, S.",
            "Song, F.",
            "Cai, T.",
            "Ring, R.",
            "Aslanides, J.",
        ),
        truncated=True,
        year=2022,
        title="Red teaming language models with language models",
        venue="Conference on Empirical Methods in Natural Language Processing",
        arxiv="2202.03286",
        url="",
    ),

    Reference(
        key="anil2024",
        authors=(
            "Anil, C.",
            "Durmus, E.",
            "Panickssery, N.",
            "Sharma, M.",
        ),
        truncated=True,
        year=2024,
        title="Many-shot jailbreaking",
        venue="Advances in Neural Information Processing Systems",
        arxiv="",
        url="https://www.anthropic.com/research/many-shot-jailbreaking",
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
