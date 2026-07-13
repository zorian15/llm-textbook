"""Single source of truth for the book's structure.

`build.py` imports `BOOK` from this module and generates one HTML file per
entry, plus the landing page. To reorder, rename, or add a chapter, edit this
file and rerun the build. If a chapter has a matching `content/<slug>.md`, that
prose is rendered; otherwise a stub page is synthesized from the `outline`
declared here, so the whole book is always navigable even before it is written.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Chapter:
    """One page of the book.

    `slug` is the file stem (used for `content/<slug>.md` and `<slug>.html`).
    `label` is the display number shown in navigation and section numbering:
    a digit for chapters, a letter for appendices, or an empty string for
    unnumbered front matter. `outline` is a list of (section_title, note)
    pairs describing what the chapter should eventually cover; it is only used
    to synthesize the stub when no drafted markdown exists yet.
    """

    slug: str
    label: str
    title: str
    outline: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class Part:
    """A titled group of chapters, rendered as a heading in the sidebar."""

    title: str
    chapters: tuple[Chapter, ...]


# Front matter sits outside any part and is unnumbered.
PREFACE = Chapter(
    slug="preface",
    label="",
    title="Preface",
    outline=(),
)


BOOK: tuple[Part, ...] = (
    Part(
        title="I · Foundations",
        chapters=(
            Chapter(
                slug="introduction",
                label="1",
                title="What Is a Large Language Model?",
                outline=(),
            ),
            Chapter(
                slug="deep-learning-refresher",
                label="2",
                title="Deep Learning, Just Enough",
                outline=(
                    (
                        "From features to representations",
                        "Why learned representations beat hand-engineered features; the manifold view.",
                    ),
                    (
                        "The training loop",
                        "Forward pass, loss, backprop, gradient step; treat autograd as bookkeeping for the chain rule.",
                    ),
                    (
                        "Optimizers and learning rates",
                        "SGD to Adam/AdamW; momentum and adaptive scaling as the two knobs that matter.",
                    ),
                    (
                        "Regularization and generalization",
                        "Weight decay, dropout, and why huge overparameterized models still generalize.",
                    ),
                    (
                        "Numerical precision",
                        "float32 vs bf16 vs fp16; dynamic range vs precision; why bf16 won for training.",
                    ),
                ),
            ),
            Chapter(
                slug="tokenization",
                label="3",
                title="Tokenization",
                outline=(
                    (
                        "Why not characters or words",
                        "The vocabulary-size vs sequence-length tradeoff; open-vocabulary problem.",
                    ),
                    (
                        "Byte-pair encoding",
                        "BPE as greedy merging of frequent pairs; a worked toy example on a tiny corpus.",
                    ),
                    (
                        "Byte-level BPE and friends",
                        "Byte-level fallback, WordPiece, Unigram, SentencePiece; what production tokenizers use.",
                    ),
                    (
                        "Consequences of tokenization",
                        "Number handling, non-English penalties, prompt-injection surface, cost per token.",
                    ),
                ),
            ),
            Chapter(
                slug="transformer",
                label="4",
                title="The Transformer",
                outline=(),
            ),
            Chapter(
                slug="modern-architectures",
                label="5",
                title="How Modern Architectures Differ",
                outline=(
                    (
                        "Positional information: RoPE",
                        "From learned/absolute positions to rotary embeddings; why rotation encodes relative position.",
                    ),
                    (
                        "Normalization: RMSNorm and pre-norm",
                        "Pre-norm vs post-norm stability; why RMSNorm dropped the mean-centering.",
                    ),
                    (
                        "Activations: SwiGLU",
                        "Gated linear units; the width bookkeeping that keeps parameter count fair.",
                    ),
                    (
                        "Attention variants: MQA and GQA",
                        "Sharing key/value heads to shrink the KV cache; the quality-vs-memory tradeoff.",
                    ),
                    (
                        "Mixture of experts",
                        "Sparse routing; total vs active parameters; the intuition that capacity is cheap but compute is not.",
                    ),
                    (
                        "A tour of a current open model",
                        "Read one real config (e.g. a Llama- or Qwen-class model) end to end.",
                    ),
                ),
            ),
        ),
    ),
    Part(
        title="II · Pretraining",
        chapters=(
            Chapter(
                slug="pretraining-objective",
                label="6",
                title="The Pretraining Objective and the Data",
                outline=(
                    (
                        "Next-token prediction as compression",
                        "Cross-entropy as bits-per-token; the 'prediction is understanding' intuition and its limits.",
                    ),
                    (
                        "Where the data comes from",
                        "Web crawl, code, books, curated sources; scale in tokens, not gigabytes.",
                    ),
                    (
                        "Cleaning and deduplication",
                        "Quality filtering, dedup, decontamination against evals; garbage-in effects.",
                    ),
                    (
                        "Data mixtures and curriculum",
                        "Domain weighting, upsampling code/math, later-stage annealing on high-quality data.",
                    ),
                ),
            ),
            Chapter(
                slug="training-dynamics",
                label="7",
                title="Training Dynamics at Scale",
                outline=(
                    (
                        "Batch size, learning rate, and warmup",
                        "The warmup-then-decay schedule; linear scaling rule and where it breaks.",
                    ),
                    (
                        "Loss curves and what they tell you",
                        "Reading a loss curve; spikes, divergence, and the usual culprits.",
                    ),
                    (
                        "Stability tricks",
                        "Gradient clipping, z-loss, careful init, QK-norm; keeping a long run alive.",
                    ),
                    (
                        "Checkpointing and reproducibility",
                        "Determinism, resumption, and why exact reproducibility is hard on GPUs.",
                    ),
                ),
            ),
            Chapter(
                slug="distributed-training",
                label="8",
                title="Distributed Training",
                outline=(
                    (
                        "Why one GPU is never enough",
                        "The memory budget: parameters, gradients, optimizer states, activations.",
                    ),
                    (
                        "Data parallelism and ZeRO/FSDP",
                        "Replicating vs sharding state across ranks; the ZeRO stages as a memory ladder.",
                    ),
                    (
                        "Tensor and pipeline parallelism",
                        "Splitting a layer vs splitting the stack; bubbles and communication cost.",
                    ),
                    (
                        "Putting it together: 3D parallelism",
                        "How the parallelisms compose; a back-of-envelope for a real cluster.",
                    ),
                    (
                        "The communication primitives",
                        "All-reduce, all-gather, reduce-scatter; why interconnect bandwidth is destiny.",
                    ),
                ),
            ),
            Chapter(
                slug="scaling-laws",
                label="9",
                title="Scaling Laws",
                outline=(
                    (
                        "The empirical power laws",
                        "Loss as a power law in parameters, data, and compute; the Kaplan picture.",
                    ),
                    (
                        "Compute-optimal training: Chinchilla",
                        "The tokens-per-parameter rule of thumb and why earlier models were undertrained.",
                    ),
                    (
                        "Inference-aware scaling",
                        "Why serving cost pushes you to smaller models trained on more tokens.",
                    ),
                    (
                        "What scaling laws do and don't predict",
                        "Smooth loss vs emergent capabilities; the ongoing debate.",
                    ),
                ),
            ),
        ),
    ),
    Part(
        title="III · Post-training and Alignment",
        chapters=(
            Chapter(
                slug="sft",
                label="10",
                title="Supervised Fine-Tuning",
                outline=(
                    (
                        "From a document completer to an assistant",
                        "Why the base model is not yet an assistant; the persona/format gap.",
                    ),
                    (
                        "Instruction data and chat templates",
                        "The structure of SFT data; special tokens and role formatting; loss masking on prompts.",
                    ),
                    (
                        "How much data, and of what quality",
                        "The 'less but better' finding; the risk of teaching style over substance.",
                    ),
                    (
                        "Failure modes",
                        "Catastrophic forgetting, sycophancy seeds, format overfitting.",
                    ),
                ),
            ),
            Chapter(
                slug="rlhf",
                label="11",
                title="RLHF and Reward Modeling",
                outline=(
                    (
                        "Why supervised learning is not enough",
                        "You can rank outputs you can't write; the demonstration-vs-preference distinction.",
                    ),
                    (
                        "Reward models from human preferences",
                        "The Bradley-Terry model; turning pairwise comparisons into a scalar reward.",
                    ),
                    (
                        "Policy optimization with PPO",
                        "The RL loop; the KL penalty as a leash keeping the policy near the reference.",
                    ),
                    (
                        "Reward hacking and over-optimization",
                        "Goodhart's law in practice; why more reward can mean worse behavior.",
                    ),
                ),
            ),
            Chapter(
                slug="preference-optimization",
                label="12",
                title="Preference Optimization Without RL",
                outline=(
                    (
                        "DPO: skipping the reward model",
                        "The key identity that turns preferences into a direct classification loss.",
                    ),
                    (
                        "The DPO family",
                        "IPO, KTO, ORPO, SimPO; what each changes and why.",
                    ),
                    (
                        "RLHF vs DPO in practice",
                        "Stability, cost, and quality tradeoffs; when teams pick which.",
                    ),
                    (
                        "Constitutional AI and RLAIF",
                        "Using models to generate the preference signal; self-critique and revision.",
                    ),
                ),
            ),
            Chapter(
                slug="peft",
                label="13",
                title="Parameter-Efficient Fine-Tuning",
                outline=(
                    (
                        "Why full fine-tuning hurts",
                        "Memory cost and the one-copy-per-task problem.",
                    ),
                    (
                        "LoRA",
                        "Low-rank update matrices; the intuition that adaptation lives in a small subspace.",
                    ),
                    (
                        "QLoRA and quantized adapters",
                        "4-bit base weights plus adapters; how a big model fine-tunes on one GPU.",
                    ),
                    (
                        "Choosing rank, alpha, and targets",
                        "Practical knobs; where adapters help and where they plateau.",
                    ),
                    (
                        "Serving many adapters",
                        "Hot-swapping LoRAs at inference; the multi-tenant story.",
                    ),
                ),
            ),
        ),
    ),
    Part(
        title="IV · Inference and Serving",
        chapters=(
            Chapter(
                slug="decoding",
                label="14",
                title="Decoding and Sampling",
                outline=(
                    (
                        "Greedy, beam, and why beam lost",
                        "Search vs sampling for open-ended generation; the degeneration problem.",
                    ),
                    (
                        "Temperature, top-k, top-p",
                        "Reshaping the distribution; intuitive pictures of each knob.",
                    ),
                    (
                        "Newer samplers",
                        "Min-p, typical, repetition penalties; when defaults are wrong.",
                    ),
                    (
                        "Structured and constrained decoding preview",
                        "Pointer forward to grammars and JSON-mode in the harness part.",
                    ),
                ),
            ),
            Chapter(
                slug="inference-optimization",
                label="15",
                title="Making Inference Fast",
                outline=(
                    (
                        "Prefill vs decode",
                        "The two phases; compute-bound vs memory-bandwidth-bound; why tokens/sec is dominated by decode.",
                    ),
                    (
                        "The KV cache",
                        "What it stores and why it grows; the memory math per token.",
                    ),
                    (
                        "Batching and PagedAttention",
                        "Continuous batching and paged KV memory; the vLLM idea in one picture.",
                    ),
                    (
                        "Speculative decoding",
                        "A small draft model proposing tokens a big model verifies; the free-lunch intuition.",
                    ),
                    (
                        "FlashAttention",
                        "IO-aware attention; recomputation to stay in fast memory.",
                    ),
                ),
            ),
            Chapter(
                slug="quantization",
                label="16",
                title="Quantization",
                outline=(
                    (
                        "Bits, ranges, and where the memory goes",
                        "Why weights dominate memory; the appeal of fewer bits per weight.",
                    ),
                    (
                        "Post-training quantization",
                        "GPTQ, AWQ, and the outlier-feature problem; calibration data.",
                    ),
                    (
                        "Formats you'll meet: GGUF, bitsandbytes, MLX",
                        "What each is for; K-quants and mixed-precision schemes.",
                    ),
                    (
                        "Quantization-aware training and the frontier",
                        "QAT, 4-bit and below, and the accuracy cliff.",
                    ),
                ),
            ),
            Chapter(
                slug="serving-systems",
                label="17",
                title="Serving Systems in Production",
                outline=(
                    (
                        "What a serving stack must do",
                        "Throughput, latency (TTFT and TPOT), and cost as competing goals.",
                    ),
                    (
                        "The engines",
                        "vLLM, TensorRT-LLM, SGLang, TGI; what differentiates them.",
                    ),
                    (
                        "Multi-request scheduling",
                        "Prefix caching, prioritization, and fairness across users.",
                    ),
                    (
                        "Autoscaling and the economics",
                        "GPU utilization, cold starts, and the unit economics of tokens.",
                    ),
                ),
            ),
        ),
    ),
    Part(
        title="V · The Harness: Making Models Behave",
        chapters=(
            Chapter(
                slug="prompting",
                label="18",
                title="Prompting and System Prompts",
                outline=(
                    (
                        "The system prompt as configuration",
                        "How a provider sets persona, rules, and defaults; the layering of system/developer/user roles.",
                    ),
                    (
                        "In-context learning",
                        "Zero-, few-shot, and why examples work without weight updates.",
                    ),
                    (
                        "Chain-of-thought and its descendants",
                        "Eliciting reasoning; when it helps and when it is theater.",
                    ),
                    (
                        "Prompt injection and the trust boundary",
                        "Why everything from tools and the web is untrusted data, not instructions.",
                    ),
                ),
            ),
            Chapter(
                slug="tool-use",
                label="19",
                title="Tool Use and Function Calling",
                outline=(
                    (
                        "Turning language into actions",
                        "The request/tool-call/result loop; the model proposes, the harness executes.",
                    ),
                    (
                        "How tool calling is trained and formatted",
                        "Schemas, JSON arguments, and the round-trip protocol.",
                    ),
                    (
                        "MCP and standardized tool interfaces",
                        "Why a protocol for tools emerged; servers, clients, and capabilities.",
                    ),
                    (
                        "Reliability and safety of actions",
                        "Confirmation, permissions, and side effects; the human-in-the-loop line.",
                    ),
                ),
            ),
            Chapter(
                slug="structured-output",
                label="20",
                title="Structured Output",
                outline=(
                    (
                        "Why free text is not an API",
                        "The parsing problem; downstream systems need schemas.",
                    ),
                    (
                        "Constrained decoding",
                        "Masking logits to follow a grammar; guaranteed-valid JSON.",
                    ),
                    (
                        "Grammars and schemas",
                        "JSON Schema, regex, and context-free grammars as decoding constraints.",
                    ),
                    (
                        "Costs and pitfalls",
                        "How constraints can hurt quality; the format-vs-content tension.",
                    ),
                ),
            ),
            Chapter(
                slug="rag",
                label="21",
                title="Retrieval-Augmented Generation",
                outline=(
                    (
                        "Why retrieval at all",
                        "Knowledge cutoff, private data, and grounding to reduce hallucination.",
                    ),
                    (
                        "Embeddings and vector search",
                        "Dense retrieval; cosine similarity; the ANN index in one picture.",
                    ),
                    (
                        "Chunking, reranking, and hybrid search",
                        "The unglamorous engineering that makes or breaks RAG; BM25 + dense.",
                    ),
                    (
                        "Evaluating and failing gracefully",
                        "Faithfulness vs relevance; what to do when retrieval returns nothing good.",
                    ),
                    (
                        "Long context vs retrieval",
                        "When a big context window replaces RAG, and when it doesn't.",
                    ),
                ),
            ),
            Chapter(
                slug="agents",
                label="22",
                title="Agents",
                outline=(
                    (
                        "What 'agent' actually means",
                        "A loop of plan-act-observe; autonomy as a spectrum, not a binary.",
                    ),
                    (
                        "Planning and memory",
                        "Decomposition, scratchpads, and external memory; ReAct-style loops.",
                    ),
                    (
                        "Multi-agent systems",
                        "Orchestrator/worker patterns; when more agents help and when they just add noise.",
                    ),
                    (
                        "Why agents are hard",
                        "Error compounding over long horizons; evaluation and cost.",
                    ),
                ),
            ),
            Chapter(
                slug="safety-guardrails",
                label="23",
                title="Safety, Guardrails, and Moderation",
                outline=(
                    (
                        "Layers of defense",
                        "Alignment training, system prompts, input/output classifiers, and monitoring.",
                    ),
                    (
                        "Jailbreaks and red-teaming",
                        "The adversarial cat-and-mouse; common attack shapes.",
                    ),
                    (
                        "Content moderation classifiers",
                        "Separate models guarding the main one; precision/recall tradeoffs.",
                    ),
                    (
                        "Governance and the harness's responsibility",
                        "Where provider policy lives; refusal behavior and its costs.",
                    ),
                ),
            ),
        ),
    ),
    Part(
        title="VI · Evaluation",
        chapters=(
            Chapter(
                slug="evaluation",
                label="24",
                title="Evaluating Language Models",
                outline=(
                    (
                        "Why evaluation is the hard part",
                        "Open-ended output has no single ground truth; the metric problem.",
                    ),
                    (
                        "Benchmarks and their discontents",
                        "MMLU-style suites, contamination, saturation, and gaming.",
                    ),
                    (
                        "LLM-as-judge",
                        "Using a model to grade a model; bias, calibration, and when to trust it.",
                    ),
                    (
                        "Human eval and arenas",
                        "Pairwise human preference, Elo-style ranking, and their limits.",
                    ),
                    (
                        "Building evals for your own product",
                        "Task-specific eval sets; the flywheel of logging, labeling, and regression tests.",
                    ),
                ),
            ),
        ),
    ),
    Part(
        title="VII · The Frontier",
        chapters=(
            Chapter(
                slug="reasoning",
                label="25",
                title="Reasoning and Test-Time Compute",
                outline=(
                    (
                        "Spending compute at inference",
                        "The shift from bigger models to longer thinking; the o1/R1 turn.",
                    ),
                    (
                        "RL on verifiable rewards",
                        "Training reasoning with automatically checkable answers (math, code).",
                    ),
                    (
                        "Search, self-consistency, and verification",
                        "Sampling many chains and selecting; process vs outcome reward.",
                    ),
                    (
                        "What this changes",
                        "New scaling axis, new costs, and the blurring of train/inference.",
                    ),
                ),
            ),
            Chapter(
                slug="open-problems",
                label="26",
                title="Open Problems (Conclusion)",
                outline=(
                    (
                        "Hallucination and calibration",
                        "Why models are confidently wrong; what partial fixes exist.",
                    ),
                    (
                        "Interpretability",
                        "Features, circuits, and the dream of reading a model's mind.",
                    ),
                    (
                        "Alignment at and beyond human level",
                        "Scalable oversight, deception, and evaluation we can trust.",
                    ),
                    (
                        "Data, efficiency, and the wall question",
                        "Running out of tokens; synthetic data; whether scaling continues.",
                    ),
                    (
                        "Where the field might go",
                        "A sober take on the next few years and what to keep learning.",
                    ),
                ),
            ),
        ),
    ),
)


APPENDICES: tuple[Chapter, ...] = (
    Chapter(
        slug="local-llms",
        label="A",
        title="Running LLMs Locally",
        outline=(),
    ),
    Chapter(
        slug="math-reference",
        label="B",
        title="Math and Notation Reference",
        outline=(
            (
                "Linear algebra you actually need",
                "Matrix multiply as the workhorse; shapes as a type system.",
            ),
            (
                "Probability and information",
                "Softmax, cross-entropy, KL divergence, perplexity in one place.",
            ),
            (
                "Notation conventions used in this book",
                "Symbols, subscripts, and shape conventions collected for reference.",
            ),
        ),
    ),
    Chapter(
        slug="glossary",
        label="C",
        title="Glossary",
        outline=(
            ("A–M", "Alphabetical terms, each defined in one or two sentences."),
            ("N–Z", "Alphabetical terms, each defined in one or two sentences."),
        ),
    ),
)


def all_pages() -> tuple[Chapter, ...]:
    """Return every page in reading order: preface, chapters, then appendices."""
    pages: list[Chapter] = [PREFACE]
    for part in BOOK:
        pages.extend(part.chapters)
    pages.extend(APPENDICES)
    slugs = [page.slug for page in pages]
    assert len(slugs) == len(
        set(slugs)
    ), "Duplicate slug detected in the table of contents."
    return tuple(pages)
