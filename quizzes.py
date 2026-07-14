"""Single source of truth for the end-of-chapter "Check yourself" quizzes.

Each chapter closes with a short set of challenging, interview-style
multiple-choice questions. `build.py` renders the quiz for a chapter after its
References section, and a small inline script makes it interactive: the reader
picks an option, the choice is marked right or wrong, the correct answer is
revealed, and an explanation appears.

The questions are meant to be *hard* in the way an interview is hard: the
distractors are the plausible misconceptions, and each explanation carries a
second-layer detail the prose only gestures at. Keep them aligned with the
chapter's content so the quiz reinforces the reading.

Question strings are **plain text**. `build.py` HTML-escapes everything, which
would neutralize MathJax delimiters, so do not write `$...$` or `\\(...\\)`
here; phrase math in words or with plain symbols instead.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Question:
    """One multiple-choice question.

    `options` are shown in order; `answer` is the index of the single correct
    option. `explanation` is revealed after the reader answers and should teach,
    not just confirm.
    """

    prompt: str
    options: tuple[str, ...]
    answer: int
    explanation: str

    def __post_init__(self) -> None:
        assert self.prompt.strip(), "Question has an empty prompt."
        assert len(self.options) >= 3, f"{self.prompt!r}: need at least 3 options."
        assert all(o.strip() for o in self.options), f"{self.prompt!r}: empty option."
        assert (
            0 <= self.answer < len(self.options)
        ), f"{self.prompt!r}: answer index {self.answer} is out of range."
        assert self.explanation.strip(), f"{self.prompt!r}: empty explanation."


_QUIZZES: dict[str, tuple[Question, ...]] = {
    "introduction": (
        Question(
            prompt=(
                "A freshly pretrained base model is given the prompt "
                '"What is the capital of France?" and often replies with more '
                "questions rather than an answer. Why?"
            ),
            options=(
                "It has not yet learned the fact and is asking for help.",
                "On its training data, questions are frequently followed by more "
                "questions, so continuing that pattern is the high-probability "
                "completion.",
                "Its temperature is set too high, so it samples randomly.",
                "The tokenizer failed to encode the question mark correctly.",
            ),
            answer=1,
            explanation=(
                "A base model only ever learned to continue text in the style of "
                "its corpus. It very likely knows the fact; it just has not been "
                "taught that a prompt is a request to be answered rather than a "
                "document to be continued. That is exactly what post-training "
                "(SFT, then preference optimization) fixes."
            ),
        ),
        Question(
            prompt=(
                "An interviewer asks whether the 'emergent abilities' of large "
                "models are real. What is the strongest version of the skeptical "
                "view?"
            ),
            options=(
                "Larger models memorize the benchmarks, so the abilities are "
                "contamination artifacts.",
                "Sharp capability jumps can be an artifact of discontinuous "
                "metrics (like exact-match accuracy); on a smooth per-token "
                "metric the same skill often improves gradually.",
                "Emergence violates scaling laws, which predict the loss must "
                "plateau.",
                "The abilities only appear at temperatures near zero, so they are "
                "a decoding artifact.",
            ),
            answer=1,
            explanation=(
                "The mirage argument (Schaeffer et al., 2023) is that a "
                "nonlinear or all-or-nothing metric can manufacture a sudden "
                "jump out of a smoothly falling loss: switch to a continuous "
                "measure and the 'emergence' often smooths out. It does not deny "
                "that capabilities appear with scale, only that the sharp "
                "threshold can be a measurement effect."
            ),
        ),
        Question(
            prompt=(
                "Cross-entropy loss is often reported as perplexity. If a model "
                "reaches a cross-entropy of 3 bits per token, its perplexity is "
                "8. What does that number mean intuitively?"
            ),
            options=(
                "The model is wrong about one token in eight.",
                "The model is, on average, as uncertain as if it were choosing "
                "uniformly among 8 equally likely tokens.",
                "The model needs 8 tokens of context to make a prediction.",
                "The vocabulary has been compressed to 8 tokens.",
            ),
            answer=1,
            explanation=(
                "Perplexity is 2 raised to the cross-entropy in bits (or e to the "
                "nats), i.e. the effective number of equally-likely choices the "
                "model is deciding among per token. Lower perplexity means the "
                "model concentrates probability on fewer candidates. It is a "
                "monotone rescaling of the loss, not new information."
            ),
        ),
        Question(
            prompt=(
                "Why is preference optimization (RLHF or DPO) used at all, when "
                "supervised fine-tuning already shows the model good answers?"
            ),
            options=(
                "It is faster to train than supervised fine-tuning.",
                "It removes the need for any human-labeled data.",
                "People can reliably rank which of two answers is better even "
                "when they cannot write the single best answer, so preferences "
                "are a richer and more scalable signal than fixed demonstrations.",
                "It lets the model update its factual knowledge after " "pretraining.",
            ),
            answer=2,
            explanation=(
                "The core asymmetry: recognition is easier than generation. "
                "Preference methods learn from comparisons among the model's own "
                "outputs, which covers far more of the output space than a fixed "
                "set of gold demonstrations and does not cap quality at what the "
                "annotators could themselves write."
            ),
        ),
        Question(
            prompt=(
                "In this book's framing, the 'harness' around a model is best "
                "described as which of the following?"
            ),
            options=(
                "The distributed system used to train the weights across many " "GPUs.",
                "Everything a provider wraps around the weights at inference "
                "time (system prompt, tools, retrieval, structured output, "
                "guardrails) that decides what context the model sees and what "
                "happens to its output.",
                "The reinforcement-learning loop that aligns the model.",
                "The quantization and serving stack that makes inference cheap.",
            ),
            answer=1,
            explanation=(
                "The harness does not change the weights at all; it shapes the "
                "model's inputs and mediates its outputs. Much of what feels like "
                "'the model just behaves' comes from the harness, which is why "
                "Part V is devoted to it and why most LLM engineering roles in "
                "2026 are really about building harnesses."
            ),
        ),
    ),
    "deep-learning-refresher": (
        Question(
            prompt=(
                "Your model runs a forward pass on a batch just fine, but the "
                "same batch runs out of memory during training. What is the "
                "primary reason, and which lever most directly addresses it?"
            ),
            options=(
                "The optimizer is too slow; switch from AdamW to SGD.",
                "Training also stores activations for backprop plus gradients and "
                "optimizer state (Adam's two moments and often an fp32 master "
                "copy); gradient checkpointing trades compute to shrink the "
                "activation memory.",
                "The learning rate is too high; lower it to reduce memory.",
                "The batch contains longer sequences at train time; truncate " "them.",
            ),
            answer=1,
            explanation=(
                "Inference needs only the weights and a single layer's "
                "activations. Training additionally holds every layer's "
                "activations (for the backward pass), the gradients, and the "
                "optimizer state; for AdamW that is two moments plus commonly an "
                "fp32 master copy, roughly 16 bytes per parameter before "
                "activations. Gradient checkpointing recomputes activations in "
                "the backward pass instead of storing them; sharding (ZeRO, "
                "Chapter 8) and gradient accumulation are the other levers."
            ),
        ),
        Question(
            prompt=(
                "What is the actual difference between L2 regularization folded "
                "into the loss and AdamW's decoupled weight decay?"
            ),
            options=(
                "They are mathematically identical for any optimizer.",
                "With Adam, an L2 penalty enters the gradient and is then divided "
                "by Adam's per-parameter denominator, so high-gradient "
                "parameters get less effective decay; AdamW instead decays the "
                "weights directly, independent of the gradient scale.",
                "AdamW applies weight decay only to the bias terms.",
                "L2 regularization increases the learning rate over time, while "
                "AdamW keeps it fixed.",
            ),
            answer=1,
            explanation=(
                "Adam normalizes each parameter's update by a running estimate "
                "of its gradient magnitude. If weight decay rides in through the "
                "gradient (L2 in the loss), it gets rescaled by that same "
                "denominator and becomes uneven across parameters. AdamW "
                "decouples the two, subtracting a fixed fraction of each weight "
                "directly, which is why it is the default for transformers."
            ),
        ),
        Question(
            prompt=(
                "The same overparameterized network can be trained to 100% "
                "accuracy on data with completely random labels, yet it "
                "generalizes well on real labels. What does this most directly "
                "imply?"
            ),
            options=(
                "The network is not actually overparameterized.",
                "Parameter count alone does not determine generalization; the "
                "structure of the data and what the optimizer finds first matter "
                "more than raw capacity.",
                "Random labels prove the model is broken.",
                "Weight decay is the only reason real-data training generalizes.",
            ),
            answer=1,
            explanation=(
                "Zhang et al. (2017) showed capacity is enough to memorize noise, "
                "so classical bounds tying capacity to generalization cannot be "
                "the whole story. On structured data the optimizer tends to find "
                "simpler, better-generalizing solutions first. In pretraining, a "
                "near-single pass over a huge corpus also removes most of the "
                "opportunity to memorize."
            ),
        ),
        Question(
            prompt=(
                "Why did bfloat16 become the default over float16 for training, "
                "even though float16 has more mantissa bits?"
            ),
            options=(
                "bfloat16 is faster to multiply on all hardware.",
                "bfloat16 keeps float32's full 8-bit exponent, so its dynamic "
                "range matches float32 and gradients do not overflow or "
                "underflow; the lost mantissa precision disappears into gradient "
                "noise the run already has.",
                "float16 cannot represent negative numbers.",
                "bfloat16 uses half the memory of float16.",
            ),
            answer=1,
            explanation=(
                "Training tolerates noise but not clipped range. float16's narrow "
                "5-bit exponent forces fragile loss-scaling to keep gradients "
                "representable; bfloat16 spends its bits on range instead of "
                "precision, so nothing silently flushes to zero or infinity. Both "
                "are 16 bits, so memory is the same."
            ),
        ),
        Question(
            prompt=(
                "Why is Adam(W) preferred over plain SGD for transformers "
                "specifically?"
            ),
            options=(
                "Adam always finds a lower final loss than SGD on any problem.",
                "Gradient magnitudes vary enormously across a transformer's "
                "parameters (rare-token embeddings versus constantly-active norm "
                "gains); Adam's per-parameter normalization equalizes these so a "
                "single global learning rate works.",
                "Adam does not require a learning rate.",
                "SGD cannot be used with mini-batches.",
            ),
            answer=1,
            explanation=(
                "A transformer mixes parameters whose gradients differ by orders "
                "of magnitude and by how often they are exercised. Adam gives "
                "each parameter an effective step size scaled by its own recent "
                "gradient magnitude, so one learning rate serves all of them; "
                "plain SGD would need impractical per-layer tuning."
            ),
        ),
    ),
    "tokenization": (
        Question(
            prompt=(
                "How does SentencePiece make detokenization exactly reversible "
                "without language-specific rules?"
            ),
            options=(
                "It stores the original string alongside the token ids.",
                "It replaces each space with a visible meta-symbol (the underscore "
                "glyph) and treats text as a raw stream, so decoding is just "
                "concatenation with the meta-symbol mapped back to a space.",
                "It only supports languages that separate words with spaces.",
                "It appends a checksum token that encodes the whitespace.",
            ),
            answer=1,
            explanation=(
                "By escaping whitespace into a normal symbol in the vocabulary, "
                "SentencePiece removes the special-case handling of spaces; the "
                "sequence of pieces reconstructs the input by direct "
                "concatenation. This is why a leading space often binds to the "
                "following word as one token."
            ),
        ),
        Question(
            prompt=(
                "WordPiece and BPE both merge subword pairs greedily. What "
                "distinguishes WordPiece's merge criterion?"
            ),
            options=(
                "WordPiece merges the pair with the highest raw frequency, like "
                "BPE.",
                "WordPiece merges the pair that most increases the training "
                "corpus likelihood, which scores a pair roughly by its frequency "
                "divided by the product of its parts' frequencies, favoring "
                "above-chance co-occurrence.",
                "WordPiece never merges across morpheme boundaries.",
                "WordPiece chooses merges at random and prunes later.",
            ),
            answer=1,
            explanation=(
                "BPE picks the most frequent adjacent pair; WordPiece picks the "
                "pair whose merge best improves a unigram language model's "
                "likelihood. That criterion normalizes by how common the parts "
                "already are, so a pair that co-occurs more than chance is "
                "preferred over one that is merely frequent because its parts are."
            ),
        ),
        Question(
            prompt=(
                "You want a tokenizer-level trick that makes a translation model "
                "more robust to how words get split. What is BPE-dropout, and "
                "when does it apply?"
            ),
            options=(
                "It removes rare tokens from the vocabulary to save memory.",
                "During training it randomly skips some merges so a word is seen "
                "under multiple segmentations (a regularizer); at inference the "
                "tokenizer runs deterministically as usual.",
                "It drops the least-frequent 10% of training sentences.",
                "It applies dropout to the embedding of each token.",
            ),
            answer=1,
            explanation=(
                "BPE-dropout (Provilkov et al., 2020), a form of subword "
                "regularization (Kudo, 2018), stochastically omits merges while "
                "training so the model sees several valid tokenizations of the "
                "same string and learns more robust subword representations. "
                "Inference stays deterministic, so throughput is unaffected."
            ),
        ),
        Question(
            prompt=(
                "Why do modern tokenizers often split numbers into fixed-size "
                "digit groups, sometimes grouped from the right?"
            ),
            options=(
                "To make numbers take fewer tokens than words.",
                "So that place value stays aligned across different numbers, "
                "giving arithmetic-like tasks a more uniform representation than "
                "if 1234 were one token and 1235 were two.",
                "Because bytes cannot represent digits directly.",
                "To hide numbers from the model for privacy.",
            ),
            answer=1,
            explanation=(
                "If nearby numbers tokenize into wildly different shapes, the "
                "model has to learn each as an unrelated symbol. Forcing digits "
                "into consistent groups (and grouping from the least-significant "
                "digit) keeps ones, tens, and hundreds aligned across examples, "
                "which measurably helps arithmetic. It is a tokenizer choice "
                "changing a 'reasoning' ability."
            ),
        ),
        Question(
            prompt=(
                "A model reliably miscounts the letters in 'strawberry'. What is "
                "the real cause?"
            ),
            options=(
                "The model has too few parameters to count.",
                "Tokenization hands the model opaque multi-character chunks, so "
                "the individual letters are not part of its input unless it has "
                "memorized spelling facts about its own tokens.",
                "The attention mechanism cannot compare positions.",
                "The context window is too short to hold the word.",
            ),
            answer=1,
            explanation=(
                "The evidence a letter-counting task needs is destroyed before "
                "the model sees it: 'strawberry' might arrive as two or three "
                "tokens with no exposed characters. Success is inconsistent "
                "across words precisely because it depends on memorized "
                "token-spelling facts, not on any counting ability."
            ),
        ),
    ),
    "transformer": (
        Question(
            prompt=(
                "You need high-quality sentence embeddings for retrieval and "
                "classification over inputs that are always fully available. Why "
                "might a BERT-style encoder beat a decoder-only LLM here?"
            ),
            options=(
                "Encoders have more parameters than decoders.",
                "An encoder attends bidirectionally (each token sees the whole "
                "input) and is trained with masked-language-modeling, which suits "
                "understanding a complete input; a decoder is causal, built for "
                "left-to-right generation.",
                "Decoders cannot produce embeddings at all.",
                "Encoders are always faster at inference.",
            ),
            answer=1,
            explanation=(
                "The families differ by attention and objective: encoders "
                "(BERT-style, bidirectional MLM) see both directions and excel at "
                "encoding a fixed input; decoders (GPT-style, causal) are made for "
                "generation; encoder-decoder models (T5-style) split the two for "
                "sequence-to-sequence. Matching the architecture to 'understand a "
                "whole input' versus 'generate a continuation' is the point."
            ),
        ),
        Question(
            prompt=(
                "Why is dividing the attention scores by the square root of the "
                "head dimension important?"
            ),
            options=(
                "It normalizes the values so they sum to one.",
                "For large head dimension the dot products have variance that "
                "grows with the dimension, pushing softmax into saturated "
                "regions with vanishing gradients; the scaling keeps the score "
                "variance near one.",
                "It converts the scores from logits into probabilities.",
                "It compensates for the causal mask removing half the entries.",
            ),
            answer=1,
            explanation=(
                "Query-key dot products sum over the head dimension, so their "
                "variance grows with it. Without the square-root scaling, large "
                "scores saturate the softmax, gradients vanish, and training "
                "destabilizes. It is a classic source of instability when omitted."
            ),
        ),
        Question(
            prompt=(
                "The causal mask lets a transformer train on every position of a "
                "sequence at once. What is the name for conditioning each "
                "prediction on the ground-truth previous tokens, and what "
                "asymmetry does it create?"
            ),
            options=(
                "Beam search; training and generation both explore multiple "
                "hypotheses.",
                "Teacher forcing; training conditions on the true previous tokens "
                "(all positions in parallel), while generation must condition on "
                "the model's own sampled tokens, one step at a time.",
                "Label smoothing; it softens the targets during training only.",
                "Curriculum learning; easy tokens are shown before hard ones.",
            ),
            answer=1,
            explanation=(
                "Because the mask guarantees each position only sees earlier "
                "ones, every position can be trained simultaneously against the "
                "real next token (teacher forcing). At generation time there is "
                "no ground truth, so the model feeds on its own outputs "
                "sequentially, the source of the train/generate asymmetry (and of "
                "exposure bias)."
            ),
        ),
        Question(
            prompt=(
                "Roughly where do a transformer's parameters live, and why does "
                "it matter for mixture-of-experts models?"
            ),
            options=(
                "Mostly in the attention projections; MoE therefore replaces the "
                "attention heads.",
                "Roughly two-thirds in the feed-forward networks and one-third in "
                "the attention projections; MoE adds capacity cheaply by swapping "
                "the single FFN for many experts and activating only a few per "
                "token.",
                "Evenly split across embeddings, attention, and FFNs; MoE "
                "duplicates the embeddings.",
                "Mostly in the token embeddings; MoE shrinks the vocabulary.",
            ),
            answer=1,
            explanation=(
                "For typical shapes the FFNs hold about two-thirds of the "
                "parameters and, many argue, most of the stored knowledge. That "
                "is exactly why MoE targets the FFN: replace one FFN with many "
                "experts, route each token to a couple, and total capacity grows "
                "while per-token compute stays roughly fixed."
            ),
        ),
        Question(
            prompt=(
                "A candidate claims the O(n^2) cost of attention means "
                "long-context models are fundamentally limited by compute. What "
                "is the more precise picture?"
            ),
            options=(
                "The claim is exactly right; nothing can reduce the quadratic " "cost.",
                "The quadratic term is real, but in practice attention is often "
                "bottlenecked by memory movement, not FLOPs; IO-aware methods "
                "like FlashAttention compute the exact result tile-by-tile "
                "without materializing the n-by-n matrix (details in Chapter 15).",
                "Attention is actually linear in sequence length once cached.",
                "The quadratic cost only applies during training, never at "
                "inference.",
            ),
            answer=1,
            explanation=(
                "The n-by-n score matrix drives the quadratic compute, but the "
                "practical wall is often reading and writing that matrix to "
                "memory. FlashAttention reorders the computation to keep tiles in "
                "fast on-chip memory and never stores the full matrix, cutting "
                "memory traffic without changing the math. The serving-side "
                "consequences are Chapter 15's subject."
            ),
        ),
    ),
    "modern-architectures": (
        Question(
            prompt=(
                "Rotary position embeddings (RoPE) encode position by rotating "
                "query and key vectors. Why does this give the model relative "
                "position for free?"
            ),
            options=(
                "Because the rotation angle is the same for every position.",
                "Because the dot product between a query at position m and a key "
                "at position n ends up depending only on the offset m minus n, "
                "not on the absolute positions.",
                "Because RoPE adds a learned position vector to each embedding.",
                "Because rotations make all attention scores positive.",
            ),
            answer=1,
            explanation=(
                "RoPE rotates each 2D sub-pair of the query and key by an angle "
                "proportional to position. In the score, the two rotations "
                "combine so only the difference of angles, i.e. the relative "
                "offset, survives. Grammar cares about how far apart tokens are, "
                "which is what this encodes."
            ),
        ),
        Question(
            prompt=(
                "A model trained with an 8K context must serve 32K prompts. Which "
                "family of techniques directly addresses this, and what is the "
                "underlying idea?"
            ),
            options=(
                "Increasing the number of attention heads so more positions fit.",
                "RoPE context-extension methods (linear position interpolation, "
                "NTK-aware scaling, YaRN) that remap the rotary frequencies so "
                "positions beyond the training length fall within a range the "
                "model has seen.",
                "Switching from RoPE to learned absolute positions at serving " "time.",
                "Lowering the temperature so the model attends to fewer tokens.",
            ),
            answer=1,
            explanation=(
                "Because RoPE stores position in rotation frequencies, you can "
                "stretch or reinterpret those frequencies to cover longer "
                "contexts: linear interpolation squeezes positions into the "
                "trained range, and NTK-aware and YaRN (Peng et al., 2023) scale "
                "frequencies more carefully to preserve short-range resolution. "
                "The serving-cost side is Chapter 15."
            ),
        ),
        Question(
            prompt=(
                "Grouped-query attention shrinks the KV cache by sharing KV heads "
                "across groups of query heads. Name another distinct architectural "
                "lever for the same goal."
            ),
            options=(
                "Increasing the number of layers.",
                "Sliding-window (local) attention, where each token attends only "
                "to the last w tokens so the cache and per-step cost stay bounded "
                "at long context, often interleaved with a few global layers.",
                "Raising the RoPE base frequency.",
                "Using SwiGLU instead of a ReLU feed-forward network.",
            ),
            answer=1,
            explanation=(
                "GQA cuts the cache by the sharing factor; sliding-window "
                "attention (as in Mistral, Jiang et al., 2023) bounds it by "
                "limiting how far back each token looks. Multi-head latent "
                "attention, which compresses KV into a low-rank latent, is a "
                "third lever. They are complementary ways to fight KV-cache "
                "growth, the dominant long-context memory cost."
            ),
        ),
        Question(
            prompt=(
                "In a mixture-of-experts layer, what is the role of the "
                "load-balancing (auxiliary) loss, and what problem does it "
                "prevent?"
            ),
            options=(
                "It penalizes wrong predictions harder for rare experts.",
                "It pushes the router to spread tokens across experts, preventing "
                "a collapse where a few experts get most of the traffic (and are "
                "overloaded past their capacity) while others go unused.",
                "It forces every token to use all experts for stability.",
                "It reduces the total number of parameters held in memory.",
            ),
            answer=1,
            explanation=(
                "Left alone, routers tend to collapse onto a few favored experts, "
                "which then exceed their capacity (dropping tokens) while the rest "
                "sit idle, wasting capacity. An auxiliary balancing loss "
                "encourages even routing; newer models (e.g. DeepSeek-V3) pursue "
                "auxiliary-loss-free balancing via learned routing biases to "
                "avoid the aux-loss's tug on quality."
            ),
        ),
        Question(
            prompt=(
                "A config shows num_attention_heads: 64 and num_key_value_heads: "
                "8. What is the interviewer looking for?"
            ),
            options=(
                "The model uses 8 layers of attention.",
                "This is grouped-query attention with 8 groups, so the KV cache "
                "is 8 times smaller than full multi-head attention would need, "
                "which matters because decode is memory-bandwidth-bound.",
                "The model has 8 experts in a mixture-of-experts layer.",
                "The head dimension is 64 divided by 8.",
            ),
            answer=1,
            explanation=(
                "Sixty-four query heads sharing eight KV heads means groups of "
                "eight, an 8x smaller KV cache. The 'why it matters' is the "
                "payoff: decode streams the cache from memory every step, so a "
                "smaller cache buys longer context and larger batches at a small "
                "quality cost."
            ),
        ),
    ),
}


def _validate(
    quizzes: dict[str, tuple[Question, ...]],
) -> dict[str, tuple[Question, ...]]:
    """Assert every chapter's quiz has a sensible number of questions."""
    for slug, questions in quizzes.items():
        assert (
            4 <= len(questions) <= 6
        ), f"Chapter '{slug}' must have 4-6 questions, has {len(questions)}."
    return quizzes


QUIZZES: dict[str, tuple[Question, ...]] = _validate(_QUIZZES)
