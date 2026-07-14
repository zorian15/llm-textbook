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
    "pretraining-objective": (
        Question(
            prompt=(
                "Model A (32k-token vocabulary) reports perplexity 9; model B "
                "(128k-token vocabulary) reports perplexity 12 on the same "
                "text. What can you conclude about which predicts the text "
                "better?"
            ),
            options=(
                "Model A is better, since 9 is lower than 12.",
                "Nothing yet: perplexity is per token, and B's larger "
                "vocabulary packs more text into each token, raising "
                "per-token uncertainty; renormalize both to bits per byte of "
                "raw text before comparing.",
                "Model B is better, because larger vocabularies always mean "
                "stronger models.",
                "Nothing ever, because perplexity is not related to "
                "prediction quality.",
            ),
            answer=1,
            explanation=(
                "A tokenizer choice changes the unit of measurement. Each of "
                "B's tokens carries more text, so its per-token perplexity is "
                "naturally higher even if it compresses the text better "
                "overall. Converting both models' total cross-entropy to bits "
                "per byte of the same underlying text removes the tokenizer "
                "from the comparison - which is exactly the compression view "
                "of the loss."
            ),
        ),
        Question(
            prompt=(
                "Beyond wasting compute on repeats, why does deduplication "
                "measurably improve a pretrained model?"
            ),
            options=(
                "It shrinks the vocabulary the tokenizer needs.",
                "Duplicated documents act as an accidental upweighting of "
                "whatever happens to be mirrored, distorting the intended "
                "mixture, and repeated passages are far more likely to be "
                "memorized and regurgitated verbatim.",
                "It guarantees the model can no longer memorize any training " "text.",
                "It makes the loss curve smoother by removing hard examples.",
            ),
            answer=1,
            explanation=(
                "Lee et al. (2022) showed near-duplicate text both skews the "
                "effective data distribution and drives verbatim memorization, "
                "which is a privacy and quality problem. Dedup reduces "
                "memorization sharply but does not eliminate it - unique text "
                "seen once can still be memorized, especially by large models "
                "late in training."
            ),
        ),
        Question(
            prompt=(
                "Why do pretraining teams shift the data mixture toward their "
                "highest-quality sources specifically during the final "
                "learning-rate decay, rather than spreading that data evenly?"
            ),
            options=(
                "Because the data loader can only handle one mixture change "
                "per run.",
                "Because early training would overfit the high-quality data.",
                "Because the last updates, taken at the smallest learning "
                "rates, are the ones least likely to be overwritten by later "
                "training - the model's final polish comes from whatever it "
                "sees there, so the scarcest, best tokens are spent there.",
                "Because high-quality data requires a lower learning rate to "
                "parse correctly.",
            ),
            answer=2,
            explanation=(
                "Annealing pairs the mixture shift with the learning-rate "
                "decay deliberately: nothing comes after the anneal to disturb "
                "what it teaches. Llama 3 and MiniCPM both report this "
                "recipe. It is also why a mid-training checkpoint understates "
                "final model quality - the anneal's gains have not happened "
                "yet."
            ),
        ),
        Question(
            prompt=(
                "Your corpus is short of fresh tokens for the planned run. "
                "What does the evidence on repeating data say?"
            ),
            options=(
                "Any repetition causes overfitting and must be avoided.",
                "Repetition is free: forty epochs behave like fresh data.",
                "Up to roughly four epochs over a source costs little "
                "compared to fresh data; after that, returns from further "
                "repeats decay rapidly toward zero.",
                "Repetition helps only for code, never for prose.",
            ),
            answer=2,
            explanation=(
                "Muennighoff et al. (2023) fit scaling laws in the "
                "data-constrained regime: a few epochs are nearly as good as "
                "unique tokens, then value decays sharply. This is why small "
                "high-quality sources are routinely upsampled several times "
                "per web epoch, and why the 'running out of data' question is "
                "about fresh-token supply, not literal exhaustion."
            ),
        ),
        Question(
            prompt=(
                "What is the precise sense in which a language model 'is' a "
                "compressor?"
            ),
            options=(
                "Its weights are smaller than its training data, so training "
                "compressed the corpus into the weights.",
                "Its next-token distribution can drive an arithmetic coder "
                "that spends exactly minus log2 p(x) bits on each actual "
                "token x, so lower cross-entropy is literally a shorter "
                "encoding of the corpus.",
                "It removes stopwords and redundancy from text it generates.",
                "It stores a zip archive of frequent phrases in its KV cache.",
            ),
            answer=1,
            explanation=(
                "The equivalence is mechanical, not metaphorical: any "
                "predictor plus arithmetic coding is a lossless compressor, "
                "and the achieved file size is the model's cross-entropy on "
                "that text. Deletang et al. (2024) demonstrate strong LLMs "
                "out-compressing gzip this way. The weights-are-compression "
                "reading (option one) is a looser, different claim - the "
                "coding argument is the one to give in an interview."
            ),
        ),
    ),
    "training-dynamics": (
        Question(
            prompt=(
                "Why does every large training run begin with learning-rate "
                "warmup rather than starting at the peak rate?"
            ),
            options=(
                "To let the GPUs reach thermal equilibrium before full load.",
                "Because early training combines random weights, huge "
                "unrepresentative gradients, and Adam statistics estimated "
                "from almost no samples - full-size steps taken on that "
                "garbage can damage the run before it starts; tiny steps buy "
                "time for the optimizer's estimates to become trustworthy.",
                "To keep the loss curve smooth for monitoring dashboards.",
                "Because the data loader shuffles poorly in the first epoch.",
            ),
            answer=1,
            explanation=(
                "Adam divides each update by a running estimate of gradient "
                "magnitude; with a handful of samples that denominator is "
                "noise, so effective step sizes are erratic exactly when "
                "gradients are largest. Warmup is insurance against those "
                "first few thousand steps. Its length is measured in steps, "
                "not epochs - typically a fraction of a percent of the run."
            ),
        ),
        Question(
            prompt=(
                "What does the critical batch size mark, and what happens if "
                "you train beyond it?"
            ),
            options=(
                "The largest batch that fits in GPU memory; beyond it the run "
                "crashes.",
                "The point where gradient noise stops being the bottleneck: "
                "past it, doubling the batch buys almost no reduction in "
                "steps needed, so you spend twice the compute per step for "
                "the same progress.",
                "The batch size at which the learning rate must be halved.",
                "The minimum batch needed for batch normalization statistics.",
            ),
            answer=1,
            explanation=(
                "McCandlish et al. (2018) estimate it from the gradient noise "
                "scale: while noise dominates, bigger batches let you take "
                "proportionally bigger steps (the linear scaling rule); once "
                "the gradient estimate is already accurate, more averaging is "
                "waste. Since the noise scale grows as the loss falls, large "
                "runs often ramp the batch size up during training."
            ),
        ),
        Question(
            prompt=(
                "During a large run the loss spikes sharply. The team rewinds "
                "to a checkpoint and skips a few hundred batches. Replaying "
                "the same batches from a different checkpoint causes no "
                "spike. What does this reveal about spikes?"
            ),
            options=(
                "The skipped batches contained corrupted data that must be "
                "removed from the corpus.",
                "The spike was triggered by a collision of a particular model "
                "state with a particular batch, not by bad data alone - which "
                "is why skip-and-resume works and why spikes are so hard to "
                "reproduce or predict.",
                "The checkpoint file was corrupted on disk.",
                "The learning rate schedule restarted from zero.",
            ),
            answer=1,
            explanation=(
                "PaLM's team documented exactly this: the same data replayed "
                "from a different state trained fine, implying the "
                "state-times-batch interaction, not the batch, is the "
                "trigger. This is also why spike prevention is about damping "
                "mechanisms (clipping, z-loss, QK-norm) rather than data "
                "cleaning."
            ),
        ),
        Question(
            prompt="What problem does z-loss solve in large-scale training?",
            options=(
                "It prevents the attention matrix from becoming singular.",
                "Cross-entropy is invariant to adding a constant to all "
                "logits, so nothing stops the final-layer logits from "
                "drifting upward together until low-precision arithmetic "
                "misbehaves; z-loss penalizes the softmax normalizer to pin "
                "the logit scale.",
                "It normalizes gradients across data-parallel workers.",
                "It rescales the loss so different domains contribute " "equally.",
            ),
            answer=1,
            explanation=(
                "The softmax output depends only on logit differences, so the "
                "absolute scale is a free parameter that can drift over a "
                "long run - harmless in fp32, dangerous in bf16. Z-loss "
                "(PaLM, ST-MoE) adds a small penalty on log-sum-exp of the "
                "logits, anchoring the scale. It is a stability patch, not a "
                "regularizer for generalization."
            ),
        ),
        Question(
            prompt=(
                "Two launches of the same training job - same code, same "
                "data, same seeds - produce measurably different weights. "
                "What is the root cause?"
            ),
            options=(
                "Cosmic-ray bit flips in GPU memory.",
                "The random seeds silently differ between launches.",
                "Floating-point addition is not associative, and massively "
                "parallel GPU reductions sum in nondeterministic order, so "
                "each run accumulates different last-bit rounding that a "
                "chaotic system amplifies over millions of steps.",
                "The checkpoint format truncates the mantissa.",
            ),
            answer=2,
            explanation=(
                "Parallel reductions schedule their partial sums differently "
                "run to run, and (a+b)+c differs from a+(b+c) in floats. Each "
                "difference is one ulp, but training dynamics amplify rather "
                "than damp them. Teams therefore promise statistical "
                "reproducibility and bitwise-faithful checkpoint resumption, "
                "not bit-identical end-to-end runs."
            ),
        ),
    ),
    "distributed-training": (
        Question(
            prompt=(
                "In mixed-precision AdamW training, roughly how do the 16 "
                "bytes per parameter break down, and what does that imply "
                "about what to shard first?"
            ),
            options=(
                "8 bytes of weights and 8 of gradients; shard the weights " "first.",
                "2 bytes of bf16 weights, 2 of bf16 gradients, and 12 of "
                "fp32 optimizer state (master weights plus two Adam "
                "moments); the optimizer state dominates, which is why "
                "ZeRO stage 1 shards it before touching anything else.",
                "4 bytes each of weights, gradients, activations, and "
                "optimizer state; shard activations first.",
                "16 bytes of weights; quantize before sharding.",
            ),
            answer=1,
            explanation=(
                "The counterintuitive headline is that the weights are the "
                "smallest tenant of their own training run: master weights "
                "(4) plus two moments (4+4) make the optimizer state six "
                "times the bf16 weights. Sharding it across N data-parallel "
                "workers (ZeRO-1) removes most of the redundancy without "
                "adding communication to the layer-by-layer critical path."
            ),
        ),
        Question(
            prompt=(
                "A colleague worries that switching from plain data "
                "parallelism to ZeRO-3/FSDP will change the model's training "
                "trajectory. What is the correct response?"
            ),
            options=(
                "It will: sharding approximates the gradients, trading "
                "accuracy for memory.",
                "It will: each GPU now trains its shard on its own data "
                "slice, like an ensemble.",
                "It will not: every ZeRO stage computes bit-for-bit the same "
                "optimizer step as plain data parallelism - it is a storage "
                "layout, not an algorithm - so no hyperparameter interacts "
                "with it.",
                "It will not, but only if the learning rate is rescaled by "
                "the shard count.",
            ),
            answer=2,
            explanation=(
                "ZeRO's insight is that N data-parallel replicas hold N "
                "identical copies of state that can be split N ways and "
                "reassembled on demand (all-gather for parameters, "
                "reduce-scatter for gradients). The arithmetic is unchanged; "
                "only where the bytes live changes. That is what separates "
                "it from tensor or pipeline parallelism, which restructure "
                "the computation itself."
            ),
        ),
        Question(
            prompt=(
                "Why does tensor parallelism stay within a single node while "
                "pipeline parallelism crosses nodes?"
            ),
            options=(
                "Tensor parallelism requires identical GPUs, and only GPUs "
                "in one node are identical.",
                "Tensor parallelism puts all-reduces on the critical path of "
                "every layer, so it needs NVLink-class intra-node bandwidth; "
                "pipeline parallelism sends only one activation tensor per "
                "stage boundary, which slower inter-node links handle fine.",
                "Pipeline parallelism cannot function inside a node.",
                "CUDA restricts collective operations to eight GPUs.",
            ),
            answer=1,
            explanation=(
                "The placement rule is: match communication appetite to link "
                "speed. TP is chatty and latency-sensitive (per-layer, "
                "per-microbatch), PP is quiet (per-boundary), and DP syncs "
                "once per step and overlaps with compute. Inverting the "
                "order - TP across slow links - stalls every layer of every "
                "forward and backward pass."
            ),
        ),
        Question(
            prompt=(
                "A pipeline has p stages and processes m microbatches per "
                "step. What fraction of time is lost to the bubble, and what "
                "follows from the formula?"
            ),
            options=(
                "Roughly p divided by m squared; bubbles vanish for deep " "pipelines.",
                "Roughly (p-1)/(m+p-1): the idle time from filling and "
                "draining shrinks as microbatches increase, so deeper "
                "pipelines demand proportionally more microbatches to stay "
                "efficient.",
                "Exactly 50 percent regardless of configuration.",
                "Zero, if activations are checkpointed.",
            ),
            answer=1,
            explanation=(
                "While the pipeline fills and drains, early and late stages "
                "idle; with m microbatches in flight the overhead amortizes "
                "as (p-1)/(m+p-1). Doubling pipeline depth without doubling "
                "microbatches lowers utilization - one reason batch sizes at "
                "scale are large, and why pipeline schedules (interleaving, "
                "one-forward-one-backward) are an active engineering area."
            ),
        ),
        Question(
            prompt=(
                "Why is 'all-reduce = reduce-scatter + all-gather' more than " "trivia?"
            ),
            options=(
                "It proves all-reduce is twice as slow as it needs to be.",
                "It shows ZeRO adds no fundamentally new communication: "
                "sharded data parallelism runs the two halves of the "
                "all-reduce plain DP already performed, with storage "
                "rearranged between them - and each half moves about the "
                "array size per GPU regardless of cluster size.",
                "It means all-gather can replace all-reduce in any " "algorithm.",
                "It only holds for clusters with power-of-two GPU counts.",
            ),
            answer=1,
            explanation=(
                "Ring implementations of each half move (N-1)/N of the data "
                "per GPU - nearly constant in N - which makes communication "
                "budgets predictable and explains why ZeRO-2's traffic "
                "matches plain DP's. The identity turns 'exotic sharding "
                "scheme' into 'the same two collectives, reordered', which "
                "is the level of understanding interviews probe for."
            ),
        ),
    ),
    "scaling-laws": (
        Question(
            prompt=(
                "Kaplan et al. (2020) and the Chinchilla work (2022) fit "
                "scaling laws to similar data but reached different "
                "prescriptions. What actually changed?"
            ),
            options=(
                "Chinchilla used a fundamentally different loss function.",
                "Chinchilla found the earlier small-model runs had used "
                "learning-rate schedules mismatched to their training "
                "length, understating small models; with matched schedules "
                "and isoFLOP sweeps, the optimum moved from growing "
                "parameters fastest to growing parameters and tokens in "
                "equal proportion, about 20 tokens per parameter.",
                "Chinchilla trained on code, which favors smaller models.",
                "Kaplan measured perplexity while Chinchilla measured " "accuracy.",
            ),
            answer=1,
            explanation=(
                "The confound: a cosine schedule must land at the final "
                "token (Chapter 7); truncating runs mid-schedule flattered "
                "large models. The corrected fit implied GPT-3-era models "
                "were several-fold undertrained, and the 70B-on-1.4T "
                "Chinchilla model beating a 4x larger one on equal compute "
                "was the demonstration. A methodology detail moved the "
                "entire field's budgets."
            ),
        ),
        Question(
            prompt=(
                "Using the approximation that training costs about 6ND "
                "floating-point operations, what does doubling only the "
                "model size N at a fixed compute budget force you to do, and "
                "why can that hurt?"
            ),
            options=(
                "Nothing; compute depends only on N.",
                "Halve the training tokens D - and if that pushes you above "
                "the compute-optimal tokens-per-parameter ratio toward the "
                "undertrained side, the bigger model finishes with a worse "
                "loss than the smaller one would have.",
                "Double the tokens D to match.",
                "Halve the batch size to compensate.",
            ),
            answer=1,
            explanation=(
                "C = 6ND makes the tradeoff a seesaw: at fixed C, parameters "
                "and tokens trade off exactly. The isoFLOP curves are "
                "U-shaped in that split - too big means undertrained, too "
                "small means underfit - and the Chinchilla point is the "
                "bottom. The 6 decomposes as roughly 2 FLOPs per parameter "
                "for the forward pass and 4 for the backward."
            ),
        ),
        Question(
            prompt=(
                "Llama 3's 8B model trained on about 1,900 tokens per "
                "parameter - two orders of magnitude past Chinchilla's 20. "
                "What justifies this?"
            ),
            options=(
                "Nothing; it was a widely acknowledged waste of compute.",
                "Chinchilla's ratio only applies to models above 70B " "parameters.",
                "Chinchilla minimizes loss for a fixed training budget with "
                "serving priced at zero; when lifetime inference cost is "
                "included, every served token re-pays the parameter count, "
                "so overspending on training to reach target quality in "
                "fewer parameters pays for itself indefinitely.",
                "The extra tokens were needed only for multilingual " "coverage.",
            ),
            answer=2,
            explanation=(
                "This is the inference-aware amendment (Sardana et al., "
                "2024): the optimum shifts toward smaller-overtrained as "
                "expected serving volume grows. It is the visible strategy "
                "of the open-weight era, where models must run on modest "
                "hardware. The regime change to name: training cost is paid "
                "once; serving cost is paid per token, forever."
            ),
        ),
        Question(
            prompt=(
                "Fitted scaling laws include an irreducible loss term that "
                "no amount of scale removes. What is it?"
            ),
            options=(
                "The numerical error of bf16 arithmetic.",
                "The entropy of the text itself: genuinely unpredictable "
                "content costs bits even for a perfect model, so scaling "
                "only buys down the reducible gap between the model and "
                "that floor.",
                "The loss contributed by the softmax temperature.",
                "An artifact of weight decay that better optimizers remove.",
            ),
            answer=1,
            explanation=(
                "In the compression view (Chapter 6), the corpus contains "
                "real information - names, outcomes, arbitrary choices - "
                "that no predictor can anticipate; that content sets a "
                "floor measured in bits per token. Fitted laws routinely "
                "estimate this constant, and its existence is why 'loss "
                "goes to zero with enough scale' is a wrong mental model."
            ),
        ),
        Question(
            prompt=(
                "A team improves its data cleaning pipeline and observes "
                "loss well below what its fitted scaling law predicted at "
                "that compute. The team claims to have 'beaten the scaling "
                "law.' What is the sharper interpretation?"
            ),
            options=(
                "The law was fit with too few points and is simply wrong.",
                "Scaling laws are properties of a particular corpus and "
                "tokenizer, not of nature: better data shifts the whole "
                "curve, so the team is now on a different, better law - "
                "which also warns against porting published constants to "
                "your own data.",
                "The improvement is impossible and indicates evaluation "
                "contamination.",
                "Loss below the law's prediction proves the model is " "memorizing.",
            ),
            answer=1,
            explanation=(
                "The fitted exponents and constants encode the data "
                "distribution. Changing the distribution changes the law - "
                "which is precisely why data work (Chapter 6) is a "
                "first-class lever: it moves the curve that compute then "
                "slides you along. Published constants are estimates for "
                "someone else's corpus, not physics."
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
