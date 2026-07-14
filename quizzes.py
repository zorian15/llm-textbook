"""Single source of truth for the end-of-chapter "Check yourself" quizzes.

Each chapter closes with a short set of challenging, interview-style
multiple-choice questions. `build.py` renders the quiz for a chapter after its
References section, and a small inline script makes it interactive: the reader
picks an option, the choice is marked right or wrong, the correct answer is
revealed, and an explanation appears.

The questions are meant to be *hard* in the way an interview is hard. The
distractors are plausible misconceptions, stated with the same confidence and
detail as the answer, so that neither length nor specificity ever signals which
option is correct. The renderer shuffles the options on load, so the position
of the answer carries no information either — write the options in any order and
point `answer` at the right one. Each explanation carries a second-layer detail
the prose only gestures at, so the quiz teaches rather than merely confirms.

Question strings are **plain text**. `build.py` HTML-escapes everything, which
would neutralize MathJax delimiters, so do not write `$...$` or `\\(...\\)`
here; phrase math in words or with plain symbols instead.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Question:
    """One multiple-choice question.

    `options` are shuffled at render time, so their order here does not matter;
    `answer` is the index (into this tuple) of the single correct option.
    `explanation` is revealed after the reader answers and should teach, not
    just confirm. Because the display order is randomized, never write an option
    that refers to another by position (e.g. "same as A").
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
                'A base model, fresh from pretraining, is prompted "What is the '
                'capital of France?" It answers with a list of related questions '
                'instead of "Paris" — even though it will readily produce "Paris" in '
                "other contexts. What best explains this?"
            ),
            options=(
                "A bare question is most often followed by more questions in web "
                "text, so continuing that pattern is the likeliest completion.",
                "Because the model can retrieve Paris elsewhere but not here, its "
                "attention has failed to link the pronoun-free question to the "
                "France entity stored deeper in the network.",
                "Its decoding temperature defaults high on base models, "
                "flattening the distribution so the model wanders instead of "
                "committing.",
                "The question is parsed as a system instruction, so the model "
                "mirrors it back rather than treating it as a task.",
            ),
            answer=0,
            explanation=(
                "A base model only ever learned to continue text in the style of its "
                "corpus. It very likely knows the fact; it just has not been taught "
                "that a prompt is a request to be answered rather than a document to "
                "be continued. That is exactly what post-training (SFT, then "
                "preference optimization) fixes."
            ),
        ),
        Question(
            prompt=(
                "An interviewer asks whether the 'emergent abilities' of large models "
                "are real. What is the strongest version of the skeptical view?"
            ),
            options=(
                "Larger models quietly memorize the benchmark test sets during "
                "pretraining, so the abilities we measure are really "
                "contamination that would vanish on a truly held-out evaluation.",
                "A discontinuous metric like exact-match stays at zero until a "
                "whole task is right, so a smoothly improving skill can look like "
                "a sudden jump.",
                "The jumps appear only under greedy decoding; sampling at a "
                "nonzero temperature averages them away and reveals steady gains.",
                "Emergence contradicts the scaling laws, which require the loss "
                "to plateau before any new capability appears.",
            ),
            answer=1,
            explanation=(
                "The mirage argument (Schaeffer et al., 2023) is that a nonlinear or "
                "all-or-nothing metric can manufacture a sudden jump out of a "
                "smoothly falling loss: switch to a continuous measure and the "
                "'emergence' often smooths out. It does not deny that capabilities "
                "appear with scale, only that the sharp threshold can be a "
                "measurement effect."
            ),
        ),
        Question(
            prompt=(
                "Cross-entropy loss is often reported as perplexity. If a model "
                "reaches a cross-entropy of 3 bits per token, its perplexity is 8. "
                "What does that number mean intuitively?"
            ),
            options=(
                "When it generates, it gets roughly one token in every eight "
                "outright wrong, which is what the eight counts.",
                "It needs about eight tokens of prior context before it can "
                "reliably commit to predicting the next one.",
                "On average it is as unsure as if choosing uniformly among eight "
                "equally likely tokens.",
                "Its usable vocabulary has effectively collapsed to about eight "
                "distinct tokens it still emits.",
            ),
            answer=2,
            explanation=(
                "Perplexity is 2 raised to the cross-entropy in bits (or e to the "
                "nats), i.e. the effective number of equally-likely choices the model "
                "is deciding among per token. Lower perplexity means the model "
                "concentrates probability on fewer candidates. It is a monotone "
                "rescaling of the loss, not new information."
            ),
        ),
        Question(
            prompt=(
                "Why is preference optimization (RLHF or DPO) used at all, when "
                "supervised fine-tuning already shows the model good answers?"
            ),
            options=(
                "It lets the model absorb new facts that appeared after "
                "pretraining finished, filling the knowledge gaps that a fixed "
                "set of supervised demonstrations simply cannot reach.",
                "It removes the need for human labeling, since the model "
                "manufactures its own preference targets.",
                "It converges faster than supervised fine-tuning, so teams reach "
                "for it mainly to cut training compute.",
                "Judging which of two answers is better is easier than writing "
                "the best one, so preferences are a richer, more scalable signal.",
            ),
            answer=3,
            explanation=(
                "The core asymmetry: recognition is easier than generation. "
                "Preference methods learn from comparisons among the model's own "
                "outputs, which covers far more of the output space than a fixed set "
                "of gold demonstrations and does not cap quality at what the "
                "annotators could themselves write."
            ),
        ),
        Question(
            prompt=(
                "In this book's framing, the 'harness' around a model is best "
                "described as which of the following?"
            ),
            options=(
                "The inference-time layer around the weights — system prompt, "
                "tools, retrieval, guardrails — deciding what the model sees and "
                "what happens to its output.",
                "The distributed training system that shards the weights, "
                "gradients, and optimizer state across hundreds of GPUs so the "
                "model can be trained in the first place.",
                "The reinforcement-learning loop that aligns the model to human "
                "preferences during post-training.",
                "The quantization and serving stack that drives down the cost of "
                "each generated token.",
            ),
            answer=0,
            explanation=(
                "The harness does not change the weights at all; it shapes the "
                "model's inputs and mediates its outputs. Much of what feels like "
                "'the model just behaves' comes from the harness, which is why Part V "
                "is devoted to it and why most LLM engineering roles in 2026 are "
                "really about building harnesses."
            ),
        ),
    ),
    "deep-learning-refresher": (
        Question(
            prompt=(
                "Your model runs a forward pass on a batch just fine, but the same "
                "batch runs out of memory during training. What is the primary "
                "reason, and which lever most directly addresses it?"
            ),
            options=(
                "AdamW's per-step update is far heavier than SGD's because it "
                "maintains two moment estimates per weight, so switching the "
                "optimizer to plain momentum SGD removes the extra memory.",
                "Training also keeps every layer's activations, plus gradients "
                "and optimizer state; gradient checkpointing recomputes "
                "activations rather than storing them.",
                "The learning rate is set too high, which inflates activation "
                "magnitudes until they overflow; lowering it shrinks the "
                "footprint.",
                "Training pads sequences to a longer maximum length than "
                "inference does, so truncating the batch resolves it.",
            ),
            answer=1,
            explanation=(
                "Inference needs only the weights and a single layer's activations. "
                "Training additionally holds every layer's activations (for the "
                "backward pass), the gradients, and the optimizer state; for AdamW "
                "that is two moments plus commonly an fp32 master copy, roughly 16 "
                "bytes per parameter before activations. Gradient checkpointing "
                "recomputes activations in the backward pass instead of storing them; "
                "sharding (ZeRO, Chapter 8) and gradient accumulation are the other "
                "levers."
            ),
        ),
        Question(
            prompt=(
                "What is the actual difference between L2 regularization folded into "
                "the loss and AdamW's decoupled weight decay?"
            ),
            options=(
                "They are mathematically identical for every optimizer, and the "
                "AdamW name merely marks a tidier implementation of exactly the "
                "same parameter update.",
                "AdamW confines weight decay to the bias and normalization terms "
                "that plain L2 regularization is unable to touch.",
                "An L2 penalty enters through the gradient, so Adam's denominator "
                "rescales it and high-gradient weights decay less; AdamW "
                "subtracts a fixed fraction of each weight directly.",
                "L2 makes the effective learning rate grow over training, whereas "
                "AdamW holds it fixed throughout the run.",
            ),
            answer=2,
            explanation=(
                "Adam normalizes each parameter's update by a running estimate of its "
                "gradient magnitude. If weight decay rides in through the gradient "
                "(L2 in the loss), it gets rescaled by that same denominator and "
                "becomes uneven across parameters. AdamW decouples the two, "
                "subtracting a fixed fraction of each weight directly, which is why "
                "it is the default for transformers."
            ),
        ),
        Question(
            prompt=(
                "The same overparameterized network can be trained to 100% accuracy "
                "on data with completely random labels, yet it generalizes well on "
                "real labels. What does this most directly imply?"
            ),
            options=(
                "Because it can fit pure noise to perfection, the network cannot "
                "truly be overparameterized for the real task, or it would "
                "overfit there just as badly.",
                "Weight decay is doing all the work; without it, real-label "
                "training would memorize just as badly.",
                "Fitting random labels shows the training pipeline is buggy, so "
                "the real-label result should not be trusted.",
                "Raw capacity does not settle generalization; the data's "
                "structure and what the optimizer reaches first matter more than "
                "parameter count.",
            ),
            answer=3,
            explanation=(
                "Zhang et al. (2017) showed capacity is enough to memorize noise, so "
                "classical bounds tying capacity to generalization cannot be the "
                "whole story. On structured data the optimizer tends to find simpler, "
                "better-generalizing solutions first. In pretraining, a near-single "
                "pass over a huge corpus also removes most of the opportunity to "
                "memorize."
            ),
        ),
        Question(
            prompt=(
                "Why did bfloat16 become the default over float16 for training, even "
                "though float16 offers more mantissa bits (higher precision)?"
            ),
            options=(
                "bfloat16 keeps float32's 8-bit exponent, so its range matches "
                "float32 and gradients neither overflow nor underflow; the lost "
                "mantissa disappears into gradient noise.",
                "float16 cannot represent the negative values that gradients "
                "routinely take, so a large fraction of updates silently flush to "
                "zero and stall the run.",
                "bfloat16 multiplies measurably faster than float16 on current "
                "accelerators, so it wins on raw throughput.",
                "bfloat16 stores each number in half the bytes float16 uses, so "
                "batches can roughly double at equal precision.",
            ),
            answer=0,
            explanation=(
                "Training tolerates noise but not clipped range. float16's narrow "
                "5-bit exponent forces fragile loss-scaling to keep gradients "
                "representable; bfloat16 spends its bits on range instead of "
                "precision, so nothing silently flushes to zero or infinity. Both are "
                "16 bits, so memory is the same."
            ),
        ),
        Question(
            prompt=(
                "Why is Adam(W) preferred over plain SGD for transformers "
                "specifically?"
            ),
            options=(
                "Adam provably reaches a strictly lower final loss than SGD on "
                "any optimization problem, so it is the safe default for "
                "transformers as much as for anything else.",
                "A transformer's gradients span orders of magnitude across "
                "parameters, and Adam's per-parameter scaling equalizes them so "
                "one learning rate suffices.",
                "Adam needs no learning rate at all, removing the single hardest "
                "hyperparameter to tune.",
                "SGD is incompatible with mini-batching, so large-batch "
                "transformer training requires Adam.",
            ),
            answer=1,
            explanation=(
                "A transformer mixes parameters whose gradients differ by orders of "
                "magnitude and by how often they are exercised. Adam gives each "
                "parameter an effective step size scaled by its own recent gradient "
                "magnitude, so one learning rate serves all of them; plain SGD would "
                "need impractical per-layer tuning."
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
                "Alongside the token ids it keeps a compressed copy of the "
                "original string, which the decoder restores verbatim whenever it "
                "needs the exact spacing back.",
                "It supports only languages that separate words with spaces, "
                "where reconstructing the input is unambiguous.",
                "It escapes each space into an ordinary vocabulary symbol and "
                "treats text as a raw stream, so decoding is just concatenation "
                "with that symbol mapped back to a space.",
                "It appends a checksum token that records the whitespace for the "
                "decoder to reinsert.",
            ),
            answer=2,
            explanation=(
                "By escaping whitespace into a normal symbol in the vocabulary, "
                "SentencePiece removes the special-case handling of spaces; the "
                "sequence of pieces reconstructs the input by direct concatenation. "
                "This is why a leading space often binds to the following word as one "
                "token."
            ),
        ),
        Question(
            prompt=(
                "WordPiece and BPE both merge subword pairs greedily. What "
                "distinguishes WordPiece's merge criterion?"
            ),
            options=(
                "It picks merges at random during training and prunes the "
                "unhelpful ones in a later cleanup pass.",
                "It merges the single highest-raw-frequency adjacent pair exactly "
                "as BPE does, and the two methods differ only in the file format "
                "they serialize the vocabulary to.",
                "It refuses to merge across morpheme boundaries, consulting a "
                "dictionary to keep known morphemes intact.",
                "It merges the pair that most raises the corpus likelihood, "
                "scoring a pair by its frequency over the product of its parts' "
                "frequencies.",
            ),
            answer=3,
            explanation=(
                "BPE picks the most frequent adjacent pair; WordPiece picks the pair "
                "whose merge best improves a unigram language model's likelihood. "
                "That criterion normalizes by how common the parts already are, so a "
                "pair that co-occurs more than chance is preferred over one that is "
                "merely frequent because its parts are."
            ),
        ),
        Question(
            prompt=(
                "You want a tokenizer-level trick that makes a translation model more "
                "robust to how words get split. What is BPE-dropout, and when does it "
                "apply?"
            ),
            options=(
                "During training it randomly skips some merges so a word appears "
                "under several segmentations; at inference it tokenizes "
                "deterministically as usual.",
                "It prunes the rarest tokens from the vocabulary after training "
                "so the embedding table is smaller and each surviving token "
                "carries more information at inference.",
                "It discards the least-frequent tenth of the training sentences "
                "to cut noise in the parallel data.",
                "It applies ordinary dropout to each token's embedding vector on "
                "the forward pass.",
            ),
            answer=0,
            explanation=(
                "BPE-dropout (Provilkov et al., 2020), a form of subword "
                "regularization (Kudo, 2018), stochastically omits merges while "
                "training so the model sees several valid tokenizations of the same "
                "string and learns more robust subword representations. Inference "
                "stays deterministic, so throughput is unaffected."
            ),
        ),
        Question(
            prompt=(
                "Why do modern tokenizers often split numbers into fixed-size digit "
                "groups, sometimes grouped from the right?"
            ),
            options=(
                "It compresses long numbers into far fewer tokens than the "
                "equivalent spelled-out words would ever take, saving context "
                "budget on numeric text.",
                "It keeps place value aligned across numbers, so ones, tens, and "
                "hundreds line up instead of 1234 being one token and 1235 two.",
                "Raw bytes cannot encode digit characters directly, so the digits "
                "have to be regrouped before embedding.",
                "It obscures numeric values to protect private data present in "
                "the training corpus.",
            ),
            answer=1,
            explanation=(
                "If nearby numbers tokenize into wildly different shapes, the model "
                "has to learn each as an unrelated symbol. Forcing digits into "
                "consistent groups (and grouping from the least-significant digit) "
                "keeps ones, tens, and hundreds aligned across examples, which "
                "measurably helps arithmetic. It is a tokenizer choice changing a "
                "'reasoning' ability."
            ),
        ),
        Question(
            prompt=(
                "A model reliably miscounts the letters in 'strawberry', even though "
                "it spells many other words correctly. What is the real cause?"
            ),
            options=(
                "It lacks the parameters to maintain a running counter as it "
                "scans the word, so the tally drifts once a word passes a few "
                "characters in length.",
                "Attention cannot compare one position with another, so it has no "
                "way to tally repeated letters.",
                "Tokenization hands the model opaque multi-character chunks, so "
                "the letters are absent from its input unless it memorized "
                "spelling facts about those tokens.",
                "The context window is too short to hold the whole word while the "
                "model counts.",
            ),
            answer=2,
            explanation=(
                "The evidence a letter-counting task needs is destroyed before the "
                "model sees it: 'strawberry' might arrive as two or three tokens with "
                "no exposed characters. Success is inconsistent across words "
                "precisely because it depends on memorized token-spelling facts, not "
                "on any counting ability."
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
                "At equal depth an encoder carries substantially more parameters "
                "than a decoder, and that extra capacity is what yields the "
                "richer embeddings retrieval depends on.",
                "Encoders are always faster at inference because they run in a "
                "single parallel pass.",
                "Decoder-only models cannot produce embedding vectors at all; "
                "they only emit token probabilities.",
                "An encoder attends in both directions and trains with "
                "masked-language-modeling, which fits understanding a complete "
                "input; a decoder is causal, built to generate.",
            ),
            answer=3,
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
                "Why is dividing the attention scores by the square root of the head "
                "dimension important?"
            ),
            options=(
                "Dot products sum over the head dimension, so their variance "
                "grows with it; the scaling holds variance near one and keeps "
                "softmax out of its saturated regime.",
                "It normalizes each row of scores so that the attention weights "
                "along that row are guaranteed to add up to exactly one before "
                "the softmax is even applied.",
                "It converts the raw dot products into a probability "
                "distribution, doing the softmax's job ahead of time.",
                "It compensates for the causal mask, which removes roughly half "
                "of every score row.",
            ),
            answer=0,
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
                "sequence at once. What is the name for conditioning each prediction "
                "on the ground-truth previous tokens, and what asymmetry does it "
                "create?"
            ),
            options=(
                "Beam search: at both training and generation the model keeps "
                "several competing hypotheses alive and expands the most "
                "promising ones, so the two phases stay symmetric.",
                "Teacher forcing: training feeds the true previous tokens at "
                "every position in parallel, while generation must feed the "
                "model's own samples one step at a time.",
                "Label smoothing: it softens the target distribution during "
                "training but leaves generation untouched.",
                "Curriculum learning: easier tokens are shown before harder ones "
                "as training proceeds.",
            ),
            answer=1,
            explanation=(
                "Because the mask guarantees each position only sees earlier ones, "
                "every position can be trained simultaneously against the real next "
                "token (teacher forcing). At generation time there is no ground "
                "truth, so the model feeds on its own outputs sequentially, the "
                "source of the train/generate asymmetry (and of exposure bias)."
            ),
        ),
        Question(
            prompt=(
                "Roughly where do a transformer's parameters live, and why does it "
                "matter for mixture-of-experts models?"
            ),
            options=(
                "Most of the parameters sit in the attention projections, so a "
                "mixture-of-experts model adds capacity by replacing the "
                "attention heads with many routed expert heads.",
                "They split roughly evenly across embeddings, attention, and "
                "FFNs, and MoE duplicates the embedding table.",
                "About two-thirds sit in the feed-forward networks and a third in "
                "attention; MoE swaps the single FFN for many experts and fires "
                "only a couple per token.",
                "Most live in the token embeddings, so MoE shrinks the vocabulary "
                "to save memory.",
            ),
            answer=2,
            explanation=(
                "For typical shapes the FFNs hold about two-thirds of the parameters "
                "and, many argue, most of the stored knowledge. That is exactly why "
                "MoE targets the FFN: replace one FFN with many experts, route each "
                "token to a couple, and total capacity grows while per-token compute "
                "stays roughly fixed."
            ),
        ),
        Question(
            prompt=(
                "A candidate claims the O(n^2) cost of attention means long-context "
                "models are fundamentally limited by compute. What is the more "
                "precise picture?"
            ),
            options=(
                "The claim is exactly right: the quadratic compute is "
                "fundamental, and because every token must attend to every other, "
                "no reformulation can bring the cost below n squared.",
                "The quadratic cost applies only during training and disappears "
                "at inference time.",
                "Attention is really linear in sequence length once the keys and "
                "values are cached across steps.",
                "The quadratic term is real, but attention is often bound by "
                "memory traffic, not FLOPs; FlashAttention computes the exact "
                "result tile by tile without storing the n-by-n matrix.",
            ),
            answer=3,
            explanation=(
                "The n-by-n score matrix drives the quadratic compute, but the "
                "practical wall is often reading and writing that matrix to memory. "
                "FlashAttention reorders the computation to keep tiles in fast "
                "on-chip memory and never stores the full matrix, cutting memory "
                "traffic without changing the math. The serving-side consequences are "
                "Chapter 15's subject."
            ),
        ),
    ),
    "modern-architectures": (
        Question(
            prompt=(
                "Rotary position embeddings (RoPE) encode position by rotating query "
                "and key vectors. Why does this give the model relative position for "
                "free?"
            ),
            options=(
                "The score between a query at position m and a key at position n "
                "depends only on the offset m minus n, not on their absolute "
                "positions.",
                "Every position is rotated by the same fixed angle, so the "
                "absolute offsets cancel out of the dot product and only ordering "
                "information is left behind in the score.",
                "RoPE adds a learned position vector into each embedding before "
                "the attention projections run.",
                "The rotations force every attention score to come out positive, "
                "which is what encodes order.",
            ),
            answer=0,
            explanation=(
                "RoPE rotates each 2D sub-pair of the query and key by an angle "
                "proportional to position. In the score, the two rotations combine so "
                "only the difference of angles, i.e. the relative offset, survives. "
                "Grammar cares about how far apart tokens are, which is what this "
                "encodes."
            ),
        ),
        Question(
            prompt=(
                "A model trained with an 8K context must serve 32K prompts. Which "
                "family of techniques directly addresses this, and what is the "
                "underlying idea?"
            ),
            options=(
                "Adding attention heads so there is representational room for the "
                "extra positions, since each head can specialize on a different "
                "slice of the longer sequence.",
                "RoPE context-extension methods — linear interpolation, NTK-aware "
                "scaling, YaRN — that remap the rotary frequencies so far "
                "positions land in a range the model has seen.",
                "Swapping RoPE for learned absolute position embeddings at "
                "serving time to reach the new length.",
                "Lowering the decoding temperature so the model attends to fewer, "
                "nearer tokens.",
            ),
            answer=1,
            explanation=(
                "Because RoPE stores position in rotation frequencies, you can "
                "stretch or reinterpret those frequencies to cover longer contexts: "
                "linear interpolation squeezes positions into the trained range, and "
                "NTK-aware and YaRN (Peng et al., 2023) scale frequencies more "
                "carefully to preserve short-range resolution. The serving-cost side "
                "is Chapter 15."
            ),
        ),
        Question(
            prompt=(
                "Grouped-query attention shrinks the KV cache by sharing KV heads "
                "across groups of query heads. Name another distinct architectural "
                "lever for the same goal."
            ),
            options=(
                "Stacking more layers, which spreads the total KV cache across "
                "depth so that each individual layer holds a smaller share of it "
                "at any one time.",
                "Raising the RoPE base frequency so each cache entry can cover "
                "more positions.",
                "Sliding-window attention, where each token attends only to the "
                "last w tokens so cache and per-step cost stop growing, often "
                "with a few global layers mixed in.",
                "Replacing the ReLU feed-forward network with SwiGLU, which "
                "shrinks the cached tensors.",
            ),
            answer=2,
            explanation=(
                "GQA cuts the cache by the sharing factor; sliding-window attention "
                "(as in Mistral, Jiang et al., 2023) bounds it by limiting how far "
                "back each token looks. Multi-head latent attention, which compresses "
                "KV into a low-rank latent, is a third lever. They are complementary "
                "ways to fight KV-cache growth, the dominant long-context memory "
                "cost."
            ),
        ),
        Question(
            prompt=(
                "In a mixture-of-experts layer, what is the role of the "
                "load-balancing (auxiliary) loss, and what problem does it prevent?"
            ),
            options=(
                "It raises the penalty on wrong predictions for the experts that "
                "are chosen least often, so that the rare experts are pushed to "
                "catch up in quality with the popular ones.",
                "It lowers the total parameter count that must be held in memory "
                "during training.",
                "It forces every token through all experts, trading compute for "
                "stability.",
                "It pushes the router to spread tokens across experts, preventing "
                "a collapse where a few overflow their capacity while the rest "
                "sit idle.",
            ),
            answer=3,
            explanation=(
                "Left alone, routers tend to collapse onto a few favored experts, "
                "which then exceed their capacity (dropping tokens) while the rest "
                "sit idle, wasting capacity. An auxiliary balancing loss encourages "
                "even routing; newer models (e.g. DeepSeek-V3) pursue "
                "auxiliary-loss-free balancing via learned routing biases to avoid "
                "the aux-loss's tug on quality."
            ),
        ),
        Question(
            prompt=(
                "A config shows num_attention_heads: 64 and num_key_value_heads: 8. "
                "What is the interviewer looking for?"
            ),
            options=(
                "Grouped-query attention with eight groups, so the KV cache is "
                "eight times smaller than full multi-head — which matters because "
                "decode is memory-bandwidth-bound.",
                "The model stacks eight distinct attention layers, one dedicated "
                "to each of the key-value heads that the configuration file "
                "happens to list.",
                "The layer is a mixture of experts with eight experts sharing one "
                "attention block.",
                "The head dimension is 64 divided by 8, giving eight dimensions "
                "per head.",
            ),
            answer=0,
            explanation=(
                "Sixty-four query heads sharing eight KV heads means groups of eight, "
                "an 8x smaller KV cache. The 'why it matters' is the payoff: decode "
                "streams the cache from memory every step, so a smaller cache buys "
                "longer context and larger batches at a small quality cost."
            ),
        ),
    ),
    "pretraining-objective": (
        Question(
            prompt=(
                "Model A (32k-token vocabulary) reports perplexity 9; model B "
                "(128k-token vocabulary) reports perplexity 12 on the same text. What "
                "can you conclude about which predicts the text better?"
            ),
            options=(
                "Model A predicts better, because on the very same text a "
                "perplexity of 9 is plainly lower than 12, and lower perplexity "
                "always means a better predictor.",
                "Nothing yet — perplexity is per token, and B's coarser tokens "
                "each carry more text, inflating per-token uncertainty; compare "
                "in bits per byte instead.",
                "Model B predicts better, since a larger vocabulary is itself a "
                "sign of a stronger model.",
                "Nothing at all, because perplexity has no bearing on how well a "
                "model predicts text.",
            ),
            answer=1,
            explanation=(
                "A tokenizer choice changes the unit of measurement. Each of B's "
                "tokens carries more text, so its per-token perplexity is naturally "
                "higher even if it compresses the text better overall. Converting "
                "both models' total cross-entropy to bits per byte of the same "
                "underlying text removes the tokenizer from the comparison - which is "
                "exactly the compression view of the loss."
            ),
        ),
        Question(
            prompt=(
                "Beyond wasting compute on repeats, why does deduplication measurably "
                "improve a pretrained model?"
            ),
            options=(
                "Removing duplicate documents shrinks the vocabulary the "
                "tokenizer must learn, so each remaining token is exercised more "
                "often and ends up carrying more information.",
                "It guarantees the model can no longer memorize and reproduce any "
                "passage from its training set.",
                "Mirrored documents silently upweight whatever is duplicated, "
                "warping the intended mixture, and repeated passages are far more "
                "likely to be memorized verbatim.",
                "It smooths the loss curve by removing the hardest examples the "
                "model struggles on.",
            ),
            answer=2,
            explanation=(
                "Lee et al. (2022) showed near-duplicate text both skews the "
                "effective data distribution and drives verbatim memorization, which "
                "is a privacy and quality problem. Dedup reduces memorization sharply "
                "but does not eliminate it - unique text seen once can still be "
                "memorized, especially by large models late in training."
            ),
        ),
        Question(
            prompt=(
                "Why do pretraining teams shift the data mixture toward their "
                "highest-quality sources specifically during the final learning-rate "
                "decay, rather than spreading that data evenly?"
            ),
            options=(
                "Presenting the scarce high-quality data early would let the "
                "model overfit it and largely forget it again over the long tail "
                "of training that still remains.",
                "High-quality sources need a lower learning rate for the "
                "tokenizer to parse them correctly.",
                "The data loader can only apply one mixture change per run, and "
                "the end is the safest place for it.",
                "The final updates, taken at the smallest learning rates, are "
                "least likely to be overwritten later, so the model's polish "
                "comes from what it sees there.",
            ),
            answer=3,
            explanation=(
                "Annealing pairs the mixture shift with the learning-rate decay "
                "deliberately: nothing comes after the anneal to disturb what it "
                "teaches. Llama 3 and MiniCPM both report this recipe. It is also why "
                "a mid-training checkpoint understates final model quality - the "
                "anneal's gains have not happened yet."
            ),
        ),
        Question(
            prompt=(
                "Your corpus is short of fresh tokens for the planned run. What does "
                "the evidence on repeating data say?"
            ),
            options=(
                "Up to about four epochs over a source is nearly as good as fresh "
                "tokens; past that, each extra repeat adds rapidly less.",
                "Any repetition at all begins to trigger overfitting, so a "
                "well-run pretraining job must be built entirely from strictly "
                "unique, never-repeated tokens.",
                "Repetition is essentially free — even forty epochs behave like "
                "that much fresh data.",
                "Repetition helps only for code and math, and hurts for ordinary "
                "prose.",
            ),
            answer=0,
            explanation=(
                "Muennighoff et al. (2023) fit scaling laws in the data-constrained "
                "regime: a few epochs are nearly as good as unique tokens, then value "
                "decays sharply. This is why small high-quality sources are routinely "
                "upsampled several times per web epoch, and why the 'running out of "
                "data' question is about fresh-token supply, not literal exhaustion."
            ),
        ),
        Question(
            prompt=(
                "What is the precise sense in which a language model 'is' a "
                "compressor?"
            ),
            options=(
                "Because its trained weights are far smaller than the corpus they "
                "were trained on, the training process has effectively packed all "
                "that data down into them.",
                "Its next-token probabilities can drive an arithmetic coder that "
                "spends minus log2 p(x) bits per token, so lower cross-entropy is "
                "literally a shorter encoding.",
                "It strips stopwords and redundant phrasing out of the text it "
                "generates, shortening the output.",
                "It keeps a compressed archive of frequent phrases inside its "
                "key-value cache.",
            ),
            answer=1,
            explanation=(
                "The equivalence is mechanical, not metaphorical: any predictor plus "
                "arithmetic coding is a lossless compressor, and the achieved file "
                "size is the model's cross-entropy on that text. Deletang et al. "
                "(2024) demonstrate strong LLMs out-compressing gzip this way. The "
                "weights-are-smaller-than-the-data reading is a looser, different "
                "claim - the coding argument is the one to give in an interview."
            ),
        ),
    ),
    "training-dynamics": (
        Question(
            prompt=(
                "Why does every large training run begin with learning-rate warmup "
                "rather than starting at the peak rate?"
            ),
            options=(
                "It keeps the loss curve visually smooth so monitoring dashboards "
                "read cleanly early on.",
                "It gives the GPUs time to reach a stable operating temperature "
                "before they are asked to sustain the full compute load of the "
                "run's peak learning rate.",
                "Early on the weights are random, gradients are huge, and Adam's "
                "statistics rest on almost no samples; small steps buy time "
                "before the optimizer can be trusted.",
                "It compensates for the data loader shuffling the corpus poorly "
                "in the first epoch.",
            ),
            answer=2,
            explanation=(
                "Adam divides each update by a running estimate of gradient "
                "magnitude; with a handful of samples that denominator is noise, so "
                "effective step sizes are erratic exactly when gradients are largest. "
                "Warmup is insurance against those first few thousand steps. Its "
                "length is measured in steps, not epochs - typically a fraction of a "
                "percent of the run."
            ),
        ),
        Question(
            prompt=(
                "What does the critical batch size mark, and what happens if you "
                "train beyond it?"
            ),
            options=(
                "It is the largest batch that still fits in aggregate GPU memory, "
                "and pushing past it makes the run exhaust memory and crash "
                "partway through training.",
                "It is the smallest batch that yields reliable "
                "batch-normalization statistics for each layer.",
                "It is the batch size at which the learning rate must be halved "
                "to keep the run from diverging.",
                "It is the point where gradient noise stops dominating; beyond "
                "it, doubling the batch barely cuts the steps needed, so you pay "
                "double compute for the same progress.",
            ),
            answer=3,
            explanation=(
                "McCandlish et al. (2018) estimate it from the gradient noise scale: "
                "while noise dominates, bigger batches let you take proportionally "
                "bigger steps (the linear scaling rule); once the gradient estimate "
                "is already accurate, more averaging is waste. Since the noise scale "
                "grows as the loss falls, large runs often ramp the batch size up "
                "during training."
            ),
        ),
        Question(
            prompt=(
                "During a large run the loss spikes sharply. The team rewinds to a "
                "checkpoint and skips a few hundred batches. Replaying the same "
                "batches from a different checkpoint causes no spike. What does this "
                "reveal about spikes?"
            ),
            options=(
                "A particular model state collided with a particular batch, not "
                "bad data alone — which is why skip-and-resume works and why "
                "spikes barely reproduce.",
                "The skipped batches must contain a cluster of corrupted or "
                "malformed samples, and the right fix is to hunt them down and "
                "scrub them from the corpus for good.",
                "The checkpoint on disk was silently corrupted, and reloading a "
                "clean copy is what avoided the spike.",
                "The learning-rate schedule reset to zero at the spike and had to "
                "be restarted by hand.",
            ),
            answer=0,
            explanation=(
                "PaLM's team documented exactly this: the same data replayed from a "
                "different state trained fine, implying the state-times-batch "
                "interaction, not the batch, is the trigger. This is also why spike "
                "prevention is about damping mechanisms (clipping, z-loss, QK-norm) "
                "rather than data cleaning."
            ),
        ),
        Question(
            prompt="What problem does z-loss solve in large-scale training?",
            options=(
                "It keeps the attention score matrix from drifting toward "
                "singular, so that the softmax over each row stays numerically "
                "well-conditioned deep in the stack.",
                "Cross-entropy ignores a shared shift of all logits, so their "
                "absolute scale can drift up until low-precision math breaks; "
                "z-loss pins that scale.",
                "It normalizes gradients across data-parallel workers so their "
                "updates stay consistent.",
                "It rescales the loss so each training domain contributes equally "
                "to the objective.",
            ),
            answer=1,
            explanation=(
                "The softmax output depends only on logit differences, so the "
                "absolute scale is a free parameter that can drift over a long run - "
                "harmless in fp32, dangerous in bf16. Z-loss (PaLM, ST-MoE) adds a "
                "small penalty on log-sum-exp of the logits, anchoring the scale. It "
                "is a stability patch, not a regularizer for generalization."
            ),
        ),
        Question(
            prompt=(
                "Two launches of the same training job - same code, same data, same "
                "seeds - produce measurably different weights. What is the root "
                "cause?"
            ),
            options=(
                "Rare cosmic-ray bit flips in GPU memory perturb a few weights "
                "differently on each launch, and over a long run those tiny "
                "perturbations accumulate into visible divergence.",
                "The random seeds are not truly identical; one differs silently "
                "between the launches.",
                "Float addition is not associative, and parallel GPU reductions "
                "sum in varying order, so each run accrues different last-bit "
                "rounding that training amplifies.",
                "The checkpoint format truncates the mantissa on save, shedding "
                "precision each write.",
            ),
            answer=2,
            explanation=(
                "Parallel reductions schedule their partial sums differently run to "
                "run, and (a+b)+c differs from a+(b+c) in floats. Each difference is "
                "one ulp, but training dynamics amplify rather than damp them. Teams "
                "therefore promise statistical reproducibility and bitwise-faithful "
                "checkpoint resumption, not bit-identical end-to-end runs."
            ),
        ),
    ),
    "distributed-training": (
        Question(
            prompt=(
                "In mixed-precision AdamW training, roughly how do the 16 bytes per "
                "parameter break down, and what does that imply about what to shard "
                "first?"
            ),
            options=(
                "Eight bytes go to the weights and eight to the gradients, so the "
                "weights are half the footprint and are the obvious first thing "
                "to split across data-parallel workers.",
                "All sixteen bytes are the weights; quantize them before any "
                "sharding.",
                "Four bytes each for weights, gradients, activations, and "
                "optimizer state; shard the activations first.",
                "2 bytes bf16 weights, 2 bytes bf16 gradients, and 12 bytes of "
                "fp32 optimizer state; the optimizer state dominates, so ZeRO-1 "
                "shards it first.",
            ),
            answer=3,
            explanation=(
                "The counterintuitive headline is that the weights are the smallest "
                "tenant of their own training run: master weights (4) plus two "
                "moments (4+4) make the optimizer state six times the bf16 weights. "
                "Sharding it across N data-parallel workers (ZeRO-1) removes most of "
                "the redundancy without adding communication to the layer-by-layer "
                "critical path."
            ),
        ),
        Question(
            prompt=(
                "A colleague worries that switching from plain data parallelism to "
                "ZeRO-3/FSDP will change the model's training trajectory. What is the "
                "correct response?"
            ),
            options=(
                "It does not change training: every ZeRO stage computes the "
                "identical optimizer step to plain DP — only the storage layout "
                "differs, so no hyperparameter interacts with it.",
                "It does change training: by holding only a shard of the "
                "parameters at a time, each worker computes an approximate "
                "gradient, trading a little accuracy for the memory it saves.",
                "It changes training, because each GPU now learns from its own "
                "data slice like an ensemble member.",
                "It leaves training unchanged only if the learning rate is "
                "rescaled by the shard count.",
            ),
            answer=0,
            explanation=(
                "ZeRO's insight is that N data-parallel replicas hold N identical "
                "copies of state that can be split N ways and reassembled on demand "
                "(all-gather for parameters, reduce-scatter for gradients). The "
                "arithmetic is unchanged; only where the bytes live changes. That is "
                "what separates it from tensor or pipeline parallelism, which "
                "restructure the computation itself."
            ),
        ),
        Question(
            prompt=(
                "Why does tensor parallelism stay within a single node while pipeline "
                "parallelism crosses nodes?"
            ),
            options=(
                "Tensor parallelism requires strictly identical GPUs, and only "
                "the accelerators packed inside a single physical node come with "
                "that guarantee from the vendor.",
                "Tensor parallelism all-reduces on every layer's critical path, "
                "so it needs NVLink-class bandwidth; pipeline parallelism sends "
                "one activation per stage boundary, which slower links handle.",
                "Pipeline parallelism cannot run within a single node at all, so "
                "it is forced to span multiple nodes.",
                "CUDA caps collective operations at eight GPUs, which is exactly "
                "one node's worth.",
            ),
            answer=1,
            explanation=(
                "The placement rule is: match communication appetite to link speed. "
                "TP is chatty and latency-sensitive (per-layer, per-microbatch), PP "
                "is quiet (per-boundary), and DP syncs once per step and overlaps "
                "with compute. Inverting the order - TP across slow links - stalls "
                "every layer of every forward and backward pass."
            ),
        ),
        Question(
            prompt=(
                "A pipeline has p stages and processes m microbatches per step. What "
                "fraction of time is lost to the bubble, and what follows from the "
                "formula?"
            ),
            options=(
                "Roughly p divided by m squared, which shrinks so quickly that a "
                "sufficiently deep pipeline drives the idle bubble to a "
                "negligible fraction on its own.",
                "Exactly half the runtime, regardless of how the stages and "
                "microbatches are configured.",
                "Roughly (p-1)/(m+p-1): the fill-and-drain idle time shrinks as "
                "microbatches grow, so deeper pipelines need proportionally more "
                "microbatches to stay efficient.",
                "Zero, as long as activations are checkpointed and recomputed on "
                "the backward pass.",
            ),
            answer=2,
            explanation=(
                "While the pipeline fills and drains, early and late stages idle; "
                "with m microbatches in flight the overhead amortizes as "
                "(p-1)/(m+p-1). Doubling pipeline depth without doubling microbatches "
                "lowers utilization - one reason batch sizes at scale are large, and "
                "why pipeline schedules (interleaving, one-forward-one-backward) are "
                "an active engineering area."
            ),
        ),
        Question(
            prompt="Why is 'all-reduce = reduce-scatter + all-gather' more than trivia?",
            options=(
                "It proves an all-reduce is inherently twice as slow as it needs "
                "to be, and that a careful engineer could replace it with a "
                "single cheaper collective to halve the traffic.",
                "It holds only on clusters whose total GPU count is a power of " "two.",
                "It means an all-gather can substitute for an all-reduce in "
                "essentially any algorithm.",
                "It shows ZeRO adds no new communication: sharded DP just runs "
                "the two halves of the all-reduce plain DP already did, with "
                "storage rearranged between them.",
            ),
            answer=3,
            explanation=(
                "Ring implementations of each half move (N-1)/N of the data per GPU - "
                "nearly constant in N - which makes communication budgets predictable "
                "and explains why ZeRO-2's traffic matches plain DP's. The identity "
                "turns 'exotic sharding scheme' into 'the same two collectives, "
                "reordered', which is the level of understanding interviews probe "
                "for."
            ),
        ),
    ),
    "scaling-laws": (
        Question(
            prompt=(
                "Kaplan et al. (2020) and the Chinchilla work (2022) fit scaling laws "
                "to similar data but reached different prescriptions. What actually "
                "changed?"
            ),
            options=(
                "Chinchilla found the early small-model runs used learning-rate "
                "schedules mismatched to their length; refit with matched "
                "schedules, the optimum became equal growth of parameters and "
                "tokens, about 20 to 1.",
                "Chinchilla optimized a fundamentally different loss function "
                "than the one Kaplan's experiments were built around, so the two "
                "sets of exponents were never measuring the same quantity.",
                "Chinchilla trained mostly on code, which happens to favor "
                "smaller models over much larger ones.",
                "Kaplan measured perplexity while Chinchilla measured downstream "
                "accuracy, so the two simply disagree.",
            ),
            answer=0,
            explanation=(
                "The confound: a cosine schedule must land at the final token "
                "(Chapter 7); truncating runs mid-schedule flattered large models. "
                "The corrected fit implied GPT-3-era models were several-fold "
                "undertrained, and the 70B-on-1.4T Chinchilla model beating a 4x "
                "larger one on equal compute was the demonstration. A methodology "
                "detail moved the entire field's budgets."
            ),
        ),
        Question(
            prompt=(
                "Using the approximation that training costs about 6ND floating-point "
                "operations, what does doubling only the model size N at a fixed "
                "compute budget force you to do, and why can that hurt?"
            ),
            options=(
                "Nothing needs to change, because at a fixed budget the training "
                "compute is set by the parameter count N alone and is independent "
                "of how many tokens you stream through.",
                "You must halve the tokens D, and if that pushes you past the "
                "compute-optimal ratio into undertraining, the bigger model ends "
                "up worse than the smaller one would have.",
                "You must double the tokens D as well, holding the "
                "tokens-per-parameter ratio fixed.",
                "You must halve the batch size to keep total compute within the "
                "budget.",
            ),
            answer=1,
            explanation=(
                "C = 6ND makes the tradeoff a seesaw: at fixed C, parameters and "
                "tokens trade off exactly. The isoFLOP curves are U-shaped in that "
                "split - too big means undertrained, too small means underfit - and "
                "the Chinchilla point is the bottom. The 6 decomposes as roughly 2 "
                "FLOPs per parameter for the forward pass and 4 for the backward."
            ),
        ),
        Question(
            prompt=(
                "Llama 3's 8B model trained on about 1,900 tokens per parameter - two "
                "orders of magnitude past Chinchilla's 20. What justifies this?"
            ),
            options=(
                "It was a widely acknowledged overspend that the later Llama "
                "releases quietly walked back once the compute-optimal ratio was "
                "understood more clearly across the field.",
                "The Chinchilla ratio only applies to models larger than about "
                "70B parameters, not to an 8B.",
                "Chinchilla minimizes loss for a fixed training budget with "
                "serving priced at zero; once inference is counted, each served "
                "token re-pays the parameter count, so a smaller overtrained "
                "model wins.",
                "The extra tokens were needed only to cover many additional "
                "languages, not to raise quality.",
            ),
            answer=2,
            explanation=(
                "This is the inference-aware amendment (Sardana et al., 2024): the "
                "optimum shifts toward smaller-overtrained as expected serving volume "
                "grows. It is the visible strategy of the open-weight era, where "
                "models must run on modest hardware. The regime change to name: "
                "training cost is paid once; serving cost is paid per token, forever."
            ),
        ),
        Question(
            prompt=(
                "Fitted scaling laws include an irreducible loss term that no amount "
                "of scale removes. What is it?"
            ),
            options=(
                "It is the accumulated numerical error of low-precision bf16 "
                "arithmetic across the forward and backward passes, which sets a "
                "floor no larger model can drop below.",
                "A weight-decay artifact that a better-tuned optimizer would "
                "remove entirely.",
                "The loss contributed by using a nonzero softmax temperature at "
                "the output layer.",
                "The entropy of the text itself: genuinely unpredictable content "
                "costs bits even for a perfect model, so scaling only shrinks the "
                "reducible gap above it.",
            ),
            answer=3,
            explanation=(
                "In the compression view (Chapter 6), the corpus contains real "
                "information - names, outcomes, arbitrary choices - that no predictor "
                "can anticipate; that content sets a floor measured in bits per "
                "token. Fitted laws routinely estimate this constant, and its "
                "existence is why 'loss goes to zero with enough scale' is a wrong "
                "mental model."
            ),
        ),
        Question(
            prompt=(
                "A team improves its data cleaning pipeline and observes loss well "
                "below what its fitted scaling law predicted at that compute. The "
                "team claims to have 'beaten the scaling law.' What is the sharper "
                "interpretation?"
            ),
            options=(
                "Scaling laws belong to a specific corpus and tokenizer, not to "
                "nature: better data moves the whole curve, so the team is now "
                "riding a different, lower law.",
                "The original law was simply fit to too few points and is "
                "unreliable at this compute scale, so the new number just "
                "corrects a bad extrapolation rather than beating anything.",
                "Beating the predicted loss is impossible and points to "
                "contamination of the evaluation set.",
                "Loss below the fitted line is a sure sign the model has begun "
                "memorizing its data.",
            ),
            answer=0,
            explanation=(
                "The fitted exponents and constants encode the data distribution. "
                "Changing the distribution changes the law - which is precisely why "
                "data work (Chapter 6) is a first-class lever: it moves the curve "
                "that compute then slides you along. Published constants are "
                "estimates for someone else's corpus, not physics."
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
