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
"sft": (
        Question(
            prompt=(
                "A team fine-tunes a base model on 100,000 high-quality demonstrations "
                "that include many facts the base model never saw in pretraining. After "
                "SFT the model states those facts fluently but is wrong about them, and "
                "about nearby questions, more often than before. What best explains this?"
            ),
            options=(
                "SFT clones the behavior of confident answering but cannot install facts "
                "the base model lacked, and fitting those examples raises its tendency to "
                "hallucinate on related questions.",
                "The demonstrations overwrote the model's pretraining knowledge through "
                "catastrophic forgetting, so it lost facts it had previously held "
                "reliably and now confabulates in their place.",
                "The learning rate was too high, so the model memorized the "
                "demonstrations verbatim rather than generalizing from them.",
                "One hundred thousand examples is simply too few to teach new facts; "
                "scaling the SFT set another order of magnitude would resolve the errors.",
            ),
            answer=0,
            explanation=(
                "Gekhman et al. (2024) found that examples introducing new knowledge are "
                "learned slowly, and as the model finally fits them its overall "
                "hallucination rate rises. SFT shapes behavior, not knowledge, so it "
                "teaches the model to answer as confidently when guessing as when it "
                "knows. This is distinct from catastrophic forgetting, which is the "
                "erosion of general capability from over-tuning, not a fact-specific "
                "effect — and adding more of the same out-of-knowledge data makes it "
                "worse, not better."
            ),
        ),
        Question(
            prompt=(
                "Standard SFT computes the loss only on the assistant's tokens, masking "
                "the system and user tokens. What is the main consequence of instead "
                "training on every token in each conversation?"
            ),
            options=(
                "The model spends capacity learning to generate user messages and prompt "
                "boilerplate, biasing it toward parroting inputs and diluting the signal "
                "for producing good responses.",
                "It leaks future assistant tokens back into the earlier prompt positions, "
                "so the held-out loss looks artificially low while the model has secretly "
                "been reading ahead throughout training.",
                "The model can no longer learn when to stop, because the end-of-turn "
                "token would then be excluded from the loss along with the prompt.",
                "Nothing about the trained behavior changes; prompt masking is purely a "
                "compute optimization that skips gradient on some tokens.",
            ),
            answer=0,
            explanation=(
                "The task is to map a prompt to a good completion, so the gradient should "
                "come only from the completion. Training on the prompt teaches the model "
                "to predict user messages, which wastes capacity. The leakage option "
                "confuses loss masking with the causal mask, which already prevents "
                "reading ahead; and the end-of-turn token is part of the completion, so "
                "it stays unmasked precisely so the model learns to stop."
            ),
        ),
        Question(
            prompt=(
                "LIMA aligned a strong base model into a competitive assistant using only "
                "about 1,000 curated demonstrations and no reinforcement learning. Which "
                "conclusion does this most directly support?"
            ),
            options=(
                "Most capability is acquired during pretraining, and SFT mainly selects "
                "format and style, so a small, diverse, high-quality set can be enough to "
                "surface it.",
                "Alignment quality scales smoothly with the number of SFT examples, and "
                "1,000 simply happened to land on an unusually data-efficient operating "
                "point for this model.",
                "A sufficiently strong base model is already an aligned assistant, so the "
                "instruction-tuning step is essentially optional in practice.",
                "Reinforcement learning is strictly inferior to supervised fine-tuning "
                "for alignment, since LIMA matched RLHF-trained systems without using it.",
            ),
            answer=0,
            explanation=(
                "This is the superficial alignment hypothesis (Zhou et al., 2023): "
                "knowledge and skills come from pretraining, and alignment largely "
                "teaches the style in which to expose them. The base model still needs "
                "SFT — it is not aligned by default — and preference optimization "
                "(Chapters 11-12) still adds value beyond demonstrations, so the lesson "
                "is about quality and diversity, not a magic example count."
            ),
        ),
        Question(
            prompt=(
                "A well-tuned instruct model gives rambling, lower-quality answers in "
                "production, even though the identical weights scored well in your offline "
                "eval harness. What is the most likely single cause?"
            ),
            options=(
                "The serving stack wraps prompts in a chat template that differs from the "
                "one used during SFT, so the special role tokens the model depends on are "
                "missing or malformed.",
                "The model was quantized to 4-bit for serving, and low-bit quantization "
                "is widely understood to strip learned instruction-following behavior "
                "out of a fine-tuned model almost entirely.",
                "The decoding temperature is pinned at zero, which collapses generation "
                "into degenerate greedy repetition loops.",
                "The KV cache is silently corrupting earlier tokens as the context grows "
                "over the course of a long generation.",
            ),
            answer=0,
            explanation=(
                "A template mismatch between training and serving is a classic deployment "
                "bug (Chapter 18): the model keys on role tokens like the assistant "
                "marker, and without them it falls back toward base-model continuation. "
                "Quantization rarely erases instruction-following, greedy decoding is a "
                "normal default, and a correct KV cache reproduces full-context "
                "attention exactly."
            ),
        ),
        Question(
            prompt=(
                "You scale SFT by distilling 500,000 demonstrations from a stronger "
                "model. Held-out human ratings of your responses jump, but accuracy on a "
                "factual knowledge benchmark barely moves and dips slightly. What is the "
                "best explanation?"
            ),
            options=(
                "SFT mainly shapes presentation, so distillation reliably improves "
                "fluency, formatting, and confidence — which lift subjective ratings — "
                "without adding the knowledge the benchmark measures.",
                "Distilled data always inherits the teacher model's factual errors, and "
                "propagating those specific mistakes into your own model is the sole "
                "reason the benchmark accuracy declined after fine-tuning.",
                "The knowledge benchmark must be contaminated, so its scores stopped "
                "being meaningful once you fine-tuned on any new data at all.",
                "The rating gain is measurement noise; with 500,000 examples, ratings and "
                "accuracy are guaranteed to move together.",
            ),
            answer=0,
            explanation=(
                "Because SFT surfaces behavior the base model already has, polishing "
                "presentation raises perceived quality without teaching facts — the "
                "'style over substance' trap. Distillation can propagate a teacher's "
                "errors, but framing that as the 'sole' cause and 'always' true overstates "
                "it; and ratings and benchmark accuracy routinely diverge precisely "
                "because they measure different things."
            ),
        ),
    ),

"rlhf": (
        Question(
            prompt=(
                "A reward model trained with the Bradley-Terry objective scores one "
                "response 8.3 and another 2.1. A teammate concludes the first response "
                "is 'about four times as good.' What is the mistake?"
            ),
            options=(
                "The loss constrains only the difference of the two rewards, so the "
                "scale and zero point are arbitrary; higher-versus-lower is meaningful, "
                "but ratios of scores are not.",
                "The outputs are calibrated log-odds that a response is preferred, so "
                "the sound move is to exponentiate the difference of the two scores "
                "rather than divide them, which would place the comparison on a "
                "probability scale instead.",
                "Reward models are trained to regress onto human one-to-ten ratings, so "
                "the two scores are valid, but the ratio only holds once both have been "
                "renormalized onto the unit interval first.",
                "The two scores come from separate forward passes and are therefore not "
                "comparable at all unless the two responses happen to be scored "
                "together within one batch.",
            ),
            answer=0,
            explanation=(
                "The Bradley-Terry loss is a logistic loss on r_w minus r_l, so adding "
                "any constant to every reward leaves both the loss and the downstream "
                "policy update unchanged. The reward is an interval scale with no fixed "
                "zero: the difference r_w minus r_l is a preference logit, but a single "
                "score's magnitude and any ratio of scores carry no meaning."
            ),
        ),
        Question(
            prompt=(
                "In PPO-based RLHF the objective is the reward-model score minus a KL "
                "penalty against a frozen reference model. If you set the KL "
                "coefficient to zero, what happens?"
            ),
            options=(
                "The policy drifts freely toward whatever maximizes the reward model, "
                "which usually means degenerate, non-fluent text that games its blind "
                "spots rather than better answers.",
                "The policy trains faster and reaches a higher true reward, because the "
                "KL term only ever slows convergence toward the reward model's optimum "
                "without changing the quality of the final model it lands on.",
                "Training diverges almost at once, because the advantage estimates go "
                "undefined once there is no reference distribution left to normalize "
                "the policy's updates against.",
                "Little changes in practice, since the reward model already penalizes "
                "low-fluency text on its own, which makes the separate KL term largely "
                "redundant during optimization.",
            ),
            answer=0,
            explanation=(
                "The KL leash keeps the policy near a model that already writes well "
                "and knows things, so it improves within the space of good answers. "
                "Remove it and the policy sprints toward the reward model's highest-"
                "scoring strings, which are rarely fluent. Too large a coefficient is "
                "the opposite failure: the policy barely moves. The advantage estimate "
                "comes from the critic and does not depend on the reference."
            ),
        ),
        Question(
            prompt=(
                "Throughout an RLHF run, the reward-model score on your policy's own "
                "samples rises steadily. What does that tell you about the model's "
                "actual quality?"
            ),
            options=(
                "Little on its own: the proxy score can climb while true quality peaks "
                "then declines, so finding the real optimum needs an independent "
                "yardstick like held-out human evals.",
                "It confirms steady, reliable improvement, because the reward model was "
                "trained specifically to predict human preference and its score is "
                "therefore the best available proxy for the model's true quality.",
                "It signals that the KL penalty has been set too weak, because a "
                "properly regularized run should show the reward-model score plateau "
                "early rather than keep on climbing.",
                "It means the reward model has overfit its own training comparisons, so "
                "the steadily rising score reflects memorization of those pairs rather "
                "than the policy actually improving.",
            ),
            answer=0,
            explanation=(
                "Gao et al. (2023) measured this directly: as the policy moves away "
                "from the reference, the proxy reward rises monotonically while the "
                "gold reward rises, peaks, and falls. The metric you are training on is "
                "exactly the one being compromised, so it cannot audit itself, which is "
                "why over-optimization is caught only with a separate evaluator."
            ),
        ),
        Question(
            prompt=(
                "After RLHF your model's answers are noticeably longer and more hedged, "
                "and its win rate against the SFT model went up. A skeptic asks how "
                "much of the gain is real. Why is that a fair question?"
            ),
            options=(
                "Reward models systematically prefer longer answers, and much of "
                "RLHF's measured gain can be reproduced by a reward that counts only "
                "length, so verbosity inflates the apparent improvement.",
                "Longer answers cost proportionally more tokens to generate and serve, "
                "so even a perfectly genuine quality gain might not be worth the extra "
                "inference expense that all the added verbosity introduces.",
                "The win rates are judged by the very same reward model that was used "
                "during training, so they are entirely circular and cannot demonstrate "
                "any real improvement whatsoever.",
                "RLHF inherently trades response length off against factual accuracy, "
                "so the longer answers are necessarily less accurate and the entire "
                "measured gain must therefore be an illusion.",
            ),
            answer=0,
            explanation=(
                "Singhal et al. (2023) found response length is a dominant factor in "
                "RLHF's gains: a purely length-based reward reproduces most of the "
                "downstream improvement over SFT. Length is thus a form of reward "
                "hacking, which is why serious evaluations control for it (length-"
                "penalized rewards or length-controlled win rates) before crediting a "
                "real quality gain."
            ),
        ),
        Question(
            prompt=(
                "Why does PPO-based RLHF sample fresh responses from the current policy "
                "at each step, instead of reusing a fixed dataset of responses the way "
                "supervised fine-tuning does?"
            ),
            options=(
                "The useful signal is which of the policy's own current outputs are "
                "better, and that region shifts as the policy improves, so on-policy "
                "samples keep the data matched to the current policy.",
                "A fixed dataset is off-policy, and PPO's importance-sampling ratio is "
                "valid only for data drawn from the exact current policy, so any reuse "
                "of earlier samples at all makes the resulting update mathematically "
                "invalid.",
                "Reusing responses would gradually leak the reward model's parameters "
                "into the policy network, eventually causing the two separately trained "
                "models to collapse into one.",
                "SFT reuses its data only because its targets are human-written, and "
                "PPO could reuse a fixed dataset equally well but samples fresh "
                "responses purely to increase output diversity.",
            ),
            answer=0,
            explanation=(
                "Reinforcement learning optimizes the policy's own output "
                "distribution, so as the policy moves, stale samples describe a policy "
                "you no longer have. PPO does use a clipped importance-sampling ratio "
                "and can safely reuse each batch for a few inner epochs, so the "
                "'exact current policy or nothing' framing overstates the constraint, "
                "but the samples must stay close to the current policy."
            ),
        ),
    ),

"preference-optimization": (
        Question(
            prompt=(
                "DPO is often described as training with 'no reward model.' In what "
                "sense does a reward still exist in the DPO objective?"
            ),
            options=(
                "There is no reward in any form; DPO simply maximizes the likelihood "
                "of the chosen response, exactly like supervised fine-tuning "
                "restricted to the winning answer in each preference pair.",
                "The reward is implicit: beta times the log-ratio of the policy to the "
                "reference, which DPO fits through the Bradley-Terry likelihood.",
                "A reward model trained separately is loaded and frozen, so DPO skips "
                "the cost of fitting a new one but still scores each candidate "
                "response with it before every gradient update.",
                "The reward is the KL divergence between the policy and the reference, "
                "and DPO maximizes it to keep the sampled responses diverse.",
            ),
            answer=1,
            explanation=(
                "DPO reparameterizes the RLHF reward as beta*log(pi/pi_ref). Because "
                "the optimal RLHF policy is a closed-form function of the reward, you "
                "can invert it and express the reward through the policy, then plug "
                "that into Bradley-Terry so the partition function cancels. The reward "
                "never disappears; it is just never materialized as a separate network."
            ),
        ),
        Question(
            prompt=(
                "Why does DPO keep a frozen reference model in every term of its loss, "
                "rather than simply raising the probability of chosen responses and "
                "lowering rejected ones?"
            ),
            options=(
                "The reference supplies the tokenizer, vocabulary, and embedding table "
                "the policy depends on, so without it the per-token log-probabilities "
                "in the loss would be computed over mismatched output spaces.",
                "The reference is only a warm start used to initialize the policy, and "
                "it is discarded after the first gradient step, so it never actually "
                "appears in the loss being optimized.",
                "The reference lets DPO skip the supervised fine-tuning stage entirely, "
                "since it already encodes the demonstrations the policy would "
                "otherwise need to imitate first.",
                "The log-ratio against the reference is the KL leash: it bounds how far "
                "the policy may drift, so quality does not collapse while chasing a "
                "wider preference margin.",
            ),
            answer=3,
            explanation=(
                "The reference plays the role PPO's KL penalty played: beta*log(pi/"
                "pi_ref) charges the policy for moving away from the reference. Remove "
                "it and the loss rewards inflating the winner's probability without "
                "bound, which wrecks fluency. It is an anchor, not an initializer, and "
                "it is queried on every batch."
            ),
        ),
        Question(
            prompt=(
                "Users report that a DPO-tuned model gives noticeably longer answers "
                "than the SFT model it started from, without a clear gain in quality. "
                "What is the most likely cause?"
            ),
            options=(
                "The KL penalty term in DPO scales with sequence length, which "
                "pressures the policy to emit extra tokens on every response just to "
                "keep its divergence from the reference model satisfied.",
                "DPO raises the decoding temperature as a side effect of optimizing "
                "the preference margin, flattening the next-token distribution so "
                "generations tend to run on longer than before.",
                "DPO sums per-token log-probabilities, so a longer rejected response is "
                "easier to push down; the model learns that extra length looks "
                "preferred.",
                "The frozen reference model has a shorter context window than the "
                "policy, so DPO compensates by padding each response until the two "
                "windows are the same size.",
            ),
            answer=2,
            explanation=(
                "This is DPO's well-known length bias: because the implicit reward is a "
                "sum of token log-probabilities, response length leaks into the signal "
                "as a confound. Length-normalizing the reward (as SimPO does) or "
                "balancing chosen/rejected lengths in the data removes the artifact; "
                "raw win-rate gains from longer outputs are usually not real quality."
            ),
        ),
        Question(
            prompt=(
                "KTO differs from DPO most fundamentally in which respect?"
            ),
            options=(
                "It runs an on-policy sampling loop like PPO, regenerating fresh "
                "responses from the current model at every step instead of reading "
                "from a fixed dataset collected in advance.",
                "It removes the reference model and normalizes the reward by response "
                "length, which makes each update cheaper to compute than DPO's "
                "reference-anchored objective.",
                "It merges supervised fine-tuning and preference optimization into a "
                "single training stage, so no separate SFT run is required before "
                "alignment begins.",
                "It learns from unpaired responses labeled individually as desirable or "
                "undesirable, rather than from matched chosen-versus-rejected pairs.",
            ),
            answer=3,
            explanation=(
                "KTO's signature is dropping the pairing requirement: grounded in "
                "prospect theory's asymmetric weighting of gains and losses, it trains "
                "on lone good/bad labels. That matters practically because unpaired "
                "binary feedback (thumbs up/down) is far more abundant than curated "
                "pairs. Merging SFT with preference is ORPO; reference-free length "
                "normalization is SimPO; on-policy sampling is PPO."
            ),
        ),
        Question(
            prompt=(
                "Careful head-to-head studies of DPO and PPO have complicated the claim "
                "that one is simply better. What is the best summary of current "
                "understanding?"
            ),
            options=(
                "DPO is strictly superior because it optimizes exactly the same "
                "objective as PPO but with lower gradient variance, so PPO offers no "
                "measurable advantage at any compute budget you might pick.",
                "Whether training uses on-policy data matters more than the loss form: "
                "well-tuned PPO and iterative on-policy DPO both beat vanilla "
                "off-policy DPO, and data quality dominates the algorithm choice.",
                "PPO is always better because reinforcement learning actively explores "
                "the output space, whereas DPO can only imitate the fixed preference "
                "dataset it was handed at the start of training.",
                "The two methods are mathematically identical in the infinite-data "
                "limit, so any measured gap between them is attributable entirely to "
                "random seeds and evaluation noise.",
            ),
            answer=1,
            explanation=(
                "The empirical picture is that on-policy sampling and preference-data "
                "quality are the load-bearing factors. Vanilla DPO is off-policy and "
                "can drift off the data it learned from; iterative/online DPO recovers "
                "much of PPO's edge without the RL machinery. PPO can still win on hard "
                "tasks like code, but the algorithm name is not the main lever."
            ),
        ),
        Question(
            prompt=(
                "In Constitutional AI, what specifically replaces the human labeler, and "
                "what is the main risk it introduces?"
            ),
            options=(
                "A second, larger reward model replaces the annotator, and the main "
                "risk is that it becomes too expensive to train, which makes the whole "
                "method impractical below frontier scale.",
                "A written set of principles the model applies to critique, revise, and "
                "rank its own outputs; the risk is that the judge's biases and "
                "gameability leak into the preference signal.",
                "A retrieval system that pulls ground-truth answers from a curated "
                "knowledge base, and the main risk is that stale documents yield "
                "outdated or subtly incorrect preferences.",
                "Random pairing of responses with fixed heuristic scores, and the main "
                "risk is the high variance of that signal, which destabilizes the "
                "reinforcement-learning loop downstream.",
            ),
            answer=1,
            explanation=(
                "Constitutional AI substitutes a short written constitution for the "
                "human annotator: the model self-critiques and revises against it "
                "(supervised phase) and judges its own samples against it (preference "
                "phase, the RLAIF idea). The signal scales as cheaply as inference, but "
                "it inherits the judge's blind spots and can be reward-hacked to satisfy "
                "the letter of the rules, which is why scaling trustworthy oversight is "
                "still open."
            ),
        ),
    ),

"peft": (
        Question(
            prompt=(
                "A 7B model that infers comfortably on one GPU runs out of memory the "
                "moment you full-fine-tune it. LoRA fixes this. What is the primary "
                "thing LoRA removes from the memory bill?"
            ),
            options=(
                "The gradients, fp32 master copy, and two AdamW moments for the "
                "billions of frozen weights, since a frozen tensor needs none of "
                "them and only the tiny adapter carries optimizer state.",
                "The frozen base weights themselves, since a low-rank adapter can "
                "reconstruct them from its two factors on the fly, so the full-size "
                "matrix never has to sit resident in GPU memory at any point in the "
                "run.",
                "The stored activations for the backward pass, which LoRA "
                "recomputes tile by tile instead of keeping, the way gradient "
                "checkpointing does.",
                "The key-value cache, which full fine-tuning materializes for "
                "every training position but LoRA is able to stream lazily from "
                "host memory.",
            ),
            answer=0,
            explanation=(
                "Mixed-precision AdamW costs about 16 bytes per parameter, and 12 of "
                "them are optimizer state (Chapter 8). Freezing the base means those "
                "billions of weights need no gradient, master copy, or moments; only "
                "the adapter's few million parameters are taxed. The base weights are "
                "still held, just once and read-only, shared across every task."
            ),
        ),
        Question(
            prompt=(
                "An interviewer asks whether adding LoRA to a deployed model slows "
                "down inference. What is the accurate answer?"
            ),
            options=(
                "Yes, slightly and unavoidably: every adapted layer runs one extra "
                "low-rank matmul per token, a small constant tax you pay on all "
                "traffic once the adapter is attached, and no amount of fusing or "
                "compilation can remove it because the factors stay separate.",
                "No, if you merge the adapter: folding BA into the base weight "
                "yields an ordinary matrix that runs at base speed, and you keep "
                "it separate only when you deliberately want to serve many "
                "adapters at once.",
                "Yes, in proportion to the rank r, because the model must "
                "concatenate the r extra dimensions onto every hidden state before "
                "the next layer can consume it.",
                "No, and it actually speeds inference up, because the low-rank "
                "factors shrink the effective weight matrices the hardware has to "
                "stream from memory each step.",
            ),
            answer=1,
            explanation=(
                "Because W0 + (alpha/r)BA is just another weight matrix of the same "
                "shape, a merged LoRA model is bit-for-bit as fast as the base. The "
                "extra matmul only exists if you keep the adapter unmerged, which is a "
                "deliberate choice for multi-adapter serving, not an inherent cost."
            ),
        ),
        Question(
            prompt=(
                "LoRA constrains the weight update to a rank-r product BA. What is "
                "the empirical justification for expecting that to work?"
            ),
            options=(
                "Pretrained weight matrices are already close to low rank, so any "
                "full-rank update would mostly land in directions the base model "
                "cannot represent anyway and would be projected out and discarded "
                "the moment the next layer consumes it.",
                "Most weights in a trained transformer are effectively zero, so the "
                "update only needs to touch the sparse nonzero entries, which a "
                "low-rank factorization captures cheaply.",
                "The change fine-tuning makes to a pretrained model has a low "
                "intrinsic dimension: adaptation moves the weights along a handful "
                "of directions, not across the full parameter space.",
                "Attention itself is a low-rank operation, so confining the update "
                "to low rank simply matches the rank the attention mechanism was "
                "always going to impose downstream.",
            ),
            answer=2,
            explanation=(
                "The bet is about the update, not the weights. Aghajanyan et al. (2021) "
                "showed fine-tuning has a small intrinsic dimension, tuning RoBERTa to "
                "most of its performance through a few hundred projected parameters, "
                "and that this dimension shrinks as models grow. LoRA operationalizes "
                "that: force the update through a rank-r bottleneck."
            ),
        ),
        Question(
            prompt=(
                "A colleague says QLoRA 'trains the model in 4-bit.' Where is that "
                "wrong, and what actually happens?"
            ),
            options=(
                "The adapters are the 4-bit part; QLoRA keeps the base in bf16 and "
                "learns tiny 4-bit factors on top, which is what makes the trained "
                "state small enough to fit.",
                "Nothing is wrong: QLoRA performs quantization-aware training, "
                "learning weights that are specifically robust to being served at "
                "4-bit precision afterward.",
                "The base is stored in 4-bit NF4 but dequantized to bf16 for each "
                "matmul; gradients flow through those bf16 values into bf16 "
                "adapters, and the quantized weights are never updated.",
                "The 4 bits refer to the optimizer, which QLoRA quantizes down to "
                "4-bit moment states through its paged optimizers, while the base "
                "weights and the adapters alike stay in full bf16 precision "
                "throughout the entire run.",
            ),
            answer=2,
            explanation=(
                "QLoRA keeps 4 bits only to store the frozen base cheaply (Dettmers et "
                "al., 2023). Each weight is dequantized to bf16 for its multiply, and "
                "learning happens entirely in the bf16 adapters. Training a model to be "
                "good at 4-bit inference is quantization-aware training, a different "
                "problem (Chapter 16)."
            ),
        ),
        Question(
            prompt=(
                "You must serve fifty customer-specific fine-tunes of one 13B model "
                "under a tight GPU budget. Why keep the LoRA adapters unmerged, "
                "despite the small per-token overhead that adds?"
            ),
            options=(
                "Unmerged adapters let one resident base copy back all fifty tenants, "
                "and a batch mixing different adapters can share the base matmul "
                "while each request applies its own low-rank term.",
                "Merging is numerically lossy at 13B, so keeping adapters separate "
                "is the only way to preserve each fine-tune's accuracy when many "
                "of them share the same underlying weights.",
                "Unmerged adapters can be quantized independently of the base, "
                "which is what actually lets fifty of them fit, whereas a merged "
                "model can no longer be quantized at all.",
                "Merged models cannot be swapped without a full reload, so each of "
                "the fifty tenants would require its own dedicated GPU to hold a "
                "separate merged copy of the weights.",
            ),
            answer=0,
            explanation=(
                "Merging gives zero latency but a merged model serves exactly one "
                "task, so a mixed batch cannot share a forward pass. Systems like "
                "S-LoRA (Sheng et al., 2024) keep adapters separate, compute the base "
                "matmul once per batch, and apply each request's BA with a custom "
                "kernel, serving thousands of adapters near base throughput. The trade "
                "is single-tenant latency versus many-tenant throughput."
            ),
        ),
    ),
"decoding": (
        Question(
            prompt=(
                "A summarization model decoded with beam search keeps emitting the "
                "same clause over and over until truncated. What is the root cause?"
            ),
            options=(
                "Beam search maximizes sequence probability, and the "
                "maximum-probability continuation of open-ended text is itself "
                "degenerate; fluent writing keeps choosing locally improbable tokens.",
                "The beam width is set too small, so the search collapses onto a single "
                "hypothesis and cannot escape the loop; widening the beam explores more "
                "candidates and reliably breaks the repeated clause apart.",
                "The model is under-trained, so its next-token distribution has not yet "
                "sharpened, and the repetition reflects a genuine gap in its "
                "language-modeling ability rather than a property of the decoder.",
                "Beam search draws from the low-probability tail of the distribution, "
                "and that tail happens to concentrate its mass on tokens the model has "
                "already produced earlier in the sequence.",
            ),
            answer=0,
            explanation=(
                "Degeneration is a property of the decoding objective, not the model or "
                "the beam width: maximizing likelihood chases the mode, and for "
                "open-ended text the mode is a repetitive, over-safe string. Human text "
                "carries a steady stream of surprisal, which is why sampling-based "
                "decoders beat search here. Widening the beam typically makes it worse, "
                "not better, because it finds even higher-probability (more degenerate) "
                "sequences."
            ),
        ),
        Question(
            prompt=(
                "You raise the sampling temperature from 0.7 to 1.3 to make a factual "
                "assistant 'more creative,' and answers get less accurate. Why?"
            ),
            options=(
                "Temperature flattens the distribution, shifting mass onto the "
                "runners-up, so the marginal tokens truncation meant to suppress get "
                "sampled more.",
                "Temperature reorders the tokens once it climbs above 1.0, so a token "
                "that was previously unlikely can become the new argmax and then be "
                "selected by the greedy step of the decoder.",
                "Higher temperature increases the number of forward passes performed per "
                "token, and the extra passes compound floating-point error in the logits "
                "before the softmax is applied.",
                "Temperature above 1.0 effectively disables top-p truncation, because "
                "the flattened cumulative mass can no longer climb to the threshold that "
                "the nucleus needs to close.",
            ),
            answer=0,
            explanation=(
                "Temperature is a monotonic rescaling of the logits: it never reorders "
                "tokens, so the argmax (and greedy decoding) is unchanged, but it does "
                "shift mass toward the tail. It is a diversity knob, not a correctness "
                "knob. Truncation still runs, but now operates on a flatter "
                "distribution that admits more marginal tokens."
            ),
        ),
        Question(
            prompt=(
                "How does min-p sampling differ from top-p (nucleus) sampling in what "
                "it keeps?"
            ),
            options=(
                "Min-p keeps every token at least a fixed fraction of the top token's "
                "probability, so its cutoff scales with the peak; top-p keeps the "
                "smallest set reaching a fixed cumulative mass.",
                "Min-p keeps a fixed minimum count of tokens no matter what their "
                "probabilities are, while top-p keeps whatever number of tokens is "
                "needed to reach a fixed cumulative probability mass from the top.",
                "Min-p discards any token below an absolute probability floor chosen "
                "before decoding begins, while top-p sets that same floor dynamically "
                "from the running entropy of the current distribution.",
                "Min-p keeps the tokens whose surprisal sits closest to the "
                "distribution's entropy, whereas top-p simply keeps the most probable "
                "tokens until their mass reaches the chosen threshold.",
            ),
            answer=0,
            explanation=(
                "Min-p's threshold is relative to the maximum probability, so a peaked "
                "distribution admits few tokens and a flat one admits many — which "
                "makes it hold up at high temperature where top-p can let a flattened "
                "tail leak in. The 'surprisal near entropy' rule describes locally "
                "typical sampling, a different method entirely."
            ),
        ),
        Question(
            prompt=(
                "A reasoning model that solves math problems performs best with "
                "near-greedy decoding (temperature near 0), while a story generator "
                "wants temperature around 1 with top-p. What principle reconciles this?"
            ),
            options=(
                "Decoding is a per-task choice: reasoning wants the single most-likely "
                "derivation, so diversity only adds chances to leave a correct path, "
                "while open-ended writing needs the variance sampling supplies.",
                "Reasoning models are calibrated differently during training, so their "
                "logits already bake in an effective temperature, and any external "
                "temperature you apply on top of that ends up double-counting it.",
                "Greedy decoding is simply always optimal for accuracy; the story "
                "generator only raises temperature because its outputs are never graded "
                "for correctness and so nothing is lost by adding randomness.",
                "Reasoning benefits from low temperature mainly because a shorter output "
                "leaves fewer tokens for a repetition penalty to act on, so the risk of "
                "falling into a degenerate loop is correspondingly lower.",
            ),
            answer=0,
            explanation=(
                "The same weights serve both tasks; only the decoder changes. When "
                "there is a correct continuation to find (a derivation, code, an "
                "extraction), you want the model's best guess and sampling mostly adds "
                "risk. When many outputs are equally valid (prose), sampling supplies "
                "the human-like variance that maximization destroys."
            ),
        ),
        Question(
            prompt=(
                "Why does constrained decoding against a grammar guarantee valid output "
                "where a well-tuned sampler only makes it likely?"
            ),
            options=(
                "It sets the logits of every token the grammar forbids to negative "
                "infinity before sampling, so an illegal token has zero probability and "
                "cannot be drawn at any temperature.",
                "It runs the ordinary sampler repeatedly and rejects any completion that "
                "fails to parse against the grammar, retrying from the start until a "
                "syntactically valid sample finally appears.",
                "It fine-tunes the model on many examples of the target format ahead of "
                "time, so that at inference the distribution's mass has concentrated "
                "almost entirely onto the legal tokens.",
                "It drops the temperature to zero on the constrained fields so that the "
                "argmax is taken at every step, and the single most probable token is "
                "always a grammar-legal one.",
            ),
            answer=0,
            explanation=(
                "Constrained decoding masks the distribution at each step, so only "
                "grammar-legal tokens have nonzero probability — validity is enforced "
                "by construction, not by chance or retries. The subtlety is that a hard "
                "mask can distort intent: if the model wanted a forbidden token, you get "
                "a valid output it would rate poorly, which is why constraint and "
                "quality are separate concerns (Chapter 20)."
            ),
        ),
    ),

"inference-optimization": (
        Question(
            prompt=(
                "A model generates at only a few tokens per second on a GPU whose "
                "arithmetic units are nearly idle, yet the same GPU processes a long "
                "prompt almost instantly. What explains the gap?"
            ),
            options=(
                "Prefill runs the prompt through in parallel with high arithmetic "
                "intensity, so it is compute-bound; decode produces one token per "
                "pass with almost no math per byte, so it is memory-bandwidth-bound.",
                "Prefill caches its intermediate results while decode recomputes "
                "the entire attention matrix from scratch at every step, which is "
                "what stalls generation.",
                "Decode runs at a higher numerical precision than prefill to keep "
                "sampling stable, and the wider arithmetic is what throttles the "
                "token rate.",
                "Prefill is dispatched to the GPU's dedicated tensor cores while "
                "decode falls back to the slower general-purpose CUDA cores, so the "
                "two phases run on entirely different hardware units with very "
                "different peak throughput.",
            ),
            answer=0,
            explanation=(
                "The two phases have opposite arithmetic intensity. Prefill reuses "
                "each weight across all prompt tokens at once, saturating the "
                "arithmetic units; decode reads the whole model (and KV cache) to "
                "emit a single token, so it is limited by how fast memory can be "
                "read, not by FLOPs. This is why generation speed tracks memory "
                "bandwidth divided by model size, and why the arithmetic units sit "
                "idle during decode."
            ),
        ),
        Question(
            prompt=(
                "Why does serving a 128k-token context slow down every decode step, "
                "not just the initial prefill?"
            ),
            options=(
                "The rotary position embeddings must be recomputed across the entire "
                "context at every step, and because that work grows with the sequence "
                "length, each added token makes refreshing the earlier positions "
                "steadily more expensive.",
                "Longer contexts push the softmax toward saturation, so each step "
                "needs extra iterations to keep the attention weights numerically "
                "stable.",
                "Each step streams the entire KV cache out of memory to attend over "
                "it, and the cache grows linearly with context, so a bigger cache "
                "means more bytes moved per token.",
                "The context no longer fits in the GPU's SRAM, forcing every step "
                "to page attention scores to disk and back before it can continue.",
            ),
            answer=2,
            explanation=(
                "Decode is memory-bandwidth-bound, and the KV cache is memory that "
                "must be read on every step. Its size grows linearly with sequence "
                "length (and batch), so long context raises the per-token byte count "
                "for the whole generation, not once. The cache — not the weights — "
                "is usually what caps context length and batch size, which is why "
                "GQA, sliding windows, and MLA all target its size."
            ),
        ),
        Question(
            prompt=(
                "PagedAttention is often summarized as 'virtual memory for the KV "
                "cache.' What concrete problem does it actually solve?"
            ),
            options=(
                "It compresses each cached key and value into fewer bits, so the "
                "same context fits in a smaller memory footprint on every step.",
                "It lets a sequence's KV cache live in non-contiguous fixed-size "
                "blocks mapped through a table, eliminating the wasted memory of "
                "reserving a contiguous max-length region per request.",
                "It moves rarely-attended cache blocks out to CPU memory and pages "
                "them back on demand, so the GPU holds only the active portion of "
                "each context.",
                "It reorders the tokens within each request so that attention reads "
                "the cache in a single sequential sweep, converting the scattered "
                "random-access pattern into a streaming read that the memory system "
                "can prefetch far more efficiently.",
            ),
            answer=1,
            explanation=(
                "Static allocation reserves contiguous memory for each request's "
                "worst-case length, most of which goes unused — severe internal "
                "fragmentation that shrinks the batch. PagedAttention allocates "
                "small blocks on demand and maps logical to physical blocks through "
                "a per-sequence block table, so no request needs contiguous space "
                "and shared prefixes can share blocks. The freed memory becomes a "
                "larger batch, which is the throughput win — it does not compress or "
                "swap the cache."
            ),
        ),
        Question(
            prompt=(
                "In speculative decoding a small draft model proposes tokens that a "
                "large target verifies. A candidate calls it 'an approximation that "
                "trades a little quality for speed.' What is wrong with that?"
            ),
            options=(
                "It is not an approximation: a modified rejection-sampling step "
                "makes the output distribution identical to sampling from the "
                "target alone, so the speedup comes purely from fewer target "
                "forward passes.",
                "It is not an approximation because the draft is distilled to match "
                "the target's outputs exactly, so after enough training the two "
                "models agree token for token and the correction step in the "
                "algorithm never actually has to fire.",
                "It is an approximation, but the error is bounded and small, so in "
                "practice the quality loss is negligible for most prompts.",
                "It is an approximation only when sampling at nonzero temperature; "
                "under greedy decoding the draft and target are guaranteed to "
                "produce the same tokens.",
            ),
            answer=0,
            explanation=(
                "Speculative decoding is exact. The acceptance test plus resampling "
                "of the first rejected token provably reproduce the target's own "
                "distribution, so the generated text is what the target would have "
                "produced. The benefit is doing several tokens' worth of progress "
                "per expensive target pass, since verifying k proposed tokens is a "
                "parallel, prefill-like operation on a target that had spare compute."
            ),
        ),
        Question(
            prompt=(
                "You add speculative decoding with a strong, accurate draft model and "
                "see almost no speedup. What is the most likely reason?"
            ),
            options=(
                "The draft's high acceptance rate forces the target to re-verify "
                "each accepted token a second time before committing it, and that "
                "duplicated verification pass cancels out most of the tokens that "
                "speculation was supposed to save.",
                "The draft is accurate because it is large, so its per-token cost is "
                "close to the target's and the verification pass no longer runs in "
                "parallel with anything cheaper.",
                "A strong draft raises the KV cache pressure so much that the batch "
                "shrinks, and the smaller batch erases the gain from speculation.",
                "Accurate drafts push the acceptance rate so high that the rejection "
                "sampler rarely fires, and without rejections the target cannot "
                "advance more than one token per pass.",
            ),
            answer=1,
            explanation=(
                "The speedup is roughly the expected accepted tokens per step "
                "divided by the combined draft-plus-target cost. A draft strong "
                "enough to be accurate is often nearly as expensive as the target, "
                "so you pay almost a full model per proposed token and the arithmetic "
                "stops favoring you. The sweet spot is a draft that is both cheap and "
                "reasonably aligned — high acceptance is only half the equation."
            ),
        ),
        Question(
            prompt=(
                "FlashAttention is frequently described as making attention 'cheaper.' "
                "What does it actually reduce, and why is that the thing that matters?"
            ),
            options=(
                "It lowers attention's asymptotic cost below the quadratic n-squared "
                "by approximating far-apart interactions, which is what removes the "
                "long-context bottleneck.",
                "It halves the floating-point operations by fusing the query-key and "
                "value multiplies into one kernel, so the arithmetic itself is what "
                "gets smaller.",
                "It reduces memory traffic by computing the exact attention in tiles "
                "that stay in fast on-chip SRAM, never writing the n-by-n score "
                "matrix to slow HBM — the FLOP count is unchanged.",
                "It shrinks the KV cache by storing attention scores instead of keys "
                "and values, so less data has to be streamed on each decode step.",
            ),
            answer=2,
            explanation=(
                "FlashAttention is exact and still O(n-squared) in compute; it does "
                "the same (or slightly more) arithmetic. Standard attention is slow "
                "because it materializes and re-reads the n-by-n score matrix in HBM, "
                "and attention is memory-bound. Tiling Q, K, and V through SRAM with "
                "an online softmax cuts that traffic, and recomputing tiles in the "
                "backward pass avoids storing the matrix at all. The win is bytes "
                "moved, not FLOPs — the same reason decode is slow in the first place."
            ),
        ),
    ),

"quantization": (
        Question(
            prompt=(
                "Naive round-to-nearest quantization takes a large language model to "
                "4 bits and its outputs turn to garbage, yet the same recipe barely "
                "dents a vision CNN of similar size. What is the root cause specific "
                "to the LLM?"
            ),
            options=(
                "A few activation dimensions carry magnitudes far larger than the "
                "rest, so any shared scale wide enough to cover them leaves ordinary "
                "values with almost no resolution.",
                "Language models pack many more parameters into each layer, so the "
                "same fixed number of quantization rungs is spread over a wider "
                "spread of weights and every rung ends up representing a coarser "
                "step than it would in a smaller convolutional layer.",
                "The output softmax runs over a vocabulary of tens of thousands of "
                "tokens, and that layer is so numerically delicate that the tiny "
                "rounding error introduced in the weights is amplified into "
                "confidently wrong next-token predictions.",
                "Transformer weights follow a much heavier-tailed distribution than "
                "convolutional filters, so a uniformly spaced integer grid misplaces "
                "the bulk of the weights no matter which scale and zero-point you "
                "choose for the tensor.",
            ),
            answer=0,
            explanation=(
                "The culprit is emergent activation outliers (Dettmers et al., 2022): "
                "past a few billion parameters, a small set of feature dimensions blow "
                "up in magnitude, and a per-tensor scale stretched to represent them "
                "leaves everything else with almost no resolution. The heavy-tailed "
                "weights option is the tempting near-miss — weight shape does motivate "
                "formats like NF4 — but it is not what breaks naive 4-bit; the "
                "activation outliers are, which is why LLM.int8(), SmoothQuant, and AWQ "
                "all target them."
            ),
        ),
        Question(
            prompt=(
                "SmoothQuant and AWQ both fight activation outliers, but their "
                "mechanisms differ. Which pair of descriptions is correct?"
            ),
            options=(
                "SmoothQuant migrates outlier magnitude from the activations into "
                "the weights with an equivalent per-channel rescale; AWQ scales up "
                "the roughly 1% of weights that multiply large activations so they "
                "round with less error.",
                "SmoothQuant keeps the outlier dimensions in 16-bit while running the "
                "rest in INT8; AWQ walks through the weight columns in order and uses "
                "second-order Hessian information to correct the weights it has not "
                "quantized yet.",
                "SmoothQuant learns its quantization scales by backpropagation during "
                "a short fine-tuning pass over calibration data; AWQ instead solves "
                "for the scales analytically from activation statistics without doing "
                "any gradient updates at all.",
                "SmoothQuant reshapes the weight distribution to be approximately "
                "Gaussian before rounding so a uniform grid fits it; AWQ clips the "
                "top 1% of activation magnitudes so a narrower shared scale can cover "
                "everything that remains.",
            ),
            answer=0,
            explanation=(
                "SmoothQuant (Xiao et al., 2023) applies a per-channel constant that "
                "divides the hard-to-quantize activations and multiplies the "
                "corresponding weight columns, an exactly equivalent rewrite that moves "
                "the difficulty to the easy-to-quantize weights. AWQ (Lin et al., 2024) "
                "identifies salient weights from activation statistics and scales them "
                "to reduce their rounding error. The tempting scramble mixes up the "
                "method names: the keep-outliers-in-16-bit trick is LLM.int8() and "
                "the Hessian correction is GPTQ, neither of which is SmoothQuant or AWQ."
            ),
        ),
        Question(
            prompt=(
                "You quantize a 13B model weight-only to 4 bits and measure a large "
                "speedup on decoding long responses, but almost none on processing "
                "long prompts. Why?"
            ),
            options=(
                "Prefill is compute-bound and still multiplies in 16-bit because "
                "weight-only quantization dequantizes each weight first; decode is "
                "memory-bandwidth-bound, so fewer bytes per weight speed it up "
                "directly.",
                "The prompt is processed one token at a time exactly as decoding is, "
                "so both phases ought to speed up by the same factor; a flat prefill "
                "number therefore points to a benchmarking mistake rather than "
                "anything intrinsic to weight-only quantization.",
                "Quantization only shrinks the weights, and prompt processing time is "
                "dominated by building the KV cache rather than by the weight "
                "matmuls, so compressing the weights leaves the prefill phase almost "
                "entirely untouched no matter how few bits you use.",
                "The custom 4-bit kernels are tuned only for batch size one, which "
                "decoding uses, whereas prefill effectively batches every prompt "
                "token together and falls back onto the slower general-purpose "
                "16-bit code path for the matrix multiplies.",
            ),
            answer=0,
            explanation=(
                "Weight-only 4-bit stores weights in 4 bits but dequantizes them to "
                "16-bit for the actual matmul, so it buys memory and decode bandwidth, "
                "not arithmetic. Prefill is compute-bound and parallel, so it sees "
                "little benefit; decode streams every weight from memory per token, so "
                "halving the bytes roughly halves the time. To speed prefill you must "
                "quantize activations too (W8A8) and run genuine integer matmuls — a "
                "different, harder regime because of outliers."
            ),
        ),
        Question(
            prompt=(
                "In a GGUF 'Q4_K_M' build for llama.cpp, what does the K-quant scheme "
                "actually do, and why does it beat rounding every weight to a flat 4 "
                "bits?"
            ),
            options=(
                "It mixes precision within the model, spending more bits on the most "
                "sensitive tensors and fewer on the rest, so the bit budget lands "
                "where degradation hurts most.",
                "It applies a short round of quantization-aware training to the "
                "checkpoint so the weights can adapt to the coarse 4-bit grid before "
                "the file is written, which a purely post-training round of the same "
                "average width has no opportunity to do.",
                "It stores the weights in a 4-bit floating-point type whose rungs are "
                "spaced to match the roughly-Gaussian shape of the weights, putting "
                "more levels where the weights cluster instead of on the uniform "
                "integer grid a flat round would use.",
                "It leaves the weights at a flat 4 bits but keeps the activations in "
                "8-bit integers so the matrix multiplies execute in INT8, recovering "
                "the compute speedup that a weight-only flat-4-bit scheme leaves "
                "sitting on the table.",
            ),
            answer=0,
            explanation=(
                "K-quants use mixed precision across tensors — more bits for the "
                "sensitive parts (such as attention projections), fewer elsewhere — so "
                "the same average bit-width preserves more quality than a uniform round. "
                "The Gaussian-spaced-rungs option describes NF4 (bitsandbytes/QLoRA), a "
                "real but different idea, which makes it the tempting near-miss; and "
                "K-quants are post-training, not QAT, and remain weight-only rather than "
                "quantizing activations."
            ),
        ),
        Question(
            prompt=(
                "An interviewer asks whether QLoRA is a form of quantization-aware "
                "training. What is the accurate answer?"
            ),
            options=(
                "No: QLoRA stores a frozen base in 4-bit only to save memory and "
                "dequantizes each weight to 16-bit for its matmul, so the base is "
                "never actually trained at 4 bits.",
                "Yes: QLoRA rounds the base weights on the forward pass and lets the "
                "gradient flow back through that rounding into the base weights "
                "themselves, which is exactly the straight-through fake-quantization "
                "mechanism that defines quantization-aware training.",
                "Partly: the base does stay frozen, but the low-rank adapters are "
                "themselves trained as fake-quantized 4-bit tensors, so QLoRA is "
                "quantization-aware training applied to the adapters while the base "
                "is simply held in reduced precision alongside them.",
                "No, but for a different reason: QLoRA actually quantizes the "
                "activations rather than the weights, making it a weight-and-"
                "activation post-training method, whereas quantization-aware training "
                "is defined by learning the weights at low precision.",
            ),
            answer=0,
            explanation=(
                "QLoRA (Dettmers et al., 2023) keeps 4-bit NF4 purely to fit the frozen "
                "base in memory; every weight is dequantized to 16-bit for its multiply, "
                "and all learning lives in 16-bit LoRA adapters that are themselves not "
                "quantized. QAT, by contrast, trains the model to be accurate at "
                "low-precision inference via fake quantization on the forward pass. The "
                "trap is the claim that the LoRA adapters are themselves fake-quantized "
                "4-bit tensors: they are trained in full precision, not at 4 bits."
            ),
        ),
    ),

"serving-systems": (
        Question(
            prompt=(
                "Raising a replica's batch size lowers the cost of each generated "
                "token but makes individual users wait longer per token. What is the "
                "correct explanation of both effects at once?"
            ),
            options=(
                "A GPU rents at a fixed hourly price, so cost per token is that price "
                "over throughput; batching lifts throughput toward the compute-bound "
                "ceiling, while each user now shares every decode step and waits "
                "longer.",
                "Bigger batches perform more arithmetic per token, and that is what "
                "simultaneously raises throughput and raises each individual user's "
                "latency; the cost falls only because the GPU's fixed startup overhead "
                "is now amortized over a great deal more work.",
                "Bigger batches shrink the KV cache that each request needs, which is "
                "what lowers the cost, and latency rises because the shared cache must "
                "then be paged out to slower host memory.",
                "Bigger batches let later requests skip their prefill phase entirely, "
                "which is what cuts the cost, and latency rises from the time each "
                "request spends waiting for the batch to fill before it runs.",
            ),
            answer=0,
            explanation=(
                "Throughput, not per-request arithmetic, sets unit cost: the GPU rents "
                "for the same price whether it emits ten tokens a second or three "
                "thousand. Decode is memory-bandwidth-bound, so batching adds users "
                "almost for free until the GPU turns compute-bound at the knee of the "
                "curve, past which extra latency buys little throughput. This is why "
                "'keep the batch full' is the whole cost story."
            ),
        ),
        Question(
            prompt=(
                "Continuous (iteration-level) batching is the core scheduling idea in "
                "modern engines. Concretely, what does it change relative to static "
                "batching?"
            ),
            options=(
                "It fuses prefill and decode into one step so that no request ever "
                "waits for a batch to assemble, which makes the classic fill-and-drain "
                "pipeline bubble vanish from the schedule entirely, regardless of how "
                "the request lengths differ.",
                "It makes a scheduling decision every iteration, so a finished request "
                "frees its slot at once and a waiting one joins on the next step, "
                "rather than the batch idling until its slowest sequence ends.",
                "It processes each request all the way to completion before admitting "
                "the next, which is what bounds tail latency and guarantees strict "
                "fairness across users.",
                "It shares key-value heads across the requests that happen to be "
                "batched together, shrinking the KV cache so more sequences fit in "
                "memory at once.",
            ),
            answer=1,
            explanation=(
                "Orca's insight was to schedule at the granularity of a single decode "
                "iteration rather than a whole request. Because requests in a batch "
                "have different lengths, a static batch leaves finished slots idle "
                "until the longest one drains; iteration-level scheduling refills them "
                "at once. The KV-cache-sharing option describes grouped-query "
                "attention, an unrelated lever."
            ),
        ),
        Question(
            prompt=(
                "A workload sends the same 2,000-token system prompt with every "
                "request. With prefix caching enabled, why does that shared prompt "
                "become nearly free after the first request?"
            ),
            options=(
                "The scheduler groups every request that carries that identical system "
                "prompt into a single batch, so the shared prompt is prefilled just "
                "one time for that batch and its cost is amortized across all of the "
                "batch's members.",
                "The engine caches the final-layer logits that the system prompt "
                "produces and then skips re-decoding those positions for each of the "
                "later requests.",
                "The system prompt's tokens are quantized to fewer bits than the "
                "user's tokens, so the shared portion streams out of memory faster on "
                "every single request.",
                "Its KV cache is computed once during the first request's prefill and "
                "reused by every later request sharing the prefix, so they pay only to "
                "prefill their own suffix.",
            ),
            answer=3,
            explanation=(
                "Prefix caching (SGLang's RadixAttention organizes it as a tree) keys "
                "on the shared token prefix and reuses its stored KV entries across "
                "requests that arrive at different times, not just within one batch. "
                "The per-batch option is the tempting near-miss: it captures reuse but "
                "misses that the cache persists across time, which is where most of "
                "the saving comes from."
            ),
        ),
        Question(
            prompt=(
                "Some serving stacks split prefill and decode onto separate GPU pools "
                "(disaggregation), or interleave chunked prefills with decodes on one "
                "pool. What problem are both attacking?"
            ),
            options=(
                "Prefill is compute-bound and decode is memory-bandwidth-bound, so "
                "running them together lets a long prefill stall the decodes in its "
                "batch and spike inter-token latency.",
                "Prefill and decode run at different numerical precisions, so "
                "separating them avoids the conversion overhead of switching number "
                "formats in the middle of a request.",
                "Decode needs the full KV cache while prefill needs none of it, so "
                "keeping the two phases apart roughly halves the memory that any one "
                "replica must hold.",
                "Prefill is stateless while decode is stateful, so only decode can be "
                "safely batched at all, and separating the two phases lets prefill run "
                "entirely without a scheduler managing it.",
            ),
            answer=0,
            explanation=(
                "The two phases have opposite bottlenecks, so co-locating them makes "
                "each degrade the other: one big prefill step blocks every decode in "
                "the batch, wrecking TPOT. Chunked prefill (Sarathi-Serve) slices the "
                "prefill and interleaves it; disaggregation (DistServe) routes the "
                "phases to pools tuned for compute versus bandwidth. Both target "
                "interference, not precision or raw memory."
            ),
        ),
        Question(
            prompt=(
                "You are choosing a serving engine. Which situation most specifically "
                "favors SGLang over the alternatives?"
            ),
            options=(
                "You need the absolute peak throughput achievable on NVIDIA GPUs and "
                "can afford to pay an ahead-of-time kernel compilation step for each "
                "model you deploy.",
                "You swap base models frequently and mainly want mature streaming, "
                "request metrics, and safe-rollout tooling that lives inside the "
                "Hugging Face serving ecosystem.",
                "Your traffic is full of long shared prefixes — a big system prompt, "
                "few-shot preambles, a common document — so automatic prefix reuse is "
                "the dominant win.",
                "Your KV cache no longer fits in contiguous memory and you need paged "
                "allocation to stop fragmentation from wasting VRAM.",
            ),
            answer=2,
            explanation=(
                "SGLang's signature is RadixAttention, which reuses overlapping "
                "prompt prefixes automatically, so it shines when prefixes are long "
                "and widely shared. The peak-throughput-via-compilation case points to "
                "TensorRT-LLM, the ecosystem case to TGI, and paged memory is now "
                "table stakes everywhere (it originated in vLLM's PagedAttention), so "
                "it no longer distinguishes one engine."
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
