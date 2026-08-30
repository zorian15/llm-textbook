A quick-reference list of the load-bearing terms this book uses, each defined in a sentence or two and pointing to the chapter where the idea is developed. Where a term has a fuller treatment, follow the chapter reference; this page is for orientation, not depth.

## A–M

**Agent.** An LLM run in a loop of plan, act, and observe, calling tools and reacting to their results to pursue a goal over many steps; autonomy is a spectrum, not a binary (Chapter 22).

**Attention.** The mechanism that lets each position read directly from every other position, weighting what to pull in by how well a query matches each key; self-attention is the transformer's one genuinely new idea (Chapter 4).

**Autoregressive generation.** Producing text one token at a time by sampling from the model's next-token distribution and feeding each choice back in as input, so the model's own output becomes its next context (Chapter 1).

**AWQ (activation-aware weight quantization).** A post-training quantization method that protects the small fraction of weights tied to the largest activations, preserving accuracy at 4 bits by rescaling rather than dropping the important channels (Chapter 16).

**Base model.** A freshly pretrained model that continues documents rather than answering them; it mirrors its training distribution and only becomes an assistant after post-training (Chapter 1).

**Beam search.** A decoding method that keeps the several highest-probability partial sequences at each step; it suits closed-ended tasks but yields bland, repetitive text for open-ended generation, which is why sampling replaced it (Chapter 14).

**bf16 (bfloat16).** A 16-bit float with the same exponent range as float32 but fewer mantissa bits; its wide dynamic range is why it won for training, where range matters more than precision (Chapter 2).

**Byte-pair encoding (BPE).** The dominant tokenization algorithm; it starts from characters or raw bytes and greedily merges the most frequent adjacent pair over and over, learning a vocabulary of subword units (Chapter 3).

**Calibration.** How well a model's stated confidence matches its actual accuracy; a well-calibrated model is right about as often as it claims, and miscalibration is one root of confident hallucination (Chapter 26).

**Catastrophic forgetting.** The loss of pretrained knowledge or skills when fine-tuning pushes weights too far toward a narrow new task; a central risk of aggressive supervised fine-tuning (Chapter 10).

**Causal mask.** The trick that makes a transformer a valid next-token predictor; before the softmax it sets every future position's score to negative infinity, so each token attends only to the past (Chapter 4).

**Chain-of-thought (CoT).** Prompting or training a model to write intermediate reasoning steps before its answer; it helps on multi-step problems but is theater when the stated steps do not actually drive the conclusion (Chapters 18, 25).

**Chat template.** The special-token scaffolding that marks system, user, and assistant turns in a conversation; fine-tuning teaches the model this format, and mismatching it at inference degrades output (Chapter 10).

**Chinchilla.** The compute-optimal scaling result showing that earlier models were badly undertrained; for a fixed compute budget you scale parameters and tokens together, roughly 20 tokens per parameter (Chapter 9).

**Chunking.** Splitting documents into passages for retrieval; chunk size and boundaries quietly make or break RAG, since a chunk is the unit that gets embedded and returned (Chapter 21).

**Constitutional AI.** An alignment recipe that replaces human preference labels with a model critiquing and revising its own outputs against a written set of principles; the "AI feedback" behind RLAIF (Chapter 12).

**Constrained decoding.** Masking the next-token logits to only those a grammar or schema allows, guaranteeing syntactically valid output such as parseable JSON; also called guided or structured decoding (Chapters 14, 20).

**Context window.** The maximum number of tokens a model can attend to at once; everything the model can use in a call must fit inside it, and attention cost grows with its square (Chapter 4).

**Cross-entropy.** The pretraining loss; the average number of bits of surprise the model assigns to the tokens that actually came next, minimized by predicting real text well (Chapters 1, 6).

**Data parallelism.** The simplest distributed-training scheme; every GPU holds a full copy of the model and processes a different slice of the batch, and gradients are averaged across ranks (Chapter 8).

**Decode.** The generation phase, producing one token per forward pass; it is memory-bandwidth-bound because each step streams the whole model from memory, which is why it dominates tokens per second (Chapter 15).

**Decoder-only.** The LLM architecture: a causal, autoregressive transformer built to continue a sequence, as opposed to a bidirectional encoder built to understand a fixed input (Chapter 4).

**Decontamination.** Filtering pretraining data to remove text that overlaps evaluation benchmarks, so reported scores reflect generalization rather than memorized test answers (Chapters 6, 24).

**Deduplication.** Removing repeated or near-repeated documents from pretraining data; it improves quality per token and curbs memorization, and is standard before training (Chapter 6).

**DPO (Direct Preference Optimization).** Aligning a model to preferences with a single classification-style loss and no reward model or RL loop; a mathematical identity lets the policy itself stand in for the reward (Chapter 12).

**Embeddings.** Vectors that place semantically similar texts near each other, so a query can find relevant passages by nearest-neighbor search rather than keyword match; the basis of dense retrieval and RAG (Chapter 21).

**Emergent abilities.** Capabilities that seem to appear abruptly at a certain scale; the abruptness is partly an artifact of all-or-nothing metrics and smooths out under continuous measures (Chapters 1, 9).

**Encoder / encoder-decoder.** A bidirectional transformer that reads a whole input to understand it (BERT-style), or a pair that encodes a source and decodes an output (T5-style); the attention pattern and objective, not the word "transformer," are the real fork (Chapter 4).

**Feed-forward network (FFN).** The per-position sublayer of a transformer block, usually two linear layers around a nonlinearity; it holds most of the parameters and, many argue, most of the stored knowledge (Chapter 4).

**Few-shot prompting.** Including a handful of worked examples in the prompt so the model infers the task from them, with no weight updates; the in-context learning that made prompting a discipline (Chapter 18).

**FlashAttention.** An IO-aware attention implementation that computes the exact softmax tile by tile without ever writing the full score matrix to memory, trading recomputation for far less memory traffic (Chapter 15).

**FSDP (Fully Sharded Data Parallel).** Data parallelism that also shards parameters, gradients, and optimizer state across ranks, gathering each layer's weights only when needed; the practical implementation of ZeRO (Chapter 8).

**GGUF.** The model file format used by llama.cpp; it packages quantized weights with metadata and drives most local, Apple-silicon-friendly inference (Chapter 16, Appendix A).

**Goodhart's law.** When a measure becomes a target it ceases to be a good measure; the pattern behind reward hacking, where optimizing a proxy reward degrades the behavior it stood for (Chapter 11).

**GPTQ.** A post-training quantization method that quantizes weights layer by layer using second-order information from calibration data to minimize the error each rounding introduces (Chapter 16).

**Gradient.** The vector of partial derivatives of the loss with respect to every parameter; backpropagation computes it via the chain rule, and the optimizer steps against it to lower the loss (Chapter 2).

**Greedy decoding.** Always taking the single highest-probability next token; deterministic and fine for short factual answers, but prone to dull, repetitive text on open-ended generation (Chapter 14).

**Grouped-query attention (GQA).** Attention where groups of query heads share one key/value head, shrinking the KV cache by the sharing factor at a small quality cost; the current default (Chapter 5).

**Guardrails.** The classifiers and policies wrapped around a model to filter unsafe inputs and outputs, a defense layer separate from the alignment baked into the weights (Chapter 23).

**Hallucination.** A fluent, confident output that is simply false; rooted in a model trained to always produce plausible text and poorly calibrated about what it does not know (Chapter 26).

**Harness.** The system around the weights that decides what context the model sees and what happens to its output: system prompt, tools, retrieval, structured output, and guardrails; in 2026 most LLM engineering is harness engineering (Chapters 1, Part V).

**Hybrid search.** Combining sparse keyword retrieval (BM25) with dense embedding retrieval and merging the rankings, so results catch both exact terms and semantic matches (Chapter 21).

**In-context learning.** The model's ability to pick up a task from instructions or examples in the prompt alone, with no change to its weights; why prompting works at all (Chapter 18).

**Interpretability.** The effort to read a model's internal computation in human terms via features and circuits; complicated by superposition, where one neuron participates in many unrelated features (Chapter 26).

**Jailbreak.** A prompt crafted to bypass a model's safety training and elicit disallowed output; the offensive half of an ongoing red-team cat-and-mouse (Chapter 23).

**KL penalty.** The leash in RLHF; a term that penalizes the policy for drifting too far from the reference model, keeping it from exploiting the reward model into gibberish (Chapter 11).

**KV cache.** The stored keys and values of past tokens that let each decode step avoid recomputing the whole prefix; its linear growth in sequence length and batch dominates long-context inference memory (Chapter 15).

**Large language model (LLM).** Mechanically, a function that reads a sequence of tokens and outputs a probability distribution over the next one; everything else emerges from making that function large and wrapping it well (Chapter 1).

**llama.cpp.** The C/C++ inference engine most local LLM tools are built on; it runs GGUF models and has mature support for Apple's Metal GPU backend (Appendix A).

**LLM-as-judge.** Using a strong model to grade another model's outputs; scalable and often well-correlated with humans, but subject to position, verbosity, and self-preference biases you must correct for (Chapter 24).

**LoRA (Low-Rank Adaptation).** Parameter-efficient fine-tuning that freezes the base weights and learns a small low-rank update per layer, on the intuition that adaptation lives in a low-dimensional subspace; rank and alpha are its main knobs (Chapter 13).

**Loss masking.** Computing the fine-tuning loss only on the assistant's tokens, not the prompt, so the model learns to produce answers rather than to predict the user's words (Chapter 10).

**Min-p.** A sampling cutoff that keeps only tokens whose probability is at least a fraction of the top token's; it adapts to how peaked the distribution is, staying tight when the model is confident and open when it is not (Chapter 14).

**Mixture of experts (MoE).** An architecture that replaces each FFN with many experts plus a router sending each token to only a few, so total parameters grow while compute per token stays fixed; capacity is cheap, compute is not (Chapter 5).

**Model Context Protocol (MCP).** An open standard for connecting models to tools and data through servers and clients, so a capability is written once and reused across applications instead of re-integrated per product (Chapter 19).

**Multi-head attention.** Running several attention operations in parallel, each in its own lower-dimensional subspace, so different heads can track different relationships at once before their outputs are recombined (Chapter 4).

**Multi-query attention (MQA).** The aggressive end of KV-head sharing: all query heads share a single key/value head, giving the largest cache savings at the largest quality cost, with GQA as the middle ground (Chapter 5).

## N–Z

**Next-token prediction.** The single training objective of an LLM: predict the next token given the preceding ones; scaled across trillions of tokens it forces the model to absorb grammar, facts, and reasoning (Chapters 1, 6).

**PagedAttention.** The idea behind vLLM: store the KV cache in fixed-size pages like virtual memory, eliminating the fragmentation that otherwise wastes most KV memory and caps how many requests fit (Chapters 15, 17).

**Parameter-efficient fine-tuning (PEFT).** The family of methods, LoRA chief among them, that adapt a model by training a tiny set of new parameters while the base weights stay frozen, sidestepping the memory and one-copy-per-task cost of full fine-tuning (Chapter 13).

**Perplexity.** The exponential of the cross-entropy loss; the effective number of equally likely tokens the model is choosing among at each step, so lower is better (Chapters 1, 6).

**Pipeline parallelism.** Splitting a model's layers into stages across devices and streaming micro-batches through them; light on memory but prone to idle "bubbles" when the pipeline fills and drains (Chapter 8).

**Post-training.** Everything done to a base model to make it an assistant, namely supervised fine-tuning and preference optimization; it changes behavior far more than it changes knowledge (Chapters 1, 10-12).

**PPO (Proximal Policy Optimization).** The reinforcement-learning algorithm classically used in RLHF; it improves the policy against the reward model in small, clipped steps to avoid destructive updates (Chapter 11).

**Prefill.** The phase that processes the whole prompt in parallel to populate the KV cache; compute-bound and fast, it sets time-to-first-token before decode takes over (Chapter 15).

**Pre-norm.** Placing normalization before each sublayer rather than after, leaving the residual stream an unbroken identity path; the change that lets very deep transformers train without delicate warmup (Chapters 4, 5).

**Prompt injection.** An attack where untrusted text in the input, from a web page or a tool result, is read by the model as instructions; the core reason everything outside the system prompt is data, not commands (Chapter 18).

**QAT (quantization-aware training).** Training or fine-tuning a model with quantization simulated in the forward pass so it learns weights robust to low precision, buying accuracy below what post-training quantization reaches (Chapter 16).

**QLoRA.** LoRA on top of a base model quantized to 4 bits, so a large model fine-tunes on a single GPU; the frozen base is stored compactly while the adapters train in higher precision (Chapter 13).

**Quantization.** Representing weights, and sometimes activations, in fewer bits to cut memory and, because decode is bandwidth-bound, to speed generation; the central lever for fitting and running big models cheaply (Chapter 16, Appendix A).

**RAG (Retrieval-Augmented Generation).** Injecting retrieved documents into the context so the model grounds its answer in fresh or private data it never memorized, reducing hallucination on knowledge it lacks (Chapter 21).

**ReAct.** An agent pattern that interleaves reasoning traces with tool actions and observations, so the model plans, acts, and revises in one loop rather than committing to a plan up front (Chapter 22).

**Reasoning model.** A model post-trained to spend test-time compute on a long internal chain of thought before answering, often via RL on verifiable rewards; the o1/R1 turn that made "thinking longer" a scaling axis (Chapter 25).

**Red-teaming.** Deliberately attacking a model to find jailbreaks and failure modes before release; the defensive discipline that stress-tests guardrails (Chapter 23).

**Reranking.** A second-stage model, often a cross-encoder, that rescores the passages a fast retriever returned; it lifts precision by judging query and passage together rather than as separate vectors (Chapter 21).

**Residual connection.** The "add the input back" path around each sublayer that gives gradients a clean route through a deep stack; keeping this identity path unbroken is what makes dozens-of-layers-deep models trainable (Chapter 4).

**Reward hacking.** When a policy maximizes the reward model's score while degrading the behavior the reward was meant to capture; Goodhart's law inside an RLHF loop, mitigated by the KL penalty and better rewards (Chapter 11).

**Reward model.** A model trained on human preference comparisons to output a scalar score for a response; it stands in for a human rater so RLHF can optimize against it (Chapter 11).

**RLHF.** Reinforcement Learning from Human Feedback; the pipeline of a preference-trained reward model plus policy optimization that tunes a model toward outputs people prefer, using rankings rather than demonstrations (Chapter 11).

**RLVR (RL from verifiable rewards).** Training reasoning by rewarding answers an automatic checker can verify, such as a math result or passing unit tests, sidestepping the reward model and its hacking (Chapter 25).

**RMSNorm.** A cheaper normalization that divides activations by their root-mean-square with a learned gain, dropping LayerNorm's mean-centering; it trains just as well and is now standard (Chapter 5).

**RoPE (rotary position embeddings).** Encoding position by rotating query and key vectors by an angle set by their position, so the attention score depends only on the relative offset between two tokens; also the lever for extending context length (Chapter 5).

**Scaling laws.** The empirical power laws by which loss falls smoothly and predictably as parameters, data, and compute grow together; the closest thing the field has to a design equation (Chapter 9).

**Self-consistency.** Sampling several independent reasoning chains and taking the majority answer; a simple way to spend test-time compute that trades tokens for accuracy when the final answer is checkable (Chapter 25).

**SentencePiece.** A tokenizer toolkit that trains BPE or Unigram models directly on raw text with no pre-tokenization, treating whitespace as an ordinary symbol; common in multilingual models (Chapter 3).

**Softmax.** The function that turns a vector of scores into a probability distribution by exponentiating and normalizing; it produces both the attention weights and the next-token distribution (Chapters 2, 4).

**Speculative decoding.** Letting a small draft model propose several tokens that the large model verifies in one parallel pass, accepting the longest correct prefix; it speeds decode with no change to the output distribution (Chapter 15).

**Structured output.** Constraining a model to emit machine-parseable data such as valid JSON matching a schema, so its output can drive a downstream system instead of being parsed out of prose (Chapter 20).

**Superposition.** The finding that a network stores more features than it has neurons by overlapping them, so a single neuron is polysemantic; a central obstacle to interpretability, which sparse autoencoders try to undo (Chapter 26).

**Supervised fine-tuning (SFT).** Training a base model on curated request-response examples to teach the assistant format and behavior; the first and cheapest step of post-training (Chapter 10).

**SwiGLU.** The gated feed-forward layer in modern transformers; one projection gates another elementwise through a Swish activation, giving multiplicative control a plain ReLU cannot, at two-thirds width to keep parameters fixed (Chapter 5).

**System prompt.** The provider- or developer-set instructions prepended before the user's turn that fix persona, rules, and defaults; the top of a trust hierarchy above user and tool text (Chapter 18).

**Teacher forcing.** Training each position against the true next token while conditioning on the ground-truth prefix, which lets a whole sequence train in parallel; its known leak is the mismatch with generation, where the model feeds on its own samples (Chapter 4).

**Temperature.** The knob that sharpens or flattens the next-token distribution before sampling; below 1 makes output more deterministic, above 1 more random, and 0 is greedy decoding (Chapter 14).

**Tensor parallelism.** Splitting the math of a single layer, its matrices, across devices that each compute a shard and then combine results; it needs high-bandwidth interconnect because it communicates within every layer (Chapter 8).

**Test-time compute.** Spending more computation at inference, by generating long reasoning or sampling many attempts, to raise accuracy; a scaling axis distinct from model size (Chapter 25).

**Tokenization.** Converting text into the integer token ids a model consumes, and back; its choices set sequence length, cost per token, number handling, and non-English penalties (Chapter 3).

**Tool use (function calling).** The loop where the model emits a structured call, the harness executes it, and the result returns to the context; the model proposes an action, it never runs code itself (Chapter 19).

**Top-k.** Sampling restricted to the k highest-probability tokens; simpler than top-p but blind to how peaked the distribution is, so a fixed k is sometimes too tight and sometimes too loose (Chapter 14).

**Top-p (nucleus sampling).** Sampling from the smallest set of top tokens whose probabilities sum to p, so the candidate pool shrinks when the model is confident and grows when it is unsure (Chapter 14).

**TPOT (time per output token).** The average latency between generated tokens during decode; with time-to-first-token, one of the two latency numbers a serving system trades against throughput (Chapter 17).

**TTFT (time to first token).** The delay before the first output token, set mainly by prefill and queueing; the latency a user feels as responsiveness (Chapter 17).

**Unified memory.** Apple silicon's shared CPU-GPU memory pool, so the GPU can address all system RAM; capacity stops being the wall and bandwidth sets the token rate for local inference (Appendix A).

**Unigram.** A tokenization method that starts from a large vocabulary and prunes it to maximize the likelihood of a probabilistic segmentation, an alternative to BPE's greedy merging (Chapter 3).

**Vector search.** Finding the passages whose embeddings are nearest a query's, usually by cosine similarity over an approximate-nearest-neighbor index; the retrieval step of RAG (Chapter 21).

**vLLM.** A widely used inference engine built around PagedAttention and continuous batching; a reference point for high-throughput open-model serving (Chapters 15, 17).

**Vocabulary.** The fixed set of tokens a tokenizer and model share; a larger vocabulary shortens sequences but widens the embedding and output layers, a tradeoff modern models push toward 100k-plus tokens (Chapter 3).

**Warmup.** Ramping the learning rate up from near zero over the first steps before decaying it, which keeps early updates from destabilizing a fresh model; the front half of the standard schedule (Chapter 7).

**Weight decay.** Regularization that shrinks weights toward zero each step; in AdamW it is applied directly rather than folded into the gradient, which is why AdamW generalizes better than Adam with plain L2 (Chapter 2).

**ZeRO.** A family of memory optimizations that shard optimizer state, then gradients, then parameters across data-parallel ranks in three stages, cutting per-GPU memory without changing the math; the ladder FSDP implements (Chapter 8).
