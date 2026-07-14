Chapter 4 built the transformer as published in 2017 [@vaswani2017]. No frontier model ships that exact recipe anymore — and yet the changes are few, and every one of them fixes a specific pain: training instability, KV-cache memory, or compute per token. This chapter walks the modern substitutions one component at a time. By the end you can read a real model's config file line by line, which is exactly how the chapter closes. The recipe below (rotary positions, pre-norm RMSNorm, SwiGLU, grouped-query attention) crystallized in the Llama family [@touvron2023] and is now close to universal for open models.

## Positional information: RoPE

Attention is order-blind: shuffle the tokens and the $QK^\top$ scores shuffle with them. The original transformer injected order by *adding* a position vector to each token embedding at the bottom of the stack. That works, but position becomes just another ingredient blended into the residual stream, and attention only sees it indirectly.

**Rotary position embeddings (RoPE)** [@su2024] move position into the attention scores themselves. Split each query and key vector into two-dimensional pairs, and rotate each pair by an angle proportional to the token's position, with each pair rotating at its own frequency. The payoff is in the dot product: the score between a query at position $m$ and a key at position $n$ depends only on the offset $m-n$, not on the absolute positions. Relative position falls out of the geometry.

!!! intuition "Intuition"
    Think of each 2D pair as a clock hand that turns as you move through the sequence, with fast hands and slow hands. Attention compares two tokens by the *angle between* their hands, and that angle only encodes how far apart they are — which is what grammar actually cares about.

RoPE also concentrates the levers for long context: because position lives in rotation frequencies, stretching the frequency schedule is how a model trained at one context length is extended to a much longer one. The named methods — linear position interpolation, NTK-aware scaling, and YaRN [@peng2023] — differ in how carefully they remap those frequencies so that short-range resolution survives; the serving costs return in Chapter 15.

<figure class="wide">
<img src="assets/figures/rope.svg" alt="The same token drawn as a clock face at three positions, its hand rotated further at each later position; the caption notes that the attention score depends on the angle between two hands.">
<figcaption>Position as rotation. RoPE turns each query and key by an angle set by the token's position, so the score between two tokens depends only on the angle between them — that is, on how far apart they are, never on where each one sits.</figcaption>
</figure>

## Normalization: RMSNorm and pre-norm

Two separate upgrades hide under "normalization."

*Where* it sits: the 2017 transformer normalized *after* each sublayer (post-norm), which at large depth needs a delicate warmup to avoid divergence. Placing the normalization *before* each sublayer (pre-norm) keeps the residual stream itself untouched — an identity path from embedding to output — and gradients flow through deep stacks without drama [@xiong2020]. Every modern LLM is pre-norm; the block in Chapter 4 already showed it.

*What* it computes: LayerNorm [@ba2016] centers activations (subtract the mean) and scales them (divide by the standard deviation). **RMSNorm** [@zhang2019] drops the centering and simply divides by the root-mean-square, with a learned gain. It is cheaper — one fewer reduction over the hidden dimension, and no bias parameters — and models train just as well. When an ablation says the cheap half of an operation was the only half that mattered, the expensive half disappears from the recipe.

A related trick you will see in recent models is **QK-norm**: normalizing the queries and keys before their dot product to keep attention logits from blowing up at large scale [@dehghani2023].

<figure class="wide">
<img src="assets/figures/prenorm.svg" alt="Two columns: post-norm places LayerNorm on the residual stream after the add, breaking it; pre-norm places the norm on the branch, leaving the residual stream an unbroken identity path.">
<figcaption>Where normalization sits. Moving the norm off the residual stream and onto each sublayer's branch keeps the stream an unbroken identity path from embedding to output, which is what makes very deep stacks train stably.</figcaption>
</figure>

## Activations: SwiGLU

The original FFN is two matrices with a ReLU between them. Modern models use a **gated** unit instead: project the input up twice, let one projection gate the other elementwise, then project back down,

$$\text{SwiGLU}(x) = \big(\text{Swish}(xW_1) \odot xW_2\big)\,W_3.$$

The gate lets the layer modulate *how much* of each feature passes, not just whether it is positive — multiplicative control instead of a hard one-sided switch. Across GLU variants, SwiGLU reliably wins by a small margin [@shazeer2020].

There is one bookkeeping consequence you will see in every config: SwiGLU has three matrices where the classic FFN has two, so to hold the parameter count fixed, the hidden width shrinks from $4d$ to roughly $\tfrac{2}{3} \cdot 4d$. That is why real models have oddly specific FFN sizes instead of a clean $4d$.

!!! note "Note"
    Why does gating help? The SwiGLU paper declines to theorize, attributing the result to "divine benevolence" [@shazeer2020]. The honest summary is that it wins ablations consistently and costs nothing at equal parameters — which, in architecture research, is a complete argument.

<figure class="wide">
<img src="assets/figures/swiglu.svg" alt="An input x is projected two ways; a Swish-activated gate multiplies the other projection elementwise, and the product is projected back down.">
<figcaption>A gated feed-forward layer. One projection gates the other elementwise, so the layer controls how much of each feature passes rather than merely whether it is positive — multiplicative control that a single one-sided ReLU cannot express.</figcaption>
</figure>

## Attention variants: MQA and GQA

At inference time, every past token's keys and values are cached so they are not recomputed per step — the KV cache, whose memory math Chapter 15 and Appendix A work out. In full multi-head attention, *every* head stores its own keys and values, so the cache scales with the head count, and at long context it competes with the weights for memory and bandwidth.

The fix is to share. **Multi-query attention (MQA)** keeps all the query heads but gives them one shared key/value head [@shazeer2019] — a large cache reduction with a measurable quality cost. **Grouped-query attention (GQA)** interpolates: query heads are divided into groups, each group sharing one KV head [@ainslie2023]. With, say, 32 query heads and 8 KV heads, the cache shrinks 4× while quality stays near full multi-head attention. GQA is the current default.

Sharing heads is not the only lever on the cache. **Sliding-window (local) attention** bounds it a different way: each token attends only to the last $w$ tokens, so cost and cache stop growing once the window is full, often with a few global layers interleaved to preserve long-range reach [@jiang2023]. And **multi-head latent attention (MLA)**, the successor probe past GQA, compresses the keys and values into a small shared latent, shrinking the cache below what GQA achieves [@deepseekv2].

<figure class="wide">
<img src="assets/figures/attention-sharing.svg" alt="Three layouts of query heads wired to key/value heads: multi-head attention has one KV head per query head, grouped-query attention shares one KV head per group of four query heads, and multi-query attention shares a single KV head.">
<figcaption>Sharing key/value heads. Full multi-head attention stores a KV head for every query head; GQA shares one across a group and MQA across all of them. Fewer KV heads means a smaller cache — longer context and bigger batches — at a small, controllable quality cost.</figcaption>
</figure>

!!! interview "Interview"
    *Why do modern models use GQA instead of full multi-head attention?* Because decode is bound by memory, not compute (Chapter 15), and the KV cache is the memory that grows with context and batch size. Sharing KV heads across groups of query heads cuts the cache by the sharing factor — allowing longer contexts and larger batches — at a quality cost small enough that everyone pays it.

## Mixture of experts

Everything so far tweaks the dense recipe. **Mixture of experts (MoE)** changes the deal: replace each FFN with many parallel FFNs (*experts*) plus a small router that sends each token to its top one or two experts [@shazeer2017; @fedus2022]. The model's *total* parameters multiply, but the compute per token only touches the chosen experts.

That splits a number you have been treating as one. A "Mixtral 8×7B" has ~47B **total** parameters (what you must hold in memory) but only ~13B **active** parameters per token (what you pay compute and, per Appendix A, bandwidth for). Scaling laws (Chapter 9) say capacity buys quality; MoE buys capacity without buying per-token compute.

!!! intuition "Intuition"
    Storing knowledge is cheap; computing with all of it on every token is not. MoE keeps a large library but only sends each token to the two most relevant shelves.

The costs are real, and the routing details are where the interviews live. Each expert has a *capacity*; tokens routed past it are dropped, so the router must be trained to spread load evenly. Classic MoE adds an auxiliary load-balancing loss for this; because that loss tugs against quality, newer models (DeepSeek-V3) pursue *auxiliary-loss-free* balancing via learned routing biases instead [@deepseekv3]. All experts must also fit in memory even though most idle per token, training is less stable than dense, and Chapter 8's parallelism grows a new dimension since experts can live on different devices.

<figure class="wide">
<img src="assets/figures/moe.svg" alt="A router sends a token to two of six experts, whose edges are highlighted while the other four are dashed; a callout contrasts six experts held in memory with two run per token.">
<figcaption>Why capacity is cheap but compute is not. A router sends each token to only its top experts, so the parameter count you hold in memory (total) grows far faster than the compute you spend per token (active). That gap is the whole appeal.</figcaption>
</figure>

## A tour of a current open model

Here is the config of a Llama-3-class 8B model [@grattafiori2024], trimmed to the load-bearing fields:

```json
{
  "hidden_size": 4096,
  "num_hidden_layers": 32,
  "num_attention_heads": 32,
  "num_key_value_heads": 8,
  "intermediate_size": 14336,
  "hidden_act": "silu",
  "rms_norm_eps": 1e-05,
  "rope_theta": 500000.0,
  "max_position_embeddings": 8192,
  "vocab_size": 128256
}
```

Read it with this chapter in hand. `num_key_value_heads: 8` against 32 attention heads is GQA with groups of four — a 4× smaller KV cache. `intermediate_size: 14336` is $3.5d$, not $4d$: the SwiGLU two-thirds bookkeeping, rounded to hardware-friendly multiples, with `hidden_act: silu` naming the Swish gate. `rms_norm_eps` tells you normalization is RMSNorm, and `rope_theta: 500000` is the rotation base — raised well above the original $10{,}000$ for longer context. The vocabulary of 128k tokens is Chapter 3's tradeoff, sized generously to shorten sequences. Every line is one of this chapter's sections, frozen into JSON.

!!! interview "Interview"
    *An interviewer shows you `num_attention_heads: 64, num_key_value_heads: 8` — what do they want to hear?* That this is grouped-query attention with eight groups, so the KV cache is 8× smaller than full multi-head attention would need; then the why — decode is memory-bandwidth-bound and the cache limits context length and batch size.

The architecture is now current. <figure class="wide">
<img src="assets/figures/config-tour.svg" alt="A model config with each field linked to a plain-language note: num_key_value_heads to GQA, intermediate_size to SwiGLU width, hidden_act silu to the Swish gate, rms_norm_eps to RMSNorm, rope_theta to the rotary base, and vocab_size to the BPE vocabulary.">
<figcaption>The chapter, frozen into JSON. Every load-bearing field in a real config is one of the choices from this chapter — GQA, SwiGLU's two-thirds width, RMSNorm, the RoPE base, the vocabulary — which is why the file reads as a summary once you know what to look for.</figcaption>
</figure>

The next part of the book takes these frozen blueprints and asks how the weights inside them are actually learned.
