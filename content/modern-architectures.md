Chapter 4 built the transformer as published in 2017 [@vaswani2017]. No frontier model ships that exact recipe anymore — and yet the changes are few, and every one of them fixes a specific pain: training instability, KV-cache memory, or compute per token. This chapter walks the modern substitutions one component at a time. By the end you can read a real model's config file line by line, which is exactly how the chapter closes. The recipe below (rotary positions, pre-norm RMSNorm, SwiGLU, grouped-query attention) crystallized in the Llama family [@touvron2023] and is now close to universal for open models.

## Positional information: RoPE

Attention is order-blind: shuffle the tokens and the $QK^\top$ scores shuffle with them. The original transformer injected order by *adding* a position vector to each token embedding at the bottom of the stack. That works, but position becomes just another ingredient blended into the residual stream, and attention only sees it indirectly.

**Rotary position embeddings (RoPE)** [@su2024] move position into the attention scores themselves. Split each query and key vector into two-dimensional pairs, and rotate each pair by an angle proportional to the token's position, with each pair rotating at its own frequency. The payoff is in the dot product: the score between a query at position $m$ and a key at position $n$ depends only on the offset $m-n$, not on the absolute positions. Relative position falls out of the geometry.

!!! intuition "Intuition"
    Think of each 2D pair as a clock hand that turns as you move through the sequence, with fast hands and slow hands. Attention compares two tokens by the *angle between* their hands, and that angle only encodes how far apart they are — which is what grammar actually cares about.

RoPE also concentrates the levers for long context: because position lives in rotation frequencies, stretching the frequency schedule (raising the rotation base, interpolating positions) is how models trained at one context length are extended to much longer ones — a topic that returns with the serving costs of Chapter 15.

## Normalization: RMSNorm and pre-norm

Two separate upgrades hide under "normalization."

*Where* it sits: the 2017 transformer normalized *after* each sublayer (post-norm), which at large depth needs a delicate warmup to avoid divergence. Placing the normalization *before* each sublayer (pre-norm) keeps the residual stream itself untouched — an identity path from embedding to output — and gradients flow through deep stacks without drama [@xiong2020]. Every modern LLM is pre-norm; the block in Chapter 4 already showed it.

*What* it computes: LayerNorm [@ba2016] centers activations (subtract the mean) and scales them (divide by the standard deviation). **RMSNorm** [@zhang2019] drops the centering and simply divides by the root-mean-square, with a learned gain. It is cheaper — one fewer reduction over the hidden dimension, and no bias parameters — and models train just as well. When an ablation says the cheap half of an operation was the only half that mattered, the expensive half disappears from the recipe.

## Activations: SwiGLU

The original FFN is two matrices with a ReLU between them. Modern models use a **gated** unit instead: project the input up twice, let one projection gate the other elementwise, then project back down,

$$\text{SwiGLU}(x) = \big(\text{Swish}(xW_1) \odot xW_2\big)\,W_3.$$

The gate lets the layer modulate *how much* of each feature passes, not just whether it is positive — multiplicative control instead of a hard one-sided switch. Across GLU variants, SwiGLU reliably wins by a small margin [@shazeer2020].

There is one bookkeeping consequence you will see in every config: SwiGLU has three matrices where the classic FFN has two, so to hold the parameter count fixed, the hidden width shrinks from $4d$ to roughly $\tfrac{2}{3} \cdot 4d$. That is why real models have oddly specific FFN sizes instead of a clean $4d$.

!!! note "Note"
    Why does gating help? The SwiGLU paper declines to theorize, attributing the result to "divine benevolence" [@shazeer2020]. The honest summary is that it wins ablations consistently and costs nothing at equal parameters — which, in architecture research, is a complete argument.

## Attention variants: MQA and GQA

At inference time, every past token's keys and values are cached so they are not recomputed per step — the KV cache, whose memory math Chapter 15 and Appendix A work out. In full multi-head attention, *every* head stores its own keys and values, so the cache scales with the head count, and at long context it competes with the weights for memory and bandwidth.

The fix is to share. **Multi-query attention (MQA)** keeps all the query heads but gives them one shared key/value head [@shazeer2019] — a large cache reduction with a measurable quality cost. **Grouped-query attention (GQA)** interpolates: query heads are divided into groups, each group sharing one KV head [@ainslie2023]. With, say, 32 query heads and 8 KV heads, the cache shrinks 4× while quality stays near full multi-head attention. GQA is the current default.

!!! figure "Figure 5.1. Multi-head, grouped-query, and multi-query attention."
    Three columns of query heads connected to key/value heads: MHA with a KV head per query head, GQA with one KV head per group of four query heads, and MQA with a single shared KV head. The KV cache size shrinks left to right.

!!! interview "Interview"
    *Why do modern models use GQA instead of full multi-head attention?* Because decode is bound by memory, not compute (Chapter 15), and the KV cache is the memory that grows with context and batch size. Sharing KV heads across groups of query heads cuts the cache by the sharing factor — allowing longer contexts and larger batches — at a quality cost small enough that everyone pays it.

## Mixture of experts

Everything so far tweaks the dense recipe. **Mixture of experts (MoE)** changes the deal: replace each FFN with many parallel FFNs (*experts*) plus a small router that sends each token to its top one or two experts [@shazeer2017; @fedus2022]. The model's *total* parameters multiply, but the compute per token only touches the chosen experts.

That splits a number you have been treating as one. A "Mixtral 8×7B" has ~47B **total** parameters (what you must hold in memory) but only ~13B **active** parameters per token (what you pay compute and, per Appendix A, bandwidth for). Scaling laws (Chapter 9) say capacity buys quality; MoE buys capacity without buying per-token compute.

!!! intuition "Intuition"
    Storing knowledge is cheap; computing with all of it on every token is not. MoE keeps a large library but only sends each token to the two most relevant shelves.

The costs are real: all experts must fit in memory even though most idle per token, the router must be trained to spread load evenly (an auxiliary loss does this), and training is less stable than dense. Chapter 8's parallelism strategies also grow a new dimension, since experts can live on different devices.

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

The architecture is now current. The next part of the book takes these frozen blueprints and asks how the weights inside them are actually learned.
