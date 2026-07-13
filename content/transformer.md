The transformer [@vaswani2017] is the architecture underneath every modern LLM. Its one genuinely new idea is **self-attention**: a mechanism that lets every position in a sequence look directly at every other position and decide, per token, what to pull in. This chapter builds a decoder-only transformer — the LLM kind — from that idea outward.

## The problem attention solves

Before transformers, sequence models processed tokens one at a time, carrying a running summary in a hidden state (RNNs, LSTMs [@hochreiter1997]). Two problems followed: information from far back got squeezed through a bottleneck and faded, and the sequential dependency made training hard to parallelize.

Attention removes the bottleneck. Instead of forcing everything through a single evolving state, it lets each token reach back and read *directly* from any earlier token, with the strength of each read learned from context — an idea that first appeared as a bolt-on to translation RNNs [@bahdanau2015] before the transformer made it the whole machine.

!!! analogy "Analogy"
    Reading a mystery novel, you hit the word "she" and instantly glance back to whichever earlier name it refers to — maybe twenty pages ago, maybe one sentence. You are not replaying every page; you jump straight to the relevant spot. Attention is that glance, computed for every word at once. It leaks in that the model has no true memory across separate calls — each glance only reaches within the current context window.

## Queries, keys, and values

Each token's vector is projected into three roles. For a token embedding $x_i$:

- a **query** $q_i = x_i W_Q$ — what this token is looking for,
- a **key** $k_j = x_j W_K$ — what token $j$ advertises about itself,
- a **value** $v_j = x_j W_V$ — what token $j$ will hand over if attended to.

Token $i$ scores every token $j$ by the dot product $q_i \cdot k_j$: high when the query and key point the same way. Those scores become weights via softmax, and the output for token $i$ is the weighted sum of values.

!!! analogy "Analogy"
    It is a soft dictionary lookup. The **query** is your search term, each **key** is an entry's label, and the **value** is its content. A hard dictionary returns the one exact match; attention returns a blend of all entries, weighted by how well each label matches — a lookup with no misses, only degrees of relevance.

Stacked over the whole sequence, this is a few matrix multiplies:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V.$$

The $QK^\top$ term is an $n \times n$ matrix of every-token-against-every-token scores — which is exactly why cost grows with the *square* of sequence length, the fact behind every long-context headache in Chapter 15.

!!! interview "Interview"
    *Why divide by $\sqrt{d_k}$?* For large head dimension $d_k$, dot products have variance that grows with $d_k$, pushing softmax into saturated regions where gradients vanish. Scaling by $\sqrt{d_k}$ keeps the scores' variance near 1 so the softmax stays in a responsive range. Forgetting this term is a classic cause of training instability.

<figure>
<img src="assets/figures/attention-lookup.svg" alt="The query token 'it' sends arrows to every key, with the thickest arrow going to 'cat'; the values flow back and combine into a single output vector.">
<figcaption>Self-attention as a soft dictionary lookup. The query for <em>it</em> scores every key; the softmax turns those scores into weights (shown as arrow thickness); the output is the weighted blend of values — here, mostly whatever <em>cat</em> was carrying. Unlike a hard lookup, nothing is missed and nothing is exact: every entry contributes in proportion to how well it matches.</figcaption>
</figure>

## Multiple heads

One attention operation blends everything into a single relevance signal. That is limiting: the relationship a pronoun has to its antecedent is not the relationship a verb has to its subject. **Multi-head attention** runs several attention operations in parallel, each with its own $W_Q, W_K, W_V$ in a lower-dimensional subspace, then concatenates their outputs and projects back.

!!! intuition "Intuition"
    Heads are specialists reading the same sentence for different things: one tracks syntax, another coreference, another position. You give the model several narrow lookups instead of one muddy one, then let it combine their findings.

If the model dimension is $d$ and you use $h$ heads, each head works in dimension $d/h$, so multi-head attention costs about the same as single-head — you are partitioning the budget, not enlarging it.

## Causal masking

An LLM must predict the next token from *past* tokens only; letting position $i$ attend to position $i+1$ during training would leak the answer. The fix is a **causal mask**: before the softmax, set every score for a future position to $-\infty$ so it receives zero weight.

<figure>
<img src="assets/figures/causal-mask.svg" alt="A five-by-five attention matrix with the lower triangle shaded blue and the upper triangle greyed out and crossed.">
<figcaption>The causal mask. Row <em>i</em> is what token <em>i</em> may read; the upper triangle is set to −∞ before the softmax, so those weights come out exactly zero. This is what makes the model a valid next-token predictor — and it is also why we can train on every position of a sequence <em>simultaneously</em>, since each position already sees only the past it will have at generation time.</figcaption>
</figure>

!!! warning "Common trap"
    Break the mask and your eval loss looks magically low while the model has secretly been reading ahead. A loss that is suspiciously good early in training is the classic symptom.

## The rest of the block

Attention moves information *between* positions. A transformer block pairs it with a feed-forward network (FFN) that processes *each* position independently, plus the connective tissue that makes deep stacks trainable:

```python
# One pre-norm transformer block (schematic, not optimized).
def block(x):
    x = x + attention(rms_norm(x))       # Mix information across positions.
    x = x + feed_forward(rms_norm(x))    # Transform each position on its own.
    return x
```

Three pieces are doing quiet but essential work:

- **Residual connections** (the `x + ...`) give gradients a clean path around each sublayer, so stacking 80 blocks does not kill training [@he2016]. Think of them as an express lane the signal can always take.
- **Normalization** keeps activations in a stable range. Modern models normalize *before* each sublayer (pre-norm) rather than after, which trains more stably at depth [@xiong2020]; Chapter 5 covers why RMSNorm replaced LayerNorm.
- **The FFN** is usually two linear layers with a nonlinearity, widening to roughly $4d$ and back. It is where most of the parameters — and, many argue, most of the stored *knowledge* — live. Attention decides what to look at; the FFN decides what to make of it.

!!! interview "Interview"
    *Where are a transformer's parameters?* Roughly two-thirds sit in the FFNs and one-third in the attention projections (for typical shapes). This is why mixture-of-experts models (Chapter 5) target the FFN when they want to add capacity cheaply — they swap the single FFN for many, activating only a few per token.

## The full stack

A decoder-only transformer is then just:

1. **Embed** each token id into a vector, and add positional information (Chapter 5 explains why modern models use rotary embeddings instead of the original sinusoids).
2. **Repeat the block** $L$ times — 32 for a small model, 80-plus for a large one.
3. **Un-embed**: a final linear layer maps the last position's vector to a score for every vocabulary token, and softmax turns those into the next-token distribution.

!!! intuition "Intuition"
    The whole network is a tall stack of the same move: *mix across positions with attention, then refine each position with an FFN, over and over.* Depth lets early layers handle surface patterns and later layers assemble them into meaning.

That is the architecture every chapter after this one takes for granted. Chapter 5 shows how today's models tweak each component — positions, normalization, activations, and attention itself — for stability and efficiency at scale.
