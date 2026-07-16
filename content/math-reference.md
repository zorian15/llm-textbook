This book leans on a surprisingly small amount of mathematics, used over and over.
A handful of linear-algebra moves, one probability distribution, and a family of information-theoretic quantities account for nearly every equation in it.
This appendix collects that machinery in one place so you can flip back to it, and pins down the notation the chapters use.
It is a reference, not a course; for the full development the standard text is still Goodfellow, Bengio, and Courville [@goodfellow2016].

## Linear algebra you actually need

The workhorse is the matrix multiply.
Almost everything a transformer does — projecting embeddings, computing attention scores, running a feed-forward layer, un-embedding to logits — is one or more matmuls with a nonlinearity between them.
So it pays to think in *shapes*, treating them the way a type checker treats types: a matmul is legal only when the inner dimensions agree, and the outer dimensions tell you the shape of the result.
For $X$ of shape $(n \times d)$ and $W$ of shape $(d \times m)$, the product $XW$ has shape $(n \times m)$: the shared $d$ cancels, and $n$ and $m$ survive.
If two shapes do not line up, you have a bug, and reading the shapes usually finds it faster than reading the code.

<figure>
<img src="assets/figures/matmul-shapes.svg" alt="Three matrices in a row: X sized n by d, times W sized d by d_ff, equals XW sized n by d_ff. The shared inner dimension d is highlighted as the one that cancels.">
<figcaption>Shapes as a type system. A matmul is legal only when the inner dimensions match; that shared dimension is contracted away, and the outer dimensions pass through to the result. Most transformer bugs are shape mismatches, and reading the shapes finds them.</figcaption>
</figure>

Three operations cover most of what you will meet, and each is really just a shape transformation:

- **A linear layer** maps $(n \times d) \to (n \times m)$ by right-multiplying with a weight matrix $W \in \mathbb{R}^{d \times m}$ (plus an optional bias broadcast over the $n$ rows).
  It transforms each of the $n$ vectors independently; the feed-forward network of Chapter 4 is two of these, widening $d \to 4d$ and back.
- **An embedding lookup** turns a token id into a vector by selecting one row of the embedding matrix $E \in \mathbb{R}^{|V| \times d}$.
  It is a matmul in disguise — a one-hot vector times $E$ — but implemented as an indexed read, which is why the same matrix can be *tied* to the output layer (Chapter 4).
- **Attention** is two matmuls around a softmax.
  The scores $QK^\top$ multiply $(n \times d_k)$ by $(d_k \times n)$ to give the $(n \times n)$ every-token-against-every-token matrix of Chapter 4; after the softmax, multiplying by $V$ contracts the $n$ back down to $(n \times d_v)$.
  That first product's $n \times n$ shape is the whole reason attention costs grow with the square of sequence length.

!!! intuition "Intuition"
    Track shapes, not values.
    A transformer is a pipeline of shape transformations, and knowing what shape enters and leaves each block tells you what it can and cannot do — long before you know a single weight.

The other operations are minor by comparison.
The **dot product** $q \cdot k = \sum_i q_i k_i$ measures alignment between two vectors and is the atom of an attention score.
**Broadcasting** stretches a smaller array over a larger one so a bias vector or a per-head scale applies across a batch without copying.
A **batch** dimension simply rides in front of every shape above: real tensors are $(\text{batch} \times n \times d)$, and the matmuls apply independently across it.

## Probability and information

A language model's output is a probability distribution over the vocabulary, and the tools for producing and scoring that distribution are all here.

**Softmax** turns a vector of real-valued logits $z$ into a distribution:

$$\text{softmax}(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}}.$$

It is monotonic, so it never reorders the logits; it just squashes them onto the simplex.
Dividing the logits by a **temperature** $T$ before the exponential sharpens the distribution when $T < 1$ and flattens it when $T > 1$ — the sampling knob of Chapter 14.

**Cross-entropy** is the training loss.
For a true next-token distribution $p$ (a one-hot on the actual token) and the model's predicted distribution $q$, it is

$$H(p, q) = -\sum_i p_i \log q_i,$$

which for a one-hot target collapses to $-\log q_{\text{true}}$: the negative log-probability the model assigned the correct token.
This is the loss in the training loop of Chapter 2, averaged over positions and the batch.

**Entropy** $H(p) = -\sum_i p_i \log p_i$ is the average surprise of $p$ itself — the irreducible uncertainty in the data.
**KL divergence** measures how far the model's $q$ sits from the truth $p$:

$$D_{\text{KL}}(p \,\|\, q) = \sum_i p_i \log \frac{p_i}{q_i} \ge 0,$$

zero only when $q = p$, and asymmetric ($D_{\text{KL}}(p\|q) \neq D_{\text{KL}}(q\|p)$).
The three fit together in one identity worth memorizing:

$$H(p, q) = H(p) + D_{\text{KL}}(p \,\|\, q).$$

Cross-entropy is the entropy floor you can never beat plus the KL penalty for being wrong.
Training minimizes cross-entropy, but since $H(p)$ is fixed by the data, it is really driving the KL term toward zero.
KL returns as an explicit leash in alignment, where the RLHF objective penalizes drift from the reference policy (Chapter 11).

<figure>
<img src="assets/figures/cross-entropy-kl.svg" alt="Stacked bars for three models of increasing quality. Every bar shares a constant blue entropy floor; an amber KL segment stacked on top shrinks as the model improves, so the total cross-entropy falls toward the floor.">
<figcaption>Cross-entropy = entropy + KL. The entropy floor is fixed by the data and shared by every model; training shrinks only the KL term stacked on top. Perplexity, printed above each bar, is just the cross-entropy exponentiated back into an average number of choices.</figcaption>
</figure>

**Perplexity** is cross-entropy made legible.
It is the loss exponentiated,

$$\text{perplexity} = \exp\!\big(H(p, q)\big),$$

and reads as an effective branching factor: a perplexity of 8 means the model is, on average, as unsure as if it were choosing uniformly among 8 tokens.
Lower is better, it scales with model size (Chapter 9), and it is a standard intrinsic eval (Chapter 24).

One recurring gotcha is the log's base, which sets the **unit**.
Base $e$ gives **nats**, base $2$ gives **bits**, and they differ only by the constant $\log_2 e \approx 1.443$.
Cross-entropy losses are almost always reported in nats (natural log); information budgets and "bits per token" are in bits.
Exponentiate with the matching base — $e^{H}$ for nats, $2^{H}$ for bits — or perplexity comes out wrong.

!!! interview "Interview"
    *Your model's loss is 2.1 — good or bad?*
    You cannot say without the unit and the vocabulary.
    In nats that is a perplexity of $e^{2.1} \approx 8.2$; a random model over a 50k-vocabulary would score $\log 50000 \approx 10.8$ nats, so 2.1 is far better than chance but says nothing on its own until you compare it to a baseline on the *same* tokenizer and data.

## Notation conventions used in this book

The book keeps its symbols consistent so equations read the same across chapters.
Matrices are uppercase ($W$, $Q$, $K$, $V$), vectors lowercase ($q$, $k$, $x$), and scalars lowercase italic ($n$, $d$).
A learned weight matrix is a $W$ with a subscript naming its role.
Shapes are written $(\text{rows} \times \text{cols})$, with an implicit leading batch dimension unless it matters.

| Symbol | Meaning | Typical shape / range |
|---|---|---|
| $n$ | sequence length (number of tokens in the context) | up to the context window |
| $d$, $d_\text{model}$ | model (residual-stream) dimension | 512 to 16k |
| $h$ | number of attention heads | 8 to 128 |
| $d_k$, $d_v$ | per-head key/value dimension, usually $d/h$ | 64 to 128 |
| $L$ | number of transformer blocks (depth) | 12 to 100+ |
| $\lvert V \rvert$ | vocabulary size | 32k to 256k |
| $d_\text{ff}$ | feed-forward hidden width, usually $\approx 4d$ | $4d$ |
| $N$ | total parameter count | millions to trillions |
| $x_i$ | the vector for token $i$ | $(d,)$ |
| $q_i, k_j, v_j$ | query, key, value vectors | $(d_k,)$ or $(d_v,)$ |
| $W_Q, W_K, W_V$ | query/key/value projection matrices | $(d \times d_k)$ |
| $W$ (unscripted) | a generic learned weight matrix | $(\text{in} \times \text{out})$ |
| $E$ | token embedding matrix | $(\lvert V \rvert \times d)$ |
| $z$ | logits (pre-softmax scores) | $(\lvert V \rvert,)$ |
| $p, q$ | true and predicted distributions | sum to 1 |
| $\theta$ | the model's parameters, collectively | $(N,)$ |
| $\eta$ | learning rate | $10^{-5}$ to $10^{-3}$ |
| $\odot$ | elementwise (Hadamard) product | shape-preserving |
| $\lVert \cdot \rVert$ | vector norm (Euclidean unless noted) | scalar |

A few conventions worth stating outright.
Vectors are rows, so a linear layer is $xW$ (right-multiply), matching the code in PyTorch and JAX where the feature dimension is last.
"Log" means the natural log unless a base is written.
And a subscript indexes a token or position ($x_i$), while a superscript, when it appears, indexes a layer — never an exponent in this book, which writes powers explicitly.
