A trained model is not fast or slow on its own; how you *run* it decides. This chapter is organized around a single fact that shapes every technique in it: generating text is bottlenecked by **memory bandwidth**, not arithmetic. The GPU can multiply far faster than it can fetch the numbers to multiply, so the game at inference time is to move fewer bytes and to do more useful work per byte moved. Read prefill-versus-decode, the KV cache, batching, speculation, and FlashAttention as five different answers to that one problem.

## Prefill and decode are two different machines

Serving a request splits into two phases with opposite characters. **Prefill** reads the whole prompt at once: every prompt token flows through the model in parallel, in one big batched matrix multiply, exactly like a training forward pass. **Decode** then generates the answer one token at a time, and each step must run a full forward pass to produce a *single* new token before it can even begin the next.

The difference that matters is **arithmetic intensity** — floating-point operations performed per byte read from memory. Prefill reuses each weight across all the prompt's tokens, so it does a lot of math per byte and saturates the GPU's arithmetic units: it is **compute-bound**. Decode reads the entire model — every weight, plus the growing KV cache — to compute one token, so it does almost no math per byte and the arithmetic units sit idle waiting on memory: it is **memory-bandwidth-bound**. This is the same wall Appendix A hits from the hardware side, where generation speed came out as bandwidth divided by model size.

!!! intuition "Intuition"
    Prefill is reading a page you already have; decode is writing the next word, then walking back to re-read the entire book before writing the word after that. The reading dominates, and it is reading from slow memory.

Because prefill happens once but decode runs once per output token, end-to-end speed on any nontrivial generation is dominated by decode. That is why almost everything downstream — batching, the KV cache, quantization (Chapter 16) — is really about making the memory-bound decode phase move fewer bytes.

!!! interview "Interview"
    *A serving team reports two latency numbers, TTFT and TPOT — what are they measuring?* Time-to-first-token is prefill: how long until the prompt is processed and the first token appears. Time-per-output-token is decode: the steady-state cost of each subsequent token, set by memory bandwidth. They trade off differently, which is why serving systems (Chapter 17) schedule the two phases separately.

<figure class="wide">
<img src="assets/figures/prefill-decode-roofline.svg" alt="A roofline chart: a diagonal memory-bandwidth roof rises to a horizontal compute roof at the ridge point; prefill sits on the compute plateau while decode sits far down the memory ramp, its arithmetic units idle.">
<figcaption>Why decode is the slow phase. On the roofline, prefill has high arithmetic intensity and rides the compute ceiling; decode has almost none and is pinned to the memory-bandwidth ramp, leaving most of the GPU's arithmetic idle. Every decode-time optimization is an attempt to climb this ramp.</figcaption>
</figure>

## The KV cache turns recomputation into memory

Attention at position $t$ needs the keys and values of every earlier token. Naively, generating each new token would recompute the keys and values for the entire prefix, making generation cost grow with the *square* of the sequence — the $O(n^2)$ of Chapter 4, paid again at every step. The **KV cache** removes that: keys and values, once computed, are stored and reused, so each decode step computes the K and V for only the one new token and reads the rest from the cache.

The catch is that you have traded compute for memory, and that memory grows. The cache holds, for every layer and every past token, one key vector and one value vector:

$$\text{cache bytes} = 2 \times L \times n_{\text{kv}} \times d_{\text{head}} \times \text{bytes} \times \text{seq len} \times \text{batch}.$$

The leading 2 is keys and values; it grows **linearly in both sequence length and batch size**. Work it out for a Llama-3-8B-class model (32 layers, 8 KV heads, head dimension 128, bf16): about 128 KB per token, so a single 128k-token context holds roughly 16 GB of cache — comparable to the 16 GB of weights themselves. Multiply by batch size and the cache, not the weights, is what fills the GPU and caps how many requests you can serve at once.

!!! interview "Interview"
    *Why does a long context slow decoding down even after prefill is done?* Because every decode step streams the whole KV cache out of memory to attend over it, and the cache grows with context length. Decode is bandwidth-bound, so a bigger cache means more bytes per token means fewer tokens per second — the cost of long context is paid on every single step, not just once at prefill.

This is exactly why the attention variants of Chapter 5 exist. Grouped-query and multi-query attention shrink $n_{\text{kv}}$ directly [@ainslie2023; @shazeer2019]; sliding-window attention caps the effective sequence length [@jiang2023]; multi-head latent attention compresses K and V into a small shared latent [@deepseekv2]. Quantizing the cache to fewer bytes per entry (Chapter 16) is a fourth lever. All four attack the same term in the formula above.

<figure class="wide">
<img src="assets/figures/kv-cache-growth.svg" alt="KV cache size in gigabytes rising linearly with sequence length, with a steep line for full multi-head attention and a shallower one for grouped-query attention; a dashed horizontal line marks the model's weight memory, which both cross at long context.">
<figcaption>The cache overtakes the weights. KV memory grows linearly with context, and grouped-query attention buys back a constant factor by storing fewer KV heads. At long enough context the cache rivals the model itself — and batch size multiplies the whole picture.</figcaption>
</figure>

## Batching and PagedAttention

Since decode re-reads the weights for every token, the way to get throughput is to make each read count for more: **batch** requests so one weight-read serves many sequences at once. Batching is the single biggest throughput lever precisely because it amortizes the memory-bound decode over more useful work.

Naive static batching wastes most of that promise. Requests in a batch finish at different times, so a batch sized for the slowest request leaves the rest idling; worse, each sequence needs contiguous KV memory reserved for its *maximum* possible length, most of which is never used. The result is severe internal fragmentation and small effective batches. Two ideas fix it. **Continuous (in-flight) batching** swaps finished sequences out and new ones in at the granularity of individual tokens, keeping the batch full (its origin, Orca, is Chapter 17's to develop). And **PagedAttention** solves the memory side by borrowing the operating system's oldest trick [@kwon2023].

!!! analogy "Analogy"
    PagedAttention treats KV memory the way an OS treats RAM: split it into fixed-size blocks, hand them out on demand, and keep a per-sequence *block table* mapping a request's logical blocks to scattered physical ones. A sequence never needs contiguous space, so fragmentation nearly vanishes and shared prefixes can share physical blocks copy-on-write. The analogy leaks in that KV blocks are only ever appended to, never randomly written, so there is no true page-replacement policy — nothing gets swapped to disk mid-attention.

The payoff is concrete: vLLM's paged cache raised serving throughput several-fold over contiguous allocation at the same latency, by fitting far larger batches into the same memory [@kwon2023]. Paging is now standard in every serious inference engine.

<figure class="wide">
<img src="assets/figures/paged-attention.svg" alt="Top: two requests each reserve a long contiguous strip of KV memory, most of it empty and wasted. Bottom: the same requests use small logical blocks mapped through a block table to non-contiguous physical blocks in a shared pool, with no wasted space.">
<figcaption>KV memory as virtual memory. Reserving a contiguous block per request for its worst-case length wastes most of it; paging hands out small blocks on demand and maps them through a block table, so the freed memory becomes a bigger batch — which is the throughput win.</figcaption>
</figure>

## Speculative decoding buys parallelism back

Decode is slow because it is *sequential*: one forward pass per token, each waiting on the last. Speculative decoding breaks that dependency with a bet. A small, cheap **draft** model proposes the next few tokens quickly; the large **target** model then verifies all of them in a *single* forward pass, checking each proposed token against what it would have produced [@leviathan2023; @chen2023]. Verifying $k$ tokens at once is a parallel, prefill-like operation, so it is nearly as cheap as generating one — and the target had spare compute to burn, being memory-bound. Accept the longest correct prefix, resample the first wrong token, and repeat.

The remarkable part is that this is exact. A carefully constructed rejection-sampling step makes the output distribution *identical* to sampling from the target alone — you get the same tokens the big model would have produced, using fewer of its forward passes [@leviathan2023]. That is the free-lunch intuition: no quality cost, only fewer expensive steps.

!!! intuition "Intuition"
    The draft model does the easy guessing; the big model just grades the guesses, and grading a whole batch of them costs about as much as producing one. When the guesses are mostly right, you advance several tokens per expensive step.

The lunch is not unconditional. The speedup is the expected number of accepted tokens per verification, which depends on the **acceptance rate** — how often the draft agrees with the target — and on the draft being genuinely cheap. A weak draft gets rejected constantly and you pay its cost for little gain; too strong a draft is slow enough to erase the win. Acceptance also collapses on hard or high-entropy text where even the target is unsure. A popular variant removes the separate draft entirely: **Medusa** and other self-speculation methods bolt extra decoding heads onto the target model itself to predict several tokens ahead, then verify with tree attention [@cai2024].

<figure class="wide">
<img src="assets/figures/speculative-decoding.svg" alt="A small draft model emits four proposed tokens; the large target model verifies all four in one parallel pass; the first three match and are accepted while the fourth is rejected and replaced by the target's own token.">
<figcaption>Four tokens from one expensive pass. The cheap draft proposes a short run; the target verifies the whole run in a single parallel forward pass, accepts the correct prefix, and corrects the first miss. The average number accepted per pass is the entire speedup.</figcaption>
</figure>

## FlashAttention moves less memory, not fewer FLOPs

The last lever returns to Chapter 4's $O(n^2)$ and reads it correctly. The quadratic cost is real, but a standard attention implementation is slow for a subtler reason: it builds the full $n \times n$ score matrix in the GPU's large, slow memory (HBM), writes it out, reads it back to apply softmax, and reads it yet again to multiply by the values. Attention is dominated by that memory traffic, not by the arithmetic.

**FlashAttention** is IO-aware: it computes the *exact* same attention without ever materializing the score matrix [@dao2022]. It tiles Q, K, and V into blocks small enough to fit in the GPU's fast on-chip memory (SRAM), and streams over them, maintaining a running "online" softmax that updates the output block by block. Nothing $n \times n$ is ever written to HBM; on the backward pass it recomputes the needed blocks rather than storing them, trading a little extra arithmetic for a large cut in memory traffic.

!!! warning "Common trap"
    FlashAttention does not reduce the FLOP count — it does the same (or slightly more) arithmetic. It is faster because attention was memory-bound, so cutting HBM reads and writes is what actually moves the clock. Calling it an "approximation" or a "lower-complexity attention" is the giveaway that a candidate has the mental model wrong: it is exact, and still $O(n^2)$ in compute.

The follow-on **FlashAttention-2** rebalances the work across the GPU's thread blocks and warps to keep more of the hardware busy, closing much of the remaining gap to peak [@dao2023]; later versions specialize further for newer accelerators. All of them are the same idea: keep the working set in fast memory and never spill the big matrix.

<figure class="wide">
<img src="assets/figures/flash-attention-tiling.svg" alt="On the left, HBM holds Q, K, and V, with the n-by-n score matrix crossed out as never built. On the right, small tiles of Q, K, and V are loaded into fast SRAM where an online softmax accumulates the output, which is written back to HBM.">
<figcaption>Where the time actually goes. Standard attention pays to write and re-read an n-by-n matrix in slow HBM; FlashAttention keeps tiles in fast SRAM and accumulates the exact result with an online softmax, so the same math costs a fraction of the memory traffic.</figcaption>
</figure>

Together these techniques decide what a model costs to run. The next chapter cuts the bytes themselves with quantization (Chapter 16), and Chapter 17 assembles all of it into a production serving stack.
