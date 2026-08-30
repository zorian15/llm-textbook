A trained, quantized model is a file on disk. A *serving system* is the software that turns that file into a service — one that answers thousands of users at once, keeps its promises about speed, and does not cost more to run than it earns. This is the layer between the weights and the API, and it is where the ideas of the last three chapters — how you decode (Chapter 14), the prefill/decode split and the KV cache (Chapter 15), and fewer bits per weight (Chapter 16) — stop being properties of one request and start being properties of a fleet. The whole chapter turns on one tension: the tricks that make a *single* request fast can make the *system* slow, and resolving that tradeoff is the serving engineer's real job.

## What a serving stack must do

Start with what a user actually feels, because it splits into two separate clocks. **Time to first token (TTFT)** is the wait from sending a prompt to seeing the first word appear; it is set by *prefill*, the one-shot pass that reads the whole prompt, so it grows with prompt length. **Time per output token (TPOT**, also called inter-token latency**)** is the pace of every word after that; it is set by *decode*, the memory-bandwidth-bound step that streams the KV cache once per token (Chapter 15). A chat feels sluggish if either clock is slow, but for different reasons and with different fixes.

<figure class="wide">
<img src="assets/figures/latency-metrics.svg" alt="A request timeline: a prefill block runs from arrival to the first token, setting TTFT; then decode emits tokens at a steady cadence, and the gap between two of them is TPOT.">
<figcaption>Two clocks, one request. TTFT is the wait for the first word, paid during prefill; TPOT is the cadence of the rest, paid once per decode step. Optimizing one does not automatically help the other, which is why a serving stack reports both.</figcaption>
</figure>

Against these sit **throughput** — total tokens per second across *all* concurrent users — and **cost**, which for a fixed GPU is just the inverse of throughput. The tension is that latency and throughput pull opposite ways: the batching that lifts throughput makes each user wait a little longer in the queue and share the GPU. So you do not "make it fast"; you commit to **service-level objectives (SLOs)** — say, TTFT under 500 ms and TPOT under 50 ms at the 99th percentile — and then maximize throughput *subject to* meeting them.

!!! intuition "Intuition"
    A serving system is a restaurant kitchen, not a personal chef. One diner served alone gets their meal fastest, but the kitchen earns nothing; the art is cooking many orders at once so no single table waits past what it will tolerate.

!!! interview "Interview"
    *Your TTFT is fine but users complain the model "types slowly." Which knob?* That is a TPOT problem, so look at decode, not prefill. Decode is bound by streaming the KV cache from memory each step, so the levers are a smaller cache (GQA, quantized KV; Chapters 5 and 16), a smaller batch on that replica, or speculative decoding (Chapter 15). Throwing more prompt-side compute at it does nothing, because prefill already finished.

## The engines

You almost never write this layer yourself; you pick an **inference engine**. Four dominate in 2026, and each is known for one signature idea. **vLLM** popularized **PagedAttention**, which stores the KV cache in fixed-size pages so memory is not fragmented by variable-length sequences, and pairs it with continuous batching [@kwon2023]; it is the open-source default. **TensorRT-LLM** compiles a model into fused, hardware-specific kernels ahead of time, trading flexibility for peak throughput on NVIDIA hardware. **SGLang** is built around **RadixAttention**, a prefix cache that reuses shared prompt prefixes automatically, and fast structured-output decoding, which makes it strong for agents and heavily templated workloads [@zheng2024]. **Text Generation Inference (TGI)** is Hugging Face's production wrapper, valued less for a novel kernel than for streaming, metrics, and safe rollout inside that ecosystem.

<figure class="wide">
<img src="assets/figures/serving-engines.svg" alt="Four cards — vLLM, TensorRT-LLM, SGLang, TGI — each labeled with its signature technique, all resting on a shared foundation of paged KV cache and FlashAttention kernels.">
<figcaption>What differentiates the engines is a thin layer on top of a shared foundation. Paged KV memory and FlashAttention kernels (Chapter 15) are table stakes; the choice is really about compilation, prefix reuse, or ecosystem fit.</figcaption>
</figure>

The important point is that these engines share more than they differ: nearly all now do paged KV memory, continuous batching, prefix caching, and FlashAttention-style kernels [@dao2022]. So the choice is rarely about raw speed on a single benchmark and usually about fit — TensorRT-LLM if you will pay a compile step for the last 20% on NVIDIA, SGLang if your traffic has long shared prefixes, TGI or vLLM if you want to swap models freely.

!!! interview "Interview"
    *Why not just call a model's `generate()` in a loop behind a web server?* Because it serves one request at a time and re-runs prefill from scratch on every call, so your GPU sits mostly idle and your cost per token is an order of magnitude too high. Everything an engine adds — continuous batching, a paged and reusable KV cache, scheduling — exists to keep that expensive GPU busy across many requests at once.

## Multi-request scheduling

The engine's real intelligence is in *scheduling* many requests onto one GPU. The foundational move is **continuous batching** (also called in-flight or iteration-level batching), introduced by Orca: instead of freezing a batch until every request in it finishes, the scheduler makes decisions *every iteration*, so a request that completes early releases its slot immediately and a newly arrived one joins on the next step [@yu2022]. Static batching wastes the GPU whenever requests in a batch have different lengths, which they always do.

<figure class="wide">
<img src="assets/figures/continuous-batching.svg" alt="Two Gantt-style schedules. In static batching, short requests finish early but their slots sit idle until the longest request ends. In continuous batching, a new request fills each freed slot at the next iteration, leaving little idle space.">
<figcaption>Why continuous batching wins. Static batching holds a finished request's slot idle until the slowest sequence in the batch drains; iteration-level scheduling refills that slot at once. Same work, far less idle GPU.</figcaption>
</figure>

The second lever is **prefix caching**: when many requests share a long common prefix — a system prompt, a few-shot preamble, a document everyone asks about — you prefill that prefix *once* and reuse its KV cache for all of them. SGLang's RadixAttention organizes cached prefixes in a tree so overlaps are found and shared automatically [@zheng2024]. A stable system prompt is therefore nearly free after the first request warms it, which is why prompt design and serving cost are linked.

<figure>
<img src="assets/figures/prefix-cache.svg" alt="A tree with a shared system prompt at the root, whose KV cache is computed once, branching into several user turns that each add only their own new tokens.">
<figcaption>Reuse, not recompute. The long shared prefix is prefilled a single time at the root; each request pays only for its own suffix. The more your traffic shares, the more this saves.</figcaption>
</figure>

Two refinements handle the messiness of mixed traffic. A long prefill can stall everyone else's decode, since one big compute step blocks the batch; **chunked prefill** slices a prompt into pieces and interleaves them with ongoing decodes, smoothing tail latency [@agrawal2024]. Going further, **prefill/decode disaggregation** runs the two phases on *different* GPU pools — prefill is compute-bound, decode is memory-bound, so co-locating them makes each interfere with the other — and routing them separately lets you scale and tune each for its own bottleneck [@zhong2024]. On top of all this sits ordinary **prioritization and fairness**: without it, one user streaming a 100K-token prompt starves everyone behind them.

!!! analogy "Analogy"
    Continuous batching is a ride-share van that picks up and drops off along the route instead of waiting to fill and empty all at once. The analogy leaks in that van seats are interchangeable, but a request also carries its growing KV cache, so "seating" a new rider costs memory that a real van never charges — which is exactly the pressure PagedAttention was built to relieve.

## Autoscaling and the economics

Zoom out to the fleet and one number governs everything: **GPU utilization**. A GPU costs the same whether it serves one token per second or three thousand, so the unit cost of a token is essentially *rental price divided by throughput*, and throughput is set almost entirely by how full your batches are. This is why batching is not one optimization among many — it *is* the cost story.

<figure class="wide">
<img src="assets/figures/throughput-frontier.svg" alt="A curve of per-user latency against system throughput as batch size grows from 1 to 256: throughput rises steeply then saturates while latency creeps up and then climbs sharply, with an SLO ceiling marking the largest usable batch.">
<figcaption>The frontier you actually tune on. Bigger batches buy throughput — and cheaper tokens — by spending per-user latency. The operating point is the largest batch whose latency still clears your SLO; past the knee you pay a lot of latency for little extra throughput.</figcaption>
</figure>

**Autoscaling** adjusts replica count to demand, usually off queue depth or TTFT rather than raw GPU percentage, because a GPU can look busy while delivering terrible latency. The hard part is the **cold start**: spinning up a new replica means loading tens of gigabytes of weights into VRAM and warming caches, which takes tens of seconds to minutes — an eternity when a traffic spike is happening *now*. Teams fight this by keeping warm pools, quantizing to shrink load time (Chapter 16), and scaling ahead of demand instead of reacting to it.

!!! interview "Interview"
    *Where does the cost of a served token actually go, and what is the biggest lever?* Cost per token is GPU-hour price divided by tokens produced per hour, so the dominant lever is throughput at fixed hardware, and the dominant lever *on throughput* is effective batch size — keeping the GPU full. Quantization and a smaller KV cache help mainly because they let you fit a *bigger* batch in the same memory, not because the arithmetic itself got cheaper. A team obsessing over kernel microseconds while running batch sizes of two is optimizing the wrong end.

Serving is where the whole book meets the electricity bill. A model is only as good as the service wrapped around it can deliver, affordably, at the ninety-ninth percentile — and Part V now turns to the other half of that wrapper: the prompts, tools, and guardrails that decide what the model is actually asked to do.
