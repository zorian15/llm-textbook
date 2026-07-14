Distributed training sounds like an infrastructure appendix, but it is where interviews reliably go deep, and it runs on one plain idea: a model's training state is too big for any single device, so you must decide what to **replicate** and what to **shard** — and every choice buys memory with communication. This chapter builds the standard answers in the order teams reach for them: shard the optimizer, then split the layers, then split the stack, then compose all three; and it ends with the communication primitives that decide what any of it costs.

## Why one GPU is never enough

Chapter 2's interview question established that training holds far more than weights. Make it concrete with mixed-precision AdamW, per parameter: bf16 weights (2 bytes) and gradients (2 bytes), plus an fp32 master copy of the weights (4) and Adam's two moments (4 + 4) — **16 bytes per parameter** before a single activation is stored. A 70B-parameter model therefore carries about 1.1 TB of training state against an 80 GB accelerator: fourteen devices just to *hold* the problem, before the batch's activations — which scale with batch size, sequence length, and depth, and at long context can rival the fixed state — claim their share.

Notice the villain ranking: the optimizer state (12 of the 16 bytes) dwarfs the weights you actually wanted. That single observation motivates the entire next section.

!!! intuition "Intuition"
    Distributed training is a memory problem first and a speed problem second. You do not scale to a thousand GPUs because you are impatient; you scale because the training state does not physically fit on one, and then you fight to keep the thousand busy.

<figure class="wide">
<img src="assets/figures/memory-budget.svg" alt="A stacked bar of training memory per parameter: 2 bytes of bf16 weights, 2 of gradients, and 12 of fp32 optimizer state, totalling 16 bytes per parameter, with activations added on top; a marker shows an 80 GB GPU dwarfed by a 70B model's 1.1 TB of state.">
<figcaption>Where the memory actually goes. The weights are the smallest slice of their own training run — the optimizer state is six times larger — so the first thing to shard is not the model at all.</figcaption>
</figure>

## Data parallelism and ZeRO/FSDP

**Data parallelism (DP)** is the default: copy the full model to every GPU, give each a different slice of the batch, and after the backward pass **all-reduce** the gradients so every copy takes the identical optimizer step. Compute scales beautifully. Memory does not improve at all — every GPU still holds all 16 bytes per parameter, so plain DP cannot train a model that does not already fit.

**ZeRO**'s observation is that the replication is redundant [@rajbhandari2020]: across $N$ data-parallel GPUs, there is no reason to keep $N$ identical copies of state that could be split $N$ ways. Its three stages form a memory ladder, each rung shard­ing the next-largest tenant:

1. **Stage 1** shards the optimizer state (the 12-byte villain). Each GPU updates only its shard of the parameters.
2. **Stage 2** also shards the gradients: instead of all-reducing full gradients, each GPU receives only the reduced gradient for its own shard (a **reduce-scatter**).
3. **Stage 3** also shards the parameters themselves. Nothing is replicated; each layer's weights are **all-gathered** just in time for its forward or backward pass, used, and discarded.

Stage 3 is what PyTorch ships as **FSDP** — fully sharded data parallel [@zhao2023]. Per-GPU memory falls almost $N$-fold, at the price of moving parameters over the wire every step; the arithmetic of whether that price is payable is the interconnect question the last section answers.

!!! note "Note"
    ZeRO does not change the math. Every stage computes bit-for-bit the same optimizer step as plain DP — it is a storage layout, not an algorithm. That is its charm: no hyperparameter interacts with it, so it composes silently with everything else in this chapter.

<figure class="wide">
<img src="assets/figures/zero-stages.svg" alt="Four rows show what each of four GPUs holds under plain data parallelism and ZeRO stages 1 to 3: replication of optimizer state, gradients, and parameters is progressively replaced by one shard each, shrinking per-GPU memory at each stage.">
<figcaption>The ZeRO ladder. Plain data parallelism keeps N identical copies of everything; each stage deletes the next-largest redundancy — optimizer state, then gradients, then the parameters themselves — until every GPU holds only its 1/N slice.</figcaption>
</figure>

## Tensor and pipeline parallelism

ZeRO shards *storage* but still runs every layer's full computation on each GPU. The model-parallel families split the computation itself, along two orthogonal cuts.

**Tensor parallelism (TP)** slices *within* a layer [@shoeybi2019]. A weight matrix is split column-wise across GPUs; each computes its slice of the matmul, and an all-reduce reassembles the result. Megatron's arrangement needs only two all-reduces per transformer block by pairing a column-split matrix with a row-split one — but those all-reduces sit on the critical path of *every layer, every microbatch*. TP is therefore chatty and latency-sensitive, and in practice lives only *inside* a node, where NVLink-class bandwidth makes the chatter affordable.

**Pipeline parallelism (PP)** slices *between* layers [@huang2019]: GPU 0 takes blocks 1–8, GPU 1 takes 9–16, and activations flow stage to stage. Communication is tiny — one activation tensor per boundary — so pipelines happily cross nodes. The tax is the **bubble**: stage $k$ idles until work reaches it, and idles again as the pipeline drains. The fix is to chop the batch into $m$ microbatches so stages overlap; with $p$ stages the idle fraction is roughly $(p-1)/(m+p-1)$, which is why pipeline configurations always come with large microbatch counts and why deeper pipelines demand them.

!!! analogy "Analogy"
    A pipeline is an assembly line for batches: the line is only efficient once every station is busy, so you feed it many small jobs rather than one big one — and the first and last moments of a shift, when the line is filling and draining, are pure overhead. The analogy leaks at the backward pass: unlike a real assembly line, every job must also flow *backwards* through the same stations, which is what makes pipeline schedules a genuine scheduling-theory problem.

<figure class="wide">
<img src="assets/figures/tensor-pipeline.svg" alt="Two cuts through the same stack of transformer blocks: tensor parallelism splits each layer's matrices across GPUs and reassembles results with per-layer all-reduces; pipeline parallelism assigns contiguous groups of layers to different GPUs, with microbatches and an idle bubble at the pipeline's start.">
<figcaption>Two orthogonal cuts. Tensor parallelism splits every matrix and pays constant communication inside each layer; pipeline parallelism splits the stack and pays in idle bubble time instead. One is bounded by bandwidth, the other by scheduling.</figcaption>
</figure>

## Putting it together: 3D parallelism

The three parallelisms answer different constraints, so frontier training uses all of them at once [@narayanan2021], and the composition follows the hardware's own hierarchy:

- **TP** spans the GPUs *within* one node, where bandwidth is highest — typically 8.
- **PP** spans nodes, because stage boundaries tolerate slower links — say 16 stages.
- **DP** (with ZeRO sharding as needed) replicates that whole 128-GPU model instance and splits the global batch across replicas.

The back-of-envelope for a Llama-3-405B-class run [@grattafiori2024]: $8 \text{ (TP)} \times 16 \text{ (PP)} = 128$ GPUs to hold one model instance, times $128$ data-parallel replicas $= 16{,}384$ GPUs, with the global batch divided among replicas and each replica's share divided into microbatches to keep its pipeline full. Long-context training adds a fourth axis (splitting the *sequence* itself), and mixture-of-experts models add expert parallelism (Chapter 5) — but every axis is still the same one decision, replicate or shard, applied to a different dimension of the problem.

!!! interview "Interview"
    *How would you place TP, PP, and DP on a cluster, and why in that order?* Match communication appetite to link speed: TP all-reduces on every layer, so it gets the intra-node NVLink; PP sends one activation per stage boundary, so it can cross nodes; DP synchronizes once per step and overlaps with compute, so it tolerates the slowest links. An answer that inverts this ordering fails the question regardless of its other details.

<figure class="wide">
<img src="assets/figures/parallelism-3d.svg" alt="A grid of GPUs organized as tensor parallelism within each node, pipeline stages across nodes forming one model instance, and data-parallel replicas of that instance; each axis is labeled with its communication pattern and preferred link.">
<figcaption>The composition follows the hardware. The chattiest parallelism gets the fastest links: tensor within the node, pipeline across nodes, data parallelism across replicas. A frontier run is this diagram times a few thousand GPUs.</figcaption>
</figure>

## The communication primitives

Every scheme above reduces to three collective operations, and knowing them turns hand-waving into arithmetic:

- **All-reduce**: every GPU contributes an array; every GPU ends with the elementwise sum. Data parallelism's gradient sync.
- **Reduce-scatter**: the same sum, but each GPU keeps only its $1/N$ shard of the result. ZeRO-2's gradient move.
- **All-gather**: each GPU contributes a shard; every GPU ends with the concatenation. How ZeRO-3/FSDP rematerializes parameters.

The identity worth memorizing: **all-reduce = reduce-scatter + all-gather**. Ring implementations of the halves each move about $(N-1)/N$ of the data volume per GPU — near bandwidth-optimal, and nearly independent of $N$ — so an all-reduce costs roughly $2\times$ the array size in traffic per GPU, whatever the cluster size. ZeRO's stages are not exotic protocols; they are the two halves of the all-reduce DP already performed, with storage rearranged between them.

That constant factor is why *interconnect bandwidth is destiny*. The traffic per step is fixed by model size and parallelism layout; the wall-clock it costs is that traffic divided by link speed — and the links span orders of magnitude, from NVLink-class intra-node fabric (hundreds of GB/s per GPU) to inter-node InfiniBand (tens). Whether communication hides behind computation or dominates it is decided by which link each collective runs over; that is the entire content of the placement rules above, and it is why clusters are bought around their network as much as their FLOPs. The same bandwidth obsession returns at serving time (Chapter 15), where the wire is replaced by GPU memory itself.

<figure class="wide">
<img src="assets/figures/collectives.svg" alt="Four GPUs illustrate the three collectives: all-reduce leaves every GPU with the full sum, reduce-scatter leaves each with one summed shard, and all-gather reassembles shards everywhere; a bracket notes that all-reduce equals reduce-scatter followed by all-gather.">
<figcaption>Three moves, one identity. All-reduce is reduce-scatter followed by all-gather, and each half costs about the array size in per-GPU traffic regardless of cluster size — the constant that makes communication budgets predictable.</figcaption>
</figure>

With the machinery to hold and feed a model at any size, one question is left hanging over Part II: how big should the model be, and how many tokens should it eat? That question turns out to have an equation.
