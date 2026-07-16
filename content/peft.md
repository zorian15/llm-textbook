Full fine-tuning updates every weight in the model, which is the obvious way to adapt a pretrained model to a new task and, for large models, an expensive one. It demands the same memory as a training run from scratch, and it leaves you with a full-size copy of the model for every task you fine-tune. Parameter-efficient fine-tuning (PEFT) refuses that bargain: it freezes the pretrained weights and trains a tiny add-on, a few million parameters instead of billions. This chapter is mostly about one method — **LoRA** — that has become the default because it costs almost nothing to train, nothing at all to serve once merged, and barely dents quality.

## Why full fine-tuning hurts

The cost is not the weights; it is everything training drags along with them. Chapter 8's memory budget is the whole argument: mixed-precision AdamW carries about 16 bytes per parameter — the bf16 weights, their gradients, an fp32 master copy, and two optimizer moments — so the optimizer state alone is six times the size of the model. Fine-tuning a 7B model this way needs well over 100 GB of state before activations, which is why a model that *runs* on one GPU will not *train* on it. And the moment you fine-tune a second task, you own a second full checkpoint: ten specialized 70B models are ten times 140 GB to store and to load.

PEFT attacks both problems at once. Freeze the pretrained weights and they need no gradients, no master copy, and no optimizer moments — they sit in memory as a single read-only tensor, shared across every task. Only the small add-on gets an optimizer, so the 12-bytes-per-parameter tax falls on millions of parameters rather than billions. The base model becomes a fixed substrate; each task is a lightweight overlay.

!!! interview "Interview"
    *Fine-tuning ran out of memory but inference on the same model was fine. Why, and what does PEFT change?* Inference holds only weights and one layer's activations; training adds gradients, an fp32 master copy, and two AdamW moments — roughly 16 bytes per parameter (Chapter 2). Freezing the base removes the last three for almost all of the model, so PEFT's footprint is dominated by the frozen weights (which you were already paying for) plus a sliver of trainable state.

<figure class="wide">
<img src="assets/figures/peft-memory.svg" alt="Two stacked bars of GPU memory to fine-tune a 7B model. Full fine-tuning stacks bf16 weights, gradients, and fp32 optimizer state to over 100 GB. LoRA keeps only the frozen bf16 base plus a sliver of adapter state, near 14 GB.">
<figcaption>The optimizer state, not the weights, is what full fine-tuning cannot afford. Freezing the base deletes the gradient and optimizer tax on billions of parameters, and leaves one shared copy of the weights instead of a fresh checkpoint per task.</figcaption>
</figure>

## LoRA

LoRA (Low-Rank Adaptation) rests on a hypothesis about *how* a model changes when you fine-tune it: the update is intrinsically low-rank [@hu2021]. Adapting a base model to a task does not rewire it — it nudges it along a handful of directions. Aghajanyan and colleagues measured this directly, tuning RoBERTa to most of its full performance through only a few hundred parameters projected into weight space, and found the "intrinsic dimension" of fine-tuning is small and shrinks as models grow [@aghajanyan2021].

So instead of learning a full weight change $\Delta W$ (a $d \times k$ matrix), LoRA forces it through a rank-$r$ bottleneck. It writes the update as a product of two thin matrices, $\Delta W = BA$ with $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times k}$ and $r$ as small as 8, and the adapted layer computes

$$h = W_0 x + \frac{\alpha}{r} BA x.$$

Only $A$ and $B$ train; $W_0$ stays frozen. For a $4096 \times 4096$ projection, a full $\Delta W$ is 16 million numbers; at $r = 8$ the two factors together are about 65 thousand — a 250-fold cut. $B$ starts at zero, so the adapter begins as the identity and training only ever adds to a working model. The scalar $\alpha / r$ decouples the update's magnitude from the rank you happened to choose.

!!! intuition "Intuition"
    A full update lets the model move in every direction in weight space; LoRA gives it a few well-chosen dials instead. The bet is that a task's worth of change was never high-dimensional to begin with, so a few dials reach almost the same place at a fraction of the cost.

The payoff at inference is that there is none to pay. Because $W_0 + \frac{\alpha}{r}BA$ is just another $d \times k$ matrix, you can fold the adapter into the base weights once and serve a model that is bit-for-bit the shape of the original — no extra layers, no added latency. This is the sharp contrast with older adapter methods that inserted extra modules into the forward pass and taxed every token forever.

!!! interview "Interview"
    *Does LoRA slow down inference?* Not if you merge it: adding $BA$ into $W_0$ leaves a standard weight matrix, so a merged LoRA model runs exactly as fast as the base. You only pay a small overhead if you deliberately keep the adapter *separate* at inference — which, as the last section shows, is a trade you sometimes make on purpose to serve many adapters at once.

<figure>
<img src="assets/figures/lora-update.svg" alt="A large frozen weight matrix W-zero, in grey and locked, sits beside a thin tall matrix B and a thin wide matrix A whose product is a low-rank update added to W-zero. Only B and A are colored as trainable.">
<figcaption>LoRA routes the whole weight update through a rank-r bottleneck. The frozen matrix keeps everything the model already knew; the two thin trainable factors carry the task, and at r far below the layer width they hold a tiny fraction of the parameters.</figcaption>
</figure>

## QLoRA and quantized adapters

LoRA shrinks the *trainable* state, but you still hold the frozen base in memory, and at 65B parameters even a read-only bf16 copy is 130 GB. **QLoRA** removes that wall by storing the frozen base in 4 bits while training the LoRA adapters in higher precision [@dettmers2023]. Three ideas make it work without hurting quality: **NF4** (NormalFloat4), a 4-bit type shaped for the roughly-Gaussian distribution of neural network weights; **double quantization**, which quantizes even the quantization constants to shave a little more; and **paged optimizers**, which spill optimizer memory to CPU RAM to survive the spikes that would otherwise crash the run.

The mechanics are worth stating plainly. The base weights *live* in 4 bits, but each is dequantized back to bf16 on the fly for its matmul; gradients then flow through those dequantized values into the adapters, never into the frozen quantized weights. The result is that a 65B model fine-tunes on a single 48 GB GPU, and QLoRA reports matching full 16-bit fine-tuning quality on its benchmarks. Chapter 16 covers 4-bit quantization for its own sake; here it is a means to fit the frozen substrate.

!!! warning "Common trap"
    QLoRA is not "train a 4-bit model." Training a genuinely quantized model to be *good* at 4-bit inference is quantization-aware training, a different and harder problem (Chapter 16). QLoRA keeps 4 bits only to *store* the frozen base cheaply; the learning still happens in bf16 adapters, and you can merge and re-quantize afterward as a separate step.

<figure>
<img src="assets/figures/qlora-stack.svg" alt="A frozen base weight stored in 4-bit NF4 is dequantized to bf16 for its matmul; small bf16 LoRA adapters sit alongside. A backward-pass arrow flows only into the adapters, not the quantized base.">
<figcaption>QLoRA fits the frozen model by storing it in 4 bits and dequantizing each weight only for its multiply. The gradient path touches only the bf16 adapters, so the heavy state a run must hold is proportional to the adapters, not the base.</figcaption>
</figure>

## Choosing rank, alpha, and targets

Three knobs decide how a LoRA run goes. **Rank** $r$ sets the adapter's capacity: too low starves a hard task, but returns flatten quickly, and much of the benchmark literature finds little gain past $r = 16$ for typical instruction tuning. **Alpha** scales the update; the common practice of holding $\alpha / r$ roughly constant keeps the effective learning rate steady as you sweep rank. **Targets** name which matrices get adapters — the original work put LoRA only on the attention projections, but adapting the FFN matrices too (Chapter 4's two-thirds of the parameters) often helps, at proportional cost.

Where PEFT plateaus is the honest limit. Adapters excel at teaching *behavior and format* — the assistant persona, a domain's style, a tool-calling convention — which is exactly what SFT (Chapter 10) mostly does. They are weaker at cramming large amounts of new *knowledge* into a model, where the low-rank bottleneck bites and full fine-tuning still wins. Successor methods chip at the gap: **DoRA** splits each weight into a magnitude and a direction and applies LoRA only to the direction, closing much of the remaining distance to full fine-tuning at similar parameter cost [@liu2024].

!!! interview "Interview"
    *A LoRA run underperforms full fine-tuning on your task. What do you try before giving up on PEFT?* Raise the rank; add the FFN and output projections to the target set, not just attention; retune the learning rate, since adapters like higher rates than full fine-tuning; and consider DoRA. If a large gap survives all of that, the task is likely knowledge-injection rather than behavior-shaping — the regime where the low-rank assumption genuinely fails.

<figure>
<img src="assets/figures/lora-rank.svg" alt="Task accuracy rising with LoRA rank on a log axis, climbing steeply from rank 1 and flattening by rank 16 just below a dashed full-fine-tuning reference line.">
<figcaption>Most of what LoRA can buy is bought by a small rank. Accuracy climbs fast and then flattens near full fine-tuning, so the practical question is rarely "how high can rank go" but "which modules to adapt."</figcaption>
</figure>

## Serving many adapters

Keeping the adapter *unmerged* turns LoRA's one weakness into its best serving trick. If every task shares one frozen base and differs only by a few megabytes of $A$ and $B$, a server can hold the base once and swap adapters per request — hot-swapping a customer's fine-tune in milliseconds instead of loading a whole new model. The multi-tenant story is dramatic: thousands of specialized models backed by a single copy of the expensive weights.

The subtlety is batching. Merging fuses the adapter into the base and gives zero latency, but a merged model serves exactly one task, so a batch of requests for different adapters cannot share a forward pass. Systems like **S-LoRA** keep adapters separate and compute the base matmul once for the whole batch while applying each request's low-rank term with a custom kernel, serving thousands of concurrent adapters at throughput close to the unadapted model [@sheng2024]. It is the same merge-versus-separate trade from the LoRA section, now decided by whether you optimize a single tenant's latency or many tenants' shared throughput — the serving concern of Chapter 17, and one more reason the 4-bit base of Chapter 16 matters.

!!! analogy "Analogy"
    One frozen base with swappable adapters is a game console: an expensive machine you buy once, and cheap cartridges that reprogram it per session. The analogy leaks in that a cartridge is inert data while a LoRA adapter is live weights in the forward pass — which is exactly why batching many "cartridges" against one console at the same time takes the special kernels S-LoRA provides.

<figure class="wide">
<img src="assets/figures/adapter-serving.svg" alt="A single frozen base model in GPU memory serving a batch of three requests, each tagged with a different small LoRA adapter that is applied on top of the shared base forward pass.">
<figcaption>Unmerged adapters make one base model behave like thousands. The costly weights are loaded once and shared across a batch; each request contributes only its own few-megabyte adapter, applied alongside the shared computation.</figcaption>
</figure>

PEFT changes the economics of the whole post-training story: adaptation stops being a per-task copy of an enormous model and becomes a small file you train cheaply, store by the thousand, and serve against one shared base. The next part turns from making models behave to making them fast — the serving and quantization machinery these adapters lean on.
