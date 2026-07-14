A pretraining run is Chapter 2's five-line loop with the safety rails removed: one pass over trillions of tokens, on thousands of GPUs, for weeks or months, with no chance to start over if week six goes wrong. You cannot grid-search a run you can only afford once. What replaces the grid search is a small set of scaling rules, a trained eye for the loss curve, and machinery for keeping a long run alive. That is this chapter: the knobs you set before launch, the signals you watch after it, and what you do when the signals go bad.

## Batch size, learning rate, and warmup

Every large run uses the same learning-rate shape: climb, then glide. **Warmup** ramps the learning rate linearly from zero over the first fraction of a percent of training; **decay** — usually a cosine — then lowers it smoothly to a small fraction of its peak by the final token [@loshchilov2017]. Warmup exists because the start of training is the most dangerous part: the weights are random, the first gradients are huge and unrepresentative, and Adam's per-parameter gradient statistics (Chapter 2) are estimated from almost no samples. Large steps taken on garbage statistics can wreck the run in its first thousand steps; warmup keeps steps tiny until the optimizer's bookkeeping is trustworthy. The decay end matters just as much — the final low-learning-rate stretch is where the loss quietly earns its last few percent, which is exactly why Chapter 6's data annealing is timed to it.

!!! note "Note"
    A newer alternative, warmup-stable-decay (WSD), holds the rate flat for most of training and decays only in a short final phase [@hu2024]. The appeal is operational: a checkpoint from the stable phase can be branched and annealed at any point, so one long run yields models of several sizes of training budget — with a cosine, the schedule must know its endpoint from step one.

Batch size is the other pre-launch commitment, and it obeys a diminishing-returns law. Averaging more examples per step reduces gradient noise, so bigger batches support proportionally bigger learning rates — the *linear scaling rule* [@goyal2017]. But once the batch is large enough that noise no longer dominates, doubling it again buys almost nothing: the gradient is already accurate, and you are spending twice the compute per step to take essentially the same step. The crossover point — the **critical batch size** — can be estimated from the gradient's own noise statistics, and it grows as the loss falls, which is why large runs often *increase* batch size mid-training [@mccandlish2018].

!!! intuition "Intuition"
    A batch is a poll of the data. Small polls are noisy, so more respondents help a lot at first; past a few thousand, the poll is already right, and more respondents just cost money. The critical batch size is where the poll stops being the bottleneck.

<figure class="wide">
<img src="assets/figures/warmup-cosine.svg" alt="Learning rate over training steps: a short linear warmup to the peak, then a long cosine decay to near zero; a dashed warmup-stable-decay variant holds flat and drops late.">
<figcaption>Climb, then glide. Warmup protects the run while the optimizer's statistics are still garbage; the long decay is where the final polish happens — and where the best data is spent (Chapter 6). The dashed WSD variant defers the decay so one run can be branched at many budgets.</figcaption>
</figure>

## Loss curves and what they tell you

The training loss is the run's vital sign, and healthy is boring: a fast early drop as the model learns token frequencies and grammar, then an ever-slower grind that looks like a straight line on a log-scale plot — the power law Chapter 9 makes precise. Deviations from boring are diagnoses:

- **A plateau** that arrives too early usually means the learning rate is too low, the data loader is recycling (the model has seen this before), or a preprocessing bug is feeding degenerate batches.
- **A spike** — the loss jumps sharply and then may or may not recover — is the signature event of large-scale training. PaLM's team reported roughly twenty of them across a run, each at a scale where the same data replayed from a different checkpoint caused no spike: the trigger is a bad *collision* of model state and batch, not a bad batch alone [@chowdhery2023].
- **Divergence** — the loss climbs and keeps climbing — means the run is dead and something structural was wrong: learning rate too high, a precision problem (Chapter 2's fp16 range trap), or missing normalization.

The standard spike response is unglamorous: rewind to the last good checkpoint, skip a few hundred batches, resume [@chowdhery2023]. Prevention is the next section's subject.

!!! interview "Interview"
    *Your eval loss is dramatically better than your training loss early in the run. Celebrate?* No — suspect leakage. The classic causes are eval data present in training (Chapter 6's decontamination failed) or a broken causal mask letting the model read ahead (Chapter 4). A number that is too good is a bug report, not a result.

<figure class="wide">
<img src="assets/figures/loss-spike.svg" alt="A training loss curve falling smoothly on a log step axis, interrupted by a sharp spike that is handled by rewinding to a checkpoint and skipping batches; an untreated path diverges upward.">
<figcaption>Healthy is boring. The loss should grind down a power law; a spike is the field's most-watched pathology, and the standard cure — rewind, skip the offending batches, resume — is as unglamorous as it is effective.</figcaption>
</figure>

## Stability tricks

Each stability trick in the modern recipe is a patch over a specific, named failure. Four cover most of what you will meet:

- **Gradient clipping** rescales any batch gradient whose global norm exceeds a threshold (typically 1.0). It converts a rare catastrophic step into a merely wasted one — a shock absorber, not a steering input. If clipping fires *often*, the learning rate is wrong; the trick is for outliers.
- **Z-loss** addresses a slow drift in the final softmax: nothing in cross-entropy prevents the logits from all growing together, and in low precision that growth eventually misbehaves. Z-loss adds a small penalty on the softmax normalizer, pinning the logit scale [@chowdhery2023; @zoph2022].
- **QK-norm** normalizes queries and keys before their dot product, capping attention logits — the fix for attention-entropy collapse at scale, met already in Chapter 5 [@dehghani2023].
- **Careful initialization and bf16** (Chapter 2) round out the kit: residual-branch scaling keeps the residual stream's variance flat with depth, and bf16's full exponent range removes the overflow failure fp16 invited.

!!! warning "Common trap"
    These tricks are cheap insurance at small scale and load-bearing at large scale — and that asymmetry is the trap. An ablation at 1B parameters showing "z-loss makes no difference" tells you little about 100B: instabilities sharpen with scale, and the run where you learn this is the expensive one. Stability choices are made from *others'* published failures, not your own ablations.

<figure class="wide">
<img src="assets/figures/stability-map.svg" alt="Four failure modes each paired with a mechanism and its fix: gradient outliers to clipping, softmax logit drift to z-loss, attention logit growth to QK-norm, and depth-growing variance to residual-scaled initialization.">
<figcaption>Patches with addresses. Each trick in the stability kit answers one named failure mechanism — none of them is generic tuning, which is why the modern recipe carries all of them at once.</figcaption>
</figure>

## Checkpointing and reproducibility

At cluster scale, hardware failure is not a risk; it is a schedule. Chain thousands of GPUs into one synchronous computation and the run's time-between-failures shrinks to hours — Llama 3's team logged over four hundred unexpected interruptions across a 54-day run, the majority from GPU and memory faults [@grattafiori2024]; the OPT team's public logbook records the same life at 175B scale, restart by restart [@zhang2022]. The design consequence: a run is not a process that finishes but a process that *resumes*, and the checkpoint is its unit of survival.

A resumable checkpoint holds more than weights: the optimizer state (Adam's moments — twice the size of the model, per Chapter 8's accounting), the learning-rate schedule position, the data loader's exact position, and the random-number-generator states. Miss the data-loader state and resumption silently double-feeds part of the corpus — a bug that shows up in nothing but a slightly wrong model. Checkpoint frequency is its own small optimization: too rare and each failure burns hours of GPU-time; too frequent and the writes themselves throttle the run, which is why large jobs stage checkpoints asynchronously to fast local storage before shipping them off-node.

Exact reproducibility, meanwhile, is quietly abandoned at scale, for a reason worth knowing precisely: floating-point addition is not associative — $(a+b)+c$ and $a+(b+c)$ can differ in the last bit — and GPU kernels sum in whatever order the hardware schedules, which varies run to run. Those last-bit differences feed a chaotic system millions of times, so two "identical" runs part ways. What teams actually guarantee is *statistical* reproducibility — same data, same seeds, same curve shape — plus bitwise-faithful *resumption* from a checkpoint, which is a much narrower promise.

!!! interview "Interview"
    *Why is bit-exact reproducibility hard on GPUs, and what do teams promise instead?* Because massively parallel reductions sum floats in nondeterministic order, and float addition is not associative, so each run accumulates different rounding — amplified over millions of steps. Teams promise determinism where it is cheap (fixed seeds, deterministic data order, faithful checkpoint resumption) and accept run-to-run weight divergence as physics.

<figure class="wide">
<img src="assets/figures/checkpoint-timeline.svg" alt="A training timeline interrupted by hardware failures; after each failure the run rewinds to the last checkpoint and replays the lost interval, shown as shaded rework.">
<figcaption>A run is a process that resumes. At thousands of GPUs, failures arrive on a schedule of hours; the checkpoint cadence decides how much work each one destroys, which makes checkpointing a first-order design choice rather than hygiene.</figcaption>
</figure>

Everything here assumed the model already fits on the hardware and the steps already run fast. Making that true is its own discipline — the next chapter's subject.
