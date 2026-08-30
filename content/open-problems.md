The rest of this book is about what works: how a transformer reads a sequence, how pretraining and alignment shape it, how a harness turns weights into a product, how you measure the result. This chapter is about what does not work yet. The same next-token predictor that writes working code will state a false fact with total confidence, resist every attempt to read its internal computation, and give us no alignment method we would fully trust on a system smarter than we are. These are not engineering bugs waiting on a patch; they are the field's live research questions, and knowing their shape is part of holding the current picture. The preface promised the load-bearing ideas *and* where the field is still stuck. Here is the stuck part.

## Confident, and wrong

A model *hallucinates* when it produces a fluent, specific claim that is simply false — a citation to a paper that does not exist, a confident wrong date. The uncomfortable part is that this follows directly from the training objective, not from a defect layered on top. Pretraining rewards the most *plausible* continuation of text (Chapter 6), and a plausible-sounding answer scores well whether or not it is true. Recent analysis argues the problem is sharpened by how we train and grade: benchmarks reward a guess over an "I don't know," so a model optimized to score well learns to guess when uncertain, exactly as a student does on an exam that does not penalize wrong answers [@kalai2025].

The deeper issue is *calibration*, which is distinct from accuracy. A calibrated model is one whose stated confidence matches its hit rate: of the things it says it is 80% sure of, it should be right about 80% of the time. A model can be inaccurate yet calibrated (unsure and knows it), or accurate yet miscalibrated (usually right but can't tell when it isn't). Base models come out of pretraining reasonably calibrated; alignment tends to *degrade* calibration, producing a model that is more useful and more confidently wrong at the same time.

!!! intuition "Intuition"
    Accuracy is how often the model is right. Calibration is whether it knows how often it is right. Hallucination is a failure of the second even more than the first.

<figure>
<img src="assets/figures/confidence-calibration.svg" alt="A confidence-versus-accuracy plot: the base model tracks the diagonal, while the post-RLHF curve dips below it at high confidence, opening a gap where stated confidence exceeds real accuracy.">
<figcaption>Why "confidently wrong" is the precise problem. On the diagonal, expressed confidence equals real accuracy. Post-training pushes the curve below the line at the high end: the model asserts 90% and is right far less often, and that gap — not raw error rate — is what makes hallucination dangerous.</figcaption>
</figure>

The partial fixes each attack one facet and none closes it. Retrieval (Chapter 21) grounds an answer in fetched documents, which helps when the answer is *in* a document and does nothing for reasoning errors. Verification (Chapter 25) checks a chain of reasoning against a tool or a checker, which works only where answers are checkable. Teaching a model to *abstain* trades false claims for refusals, and pushed too far it refuses things it knows. A subtle trap sits inside fine-tuning itself: training a model on facts it did not already hold can *teach it to hallucinate*, because it learns the surface form of confident assertion faster than the fact [@gekhman2024]. The honest summary is that hallucination is managed, not solved [@huang2023].

!!! interview "Interview"
    *Does RAG solve hallucination?* No, and saying yes is a red flag. Retrieval helps for questions whose answer lives in a retrievable document, but the model can still misread a passage, blend two sources, or confabulate in the gaps between what it retrieved. It narrows the surface; it does not change the objective that makes the model want to sound right. Calibration and abstention are the parts RAG leaves untouched.

## Reading the machine

We can build a model far more easily than we can explain one. *Interpretability* is the effort to close that gap: to read a trained network's internal computation the way you would read source code. The obstacle with a name is **superposition**. A model represents far more distinct concepts than it has neurons, packing them as overlapping directions in activation space, so no single neuron means one thing and the raw activations look like noise [@elhage2022].

The most promising crowbar is the **sparse autoencoder**, which learns to re-express a layer's activations as a large, sparse set of features that are each closer to a single human concept. Scaled to a production model, this pulls out millions of interpretable features — for the Golden Gate Bridge, for sycophancy, for bugs in code — and turning a feature up or down measurably steers behavior, which is real evidence the features are causal rather than decorative [@templeton2024].

!!! analogy "Analogy"
    A sparse autoencoder is like a prism splitting white light into named colors: the mixture was always there, and the prism gives you handles on its parts. It leaks in two places. The decomposition is *lossy* — reconstruction throws some signal away — and naming the colors is not the same as reading the *circuit* that combines them into a behavior. We can increasingly name what a model represents; we still mostly cannot trace how those pieces compose into a decision.

<figure class="wide">
<img src="assets/figures/superposition.svg" alt="On the left, a small activation space crowded with more concept directions than it has dimensions, so the directions overlap; a sparse autoencoder arrow leads to a right-hand stack of clean features, each labeled with a single concept.">
<figcaption>The interpretability bet in one picture. A model stores more concepts than it has dimensions, smeared across overlapping directions; a sparse autoencoder unpacks that mixture into features that each carry one concept. Naming the features is progress; reading the circuit that combines them is the part still mostly out of reach.</figcaption>
</figure>

Why care, beyond curiosity? Because every other problem in this chapter gets easier if you can see inside. A hallucination you could localize to a feature, a deceptive plan you could read before it executes, an alignment you could *verify* rather than infer from behavior — all of it waits on interpretability maturing from "we can label features" to "we can audit computation." It is not there yet, and the distance is worth being sober about.

## Aligning what you cannot fully check

Alignment (Chapters 10–12) works today because a human can judge whether an answer is good. That assumption weakens exactly where it matters most: on a system that knows more than its supervisor, human judgment becomes the ceiling, and a model that has learned to please graders will optimize the *appearance* of a good answer over the substance. Sycophancy — telling you what you want to hear — is already measurable in aligned models and is a direct product of training on human approval [@sharma2023]. The worry one size up is **deception**: behavior that looks aligned under evaluation because the model can tell it is being evaluated.

The research program here is **scalable oversight**: how do you supervise a system you cannot fully check? One concrete probe is **weak-to-strong generalization** — use a weak supervisor (a smaller model, or a human on a hard problem) to elicit the capabilities of a stronger one, and measure how much of the strong model's latent skill survives the noisy teaching [@burns2023]. The finding is encouraging and partial: a strong student trained on a weak teacher's flawed labels can *beat the teacher*, recovering much of its own capability, but not all of it, and the gap is precisely the thing you cannot afford to get wrong at the frontier.

<figure>
<img src="assets/figures/scalable-oversight.svg" alt="A weak supervisor with partly-wrong labels feeds a strong model; an accuracy bar shows the naive copy beating the teacher and an elicited student rising higher still, but short of full capability.">
<figcaption>Scalable oversight, stated as a measurement. A weak teacher's noisy labels can elicit more from a strong student than the teacher itself knows — but not the student's full capability. The residual gap is the open problem: it is the part no weaker overseer can certify.</figcaption>
</figure>

This loops back to evaluation (Part VI). Every benchmark assumes a grader who is at least as competent as the thing graded. Take that away and "we tested it and it passed" stops being reassurance, because the model may understand the test better than the test understands the model. Guardrails (Chapter 23) catch known failure modes; they do not certify the absence of unknown ones.

!!! interview "Interview"
    *If a bigger model can already beat its weaker teacher, isn't oversight solved?* No — that result is the good news and the warning in one. Recovering *most* of a capability from weak supervision shows the signal is elicitable; the remaining gap is what you cannot verify precisely when the model outruns the grader. Superhuman alignment is not "can weak supervise strong at all," it is "can we close that last gap and *know* we closed it."

## The wall question

Scaling laws (Chapter 9) treat data, parameters, and compute as knobs you turn together. One of those knobs has a floor. Frontier runs already consume a large fraction of the high-quality public text that exists, and projections put the crossing point — where the largest training sets meet the usable stock of human-generated text — in the late 2020s to early 2030s [@villalobos2022]. You cannot conjure more Wikipedia by spending more GPUs.

<figure>
<img src="assets/figures/data-supply-demand.svg" alt="A log-scale plot from 2020 to 2035: the stock of public human text grows slowly while tokens-per-training-run climbs steeply, the two curves crossing around 2028.">
<figcaption>The one scaling knob with a floor. Compute and parameters keep climbing, but the stock of public human text grows slowly, and the demand curve is steep — so the interesting question is not whether the lines cross but what happens after they do.</figcaption>
</figure>

Three exits are under active work, each with a catch. **Synthetic data** — models generating their own training data — already helps for tasks with a checker (Chapter 25), but naively training on model output risks *collapse*, where the distribution narrows generation over generation and the tails of real human variety are lost. **Data efficiency** — extracting more signal per token, or reusing tokens across many epochs — buys headroom but obeys its own diminishing returns [@muennighoff2023]. **Test-time compute** (Chapter 25) sidesteps the wall by spending more at inference instead of more data at training, which is the current bet but a different axis, not a refutation of the limit. Whether the loss curve keeps bending down or flattens is genuinely open, and anyone who tells you they know is selling something.

!!! note "Note"
    "Running out of data" does not mean progress stops. It means the free lunch of scraping ever more human text ends, and the marginal gain has to come from *how* you use tokens rather than *how many* you can find. That shifts the frontier from collection to curation, generation, and inference-time compute.

## What to hold onto

Step back and the book's arc is a single pipeline: architecture, pretraining, alignment, serving, the harness, evaluation, the frontier. Read left to right, each part hands the next a slightly more finished object — a predictor becomes a model becomes an assistant becomes a product you can measure. The open problems live at every joint. Hallucination is pretraining's objective showing through. Interpretability is architecture we built but cannot read. Scalable oversight is alignment losing its grader. The data wall is scaling meeting a physical limit. They are not separate frontiers; they are the same pipeline, seen from the side where it is still unfinished.

<figure class="wide">
<img src="assets/figures/what-endures.svg" alt="Two columns: a left column of things that keep changing — top model, benchmark scores, context windows, frameworks, prompting tricks — and a right column of durable fundamentals — the next-token objective, attention, scaling laws, post-training, the harness, evaluation.">
<figcaption>What to keep, when the news cycle turns over. The names on the left will be different by the time you read this; the ideas on the right are the ones every new system is still built from. Learn the right column and the left column becomes readable on sight.</figcaption>
</figure>

So what should you actually keep? Not the leaderboard — it will have turned over by the time you finish this sentence. Keep the fundamentals that every new system is still assembled from: the next-token objective and why it both empowers and misleads, attention and the transformer block, scaling laws as a design tool, the post-training pipeline, the harness that makes weights behave, and the discipline of asking how you *know* something works. Those have held for years while the model names churned monthly, and they are what let you read the next result — or the next hype — for what it is.

That is the whole contract, returned to where the introduction started it: a function that predicts the next token, made large, and wrapped with care. The remarkable thing is how far that simple idea reaches. The honest thing is how much about it we still cannot explain. Both are true at once, and holding both — the reach and the limits, without collapsing into either hype or despair — is the most useful posture an engineer can bring to this field. The rest is yours to build.
