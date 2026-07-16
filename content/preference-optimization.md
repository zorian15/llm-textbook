Chapter 11 aligned a model the hard way: fit a reward model to human preferences, then chase that reward with reinforcement learning. It works, and frontier labs still run it, but it is a heavy machine — a second network to train, a full RL loop to babysit, and a reward signal you can over-optimize. This chapter is the counter-argument. It turns out the preference data can train the policy *directly*, with an ordinary supervised loss and no reward model in sight. That reframing, **Direct Preference Optimization**, is the default alignment recipe for most teams in 2026, and the family of variants around it is a map of what alignment actually needs.

## DPO: skipping the reward model

Start from the exact objective Chapter 11 optimized: maximize reward while a KL penalty keeps the policy close to the reference (the SFT model). That objective has a *known closed-form solution* — the optimal policy is the reference reweighted by the exponentiated reward. DPO's move is to read that equation backwards [@rafailov2023]. If the optimal policy is a function of the reward, then the reward is a function of the policy:

$$\hat r(x, y) = \beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \text{(a term that cancels)}.$$

The reward is *implicit* — it is just how much more likely the policy makes a response than the reference does. Substitute this into the Bradley-Terry preference model from Chapter 11, and the awkward normalizing term cancels between the chosen and rejected responses. What remains is a plain classification loss on preference pairs $(x, y_w, y_l)$:

$$\mathcal{L}_{\text{DPO}} = -\log \sigma\!\left(\beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\right).$$

Read it in words: push up the log-probability of the winner, push down the loser's, and measure both against a frozen reference so the model cannot cheat by inflating everything. No reward model, no sampling loop — one forward pass over a fixed dataset, trained like any classifier.

!!! intuition "Intuition"
    The reward model was never the point; it was a middleman between preferences and the policy. DPO cuts the middleman and lets the preference pairs adjust the policy's own token probabilities.

!!! interview "Interview"
    *If there is no reward model, what is $\beta$ doing, and why keep the reference model at all?* $\beta$ is the same KL leash from PPO, now baked into the loss: it sets how far the policy may move from the reference per unit of preference. The reference appears in every term as the anchor the log-ratio is measured against — drop it and the loss rewards raising the winner's probability without bound, which degrades fluency fast. The gradient also self-weights: pairs the implicit reward already gets *wrong* push hardest, so DPO spends its capacity where it is currently mistaken.

<figure class="wide">
<img src="assets/figures/dpo-vs-rlhf.svg" alt="Two tracks from the same preference dataset: the RLHF track passes through a reward model and a PPO loop before reaching the policy; the DPO track goes through a single direct-preference loss.">
<figcaption>What DPO removes. The same preference pairs reach the same goal, but DPO replaces the reward model and the RL loop with one classification-style loss — the reward model turns out to have been an indirection you can skip.</figcaption>
</figure>

## The DPO family: IPO, KTO, ORPO, SimPO

DPO makes several assumptions, and each variant relaxes one. **IPO** attacks overfitting: DPO's logistic loss keeps rewarding a wider and wider margin, so when preferences are near-deterministic the log-ratio runs off to extremes and the KL leash slips. IPO replaces the objective with a bounded, squared target that pins the margin to a fixed value, which regularizes without a separate early-stopping trick [@azar2024]. **KTO** attacks the *data* requirement. DPO needs matched pairs; KTO, drawing on prospect theory's asymmetric treatment of gains and losses, learns from lone thumbs-up / thumbs-down labels with no pairing at all — which matters because unpaired binary feedback is far cheaper and more plentiful than curated comparisons [@ethayarajh2024].

The other two drop the reference model. **ORPO** folds supervised fine-tuning and preference optimization into a single stage: it adds a small odds-ratio penalty to the ordinary SFT loss, so one pass over demonstrations both teaches the format and discourages the disfavored style — no separate preference phase, no reference copy [@hong2024]. **SimPO** keeps two responses but makes the implicit reward the *length-normalized* average log-probability of a sequence, plus a target margin. Dropping the reference makes it leaner, and the length normalization directly attacks DPO's best-known failure — a drift toward longer answers [@meng2024].

!!! warning "Common trap"
    DPO quietly rewards *length*: because it sums token log-probabilities, a longer rejected response is easier to push down, so the model learns that more tokens look preferred. Verbose DPO outputs are usually this artifact, not a genuine quality gain — length-normalized rewards (SimPO) or length-balanced data are the fix.

<figure class="wide">
<img src="assets/figures/preference-methods.svg" alt="A table of five methods — DPO, IPO, KTO, ORPO, SimPO — with columns for whether each needs a reference model, paired data, and a separate SFT stage, plus a one-line note on what each changes.">
<figcaption>The family, at a glance. Each variant is DPO minus one assumption — a reference model, paired data, a separate SFT stage — not a different paradigm. Reading the columns tells you which constraint your setup can afford to drop.</figcaption>
</figure>

## RLHF vs. DPO in practice

The honest tradeoff is simplicity against ceiling. DPO trains like supervised learning: one loss, one model plus a frozen reference, stable and cheap. PPO holds up to four networks in memory at once — policy, reference, reward model, and value head — and RL training is famously twitchy. For most teams that settles it: DPO gets you most of the way at a fraction of the engineering cost. But the comparison hides a deeper axis. DPO is **off-policy** — it learns from a dataset collected once, so as training shifts the policy, it can drift into regions the preference pairs never covered, where the loss has nothing to say. PPO is **on-policy**: it samples fresh responses from the current model every step, so the signal always lands where the policy actually is.

<figure class="wide">
<img src="assets/figures/on-off-policy.svg" alt="Two panels: on the left, a fixed preference-data region with the DPO policy drifting outside it into an uncovered zone; on the right, on-policy PPO with fresh samples re-covering the policy each step.">
<figcaption>The real fork is on-policy versus off-policy. DPO's fixed dataset can fail to cover a policy that has moved; online sampling keeps the training signal centered on the current model. This, more than the loss form, drives the quality gap when there is one.</figcaption>
</figure>

This is why "is DPO actually worse than PPO" has a nuanced answer. Careful head-to-head studies find that well-tuned PPO can beat DPO on hard benchmarks like code generation [@xu2024] — but the decisive factor is usually the *data*, not the algorithm: preference quality and on-policy sampling matter more than the choice of loss [@ivison2024]. The practical upshot, and what an interviewer wants, is that the gap largely closes with **iterative (online) DPO** — regenerate fresh pairs from the current policy, relabel, and repeat — which buys back the on-policy benefit while keeping DPO's simplicity.

!!! interview "Interview"
    *A team asks whether to use DPO or PPO. What do you tell them?* Default to DPO for the simplicity and stability, and reach for online or iterative DPO before PPO if quality plateaus. Choose full PPO only when you have the infrastructure and a reward signal worth chasing hard — and note that in *verifiable* domains (math, code) the field has largely moved to on-policy RL against programmatic rewards, which sidesteps the reward-model problem entirely (Chapter 25).

## Constitutional AI and RLAIF

Every method so far assumes a human wrote the preference labels, which is the real bottleneck: it is slow, expensive, and caps quality at what annotators can judge. The last idea in this chapter is to let a *model* supply the signal. **Constitutional AI** runs in two phases [@bai2022]. In the supervised phase, the model critiques its own response against a short written list of principles — the "constitution" — then revises it, and you fine-tune on the revisions. In the preference phase, the model plays judge: it labels which of two responses better follows the constitution, producing the exact preference pairs DPO or RLHF consumes. Because the labeler is now an LLM, the same idea generalizes to **RLAIF**, where AI feedback stands in for human feedback and, at least for helpfulness and harmlessness, matches it at a fraction of the cost [@lee2023].

!!! analogy "Analogy"
    The constitution is like a style guide handed to a copy editor: the writer drafts, then checks each line against the rules and fixes what violates them. It leaks in that a copy editor brings outside judgment, while here the same model is author, editor, and rulebook-follower — so any blind spot it has, it applies uniformly and cannot catch in itself.

<figure class="wide">
<img src="assets/figures/constitutional-ai.svg" alt="Two rows: a supervised phase where a response is self-critiqued against a principle and revised into an SFT target, and a preference phase where an AI judge ranks two responses per the constitution to make a preference pair for DPO or RLHF.">
<figcaption>Constitutional AI swaps the human labeler for a written principle. The model critiques and revises itself, then judges its own samples against the constitution — turning the preference signal into something that scales as cheaply as inference.</figcaption>
</figure>

The catch is that AI feedback inherits the judge's biases and can be gamed: a policy may learn to satisfy the *letter* of the constitution while a human would still object, the same reward-hacking pressure Chapter 11 warned about, now aimed at a model's judgment instead of a reward model's. And it only works when the judge is capable enough to tell the responses apart — which is exactly why **scaling oversight**, using models to help supervise tasks that outstrip human judgment, is an open frontier (Chapters 23 and 25). Cheap preference signal is the prize; a trustworthy one is the hard part.
