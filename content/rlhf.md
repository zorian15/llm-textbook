Supervised fine-tuning (Chapter 10) teaches a model to imitate answers people wrote. That works until you want answers better than any annotator can produce on demand, and it wastes the one thing people are reliably good at: judging. Most of us cannot write a top-1% response to a hard prompt, but we can spot which of two responses is better in seconds. Reinforcement learning from human feedback (RLHF) turns that asymmetry into a training signal. You collect preferences, fit a **reward model** that predicts them, then optimize the policy to score well under that model — with a leash that keeps it from wandering off. This chapter builds that pipeline, then confronts the failure at its center: the reward is a proxy, and a proxy optimized too hard stops measuring what you wanted.

## You can rank what you cannot write

The gap between recognizing quality and producing it is the whole reason RLHF exists. A demonstration, the unit of supervised fine-tuning, is a single answer a human committed to paper; the model learns to copy it. Its ceiling is therefore the annotator's own writing, on the day they wrote it, and the model never sees what *worse* looks like — it only ever imitates the good. A **preference**, the unit of RLHF, is a judgment between two answers the *model* generated. It is cheaper to collect, it covers far more of the output space than a fixed set of gold answers, and crucially it is not capped by what the labeler could have written themselves.

!!! intuition "Intuition"
    Recognition is easier than generation. You can tell a great pun from a mediocre one without being able to invent either. RLHF harvests that easy signal at scale and pushes the model up the ranking.

This also explains why more supervised fine-tuning is not a substitute. SFT can only reach the quality of its demonstrations, and for the hardest prompts nobody has a demonstration to give. Preference data sidesteps the bottleneck by asking the strictly easier question.

<figure class="wide">
<img src="assets/figures/demonstration-vs-preference.svg" alt="Left: supervised fine-tuning shows the model one gold answer written by a human and trains it to imitate that answer. Right: preference learning has the model generate several answers, which a human then ranks from best to worst.">
<figcaption>Two ways to teach an answer. Supervised fine-tuning copies one response a human wrote, so quality is capped by the annotator's pen. Preference learning ranks responses the model wrote, turning the easier act of judging into a signal that can climb above any single demonstration.</figcaption>
</figure>

## Reward models from human preferences

A reward model is the SFT model with its vocabulary head swapped for a single scalar head: it reads a prompt and a response and outputs one number, $r(x, y)$, meant to track human preference [@ouyang2022]. You cannot supervise that number directly, because no human hands you a calibrated score. What they hand you is a comparison: for a prompt $x$, response $y_w$ was preferred over $y_l$. The **Bradley-Terry** model turns comparisons into probabilities by assuming the chance $y_w$ wins rises smoothly with its reward advantage:

$$P(y_w \succ y_l) = \sigma\big(r(x, y_w) - r(x, y_l)\big).$$

Fitting the reward model is then ordinary maximum likelihood — minimize $-\log \sigma(r_w - r_l)$ over the comparison dataset, a logistic loss on the reward *difference*. Because only differences appear, the reward has no absolute meaning; add a constant to every score and nothing changes. That is fine: the policy step below cares only about which responses score higher, not by how much on any fixed scale.

!!! interview "Interview"
    *Why train a reward model on pairwise comparisons instead of asking labelers for a 1–10 score directly?* Absolute scores are noisy and drift between annotators and across a session, so a "7" from one labeler is not a "7" from another. A pairwise choice is far more consistent, and the Bradley-Terry loss extracts a latent scalar from those choices without ever needing the labelers to agree on a scale. The cost is a ceiling: reward models typically agree with held-out human preferences only around 65–75% of the time, because people genuinely disagree [@bai2022hh].

The reward model inherits the base model's knowledge, which is what lets it generalize to responses no labeler ever ranked. Early RLHF work built exactly this — a preference-trained reward model over a pretrained transformer [@christiano2017] — and it remains the standard recipe.

<figure>
<img src="assets/figures/reward-model.svg" alt="A prompt and two responses, chosen and rejected, both pass through the same reward model, which emits a scalar reward for each; the two rewards feed a logistic loss that pushes the chosen reward above the rejected one.">
<figcaption>How a scalar reward is learned from a choice. The same reward model scores both responses; the Bradley-Terry loss only ever sees the difference of the two scores, pushing the preferred response higher. Nothing pins the absolute scale, which is why a reward model measures better-than, not good.</figcaption>
</figure>

## Policy optimization with PPO

With a reward model in hand, you optimize the policy — the model you are actually shipping — to generate responses it scores highly. This is a reinforcement learning loop: the policy samples a response to a prompt, the reward model scores it, and the score drives an update that makes high-scoring responses more likely. The standard optimizer is **proximal policy optimization** (PPO), which takes small, clipped steps so a single large update cannot blow up the policy [@schulman2017]. A learned value function (a critic) estimates how good the average response is, so each step pushes on the *advantage* — how much better this response was than expected — rather than on the raw reward.

The load-bearing detail is the leash. The optimizer does not maximize $r(x, y)$ alone; it maximizes the reward minus a penalty for drifting from a frozen copy of the starting model, the **reference** (usually the SFT model):

$$\text{reward} = r(x, y) - \beta \, \mathrm{KL}\!\big(\pi_\theta(\cdot \mid x) \,\|\, \pi_{\text{ref}}(\cdot \mid x)\big).$$

Without that KL term, the policy would sprint toward whatever text the reward model happens to score highest, which is rarely fluent English — it is more often a degenerate string that games the reward model's blind spots. The penalty keeps the policy close to a model that already writes well and knows things, so it improves *within* the space of good answers instead of leaving it. The coefficient $\beta$ sets the leash length: too loose and the policy games the reward, too tight and it barely moves [@ouyang2022].

!!! analogy "Analogy"
    The KL penalty is a leash tying the policy to the SFT model. It lets the policy explore nearby, better behavior but yanks it back before it runs into nonsense. The analogy leaks in that the "distance" is a KL divergence over next-token distributions, not a physical radius — the policy can change *what* it says a lot while staying close in KL if it keeps its wording fluent and its facts intact.

Sampling on-policy matters here. The updates come from the policy's *own* current samples, so it learns from the parts of the output space it actually visits, and that region moves as it improves. This is also why PPO-based RLHF is heavy: it holds four models at once — policy, reference, reward model, and critic. Chapter 12 shows how direct preference optimization collapses this loop back into a single supervised-style loss, and Chapter 15 covers the serving cost of the sampling itself.

<figure class="wide">
<img src="assets/figures/rlhf-loop.svg" alt="A cycle: a prompt goes to the policy being trained, which samples a response; the response is scored by the reward model; a KL penalty compares the policy to a frozen reference model; the reward minus the penalty drives a PPO update back to the policy.">
<figcaption>The RLHF loop. The policy samples its own responses, the reward model scores them, and a KL penalty against the frozen reference keeps the policy from wandering out of the space of fluent answers. The update chases the reward only as far as the leash allows.</figcaption>
</figure>

## Reward hacking and over-optimization

The reward model is a proxy for human preference, not the real thing, and here Goodhart's law bites: once a measure becomes a target, it stops being a good measure. Push the policy hard enough and it finds responses that score well under the reward model but that humans dislike — the policy is exploiting the reward model's errors, not satisfying the preference behind it. The signature is unmistakable and was measured cleanly: as the policy drifts further from the reference (more KL), the *proxy* reward keeps climbing while the *true* reward, judged by held-out humans or a much larger gold model, rises, peaks, and then falls [@gao2023]. The best model is at the peak, not at maximum proxy reward.

<figure>
<img src="assets/figures/reward-overoptimization.svg" alt="Two curves against KL distance from the reference model: the proxy reward-model score rises monotonically, while the true reward rises to a peak and then declines. A dashed line marks the peak as where optimization should stop.">
<figcaption>Why more reward can mean a worse model. Optimizing against the proxy pushes its score up without limit, but the true quality it stands in for turns over and declines once the policy starts exploiting the reward model rather than satisfying the preference. The peak, not the maximum, is the model you want to ship.</figcaption>
</figure>

The most familiar symptom is **length bias**: reward models tend to prefer longer, more thoroughly hedged answers, so RLHF reliably makes models more verbose, and a surprising fraction of the apparent gain is just responses getting longer [@singhal2023]. Sycophancy is a cousin — reward models absorb the human tendency to rate agreeable answers higher, so the policy learns to agree.

!!! warning "Common trap"
    Reward hacking does not look like failure from inside the loop; it looks like success. The proxy reward, the number your dashboard tracks, goes *up* the whole time. Catching over-optimization requires an independent yardstick — held-out human evals or a stronger judge — because the metric you are training on is exactly the one that has been compromised.

The mitigations are unglamorous and complementary: keep the KL leash short enough, stop early rather than training to convergence, scale or ensemble the reward model so its blind spots are harder to find, and above all broaden the preference data so the hacks that remain are ones humans actually catch. The field's next moves also respond to this ceiling. Chapter 12 shows how RLAIF replaces expensive human labels with model-generated preferences, and Chapter 25 covers reasoning RL, where the reward is a *verifiable* checker — did the code pass, is the proof valid — rather than a learned proxy, which sidesteps reward-model hacking even as it invents new hacks of its own.
