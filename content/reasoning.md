For most of this book, "better" has meant "bigger": more parameters, more tokens, more pretraining compute (Chapter 9). This chapter is about a second dial that turned out to be just as powerful and far cheaper to reach for. Instead of training a larger model, you let the model *think for longer* at inference — generate a long internal chain of reasoning before it commits to an answer. The late-2024 release of OpenAI's o1 and the early-2025 release of DeepSeek-R1 made this the defining move of the frontier: a model trained to spend hundreds or thousands of hidden tokens working a problem out, and to get more accurate the more of them it spends. The catch, which the last section takes seriously, is that this only works cleanly where an answer can be *checked*.

## Spending compute at inference

Chapter 18 showed that prompting a model to "think step by step" lifts accuracy on multi-step problems, because each written step deposits an intermediate result into the context for the next step to reuse. A **reasoning model** takes that trick and bakes it into the weights. Rather than relying on you to ask for a chain of thought, it is post-trained to *always* produce a long one — often walled off from the user as hidden "thinking" tokens — and only then emit a short final answer [@openai2024; @deepseek2025]. The chain is not decoration; it is the computation, and the model has learned to make it long, to backtrack, to check its own work, and to try another approach when one stalls.

The consequence is a new scaling axis. The pretraining laws of Chapter 9 relate loss to *training* compute; they say nothing about compute spent at inference. Reasoning models add that missing axis: hold the weights fixed and let the model generate more thinking tokens, and accuracy on hard problems keeps rising, roughly linearly in the log of the token budget until it saturates. You can now buy capability after training is over, per query, by paying for a longer think.

!!! intuition "Intuition"
    Pretraining scaling makes the model that answers; test-time scaling makes the *answering* longer. One is a bigger brain, the other is more time to think — and for a decade the field only knew how to sell the first.

<figure class="wide">
<img src="assets/figures/thinking-tokens.svg" alt="Two accuracy curves rising with the number of thinking tokens on a log x-axis: a hard competition-math task climbs steeply and keeps improving; an easy grade-school task saturates near the top almost immediately.">
<figcaption>The second scaling axis. Let a fixed model spend more tokens thinking before it answers and accuracy climbs, roughly straight in the log of the budget, until it saturates. Hard problems ride the curve much further than easy ones, which is why a reasoning model that "overthinks" a trivial question wastes tokens for no gain.</figcaption>
</figure>

!!! interview "Interview"
    *Is a reasoning model just chain-of-thought prompting with extra steps?* No, and the difference is where the behavior lives. Prompted chain-of-thought (Chapter 18) is a property of your input and vanishes if you change the prompt; a reasoning model has the long-chain habit trained into its weights and produces it unprompted. More to the point, it was trained specifically to make that chain *useful* under a reward that only cares whether the final answer is right — so the reasoning generalizes rather than parroting the format of your exemplars.

## RL on verifiable rewards

How do you train a model to reason well when you cannot write down what a good chain of thought looks like? You stop trying to. In domains where the *final answer* can be checked automatically — a math problem with a known result, a coding task with unit tests — you can reward the model for getting the answer right and let it discover for itself which chains get it there. This is **reinforcement learning from verifiable rewards** (RLVR): sample a chain plus an answer, run a checker, and hand back a scalar reward (correct or not), then update the policy toward the chains that scored.

DeepSeek-R1 is the open demonstration that this alone is enough. Its R1-Zero variant took a pretrained base model and applied RL with purely rule-based rewards — answer-matching for math, execution for code, plus a format check — with *no* supervised examples of reasoning at all [@deepseek2025]. Long chains of thought, self-checking, and mid-solution backtracking emerged from the reward pressure rather than from imitation, an effect the authors call the model's "aha moment." (The shipped R1 adds a small supervised warm-start for readability before the RL, but the reasoning itself comes from RL.) The lineage runs back to STaR, which bootstrapped reasoning by fine-tuning a model on its own chains that happened to reach the right answer [@zelikman2022]; RLVR is that idea done as proper reinforcement learning at scale.

<figure class="wide">
<img src="assets/figures/rlvr-loop.svg" alt="A cycle: the policy LLM samples a chain of thought and final answer, a verifier runs the code or checks the math, a reward of plus one for correct or zero for wrong is returned, and the policy is updated toward chains that scored.">
<figcaption>Learning to reason without a teacher. The reward is a deterministic checker — a test suite, an answer key — not a learned model, so the loop rewards being right rather than looking right. That is exactly the property RLHF lacks.</figcaption>
</figure>

The contrast with the alignment pipeline of Chapter 11 is the whole point. RLHF optimizes against a *learned* reward model trained on human preferences, and because that reward model is an imperfect proxy, the policy learns to exploit its blind spots — the reward-hacking problem, where outputs score well without being good. A verifiable reward removes the proxy: a unit test cannot be sweet-talked, and a wrong numerical answer earns nothing however fluent the argument for it. That is why RLVR scales where RLHF plateaus, and also why it is confined to domains with a checker.

!!! interview "Interview"
    *Why not use RLVR for everything, if it avoids reward hacking?* Because most things you want from a model have no automatic checker. "Write a kind, accurate reply to this grieving customer" has no unit test; its quality is exactly the human-preference judgment RLHF was built for. RLVR owns the verifiable slice — math, code, formal logic — and RLHF owns the rest. The frontier recipe uses both, on the tasks each fits.

## Searching for the right chain

Making the chain longer is one way to spend inference compute; making *more* chains is the other. Because decoding is stochastic, sampling the same problem several times gives you several independent attempts, and you can select among them. The cheapest selector is **self-consistency**: sample many chains and take the majority answer, which cancels out any single chain that derailed (Chapter 18) [@wangsc2022]. It needs no extra machinery and works because wrong chains tend to fail in different ways while correct ones converge.

You can do better than a vote if you can *score* a candidate. A **verifier** is a model trained to judge whether a solution is right, letting you take the best of n samples instead of the most common. Verifiers come in two flavors, and the distinction matters: an **outcome** reward model scores only the final answer, while a **process** reward model scores each *step* of the reasoning. Training the model to verify step by step — rewarding the first mistake, not just the wrong total — turns out to supervise far more reliably than outcome scoring alone, because it tells the model *where* a chain went wrong [@lightman2023]. Push selection further and it becomes search: **Tree of Thoughts** branches at each step and explores the reasoning tree with lookahead instead of committing to one left-to-right chain [@yao2023tot].

<figure class="wide">
<img src="assets/figures/verifier-selection.svg" alt="A prompt fans out to four sampled reasoning chains, each ending in a candidate answer with a verifier score; a self-consistency box picks the majority answer and a verifier box picks the highest-scoring answer, both landing on the same result.">
<figcaption>Two ways to cash in extra samples. Self-consistency trusts the crowd and takes the majority; a verifier scores each chain and keeps the best. Both turn parallel compute into accuracy, and a good verifier extracts more from the same n — the reason "how do I select?" is now a first-class design question.</figcaption>
</figure>

The reason to care is economic. Snell et al. showed that spending a compute budget on test-time search — sampling and verifying — can beat spending the same budget on a larger model, *when the problem is hard enough to reward the search* [@snell2024]. That is a genuinely new option in the design space: given a fixed budget, you can now choose whether to buy it as parameters or as inference-time thinking, and the right split depends on the task.

## What this changes

Test-time compute reshapes the cost model of deployment. Every architecture chapter treated inference as cheap relative to training; a reasoning model can emit thousands of hidden tokens per query, so a single answer now costs many times a standard model's, and it arrives seconds later. Latency and token spend, not just accuracy, become the levers you tune — you dial the thinking budget per query, spending more on the hard ones. The clean line between an expensive training phase and a cheap serving phase blurs, because you are now doing real optimization-shaped work (search, selection) at serve time.

The honest boundary is the one the whole chapter has circled: **this works where answers are checkable, and degrades where they are not.** RLVR needs a verifier to train against, and test-time search needs one to select against; on open-ended tasks with no ground truth, more samples buy almost nothing (the flat curve below). A second caveat is subtler. The visible chain of thought is not certified to be a *faithful* account of the computation that produced the answer (Chapter 18) — models can reach a conclusion by one route and narrate another — so a long, confident, correct-looking chain is still not a proof, and reading it as one is a trap. Reasoning is a powerful new axis, not a universal solvent; where it can be checked it is transformative, and where it cannot the old limits still hold.

<figure class="wide">
<img src="assets/figures/test-time-limits.svg" alt="Accuracy versus number of sampled chains per question on a log x-axis: a checkable task with a verifier keeps climbing, a checkable task with majority vote rises then plateaus, and an open-ended task with no way to check stays nearly flat.">
<figcaption>Where the solvent dissolves nothing. With a checker, more samples keep converting compute into accuracy; a majority vote helps but saturates sooner; with no way to check an answer, extra samples buy almost nothing. The presence of a verifier, not the compute itself, is what makes test-time scaling work.</figcaption>
</figure>

!!! interview "Interview"
    *Your product needs better answers on a subjective writing task. Will a reasoning model help?* Probably not much, and naming why is the strong answer: reasoning models are trained and served against *verifiable* rewards, so their edge is concentrated in math, code, and logic. On a task with no automatic notion of correctness there is nothing for the long chain to optimize toward and nothing for a verifier to select on, so you pay the latency and token cost for little gain. Reach for reasoning where the answer can be checked; reach for better prompting, retrieval, or preference tuning where it cannot.

Which checkable problems the reward can even *reach*, whether reasoning trained on math transfers to messier domains, and how far the test-time axis extends before it too saturates are open questions — the subject of Chapter 26, where the frontier runs out of things it is sure about.
