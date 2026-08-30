Every earlier part of this book improved a model; this part asks how you would *know*. That turns out to be the harder problem. Training a better model is now often gated not by compute or data but by the ability to measure whether a change helped — a noisy, contested, quickly-saturating signal. Evaluation is where progress is actually bottlenecked, and it is the part of the stack most likely to fool you: a number can rise while the thing you care about gets worse. This chapter is about earning trust in the numbers.

## Why evaluation is the hard part

Classification has a ground truth: the label is cat or it is not, and accuracy is unambiguous. Open-ended generation does not. Ask a model to summarize a report, write a function, or explain a proof, and there are many good answers and many bad ones, with no single reference to check against. The moment output is a paragraph instead of a class, the clean metric disappears.

The old reflex is to compare against a reference string. Metrics like BLEU and ROUGE count n-gram overlap with a human-written answer, and they fail exactly where language is supposed to be flexible: a summary that is correct but differently worded scores low, and a fluent wrong answer that reuses the reference's words scores high. Overlap measures surface form, not meaning.

!!! intuition "Intuition"
    A good evaluation has to reward being *right*, but for open-ended tasks "right" is a set of outputs, not a point. Every practical method in this chapter is a different way to approximate that set — with a benchmark, a model, or a human.

The escape hatch is to force the task back into a checkable shape: multiple-choice questions, unit tests that a code patch must pass, a math answer that either equals the key or does not. This is why so much of evaluation funnels into these formats. It buys objectivity at the cost of coverage: you can only measure what you can make checkable, and the most valuable model behaviors — judgment, tone, when *not* to answer — resist it.

<figure class="wide">
<img src="assets/figures/open-ended-scoring.svg" alt="One prompt, 'summarize the findings', fans out to three differently worded but equally valid summaries, each marked correct by a human; a note observes a single-reference string metric would mark two of them wrong.">
<figcaption>Why the clean metric disappears. One open-ended prompt has a whole set of good answers; a metric that checks a single reference string rewards matching words, not being right, and punishes valid variation.</figcaption>
</figure>

## Benchmarks and their discontents

A benchmark packages a task into a fixed, scored dataset so different models get one comparable number. MMLU is the archetype: 57 subjects of multiple-choice questions from elementary to professional level, reduced to a single accuracy [@hendrycks2021]. Benchmarks made the field legible — you can rank models — and that very legibility is their weakness, because a public number is a target.

Three failure modes follow. **Saturation:** once frontier models pass human-expert accuracy, the benchmark stops discriminating; a suite that everyone scores 90 on ranks nothing. **Contamination:** the test questions are on the web, so they leak into pretraining corpora, and the model that "solves" them may be recalling them. Chapter 6 covered decontamination from the *builder's* side; from the evaluator's side the danger is that exact-match filtering never fully works, since a paraphrase of a leaked question survives it. **Gaming:** teams tune on the eval, directly or by picking data that happens to help it, and the score drifts away from the ability it once stood for.

!!! warning "Common trap"
    A rising benchmark score is consistent with the model getting better *and* with the test leaking into training. The two are indistinguishable from the number alone — which is why a suspiciously large jump on a public benchmark should raise your suspicion, not your confidence, until you can rule contamination out.

The response is an arms race toward benchmarks that are harder to saturate and harder to leak. GPQA writes "Google-proof" graduate questions that non-expert humans cannot solve even with web access, buying headroom [@rein2023]. SWE-bench abandons multiple choice entirely: a model must produce a patch that makes a real repository's real test suite pass, which is expensive to fake and grounded in execution rather than a reference answer [@jimenez2024]. Holistic suites push the other axis, measuring many models on many scenarios with accuracy, calibration, robustness, and bias side by side rather than one headline number [@liang2022].

<figure>
<img src="assets/figures/benchmark-saturation.svg" alt="Accuracy-versus-year curves for MMLU, GPQA, and SWE-bench; each climbs steeply toward a human-expert ceiling within a few years of release, with newer benchmarks starting lower and rising fast.">
<figcaption>Benchmarks expire on a schedule. Each climbs to its ceiling within a couple of years of release, so the frontier keeps minting harder successors; when a score plateaus it is usually the benchmark that is exhausted, not progress.</figcaption>
</figure>

The contamination worry is worth seeing concretely, because it changes what the number *means*, not just its size.

<figure>
<img src="assets/figures/contamination-inflation.svg" alt="A training corpus with one row highlighted as a leaked test question feeds a model; two bars compare a 62 percent held-out score against an inflated 88 percent contaminated score, the gap labeled recalled not learned.">
<figcaption>Contamination is not just inflation. When test items sit in the training data, the reported score silently stops measuring generalization and starts measuring recall — while still looking like an accuracy. The gap is memory wearing the costume of skill.</figcaption>
</figure>

## When the grader is a model

For open-ended output with no reference, the dominant method is now **LLM-as-judge**: prompt a strong model to grade another model's answer, either scoring it alone (pointwise) or picking the better of two (pairwise) [@zheng2023]. It scales to millions of examples at cents each, and on many tasks its verdicts agree with human raters about as often as two humans agree with each other — the bar that makes it usable at all.

The catch is that the judge is a language model with a language model's biases, and they are systematic, not random. It exhibits **position bias**, favoring whichever answer it sees first; **verbosity bias**, preferring longer, more padded answers even when they are no more correct; and **self-preference**, scoring outputs from its own model family higher [@zheng2023]. Systematic bias is the dangerous kind: averaging over more examples does not wash it out, it just estimates the wrong quantity more precisely.

!!! analogy "Analogy"
    An LLM judge is a fast, cheap, tireless grader who has some fixed prejudices — always likes the longer essay, always leans toward the first paper in the stack. The analogy leaks because a human grader's prejudices are somewhat independent across people, so a panel cancels them; a fleet of copies of the *same* judge shares one bias, and quantity cannot cure it.

The defenses are mechanical. Randomize or swap the order of the two answers and average, so position cancels. Control for length, and be suspicious when the judge's preference tracks word count. Never let a model be the sole judge of its own outputs in a comparison it can win by self-preference. And calibrate against a human-labeled sample before trusting a judge on a new task, because agreement on chat quality does not imply agreement on, say, factual precision or safety.

!!! interview "Interview"
    *Your new model wins 70% of pairwise judgments against the old one — ship it?* Not yet. Check that the judge saw the two answers in randomized order, that the winner is not simply longer, and that the judge is not from the same family as one of the contestants. Then confirm the win holds on a human-labeled slice. A pairwise number is only as trustworthy as the controls around the judge that produced it.

<figure class="wide">
<img src="assets/figures/judge-biases.svg" alt="Three cards labeled position, verbosity, and self-preference, each describing how an LLM judge is systematically swayed — by answer order, by length, and toward its own model family.">
<figcaption>The judge's fixed prejudices. An LLM grader is swayed by answer order, by length, and toward its own family — biases that are systematic, so more samples sharpen the wrong estimate rather than correcting it. The fixes are procedural: randomize order, control for length, never let a model grade only itself.</figcaption>
</figure>

## Human preference and the arena

When you want the ground truth of "which answer do people actually prefer," you ask people. The scalable form is pairwise: show a rater two anonymous answers to the same prompt and record which they pick. Chatbot Arena turned this into a public platform where users chat with two hidden models, vote, and the votes accumulate [@chiang2024]. Because each battle is relative, the votes are aggregated into an Elo-style rating — the same system that ranks chess players — producing one leaderboard from a flood of noisy pairwise comparisons [@zheng2023].

The arena is the closest thing the field has to a contamination-proof, hard-to-game evaluation: the prompts are fresh, live, and secret, and there is no fixed answer key to leak. But it measures a specific thing — aggregate human *preference* — and preference is not correctness. Raters reward answers that look confident, are nicely formatted, and agree with them; they cannot easily catch a subtle factual error or a citation that does not exist. It is also slow, and skewed toward the casual prompts people bring to a public demo, not the hard, domain-specific work your product may depend on. Popularity and quality correlate, but the gap between them is exactly where a fluent, well-formatted, wrong model wins.

!!! note "Note"
    A leaderboard's rank is a summary statistic, and it can be worked. Labs can privately test many variants and reveal only the best, or optimize for the format raters like — pushing the score up without a matching gain in capability. Treat a single arena rank the way you would any one number that a lot of money is trying to move: as evidence, not verdict.

<figure class="wide">
<img src="assets/figures/arena-elo.svg" alt="On the left, one anonymous battle where a user picks between hidden models A and B for the same prompt, repeated hundreds of thousands of times; on the right, the votes aggregate into an Elo leaderboard ranking four models.">
<figcaption>From battles to a ranking. Each vote is a single noisy pairwise comparison; aggregated by an Elo system over hundreds of thousands of battles they yield one relative ranking. It is a real signal, but of preference, not of correctness.</figcaption>
</figure>

## Evals for your own product

Public benchmarks tell you about general capability; they say almost nothing about whether the model does *your* job. A support assistant, a contract summarizer, and a coding agent fail in different ways, and no leaderboard measures those ways. The single highest-leverage thing a team building on LLMs can do is write its own eval set: a few dozen to a few hundred real task instances with a checkable notion of success, drawn from actual usage.

The mechanism that keeps it honest is a flywheel. Log every production interaction. Triage the failures and label them. Turn each distinct failure into a permanent case in the eval set. Run that set as a regression gate on every prompt change, model swap, or fine-tune, so a fix for one bug cannot silently reintroduce another. This is eval-driven development, and it inverts the usual order: you write the failing eval first, then change the system until it passes, exactly as you would with tests.

!!! interview "Interview"
    *How do you evaluate a RAG or agent system, where the model is one part of a pipeline?* End-to-end task success is the number that matters, but it hides *where* failure happened, so you also instrument the stages: did retrieval surface the right document (Chapter 21), did the agent call the right tool with the right arguments (Chapter 22), did generation stay faithful to what it retrieved? A single end-to-end score tells you the system is broken; per-stage evals tell you which part to fix.

There is a deeper reason evals matter more the closer you get to production: optimization pressure finds whatever you measure. This is Goodhart's law, the same force behind reward over-optimization in Chapter 11 — a metric under pressure stops being a good metric. The defense is not one perfect number but a diverse, evolving suite that is expensive to game precisely because it keeps changing as your product meets reality.

<figure>
<img src="assets/figures/eval-flywheel.svg" alt="A five-node cycle — ship, log, label, grow the eval set, regression-test — with arrows flowing around a ring labeled the flywheel.">
<figcaption>The evaluation flywheel. Production failures are logged, labeled, and folded into a growing eval set that gates the next release, so every real bug becomes a permanent test and the same regression never ships twice.</figcaption>
</figure>

Trustworthy evaluation is what lets every other chapter's improvement be believed. It is also the field's rate limiter: measuring the newest capabilities — long-horizon reasoning (Chapter 25), autonomy, honesty under pressure — is genuinely unsolved, which is why Chapter 26 returns to evaluation we can trust as one of the open problems that will shape what comes next.
