Everything in Part V made the model more capable: a system prompt gave it a role (Chapter 18), tools let it act (Chapter 19), retrieval fed it fresh knowledge (Chapter 21), and agent loops let it pursue goals over many steps (Chapter 22).
Each of those additions is also an attack surface.
This closing chapter is about the other half of the harness: the machinery that keeps a helpful model from being turned into a harmful one.
The single most important idea is that safety is not a feature you switch on but a *stack* you build, because the model you aligned in Chapter 12 is a tendency, not a guarantee, and no one layer is enough on its own.

## Layers of defense

Start from the fact that alignment training is imperfect.
Preference optimization (Chapter 12) and Constitutional AI [@bai2022] push a model hard toward refusing clearly harmful requests, but they shape a *disposition* over a distribution of inputs, not a proof about every input.
An adversary chooses the input, and can search a space the training never covered.
So production systems wrap the aligned model in additional, independent layers: a **system-prompt policy** that states the rules in natural language, **input and output classifiers** that screen the prompt before it reaches the model and the reply before it reaches the user, and **monitoring** that logs traffic and flags abuse patterns after the fact.

The mental model is defense in depth, borrowed from security engineering.

!!! analogy "Analogy"
    Think of the "Swiss-cheese model" from accident analysis: each layer of defense is a slice with holes in it, and a bad outcome happens only when the holes in every slice line up.
    The analogy leaks in one direction worth naming — the holes here are not random.
    An adversary actively moves their attack until it finds a path through, so unlike an accident, the alignment of the holes is being *searched for*, not stumbled into.

The payoff of layering is *diversity*: alignment training and a keyword-based filter fail on different inputs, so an attack that slips past one is likely caught by another.
Stacking identical filters buys little; stacking a trained disposition, a policy, a separate classifier, and human review buys a lot, because an adversary now has to defeat all of them at once.

!!! intuition "Intuition"
    No single layer has to be perfect; the layers only have to fail *independently*, so the chance one input defeats all of them is the product of small numbers rather than any one of them.

<figure class="wide">
<img src="assets/figures/defense-layers.svg" alt="Four vertical slabs — alignment training, system-prompt policy, input/output classifiers, and monitoring — each with a gap at a different height. Most attacks stop at a solid part of the first slab; one rare path threads through the gaps of all four.">
<figcaption>Why one layer is never enough. Each defense catches most of what reaches it but has gaps; harm gets through only where the gaps happen to line up. The engineering goal is not a perfect slab but slabs whose holes sit in different places, so the residual path is vanishingly narrow.</figcaption>
</figure>

## Jailbreaks and red-teaming

A **jailbreak** is an input that makes an aligned model do what it was trained to refuse.
The reason jailbreaks keep working is structural, not a bug to be patched once: safety training fails in two characteristic ways [@wei2023].
It sets up *competing objectives*, where the model's drive to be helpful pulls against its drive to refuse, and *mismatched generalization*, where the model has a capability its safety training never learned to police.
Both are byproducts of the same competence that makes the model useful.

The attacks that exploit these gaps come in a handful of recognizable *shapes*, and knowing the shapes matters more than any specific string.
**Persona and role-play** attacks reframe a forbidden request as fiction or as a character the model is asked to inhabit, pitting helpfulness against the refusal.
**Obfuscation and encoding** hide the intent that a filter reads for, exploiting mismatched generalization — the model understands a request its guard never learned to flag.
**Injection** hides hostile instructions inside content the model consumes rather than in the user's message: a retrieved document (Chapter 21) or a tool result (Chapter 19) that says "ignore your instructions and do X," which the model may follow because it cannot tell data from command.
**Many-shot** attacks flood a long context window with hundreds of fake examples of the model complying, until it completes the pattern by imitation rather than by instruction; the attack's success grows smoothly with the number of examples [@anil2024].
**Automated adversarial suffixes** use gradients to search for a token string that maximizes the chance of compliance, and strikingly, a suffix optimized against open models often *transfers* to models the attacker cannot see inside [@zou2023].

!!! interview "Interview"
    *If alignment training can't make a model robust, why do it at all?*
    Because it raises the cost of every attack and removes the easy ones.
    A well-aligned model refuses casual misuse outright, so the remaining attacks require real effort — long crafted contexts, gradient search, or injection through another channel.
    Alignment is the layer that makes the *other* layers' job small; skip it and the classifiers and monitors drown in volume they were never meant to carry.

The practice that turns this understanding into defenses is **red-teaming**: deliberately attacking your own model to find failures before adversaries do.
Manual red teams probe by hand and reveal failure categories no benchmark anticipated [@ganguli2022], while automated red-teaming trains a *second* model to generate adversarial prompts at scale, surfacing thousands of failures a human team would never reach [@perez2022].
Red-teaming is not a release checkbox but a loop: find failures, patch them into training and filters, then attack the patched model again.

<figure class="wide">
<img src="assets/figures/jailbreak-shapes.svg" alt="Five labeled attack shapes — persona/role-play, obfuscation/encoding, injection via content, many-shot/long context, and automated suffix — each with an arrow pointing at a central box labeled safety-trained model.">
<figcaption>Jailbreaks are a small set of shapes, not endless one-off tricks. Each turns the model's own competence against its rules — its helpfulness, its understanding of encodings, its instruction-following, its in-context learning. Defending the shapes generalizes; chasing individual strings does not.</figcaption>
</figure>

## Content moderation classifiers

The layer you can improve fastest is a **guard model**: a separate, usually smaller classifier that reads the prompt on the way in and the response on the way out, and blocks, redacts, or escalates when either looks unsafe.
Llama Guard is the canonical open example — a fine-tuned LLM that classifies both prompts and responses against an explicit safety taxonomy, deployed alongside the main model rather than inside it [@inan2023].
Keeping the guard separate is the point: it can be updated on its own schedule, swapped for a stricter one per deployment, and it fails differently from the model it wraps, which is exactly the independence the Swiss-cheese stack needs.

!!! intuition "Intuition"
    The main model is optimized to be helpful; the guard is optimized to be suspicious.
    Splitting those two jobs across two models lets each be good at one thing, instead of asking one model to be both eager and paranoid at once.

The hard part is not building the classifier but living with its error rates, and the trap here is the **base rate**.
Real traffic is overwhelmingly benign, so even a genuinely strong classifier produces mostly false alarms.
A filter that catches 95% of harmful messages and wrongly flags only 1% of benign ones sounds excellent — but if one message in a thousand is actually harmful, the vast majority of what it flags is innocent, because 1% of an enormous benign stream dwarfs 95% of a tiny harmful one.

!!! interview "Interview"
    *Your safety classifier has 99% accuracy. Ship it?*
    Not on accuracy alone.
    When harmful traffic is rare, a classifier that blindly approved everything would already score near 99%, so the headline number is almost meaningless.
    What matters is precision and recall at your real base rate, and the *cost asymmetry* between the two errors — a missed harm and a wrongly blocked user are not equally bad, and which you tune toward is a policy choice, not a modeling one.

The other costs are latency and money.
An output guard cannot judge a response until the response exists, so a naive design makes the user wait for the model and then the guard in series; streaming interfaces buffer or screen incrementally to hide this.
And you are now running at least two models per turn, so the guard is deliberately kept small.

<figure class="wide">
<img src="assets/figures/guard-classifiers.svg" alt="A pipeline: user to input guard to assistant to output guard to delivered response, with each guard able to divert down to a box labeled refuse, redact, or escalate.">
<figcaption>Guard models bracket the assistant. A cheap input classifier screens the prompt before the expensive model runs, and a cheap output classifier screens the reply before the user sees it; either can stop the turn. Because the guards are separate models, they can be retrained and tuned without touching the assistant.</figcaption>
</figure>

<figure>
<img src="assets/figures/moderation-base-rate.svg" alt="A curve of classifier precision against the true share of harmful traffic on a log axis, falling steeply as harm becomes rarer, with markers at one in ten, one in a hundred, and one in a thousand.">
<figcaption>Why a strong filter still cries wolf. Holding recall and false-positive rate fixed, precision collapses as harmful traffic gets rarer, because a small false-positive rate applied to a huge benign stream swamps the true positives. This is the base-rate problem, and it is why "accuracy" is the wrong headline metric for moderation.</figcaption>
</figure>

## Governance and the harness's responsibility

Every layer so far has a dial, and turning them all toward maximum safety has a cost that shows up as **over-refusal**: the model declines requests that were perfectly fine.
Over-refusal is not a minor annoyance but the direct price of harmlessness, and it is measurable — a model that refuses "how do I kill a Python process" because it pattern-matched on "kill" has failed its user as surely as one that answered a genuinely dangerous question.
The tension is fundamental, and the original RLHF work named it: helpfulness and harmlessness are partly in conflict, and you are always choosing a point on the frontier between them [@bai2022hh].

!!! warning "Common trap"
    Treating the two error types as independent knobs you can both minimize.
    They trade against each other through the same threshold: the setting that lets fewer harmful replies through also refuses more benign ones.
    A safety change that only reports how much harm it blocked, with no measurement of how much helpfulness it cost, is reporting half the ledger.

<figure class="wide">
<img src="assets/figures/helpful-harmless-frontier.svg" alt="Two curves against increasing refusal strictness: harmful replies that slip through falls toward zero, while benign requests wrongly refused rises steeply, crossing near the middle with a note that no threshold zeroes both.">
<figcaption>Harmlessness and helpfulness come from one budget. As the system refuses more aggressively, leaked harm falls but wrongly refused benign requests climb; no threshold drives both to zero. Picking where to sit on this curve is a value judgment about which error is worse, not a technical optimum.</figcaption>
</figure>

Where does that value judgment live?
Not in the model's weights alone.
It is set by **provider policy** — the written rules a lab publishes about what its models will and won't do — and enforced through the layers of this chapter: the disposition baked in by alignment, the system prompt written for a deployment, the classifiers tuned for a context.
This is why the same base model behaves differently across products: a coding assistant, a medical tool, and a children's app draw the helpful-harmless line in different places, and they do it in the harness, not by retraining.

!!! interview "Interview"
    *Who is accountable when a deployed LLM causes harm?*
    Responsibility is layered, matching the stack.
    The model provider owns the base disposition and publishes the usage policy; the deployer owns the system prompt, the guard configuration, and the choice to connect tools and data; and both own the monitoring.
    The interview-grade point is that "the model did it" is never a complete answer — every harmful output passed through a harness that someone configured, and the defensible position is defense in depth plus logging you can audit, not a claim that the model was supposed to be perfect.

That returns us to where the chapter began: safety is not a property of the model but of the whole system around it, and it is never finished, because the adversary adapts.
The job of the harness is to make the residual path through the cheese as narrow as it can be, to measure the helpfulness it spends doing so, and to keep watching — which is why the next part of the book turns from building these systems to *evaluating* them.
