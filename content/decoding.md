The model does not emit text. At every step it emits a *distribution* — a probability for every token in the vocabulary, the softmax output from Chapter 4 — and something else has to turn that distribution into an actual next token. That something is the **decoder**, and it is a separate choice from the weights: it costs no training, changes no parameter, and yet swings the output from crisp to rambling to broken. The same model, decoded two ways, is effectively two products. This chapter is about the knobs on that dial and what they actually do to the distribution.

## Greedy, beam, and why sampling won

The first instinct is to be greedy: at each step take the single most probable token, append it, repeat. A less myopic version is **beam search**, which keeps the $B$ highest-probability *partial sequences* alive at once and expands them in parallel, so a locally poor token can survive if it leads somewhere globally likely. Both are **search**: they try to find the sequence the model scores highest.

For closed-ended tasks that has a right answer to aim at, search is exactly right — machine translation and speech recognition leaned on beam search for years, because there the highest-probability output is usually the correct one. Open-ended generation breaks that assumption. Maximizing sequence probability produces text that is bland, generic, and pathologically repetitive: it falls into loops, restating the same clause until you cut it off. This is **neural text degeneration**, and its diagnosis is that natural human language is simply *not* the maximum-probability string — people constantly choose locally surprising words, and a decoder that refuses to do so drifts into a rut the model itself would never have written [@holtzman2020].

<figure class="wide">
<img src="assets/figures/beam-degeneration.svg" alt="Two curves of per-token probability over position. Human-written text zig-zags across the whole range, dipping to low-probability tokens often; beam-search text stays pinned near 1.0 and flat.">
<figcaption>Why maximizing probability backfires. Human text keeps a steady stream of locally improbable tokens (the dips), while beam search stays pinned near certainty and flattens into a loop. The most probable continuation is not the most human one — so the objective that trained the model is the wrong objective for decoding it.</figcaption>
</figure>

!!! intuition "Intuition"
    The model is a calibrated bettor, not an oracle. Its distribution says a good continuation *usually* surprises you a little; always taking the safest bet produces text no fluent writer would.

The fix is to stop searching and start **sampling**: draw the next token *from* the distribution rather than maximizing over it. Pure sampling restores the human-like variance, but it overshoots — the distribution's long tail holds thousands of low-probability tokens whose combined mass is large, and sampling one occasionally derails a sentence into nonsense. So the practical decoders are all sampling with the unreliable tail cut away, which is the next section.

!!! interview "Interview"
    *Why not just maximize likelihood at generation time?* Because the model's likelihood is trained to match the corpus *distribution*, not to make its mode a good output — and for open-ended text the mode is degenerate. Beam search still wins where the target is short and nearly deterministic (translation, transcription, constrained extraction), where higher-probability really does mean more correct. The failure is specific to open-ended, long-form generation.

## Temperature, top-k, and top-p

Every knob reshapes the same distribution before you draw from it. **Temperature** $T$ divides the logits before the softmax: $T < 1$ sharpens the distribution toward its peak (more conservative), $T > 1$ flattens it toward uniform (more random), and $T \to 0$ recovers greedy decoding. Note what temperature does *not* do — it never reorders the tokens, so it cannot make an unlikely token beat a likely one; it only changes how often the also-rans get picked.

Temperature alone still leaves the whole tail in play, so two **truncation** knobs cut it off before sampling. **Top-k** keeps the $k$ highest-probability tokens, zeros the rest, and renormalizes [@fan2018]. It is simple but rigid: $k$ is fixed whether the model is certain (where even $k=5$ admits junk) or torn (where $k=5$ throws away good options). **Top-p**, or **nucleus sampling**, fixes that by cutting on mass instead of count — keep the smallest set of tokens whose cumulative probability reaches $p$ (say $0.9$), then renormalize [@holtzman2020]. When the model is confident the nucleus is one or two tokens; when it is unsure the nucleus widens to admit many. Top-p adapts its own cutoff to the shape of the distribution, which is why it became the default.

<figure class="wide">
<img src="assets/figures/sampling-knobs.svg" alt="Four bar charts of the same ten-token next-token distribution: the raw distribution, a low-temperature sharpened version, a top-k=3 truncation, and a top-p=0.9 nucleus, with kept tokens colored and cut tokens greyed.">
<figcaption>One distribution, four knobs. Temperature rescales the whole shape without reordering it; top-k keeps a fixed count; top-p keeps a fixed probability mass, so its cutoff moves with the model's confidence. In practice you stack them — temperature first, then a truncation — rather than choosing one.</figcaption>
</figure>

!!! analogy "Analogy"
    Temperature is the thermostat on the model's boldness; top-p is a bouncer who admits guests until the room is 90% full. The analogy leaks because the bouncer counts probability mass, not heads — one dominant token can fill the room alone and shut the door, which is exactly the adaptivity top-k lacks.

!!! interview "Interview"
    *Does raising temperature change which token is most likely?* No — temperature is a monotonic rescaling, so the argmax is invariant and greedy decoding ($T=0$) is unaffected by it. What temperature changes is the *mass on the runners-up*: it is a diversity knob, not a correctness knob. A common bug is to set a high temperature to "make the model more creative" and then wonder why factual answers get worse — you widened the tail the truncation was supposed to protect you from.

## Newer samplers and the penalties

Nucleus sampling is not the last word. **Min-p** sets its threshold *relative to the top token*: keep every token whose probability is at least $p_{\min}$ times the maximum, so a peaked distribution admits a handful and a flat one admits many [@nguyen2024]. That makes it robust at high temperature — where top-p can let a flattened tail leak in, min-p's floor scales down with the peak and holds the line. **Locally typical sampling** takes a different cut entirely: instead of keeping the *most probable* tokens it keeps the *typical* ones, those whose surprisal sits close to the distribution's entropy, on the information-theoretic argument that fluent human text hugs its expected information content rather than maximizing probability [@meister2023].

<figure class="wide">
<img src="assets/figures/min-p-vs-top-p.svg" alt="Two bar-chart panels, one a peaked confident distribution and one a flat uncertain distribution, each with a horizontal min-p threshold line that sits high in the peaked case and low in the flat case, coloring the kept tokens.">
<figcaption>What min-p adds. Its cutoff is a fraction of the top token's probability, so it rises when the model is confident (keeping almost nothing but the peak) and falls when the model is unsure (keeping a broad set). The threshold tracks the model's certainty, which top-p's fixed mass does not.</figcaption>
</figure>

A separate family fights repetition directly by editing logits from the history. The **repetition penalty** divides the logit of any token already generated, discouraging reuse [@keskar2019]; OpenAI-style **presence** and **frequency** penalties subtract a flat amount for having appeared at all, or an amount scaled by how often. These break loops, but they are blunt: a strong penalty punishes the legitimate repetition that code, names, and structured text depend on.

!!! warning "Common trap"
    Reaching for a big repetition or frequency penalty to stop a loop often corrupts exactly the outputs that need to repeat — a JSON object reuses `"` and `,`, code reuses variable names, a table reuses delimiters. The penalty cannot tell a degenerate loop from a required token. Prefer fixing the sampler (min-p, a lower temperature) over leaning on penalties.

The defaults also assume open-ended prose, and that assumption is often wrong. Factual answers, extraction, and code want a *low* temperature — you are not after diversity, you are after the one right continuation. Reasoning models (Chapter 25) push this furthest: they typically want near-greedy decoding, because the goal is the single most-likely chain of thought, and sampling temperature there mostly adds a chance to wander off a correct derivation.

!!! interview "Interview"
    *A code-generation endpoint returns subtly different output every call and sometimes malformed code. What do you change first?* The decoder, before the model. Drop the temperature toward zero and widen or disable truncation so you are effectively taking the model's best guess; determinism and syntactic validity matter more than diversity here. Sampling settings are a per-task decision, not a global one — the same weights serve a creative writer and a compiler, and they should not share a temperature.

## Structured and constrained decoding: a preview

Sometimes "usually valid" is not good enough — a downstream parser needs the output to *be* a JSON object, or match a regex, or follow a grammar, every time. **Constrained decoding** guarantees this by masking: at each step it sets the logits of every token the format forbids to $-\infty$ before sampling, so only legal continuations can be drawn. The samplers from this chapter still run, but on the survivors — mask first, then apply temperature and top-p to whatever remains legal.

<figure class="wide">
<img src="assets/figures/constrained-decoding.svg" alt="The model's raw next-token distribution over five candidates passes through a JSON-schema gate that allows only the two boolean tokens; the forbidden tokens are crossed out and the remaining two are renormalized into a valid distribution.">
<figcaption>Constrained decoding as a mask. A schema expecting a boolean forbids everything but two tokens; their logits survive, the rest go to negative infinity, and the distribution renormalizes over what is left. Validity is enforced by construction, not hoped for — the model cannot emit a token the grammar disallows.</figcaption>
</figure>

The catch is that a hard mask can distort the model's intent — if the format forbids the token the model actually wanted, you get a valid output the model would rate poorly — and building the mask correctly against the tokenizer is fiddly. This is a harness concern, so the full treatment of JSON mode, regex, and context-free grammars lives in Chapter 20; here it is enough to see that decoding and format constraints compose. The other axis this chapter set aside is *cost*: every sampled token requires a full forward pass, and making that pass cheap — the KV cache, speculative decoding — is the subject of Chapter 15.
