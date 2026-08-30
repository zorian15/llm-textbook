A model's native output is a stream of tokens meant for a human to read.
A program cannot read prose.
When the model's answer has to flow into a database write, a function call, or another service, "reads well" is the wrong bar — the output has to *parse*, every time, as a specific shape the caller agreed to in advance.
This chapter is about closing that gap.
The reliable way to get machine-readable output is not a better-worded request; it is to change how the model decodes, so an invalid answer becomes impossible rather than merely unlikely.

## Free text is not an API

The first thing everyone tries is to ask.
You append "reply with a JSON object with fields `name` and `age`" to the prompt and hope.
This works surprisingly often, and it fails often enough to be dangerous.
The model wraps the object in an apology, adds a trailing comma, emits `age: "twelve"` instead of a number, or fences the whole thing in a Markdown code block your parser does not expect.
Each of those is a parse error, and each parse error costs you a retry: another full forward pass, more latency, more money, and no guarantee the second try is any better than the first.
Call this the **prompt-and-pray tax**.

<figure class="wide">
<img src="assets/figures/json-retry-loop.svg" alt="A prompt asking for JSON goes into a model, which emits free text with a trailing comma; a parser rejects it and loops back to the model as a retry, while a validated path exits to a downstream system.">
<figcaption>The tax on coaxing structure by prompting alone. The model usually returns parseable JSON, but the fraction that does not sends you around the retry loop — a second forward pass that costs latency and money and may fail again. At one call the tax is invisible; at a million calls, or inside a tool-calling agent (Chapter 19) where one malformed argument breaks the chain, it is the whole problem.</figcaption>
</figure>

The tax is easy to miss because it scales with things you only meet in production.
A 1% failure rate is fine in a demo and unacceptable when you make millions of calls a day, or when the output feeds a chain of tool calls where a single malformed argument derails everything after it.
Prompting also fights itself: the same instructions that pin down the format crowd out the room the model has to actually think, a tension we return to at the end of the chapter.

!!! interview "Interview"
    *Your service asks for JSON in the prompt and gets valid JSON 99% of the time. Why is that not good enough?* Because the 1% is not random noise you can average away — it clusters on the hard, long, or unusual inputs you most need to handle, and every failure forces a retry that doubles cost and latency for that request. Worse, "99% valid JSON" says nothing about whether the *schema* is right: the object can parse cleanly and still have the wrong field names or types. Validity you can *measure* after the fact is not the same as validity you can *guarantee* before you ship a token.

## Constrained decoding: validity by construction

The fix is to stop hoping and start forbidding.
Recall from Chapter 14 that the model does not emit text; at each step it emits a distribution over the vocabulary, and the decoder turns that distribution into a token.
**Constrained decoding** inserts one operation just before the token is chosen: given the target format and everything generated so far, it computes the set of tokens that could legally come next, and sets the logit of every *other* token to $-\infty$.

$$
\ell'_t = \begin{cases} \ell_t & \text{token } t \text{ is a legal continuation} \\ -\infty & \text{otherwise.} \end{cases}
$$

After the softmax, the forbidden tokens carry probability exactly zero, so no sample — greedy, top-p, or otherwise — can ever draw one.
The samplers from Chapter 14 still run; they just run on the survivors.
Validity stops being a property you check afterward and becomes a property of the machine: the model *cannot* emit a token the format disallows.

!!! intuition "Intuition"
    Prompting asks the model to please stay on the road; constrained decoding removes every exit but the legal ones. The model still steers, but the guardrails are what keep it on the road, not its good intentions.

There is one wrinkle that makes this harder than it sounds.
A format like JSON is defined over *characters* — a `{`, a `"`, a digit — but the model emits *tokens*, and a token is often several characters (` {"name"` might be a single token) and rarely lines up with the format's boundaries.
So "which tokens are legal next" is not a lookup in the grammar; it is a question about which tokens' character-expansions keep you inside the grammar, and answering it efficiently for a 100,000-token vocabulary at every step is the real engineering problem, handled by the machinery in the next section.

<figure class="wide">
<img src="assets/figures/logit-mask-valid-tokens.svg" alt="A next-token distribution over seven candidate tokens passes through a grammar gate that is in the state 'expecting a value'; tokens that cannot begin a JSON value are set to negative infinity and greyed out, and the remaining legal tokens are renormalized into a new distribution.">
<figcaption>Masking, one step. The grammar knows its current state — here, having just seen a key and colon, it expects the start of a value — and that state names the legal tokens. Everything else drops to negative infinity before the softmax, and the distribution renormalizes over what remains. The model's preferences among the legal tokens are preserved; its ability to leave the format is removed.</figcaption>
</figure>

## Grammars and schemas: from spec to mask

You do not write token masks by hand.
You describe the target shape once — as a **JSON Schema**, a **regular expression**, or a **context-free grammar** — and a compiler turns that description into the per-step masks.
The key idea, introduced by Willard and Louf in the Outlines library, is to treat generation as a walk over a **finite-state machine** [@willard2023].
A regex or schema compiles to an automaton whose states encode "where am I in the format so far"; you precompute, once, an index mapping each state to the set of vocabulary tokens that keep the walk alive.
At generation time, advancing the state and fetching its allowed-token set is roughly an $O(1)$ lookup, so the mask costs almost nothing per step.

<figure class="wide">
<img src="assets/figures/schema-to-fsm.svg" alt="Three stages left to right: a JSON Schema snippet, an arrow to a small finite-state machine with labeled states and transitions, and an arrow to a column showing the vocabulary split into allowed and forbidden tokens at the current state.">
<figcaption>How a schema becomes a decoding constraint. The schema is compiled once into an automaton whose state tracks the position in the format; each state carries a precomputed set of legal next tokens. Decoding is then a walk over that automaton, and the mask at every step is a table lookup rather than a fresh parse — which is what makes guaranteed-valid output cheap enough to turn on by default.</figcaption>
</figure>

Regexes and JSON Schemas cover most of what applications need, but some targets — a whole programming language, deeply nested JSON — need the extra power of a context-free grammar, expressed in a notation like llama.cpp's **GBNF**.
Grammars are heavier: a CFG's "legal next token" can depend on unbounded context (how many brackets are still open), so not every token can be pre-classified.
Modern engines close much of that gap.
XGrammar, for instance, splits the vocabulary into *context-independent* tokens it can check once ahead of time and the smaller set of *context-dependent* tokens it must resolve at runtime, which brings the per-step overhead of grammar-constrained JSON close to zero [@dong2024].
This is why structured output moved from a research trick to a default feature of serving stacks: the machinery finally became cheap.

!!! interview "Interview"
    *Why is regex-constrained decoding cheaper than grammar-constrained decoding?* A regular expression compiles to a finite-state machine with a bounded number of states, so you can fully precompute the allowed-token set for each state and never touch the grammar again at run time. A context-free grammar can require a stack — unbounded nesting — so its set of legal tokens depends on run-time context that no finite index can enumerate in advance. The practical takeaway: reach for the *weakest* formalism that expresses your target. If a regex suffices, do not pay for a grammar.

## The format tax and other pitfalls

Constraining decoding guarantees a valid *shape*, not a good *answer*, and the two can pull apart.
The mask only ever removes options; when it removes the token the model most wanted, you get an output that is perfectly well-formed and worse than what the model would have said unconstrained.
Empirically this is not hypothetical: forcing models to answer under tight format restrictions measurably degrades their reasoning, and the tighter the constraint, the larger the hit [@tam2024].
Call it the **format tax on quality**, distinct from the parsing tax of the first section.

<figure class="wide">
<img src="assets/figures/constrain-answer-not-thinking.svg" alt="Two rows. In the top row a model is forced into JSON from the first token and produces a short, shallow answer. In the bottom row the model reasons in free text first and only its final answer field is constrained, producing a better answer.">
<figcaption>Constrain the answer, not the thinking. Force the JSON from the first token and you rob the model of the scratch space where reasoning happens (top). Let it think in free text and clamp only the final answer to the schema (bottom), and you keep both the guarantee and the quality. The constraint is a gate at the exit, not a cage for the whole computation.</figcaption>
</figure>

The mitigation follows from the diagnosis.
Most of the tax comes from constraining the model *while it is still working*, so the standard move is to separate the two phases: let the model reason in free text — the chain of thought of Chapter 25 — and switch the constraint on only for the final answer.
Reasoning models make this explicit, emitting an unconstrained thinking block and then a constrained answer block.
The rule of thumb: never constrain the scratchpad; constrain only the part a program will actually read.

!!! warning "Common trap"
    Over-tight schemas cause their own damage. Pin a field to a strict `enum` and, if the true answer is not in your list, the model is forced to emit a wrong-but-valid value with full confidence — the constraint manufactures a hallucination the model would otherwise have hedged. Leave an escape hatch (an "other" option, a nullable field) for the cases your schema did not foresee, or you trade parse errors for silent semantic errors, which are worse.

The last cost is latency, and it comes in two forms.
Compiling a complex schema or grammar into its automaton takes real time, so engines cache compiled constraints and reuse them across requests rather than rebuilding per call.
The per-step mask is cheap by design (that was the point of the FSM index), but it is not free, and a poorly implemented constraint that re-parses the prefix every step can cost more than the retries it was meant to save.

!!! interview "Interview"
    *When would you deliberately not use constrained decoding?* When the task is reasoning-heavy and the output is read by a human, the format tax can outweigh the parsing convenience — you are better off letting the model answer freely and extracting structure in a cheap second pass. And on a closed API that exposes only "JSON mode" without a schema, you get validity of *syntax* but no guarantee of *fields*, so you still validate downstream. Constrained decoding is the right default for machine-consumed output, not a universal on-switch.
