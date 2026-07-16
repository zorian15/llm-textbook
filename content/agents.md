An **agent** is a model placed in a loop. Instead of answering a prompt once and stopping, it plans a step, takes an action — usually a tool call (Chapter 19) — observes what came back, and decides what to do next, repeating until the goal is met or a budget runs out. Nothing about the model changes; what changes is that the harness keeps handing it back its own consequences. That single shift, from one-shot generation to a controlled loop, is the whole idea, and almost everything hard about agents follows from the loop being long.

## The loop is the whole idea

Strip an agent down and you find a cycle the harness runs: the model **plans** (decides what to do), **acts** (emits an action the harness can execute), and **observes** (the harness runs the action and feeds the result back into the context). The model is the controller; the harness is the interpreter that closes the loop. A weather assistant that calls one API and answers has run the loop once. A coding agent that reads files, edits them, runs the tests, reads the failures, and edits again has run it thirty times.

The important correction to make early is that **autonomy is a spectrum, not a binary**. At one end sits a fixed **workflow**: the control flow is written in code, and the model fills slots along a predetermined path — classify, then route, then summarize. At the other end sits an open-ended **agent** that decides its own path, including how many steps to take and which tools to use. Anthropic draws exactly this line and gives the load-bearing advice: use the simplest thing that passes your eval, and reach for an autonomous agent only when you genuinely cannot hardcode the path [@anthropic2024agents]. Most production "agents" are mostly workflow with a small autonomous core.

!!! intuition "Intuition"
    An agent is not a smarter model. It is the same next-token predictor from Chapter 4, given a scratchpad, a set of tools, and permission to keep going until it decides it is done.

!!! interview "Interview"
    *When would you build a workflow instead of an agent?* Whenever the task's steps are known in advance. If you can draw the flowchart, code the flowchart: it is cheaper, faster, and far easier to test, because every path is fixed. Autonomy earns its cost only when the path depends on inputs you cannot see ahead of time — an open-ended research question, a debugging session — where you can still *verify* progress even though you cannot script it. The failure mode is reaching for a general agent where a three-step chain would do, and paying in latency and unpredictability for flexibility you never use.

<figure>
<img src="assets/figures/agent-loop.svg" alt="A cycle: the model plans, emits an action, the harness executes it against tools and environment, and feeds the observation back to the model, repeating until a goal or budget stops the loop.">
<figcaption>The controller and the loop it runs. The model only ever proposes the next action; the harness executes it and returns the observation, then asks again. Autonomy is how much of the path the model chooses rather than how much intelligence sits in any one step.</figcaption>
</figure>

## Planning and memory

A long loop needs two things a single forward pass does not: a way to break a goal into steps, and a way to remember more than fits in the context window.

**Planning** starts with decomposition — turning "book me a trip" into find flights, compare, hold a fare, book a hotel. The most influential recipe for interleaving this with action is **ReAct**, which has the model alternate a *reasoning* trace with an *action* [@yao2023]. The reasoning step lets it form and revise a plan and handle surprises; the action step lets it pull real information from the world instead of inventing it, which measurably curbs the hallucination that pure chain-of-thought drifts into. The two are complements: thinking without acting invents facts, acting without thinking flails.

**Memory** comes in two tiers. The model's own reasoning is a **scratchpad** — working memory that lives entirely inside the context window, so it is bounded by the window and wiped between separate sessions. To remember across that boundary, an agent writes to **external memory**: a file, a database, a scratchpad it re-reads, or a vector store it retrieves from exactly as in RAG (Chapter 21). The generative-agents work made this concrete with a memory stream that stores observations in natural language and retrieves the relevant ones on demand [@park2023]. External memory is how an agent's effective horizon exceeds its context length.

The subtlest tool here is **reflection**: after a failure, the agent writes down what went wrong in words and keeps that note for the next attempt. **Reflexion** turns this into a learning signal without any weight update — the agent verbally critiques its last trajectory, stores the critique in episodic memory, and does better on the retry [@shinn2023]. The honest caveat, and the second question an interviewer asks, is that reflection only helps when the feedback is *trustworthy*: with a real signal like a failing test or a compiler error it works well, but asking a model to grade its own reasoning with no external check can just as easily launder a wrong answer into a confident one.

<figure class="wide">
<img src="assets/figures/react-trace.svg" alt="A ReAct trajectory: alternating Thought and Action steps inside the context window, each Action producing an Observation, with a separate external memory store the agent writes to and reads from across steps.">
<figcaption>Reasoning and acting, interleaved. Each thought plans the next action; each action returns a real observation, keeping the plan tied to the world instead of to the model's guesses. The scratchpad lives in the context window and is bounded by it; external memory is what carries state past that limit.</figcaption>
</figure>

## More agents, or just more cost?

Once one agent works, the temptation is to build many. The dominant pattern is **orchestrator–worker**: a lead agent decomposes the task, spawns worker agents (often specialists — a searcher, a coder, a critic), and synthesizes their results. The workers can run in parallel, each with its own context window.

The question that matters is when this genuinely helps. It helps when subtasks are **separable and parallel** — a breadth-first research question where independent threads can be chased at once and the total information gathered would overflow a single context window. Anthropic's research system is the canonical case: parallel subagents, each with a fresh context, beat a single agent on breadth-heavy queries, but at roughly fifteen times the token cost of a normal chat [@anthropic2025multiagent]. That ratio is the whole tradeoff in one number. Multi-agent pays off only when the value of the answer dwarfs the token bill.

It hurts when tasks are **sequential or tightly coupled**. Then the agents cannot work in parallel, and every handoff is a chance to lose context: worker B does not see what worker A saw, so it duplicates work, contradicts a decision, or acts on a stale assumption. More agents multiply the token cost *and* the error surface, and the coordination itself becomes a source of bugs. A single well-tooled agent with a long context often beats a committee.

!!! analogy "Analogy"
    Think of it as staffing a project. Independent research strands parallelize across people beautifully; a delicate sequential negotiation does not, because context lost in the handoff between two people is context the work depended on. The analogy leaks in that human teammates share a persistent mental model and can interrupt each other to sync, while subagents share only the text you explicitly pass between them.

!!! interview "Interview"
    *Your multi-agent system is slower and worse than the single agent it replaced — why?* Almost always because the subtasks were not actually independent. Coordination overhead and lossy handoffs only pay off when workers explore genuinely parallel branches; on a sequential task you have added latency, token cost, and failure modes while removing the shared context that a single agent kept for free. Reach for multiple agents for parallel breadth, not to make one hard sequential task feel more organized.

<figure class="wide">
<img src="assets/figures/orchestrator-worker.svg" alt="An orchestrator agent at the top delegates to three parallel worker agents, each with its own tools and context window, which return findings that the orchestrator synthesizes into a final answer.">
<figcaption>When a committee earns its cost. An orchestrator splits a task into parallel, separable threads, each worker exploring one with its own context, then synthesizes. On sequential or tightly-coupled work the same structure just multiplies cost and adds handoffs where context leaks.</figcaption>
</figure>

## Why agents are hard

The defining difficulty of agents is arithmetic. If each step succeeds with probability $p$, a task requiring $n$ independent correct steps succeeds with about $p^n$. At a strong-sounding 95% per step, twenty steps land at $0.95^{20} \approx 0.36$ — worse than a coin flip. Errors compound, and long horizons punish even reliable models, which is why 2026's agents are impressive on tasks of a few dozen steps and fragile on tasks of hundreds. Reflection and verification claw some of this back by catching errors before they propagate, but they do not repeal the exponent.

Three more difficulties ride alongside. **Evaluation** is genuinely hard: a task has many valid trajectories, partial progress resists scoring, and a run that reaches the right answer by a lucky wrong path should not really count as success — Chapter 24 takes this up. **Cost and latency** scale with the loop: every step is a full generation plus a tool round-trip, so an agent that "thinks" for fifty steps is fifty times the bill and the wall-clock of one answer. And **security** is the sharpest edge. An agent that browses the web and takes actions is a standing target for **prompt injection** (Chapter 18): a malicious instruction hidden in a page or a document the agent reads can hijack its plan and turn its tools against the user. The danger concentrates when three things meet — access to private data, exposure to untrusted content, and the ability to communicate outward — because that combination lets an injection exfiltrate what it should not. Chapter 23 treats the defenses; the point here is that autonomy and attack surface grow together.

!!! warning "Common trap"
    Demos mislead because they show the trajectories that worked. A per-step success rate estimated from cherry-picked runs hides the compounding that dominates real usage, and an agent that clears a curated benchmark can still fail most long real tasks. Measure end-to-end success on the full horizon you actually deploy, not the reliability of a single step.

The sober 2026 read: agents are real and useful where the loop is short, the tools are reliable, and a human or a hard check sits in the loop — coding assistants, research and browsing, structured data work. They remain brittle wherever the horizon is long, the reward is unverifiable, or the environment is adversarial. The frontier (Chapter 25) is pushing per-step reliability and horizon length up together, because only both at once moves the exponent.

<figure>
<img src="assets/figures/compounding-reliability.svg" alt="End-to-end success rate plotted against number of steps, for several per-step reliabilities; even 95% per step decays below 40% by twenty steps, while 99% stays high far longer.">
<figcaption>Why long horizons are brutal. End-to-end success is roughly per-step reliability raised to the number of steps, so even a 95%-reliable step collapses over a few dozen of them. Closing the gap needs per-step reliability pushed toward 99%-plus, not just a longer leash.</figcaption>
</figure>
