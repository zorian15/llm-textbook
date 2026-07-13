A large language model is, mechanically, a function that reads a sequence of tokens and outputs a probability distribution over what the next token should be. That is the whole contract. Everything else — the apparent reasoning, the tool use, the ability to write a sonnet or a Kubernetes config — is what happens when you make that function very large, train it on a very large amount of text, and then wrap it carefully.

This chapter lays out the shape of the field so the rest of the book has somewhere to hang.

## The one function

Call the model $p_\theta$, with parameters $\theta$. Given a context of tokens $x_1, \dots, x_t$, it produces

$$p_\theta(x_{t+1} \mid x_1, \dots, x_t),$$

a distribution over the vocabulary. To generate text, you sample a token from that distribution, append it, and repeat. That is *autoregressive* generation: the model's own output becomes part of its next input.

!!! intuition "Intuition"
    An LLM is a next-token predictor run in a loop. Its "thinking" is a side effect of predicting each next token well enough that the whole sequence hangs together.

Training is nothing more exotic than teaching $p_\theta$ to assign high probability to the tokens that actually came next in a huge corpus of human text. The loss is cross-entropy — the number of *bits of surprise* the model feels at each real token. Drive that surprise down across trillions of tokens and, empirically, useful behavior falls out.

!!! analogy "Analogy"
    Think of a student who has read the entire internet and plays a relentless game of "guess the next word." To get good at the game on *arbitrary* text, they are forced to absorb grammar, facts, arithmetic, code, and the rhythms of argument. The game is trivial; being good at it is not. This leaks in one place: the student never gets to *act* in the world during study, only to predict — which is exactly why Part III exists.

## Why "large" is the whole story

Nothing about next-token prediction is new; the surprise is what happens at scale. As you increase parameters, data, and compute together, the loss falls along a smooth power law [@kaplan2020], and somewhere along that curve the model stops merely completing text and starts following instructions, doing multi-step arithmetic, and writing working code [@brown2020; @wei2022]. We spend Chapter 9 on the exact shape of these *scaling laws*, because they are the closest thing the field has to a design equation.

The practical consequence: much of LLM engineering is really *systems* engineering. Making a model bigger means splitting it across hundreds of GPUs to train (Chapter 8) and squeezing it back onto a few to serve (Part IV). A great deal of what an LLM engineer does is fight the memory and bandwidth of physical hardware.

## From a text predictor to an assistant

A freshly pretrained model is a *base model*. It will happily continue a document, but ask it a question and it might respond with a list of similar questions — because on the internet, questions are often followed by more questions. It is a mirror of its training distribution, not yet an assistant.

Turning it into ChatGPT- or Claude-like behavior takes *post-training*:

- **Supervised fine-tuning** (Chapter 10) shows it thousands of examples of the assistant format — a helpful answer following a user request.
- **Preference optimization** (Chapters 11–12), via RLHF [@christiano2017; @ouyang2022] or its RL-free cousins like DPO [@rafailov2023], tunes it toward responses humans actually prefer, using comparisons rather than demonstrations.

!!! interview "Interview"
    *Why isn't supervised fine-tuning enough — why bother with RLHF?* Because you can *recognize* a good answer more reliably than you can *write* the single best one. Preference methods learn from rankings of outputs the model itself generates, which is a richer and more scalable signal than a fixed set of gold demonstrations.

## The harness: the product around the weights

Here is the thing most architecture-focused explanations skip. The model you talk to through an API is not just the weights. A provider wraps it in a **harness**: the system that decides what context the model sees and what happens to its output.

The harness is where a lot of the "it just behaves" comes from:

- A **system prompt** sets the persona, rules, and defaults before you type anything (Chapter 18).
- **Tool calling** lets the model's text trigger real actions — searches, code execution, API calls — and feeds the results back (Chapter 19).
- **Structured output** forces valid JSON when a downstream system needs it (Chapter 20).
- **Retrieval** injects fresh or private documents into the context so the model can ground its answers (Chapter 21).
- **Guardrails** — separate classifiers and policies — filter inputs and outputs (Chapter 23).

!!! intuition "Intuition"
    The weights are an engine. The harness is the car built around it: steering, brakes, dashboard, seatbelts. A benchmark tests the engine; a product ships the car.

<figure class="wide">
<img src="assets/figures/lifecycle.svg" alt="Four stages left to right: pretraining produces a base model, supervised fine-tuning produces an instruct model, preference optimization produces an aligned model, and the harness produces the product. The first three change the weights; the harness does not.">
<figcaption>The life of a model, and the shape of this book. The first three stages change the weights and are covered in Parts II–III; the harness changes nothing about the model but determines almost everything about the product, and is covered in Part V. Most engineers work to the right of the line.</figcaption>
</figure>

Part V is devoted to the harness, because in 2026 most engineering roles that touch LLMs are really building harnesses, not training weights from scratch.

## The frontier

Two shifts define the current edge. First, **reasoning models** (Chapter 25) spend extra compute *at inference time* — generating long internal chains of thought, often trained with reinforcement learning on problems whose answers can be checked automatically [@openai2024; @deepseek2025]. Compute at test time became a new axis to scale, alongside model size.

Second, the **open problems** (Chapter 26) remain stubborn: models still hallucinate confidently, we still cannot fully read what a model has learned, and we do not yet have alignment techniques we would trust on systems smarter than us. These are where the interesting research is, and they close the book.

## A map of the rest

If you want the shortest useful path: read the transformer chapter (4), the scaling-laws chapter (9), and the post-training chapters (10–12) to understand how models are *made*; then the inference and harness parts (IV–V) to understand how they are *shipped*. The appendix on running models locally is a good place to get your hands dirty on your own machine.

Onward to tokens.
