This is a working textbook about large language models: how they are built, how they are made to behave, how they are served, and where the field is still stuck. It is written for someone who already knows the basics of deep learning — you can read a training loop, you know what a gradient is, you have used PyTorch or JAX — but who wants the specific, current picture of LLMs that an engineer is expected to hold in 2026.

## What this book assumes

You are comfortable with linear algebra, probability, and the mechanics of training a neural network. You do not need to have read any transformer papers. Where a concept from deep learning matters, we give a short refresher rather than a full course, and point you to a longer treatment if you want one.

The goal is not encyclopedic coverage. It is the *load-bearing* set of ideas: the ones that come up when you design a training run, debug an inference stack, reason about a serving bill, or sit across from an interviewer who asks why bf16 beat fp16, or what a KV cache costs.

## How it is organized

The book moves in the order things actually happen to a model. **Part I** builds the architecture from tokens up. **Part II** pretrains it. **Part III** turns a raw next-token predictor into an assistant through fine-tuning and alignment. **Part IV** serves it efficiently. **Part V** — "the harness" — covers everything a provider wraps around the weights to make the product behave: system prompts, tools, structured output, retrieval, agents, and guardrails. **Part VI** asks how we know any of it works. **Part VII** looks at the frontier and the open problems, and doubles as the conclusion.

Three appendices follow: running models on your own hardware (with the memory math worked out for an Apple-silicon machine), a compact math reference, and a glossary.

## How to read it

The margin callouts do specific jobs, and you can lean on them:

!!! intuition "Intuition"
    The one-sentence mental model. If you remember nothing else from a section, remember the intuition box.

!!! analogy "Analogy"
    A concrete, everyday comparison. Analogies are lies that fit in your head; each one leaks somewhere, and we say where.

!!! interview "Interview"
    A question you are likely to be asked, and the crisp answer behind it.

!!! note "Note"
    A useful detail that would otherwise interrupt the flow.

!!! warning "Common trap"
    A place people reliably get it wrong.

## A note on how this book is made

This is a folder of linked HTML files generated from Markdown by a small Python build script. The table of contents lives in one file (`toc.py`); prose lives in `content/`. If a chapter has not been written yet, its page is a navigable stub built from a planned outline, so the whole book is always browsable. See `CLAUDE.md` for the authoring workflow.

Nothing here is final. It is a scaffold meant to be argued with, corrected, and filled in.
