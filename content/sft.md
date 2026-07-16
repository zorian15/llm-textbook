A base model has read much of the internet, but no one has ever shown it how to *behave*. Ask it a question and it may answer, or it may reply with three more questions, or drift into a listicle — because in its training data a bare question was as often the start of a forum thread as the setup for an answer (Chapter 1). Supervised fine-tuning (SFT) is the first and simplest fix: gather thousands of worked demonstrations of an assistant responding well, and train the model to imitate them with the very same next-token objective it was pretrained on. This is behavior cloning. The move that makes it work is subtle: SFT does not so much pour new facts into the model as *select*, from everything pretraining already stored, the narrow slice of behavior that a helpful assistant exhibits.

## From a document completer to an assistant

The base model is a document completer, and that is the whole problem. It very likely knows that the capital of France is Paris — it will produce "Paris" readily in the middle of an encyclopedia article — but it has not learned that a prompt is a *request to be answered* rather than a *document to be continued*. The gap is not knowledge; it is persona and format. The model needs to be told, by example, what kind of text should follow a user's turn.

SFT supplies those examples. You assemble a dataset of demonstrations — each an instruction paired with an ideal response — and run the ordinary training loop over them, minimizing cross-entropy on the response tokens [@ouyang2022]. Because the objective is unchanged, SFT is cheap next to pretraining: a few epochs over tens of thousands to a few hundred thousand demonstrations, against the trillions of tokens of Chapter 6. Fine-tuning on instructions phrased across many tasks was the finding that first made this generalize — a model tuned to follow instructions on one set of tasks follows *unseen* instructions zero-shot [@wei2021].

<figure class="wide">
<img src="assets/figures/sft-behavior-cloning.svg" alt="Three demonstration cards, each a user instruction paired with an ideal assistant response, feed into a base model that is trained to reproduce the responses and becomes an assistant.">
<figcaption>SFT is behavior cloning. The objective never changes from pretraining — predict the next token — but now the targets are curated answers, so the model learns to copy the demonstrated behavior. It is selecting a persona it can already voice, not memorizing new facts.</figcaption>
</figure>

!!! intuition "Intuition"
    Pretraining teaches the model everything it will ever know; SFT only teaches it which of those things to say, and how, when someone asks.

!!! analogy "Analogy"
    SFT is an apprentice shadowing an expert, copying what a good answer looks like. The analogy leaks in the way that matters most: the apprentice only ever imitates behaviors it was shown and silently inherits the demonstrator's mistakes; it never learns from the consequences of its *own* attempts. Closing that gap — letting the model improve past its demonstrations — is what RLHF (Chapter 11) is for.

!!! interview "Interview"
    *Does SFT teach the model new knowledge?* Almost none. A demonstration is seen a handful of times at a small learning rate, which is nowhere near enough to install a fact the base model lacks; what SFT reliably changes is behavior — format, tone, willingness to answer, task framing. This is why teams put their factual effort into the pretraining mixture (Chapter 6) and treat SFT as behavior shaping. It is also the setup for the chapter's sharpest failure mode: demand facts the base model does not have and SFT teaches confident guessing instead.

## Instruction data and chat templates

SFT data is *conversations*, not raw text. Each example carries roles — a system message that sets policy, a user turn, an assistant turn, sometimes many turns — and the model must learn to tell whose turn it is and to generate only the assistant's part. To make roles legible, every turn is wrapped in a **chat template**: a fixed format that delimits each role with special tokens like `<|user|>` and `<|assistant|>`, added to the vocabulary as their own atoms (Chapter 3). Different model families use different templates, and this is a live deployment trap — feed a model a template it was not trained on and it quietly degrades, because the role tokens it keys on are missing (more in Chapter 18).

The second essential trick is **loss masking**: compute the loss only on the assistant's tokens and mask out the system and user tokens. You want the model to learn to *produce* responses, not to model the user's phrasing; training on the prompt spends capacity teaching it to generate instructions, which is not the job. The end-of-turn token is masked *in*, deliberately — training on it is how the model learns to **stop**, rather than answering and then rambling into an invented next turn.

<figure class="wide">
<img src="assets/figures/chat-template-masking.svg" alt="A single training example shown as three role-tagged rows — system, user, assistant — with special role tokens; a bracket marks the system and user rows as masked prompt and the assistant row as the completion that carries the loss.">
<figcaption>What the model actually trains on. The chat template marks whose turn it is with special tokens, and the loss falls only on the assistant's completion — including the end token that teaches the model to stop. The prompt is context, never a target.</figcaption>
</figure>

!!! interview "Interview"
    *Why mask the prompt tokens instead of training on the whole sequence?* The task is to map a prompt to a good completion, so the gradient should come only from the completion; training on the prompt teaches the model to *predict user messages*, which wastes capacity and can bias it toward parroting the input. On short demonstrations, unmasked training also lets the easy-to-predict boilerplate of the prompt dominate the loss. A related efficiency detail: multiple short conversations are often *packed* into one sequence, with the attention masked at the boundaries so examples cannot read across each other.

## How much data, and of what quality

Here the surprising result is how *little* data SFT needs. LIMA fine-tuned a strong base model on just 1,000 carefully curated prompt-response pairs, with no reinforcement learning, and produced an assistant competitive with far more heavily tuned systems [@zhou2023]. The authors read this as evidence for a **superficial alignment hypothesis**: a model's knowledge and capabilities are learned almost entirely in pretraining, and alignment mostly teaches the *format and style* in which to expose them — so a small, diverse, high-quality set is enough to select the right behavior.

The practical corollary is that quality and diversity beat sheer volume, and the failure it warns against is real: because SFT mostly shapes presentation, it is easy to make outputs *look* better — fluent, formatted, confident — without making them more correct. Optimizing your demonstrations for polish over substance produces exactly that. Early SFT sets were hand-written, which is expensive; modern pipelines lean on **synthetic data**, either bootstrapping instructions from the model itself [@wang2023] or distilling demonstrations from a stronger model, then filtering hard for quality and task coverage [@grattafiori2024]. Coverage — how many genuinely different kinds of request appear — tends to matter more than raw count.

<figure>
<img src="assets/figures/sft-data-quality.svg" alt="Two curves of assistant quality against the number of SFT demonstrations on a log axis: a curated-and-diverse curve rises fast and plateaus by about a thousand examples, while a scraped-and-noisy curve climbs slowly to a lower ceiling.">
<figcaption>Curation beats volume. Because SFT surfaces behavior the base model already has, a thousand clean, diverse demonstrations bank most of the gain, while a noisy set needs orders of magnitude more data to reach a lower ceiling.</figcaption>
</figure>

!!! warning "Common trap"
    A higher SFT loss on a held-out set is not the goal, and neither is a lower one — SFT quality is about *which* behaviors you cloned, not how tightly you fit them. Overfitting a large, uniform set makes the model more fluent and more templated at once, which reads as improvement right up until you probe substance.

!!! interview "Interview"
    *You have 50,000 scraped demonstrations and 1,000 hand-checked ones. Which do you train on?* The clean thousand, or the scraped set aggressively filtered down toward that quality bar. Noisy demonstrations teach noisy behavior — sloppy formats, wrong answers stated confidently, inconsistent refusals — and SFT clones all of it faithfully. The LIMA result is the license to throw data away; the win is in curation and diversity, not headcount.

## Failure modes

SFT is powerful precisely because it clones behavior faithfully, which is also how it goes wrong. **Catastrophic forgetting** comes first: train too long or too hot on a narrow set and the model erodes the broad competence pretraining gave it, trading generality for the demonstration style. The defenses are all forms of restraint — few epochs, a small learning rate, mixing in general data, or freezing most weights and tuning a thin adapter (PEFT and LoRA, Chapter 13). **Format overfitting** is the milder cousin: the model latches onto surface habits of the demonstrations, opening every answer with the same preamble or forcing everything into bullet lists.

Two failure modes are subtler and more consequential. **Sycophancy** can be seeded here: if demonstrations consistently agree with or flatter the user, the model learns agreement as the expected behavior. The effect is amplified later by preference optimization, where human raters reward answers that match their beliefs [@sharma2023], but demonstrations that never push back plant the seed (Chapter 12). And **hallucination** has a specific, well-documented root in SFT. When demonstrations require facts the base model does not hold, fine-tuning does not install them — such examples are learned slowly, and as the model finally fits them, its overall tendency to hallucinate rises measurably [@gekhman2024]. You have not taught it the fact; you have taught it to answer just as confidently when it is guessing.

<figure class="wide">
<img src="assets/figures/sft-knowledge-boundary.svg" alt="A circle labeled 'what the base model knows' holds green checkmark dots for demonstrations inside its knowledge, while red x-mark dots sit outside; a legend explains that inside demonstrations reinforce recall while outside ones teach confident guessing.">
<figcaption>Why fine-tuning on new knowledge backfires. Demonstrations inside the base model's knowledge reinforce recall and formatting; demonstrations outside it cannot add the fact, so they train away the model's hesitation and leave confident hallucination in its place.</figcaption>
</figure>

!!! intuition "Intuition"
    Keep your demonstrations inside the base model's knowledge, and where you cannot, demonstrate the honest answer — "I don't know" — so that not-knowing is one of the behaviors you clone (Chapter 26).

SFT gets you a competent imitator, and no further: the model is capped by the quality of its demonstrations and cannot learn from the consequences of its own answers. To push past the best demonstration — and to teach the model what people actually *prefer* among the answers it could give — you stop cloning behavior and start optimizing against a preference signal, which is where RLHF (Chapter 11) and preference optimization (Chapter 12) take over.
