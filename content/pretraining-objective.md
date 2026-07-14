Part I froze the architecture; this part learns the weights. The objective could not be simpler — predict the next token, everywhere, on everything — and that simplicity moves all the leverage into two questions this chapter takes in turn: what the objective really optimizes, and what "everything" should be. The second question is the underrated one. Two teams with the same architecture and the same compute routinely get different models, and the difference is almost always the data.

## Next-token prediction as compression

The training loss is cross-entropy: for each position, the negative log-probability the model assigned to the token that actually came next. Measured in base 2, that is literally a count of *bits* — and Chapter 1's perplexity is just this number re-expressed as a choice count. An old result gives the count an operational meaning: any probability distribution over the next symbol can drive an arithmetic coder that spends exactly $-\log_2 p(x)$ bits encoding the symbol $x$. A model that predicts well therefore *is* a compressor, mechanically, not metaphorically — and a strong LLM used this way out-compresses gzip on text by a wide margin [@deletang2024]. Training to minimize cross-entropy is training to compress the corpus.

This is why "it just predicts the next token" is not the dismissal it sounds like. Shannon estimated English's entropy by having people play the guess-the-next-letter game, and found that guessing well requires knowing spelling, grammar, idiom, and context [@shannon1951]. To compress arbitrary text toward its entropy, shallow statistics run out fast; the remaining bits are recoverable only from facts, structure, and reasoning. Compression pressure is what forces the model to learn them.

!!! intuition "Intuition"
    A model that has to bet on every next token, on every text ever written, can only keep winning by understanding what the text is about. The loss is the bookkeeping of those bets, in bits.

!!! interview "Interview"
    *Model A reports lower perplexity than model B — is A better?* Only if they share a tokenizer. Perplexity is per *token*, and a tokenizer with a bigger vocabulary packs more text into each token, raising per-token entropy while compressing the sequence (Chapter 3). To compare across tokenizers, renormalize to bits per byte of raw text — the compression view again, which is tokenizer-independent.

The intuition has limits, and naming them is second-question territory. Prediction rewards matching the *distribution* of the corpus, including its errors, biases, and boilerplate; a perfect predictor of internet text confidently completes falsehoods that are common online. And the loss counts every token equally, though tokens differ wildly in how much they matter — getting a name or a digit wrong is billed the same one-token price as flubbing "the." What prediction alone cannot teach — that a prompt is a request from a user, not a document to continue — is Part III's opening problem.

<figure class="wide">
<img src="assets/figures/prediction-compression.svg" alt="Two models predict the same next token. The sharp model assigns it high probability and pays about one bit; the flat model assigns it low probability and pays several bits. Summed over a corpus, the sharp model's encoding is far shorter.">
<figcaption>The loss is a file size. Each token costs the model the negative log of the probability it assigned, in bits; a sharper prediction is a shorter code. Minimizing cross-entropy over a corpus and compressing that corpus are the same act.</figcaption>
</figure>

## Where the data comes from

Scale first, because it sets the terms: modern pretraining corpora are measured in *tokens*, and the number has grown faster than model size. GPT-3 trained on roughly 300 billion tokens [@brown2020]; Chinchilla made 1.4 trillion the compute-optimal choice for its size (Chapter 9 derives why) [@hoffmann2022]; Llama 3 trained on about 15 trillion [@grattafiori2024]. Counting in gigabytes misleads — deduplication, filtering, and tokenizer choice all change bytes-per-token — so the field standardized on the unit the loss actually consumes.

Where do trillions of tokens exist? Essentially one place: the web. Common Crawl, a nonprofit's ongoing scrape of the public internet, is the base ingredient of nearly every open corpus — The Pile drew on it alongside academic and specialist sources [@gao2020], and FineWeb refined roughly a hundred crawl snapshots into about 15 trillion usable tokens [@penedo2024]. Around that base, every serious mixture adds smaller high-value streams: **code** (GitHub and its descendants), **academic text** (papers, textbooks), **books**, **encyclopedic reference**, **math**, and increasingly **multilingual** web. The web supplies volume; the curated streams supply density.

!!! note "Note"
    Frontier labs disclose less about data than about anything else — data is where the competitive edge and the copyright exposure both live. The open-data projects (The Pile, FineWeb, and successors) are how the field actually knows what works.

<figure class="wide">
<img src="assets/figures/pretraining-mix.svg" alt="A stacked horizontal bar of an illustrative pretraining mixture: filtered web text dominates at roughly half, with code, multilingual web, academic text and books, math, and reference making up the rest.">
<figcaption>What a mixture looks like. Filtered web crawl supplies most of the tokens because it is the only source with trillions to give; code, academic text, math, and reference are small by share but heavily over-represented relative to their share of the raw internet.</figcaption>
</figure>

## Cleaning and deduplication

Raw crawl is unusable. It is boilerplate, navigation menus, SEO spam, auto-generated listings, adult content, and the same page mirrored ten thousand times. Between the crawl and the training run sits a pipeline that discards most of the internet:

1. **Extraction** strips HTML to running text — an unglamorous step that changes downstream quality measurably [@penedo2024].
2. **Language identification** routes documents to the intended language mix.
3. **Quality filtering** drops junk, by hand-written heuristics (document length, symbol ratios, repetition) and by classifiers trained to score "does this look like text worth learning from."
4. **Deduplication** removes exact and near-duplicate documents, typically with hash-based fuzzy matching at scale. Duplicated text distorts the implicit data mixture, wastes compute on repeats, and makes the model far more likely to memorize and regurgitate the repeated passage [@lee2022].
5. **Decontamination** removes documents that overlap the evaluation benchmarks you intend to report. Test questions are on the internet too.

!!! warning "Common trap"
    Contamination does not just flatter a benchmark score; it silently converts an evaluation of *generalization* into an evaluation of *recall*, while the number still looks like the former. Chapter 24 returns to this from the evaluator's side — and to why decontamination by exact match is never fully clean, since paraphrases survive.

Each stage is a dial, not a switch, and the dials trade against each other: filter aggressively and you gain average quality but lose diversity and total tokens; filter lightly and you keep scale but learn spam. The striking empirical fact is how much these unglamorous choices matter — FineWeb's ablations show filtering and dedup decisions moving downstream benchmark accuracy by more than many architecture changes do [@penedo2024].

<figure class="wide">
<img src="assets/figures/data-funnel.svg" alt="A funnel from raw web crawl through extraction, language ID, quality filtering, deduplication, and decontamination, with the token count shrinking at every stage until a small refined corpus remains.">
<figcaption>Most of the internet does not survive. Each cleaning stage discards a large fraction of what enters it, and the final corpus is a small, deliberate residue of the raw crawl — the model never sees the internet, only this distillate.</figcaption>
</figure>

## Data mixtures and curriculum

Cleaned sources still have to be combined, and the weights are a first-class hyperparameter. Sampling proportionally to raw size would drown everything in web text, so high-density domains are *upsampled* — a small math corpus might be repeated several times per "epoch" of web text. Repetition is affordable in moderation: up to roughly four epochs over a source costs little versus fresh data, after which returns decay sharply [@muennighoff2023]. Tuning the weights by grid search is hopeless at full scale, so mixtures are tuned on small proxy runs, or learned — DoReMi trains a small model to find domain weights that transfer to a much larger run [@xie2023].

Code deserves its own sentence: every modern mixture over-weights it, even for models not aimed at programmers, because training on code measurably improves structured, multi-step behavior in prose — it is the largest corpus of explicit logical procedure ever written.

The mixture also need not be constant. The most common curriculum is **annealing**: in the final stretch of training, as the learning rate decays toward zero (Chapter 7), the mixture shifts toward the highest-quality sources — curated text, math, code [@grattafiori2024; @hu2024]. The last tokens a model sees under a tiny learning rate are the ones that most directly shape its final polish, so you spend your scarcest, best data there.

!!! analogy "Analogy"
    It is exam-week studying: a semester of broad reading, then the final days spent only on the best notes. The analogy leaks in one way — the model does not consolidate on its own between sessions; the annealing schedule has to *be* the consolidation, which is why its placement against the learning-rate decay matters.

!!! interview "Interview"
    *You have a fixed token budget and your model underperforms on reasoning. What is the cheapest lever?* Not architecture — mixture. Upweight code and math, consider a quality-filtered second pass on the best web data, and anneal on your highest-quality sources at the end of training. Data interventions dominate architecture interventions at fixed compute, which is why data work is where pretraining teams actually spend their time.

<figure class="wide">
<img src="assets/figures/mixture-annealing.svg" alt="Domain shares of the training mixture over the course of training: web text dominates for most of the run, then in the final annealing phase its share drops while code, math, and curated text expand.">
<figcaption>The curriculum that survived. Modern pretraining is mostly one constant mixture — until the end, where the learning-rate decay is paired with a shift toward the best data, so the final, smallest updates come from the highest-quality tokens.</figcaption>
</figure>

The objective and the diet are set. What remains is to actually run the loop from Chapter 2 a few million times without it exploding — the subject of the next chapter.
