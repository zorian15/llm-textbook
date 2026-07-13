A language model never sees text. It sees a sequence of integers, produced by a **tokenizer** that chops text into pieces from a fixed vocabulary and numbers them. The tokenizer is a contract signed before training begins: the model's entire view of language — what it can compare, count, and complete — is mediated by these pieces. That makes tokenization the source of some of the strangest LLM behavior, and of very real serving costs, so it deserves a chapter before we build the transformer that consumes the tokens.

## Why not characters or words

The two obvious units both fail.

**Words** fail on the *open-vocabulary problem*. Any fixed word list will be missing tomorrow's product names, someone's typo, rare morphology ("untranslatable"), and most of every other language. Everything missing collapses into a single unknown-word token, and the model is blind to it.

**Characters** never miss, but they are expensive. A character carries little meaning on its own, so the model burns layers reassembling words before it can think about them — and the sequences are long. Attention cost grows with the square of sequence length (Chapter 4), so 5× more positions is far more than 5× more compute. The context window, a scarce resource, fills with spelling.

Subword tokenization is the negotiated middle: a vocabulary of tens of thousands of *pieces*, where common words are a single token and rare words are built from a few fragments. The knob is vocabulary size, and it is a genuine tradeoff: a bigger vocabulary shortens sequences but grows the embedding matrix, and its rarest tokens are each seen less often during training, so their representations are worse.

!!! intuition "Intuition"
    Tokenization is compression with a fixed dictionary: frequent strings earn a short code (one token), rare strings get spelled out from pieces. Frequency in the training corpus decides who earns a short code.

<figure>
<img src="assets/figures/tokenizer-tradeoff.svg" alt="A curve of tokens per passage against vocabulary size on a log scale, falling steeply then flattening, with a highlighted subword sweet spot and the character and word regimes marked at the extremes.">
<figcaption>The knob is vocabulary size. Characters give a tiny vocabulary but long sequences; whole words give short sequences but a huge vocabulary and out-of-vocabulary misses. Subword tokenizers sit in the sweet spot where each extra token stops paying for itself.</figcaption>
</figure>

## Byte-pair encoding

The dominant algorithm, **byte-pair encoding (BPE)**, was born as a compression trick [@gage1994] and adapted for translation models [@sennrich2016]. Building the vocabulary is greedy and simple: start from single characters, count adjacent pairs across the corpus, merge the most frequent pair into a new token, and repeat until you reach the target vocabulary size.

A toy corpus makes it concrete. Take `low low low lower lowest` and start from characters:

| Step | Most frequent pair | New token | The corpus now looks like |
|------|--------------------|-----------|---------------------------|
| 0 | — | — | `l o w · l o w · l o w · l o w e r · l o w e s t` |
| 1 | `l`+`o` (5×) | `lo` | `lo w · lo w · lo w · lo w e r · lo w e s t` |
| 2 | `lo`+`w` (5×) | `low` | `low · low · low · low e r · low e s t` |
| 3 | `low`+`e` (2×) | `lowe` | `low · low · low · lowe r · lowe s t` |

The merges are the tokenizer. To tokenize new text at inference time, you replay the merge list in the order it was learned; `lowest` becomes `lowe` + `s` + `t`, even though the trainer never saw a vocabulary entry for it. Frequent whole words fused early; rare words decompose into learned fragments.

!!! analogy "Analogy"
    A tokenizer is a stenographer's shorthand: the phrases the stenographer hears every day get a single stroke, and anything unusual is spelled out sign by sign. The analogy leaks at readback: a stenographer can still see the letters inside a stroke, but the model cannot — a token is an opaque integer, and whatever characters it "contains" are invisible unless the model memorized them during training.

<figure>
<img src="assets/figures/bpe-merges.svg" alt="A merge tree for the word lowest: l and o merge into lo, then lo and w into low, while e, s, and t remain single characters; the resulting tokens are low, e, s, t.">
<figcaption>Replaying the merges. A word's tokens are the roots of its merge tree: pairs that were frequent enough in training fuse into a single token (low), and whatever never earned a merge stays a lone character (e, s, t).</figcaption>
</figure>

## Byte-level BPE and friends

Production tokenizers are variations on this theme:

- **Byte-level BPE** runs BPE over the 256 possible *bytes* rather than characters, so every string — any language, any emoji, any binary garbage — is representable and there is no unknown token, ever. GPT-2 introduced this [@radford2019] and the GPT lineage kept it.
- **WordPiece** is BPE with a different merge criterion (likelihood gain rather than raw frequency) [@schuster2012]; it is the tokenizer of BERT [@devlin2019].
- **Unigram** flips the direction: start from a large candidate vocabulary and prune it down, keeping the pieces that best explain the corpus under a probabilistic model [@kudo2018].
- **SentencePiece** is the widely used library that packages these algorithms, treating text as a raw stream (spaces included) so the pipeline is language-agnostic and exactly reversible [@kudo2018sp].

The differences matter less than the shared shape: a current open model ships a byte-level BPE or SentencePiece vocabulary of roughly 32k to 256k tokens, learned once from a corpus resembling its pretraining data, then frozen forever.

!!! warning "Common trap"
    The tokenizer is part of the model. Feed a model token ids produced by a different tokenizer and you get confident garbage, not an error — the integers all "mean" something else. Swapping or extending a vocabulary after pretraining is possible but requires retraining the embeddings involved.

<figure>
<img src="assets/figures/byte-fallback.svg" alt="A frequent word maps to one token, while a rare emoji glyph decomposes into its four UTF-8 byte tokens.">
<figcaption>Why byte-level tokenizers never see an unknown word. Falling back to the 256 possible bytes means any string — any language, any emoji — is representable, so there is no unknown token. The price is that rare characters cost several tokens each.</figcaption>
</figure>

## Consequences of tokenization

A frozen, frequency-based view of text explains a family of famous quirks:

- **Character blindness.** Ask a model how many r's are in "strawberry" and it must answer from a token like `straw`+`berry` — it does not see letters. Spelling, rhyming, and counting tasks fail not because the model is stupid but because the evidence was destroyed before it arrived.
- **Numbers.** If `1234` happens to be one token and `1235` is two, nearby numbers have wildly different shapes. Modern tokenizers force digits into fixed-size groups to make arithmetic more uniform, which helped — a tokenizer design choice visibly changing a "reasoning" ability.
- **A non-English tax.** Vocabularies learned from English-heavy corpora spend their short codes on English. The same sentence can cost several times more tokens in Thai or Telugu than in English [@petrov2023] — which means higher API cost, a smaller effective context window, and often worse quality, all before the model has done anything.
- **Glitch tokens.** A token that barely appeared in training data has an essentially untrained embedding, and feeding it to the model produces erratic behavior. Unusual byte sequences are also a place where filters and models can disagree about what text "says" — part of the injection surface Chapter 18 returns to.

!!! interview "Interview"
    *Why do LLMs struggle to count letters in a word?* Because tokenization hands the model opaque multi-character chunks; the letters inside a token are not part of the input. The model can only succeed where it has memorized spelling facts about its own tokens, which is why performance is inconsistent across words.

!!! interview "Interview"
    *You double the tokenizer vocabulary. What changes?* Sequences get shorter (cheaper attention, more effective context), but the embedding and output matrices grow, the softmax over the vocabulary costs more, and the added tokens are the rarest ones — each trained on fewer examples. Somewhere in the tens-to-hundreds of thousands the tradeoff stops paying; that is why vocabularies cluster there.

One economic point to carry forward: **the token is the billing unit**. API prices, context limits, and the serving throughput of Part IV are all denominated in tokens, so a tokenizer that spends 30% more tokens on your traffic is a 30% cost increase with no quality upside. When Chapter 15 measures tokens per second, remember that how much *text* a token buys was decided here.

<figure>
<img src="assets/figures/language-tax.svg" alt="A horizontal bar chart of token counts for the same sentence across languages: English is the shortest baseline, and Spanish, German, Hindi, Thai, and Burmese each take progressively more tokens.">
<figcaption>The non-English tax, made concrete. A vocabulary learned on English spends its short codes on English, so the same sentence can cost several times more tokens in another language — a direct hit to cost, context, and often quality, all decided before the model runs.</figcaption>
</figure>

With text turned into integers, we can now build the machine that reads them.
