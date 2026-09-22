# Concepts

Background for the extension: what happens between a string of text and a category name, and
which knob to reach for when the answer is wrong.

## The pipeline

Every classification runs the same four steps.

```
text ──▶ Tokenizer ──▶ Vocabulary lookup ──▶ Log-space scoring ──▶ normalized scores
         IBayesian       TBayesianVocabulary   TBayesMath           score() / classify()
         Tokenizer       (built by training)
```

1. **Tokenize.** An `IBayesianTokenizer` turns the document into a list of feature tokens.
   Nothing downstream knows about text — only tokens.
2. **Look up.** Training built a `TBayesianVocabulary`: per-category token counts, per-category
   document counts, and a corpus-wide document-frequency map. Classification reads it.
3. **Score.** Each candidate category accumulates a log-probability. The arithmetic is in log
   space; see [Why log space](#why-log-space).
4. **Normalize.** `TBayesMath::normalize()` turns the relative log-scores back into a
   distribution that sums to 1.

`classify()` returns the highest-scoring category name. `score()` returns the whole
distribution.

> **These are not calibrated probabilities** until you calibrate. `score()` normalizes the Naive
> Bayes log-posteriors, and Naive Bayes is notoriously overconfident: its independence
> assumption multiplies correlated evidence, so a document that is 70% likely to be spam
> routinely scores 0.99. Read the raw scores as a ranking and a margin — which category wins,
> and by how much relative to the others. To get probabilities, fit a calibration on held-out
> documents; see [Calibration](#calibration).

> **Numeric category names.** `score()` returns a PHP array keyed by category name, and PHP
> coerces a purely numeric string key to an integer — a category named `"2024"` comes back
> under the integer key `2024`. `classify()` always returns a string. The same applies to
> candidate identifiers from `TBayesianRecommender::recommend()`.

## Training

`trainOne(string $category, $document)` adds one document to one category. `train(TBayesianTrainingSet $set)`
adds many. Both do the same thing per document: tokenize it, increment the category's document
count, increment each token's count in that category, and increment each distinct token's
document frequency.

Training is **incremental** — there is no separate "fit" step, and calling `trainOne()` again
later refines the same model — and **reversible**: `untrainOne()` and `untrain()` withdraw a
document exactly. Whether several processes may train one model at once depends on the storage
layout; see [Storage → Concurrency](storage.md#concurrency). All four classifier variants train
and untrain incrementally through per-token storage; see
[Storage → Incremental training and the variants](storage.md#incremental-training-and-the-variants).

### Untraining

`untrainOne(string $category, $document)` is the inverse of `trainOne()` for the same document:
the category's document count, its per-token counts and per-token document counts, and the
corpus document frequencies all go down by what the document contributed. Every count stops at
zero; a token that no document contains any more leaves the vocabulary (so `|V|` shrinks and the
smoothing changes back); a category left without documents is removed. A model that trained and
then untrained a document is therefore indistinguishable from one that never saw it — the test
suite asserts identical scores — in every storage layout, including per-token models trained
from several processes (the deltas are atomic decrements).

Two things to get right. Pass the document as it was trained: the same text through the same
tokenizer, or the same pre-tokenized list — the model keeps no record of which documents it
saw, so a different tokenization withdraws different tokens. And untraining a document that was
never trained is not an error: counts clamp at zero and unknown tokens are ignored, but the
category's document count still drops by one, so the caller is responsible for the bookkeeping.

Two consequences worth internalizing:

- **The tokenizer is part of the model.** Change the tokenizer after training and the stored
  tokens no longer correspond to what the new tokenizer produces. Retrain, or load a model that
  was saved with that tokenizer. This is why the tokenizer class and its settings are written
  into the saved payload.
- **The vocabulary is fixed by training.** A token never seen during training is
  *out-of-vocabulary* and is **skipped** at classification time, not penalized.

### Why out-of-vocabulary tokens are skipped

The tempting alternative is to give an unseen token its smoothed probability, `α / (total + α|V|)`.
That penalty is *larger for small categories*, so every novel document drifts toward whichever
category has the least training data. Skipping unseen tokens is the convention in scikit-learn
and in Manning et al., and it is what this extension does in all three event models.

## Smoothing (Alpha)

A token that never appeared in a category would give that category probability zero, and a
single such token would veto the category no matter what the rest of the document said.
Laplace (additive) smoothing adds a constant `α` to every count:

```
P(token | category) = (count(token, category) + α) / (total_tokens(category) + α · |V|)
```

`Alpha` must be positive and finite — `setAlpha()` throws `bayesian_alpha_invalid` otherwise.
Zero or negative alpha makes the smoothed probabilities zero, negative, or undefined, and every
score degenerates. The default is a conventional add-one; lower values (0.1–0.5) sharpen the
model on large, clean corpora, higher values flatten it on small or noisy ones.

## The three event models

All three inherit from `TNaiveBayesClassifier` and differ only in how they turn a document into
a likelihood. They share the same training statistics, so the choice is about which model fits
your documents — not about training differently.

| Class | Feature | Best for |
| --- | --- | --- |
| `TMultinomialNaiveBayes` | Token **counts** | Long documents where repetition carries signal — the standard text-classification choice |
| `TBernoulliNaiveBayes` | Token **presence/absence** | Short documents (titles, queries, tags) and features that are boolean by nature |
| `TComplementNaiveBayes` | Counts, scored against the **complement** | Class-imbalanced corpora; empirically the strongest of the three |

`TNaiveBayesClassifier` itself is the multinomial model — `TMultinomialNaiveBayes` is
mathematically equivalent to it and exists so the saved payload carries a distinct `kind`
marker. That marker lets several variants share one storage backend without a model being
loaded into the wrong class; a mismatch throws `bayesian_classifier_kind_mismatch`.

### Multinomial

```
P(category | document) ∝ P(category) · ∏ P(token | category) ^ weight(token, count)
```

The weight is the occurrence count, or the TF-IDF weight when `UseTfidf` is on.

### Bernoulli

```
P(category | document) ∝ P(category) · ∏ P(t | category)^present · (1 − P(t | category))^absent
```

Here `P(token | category)` is the *fraction of documents in the category containing the token*,
not the token's share of occurrences — which is why the vocabulary records per-token document
counts alongside occurrence counts. Bernoulli scores every category over the same fixed feature
set, so absence is evidence too. `TBayesMath::logComplement()` computes the `log(1 − p)` term
without losing precision when `p` is near zero.

Taken literally that means walking the whole vocabulary for every category on every
classification. The implementation does not: the absent-token mass is a constant per category,
so it is summed once and cached, and scoring a document only corrects that constant for the
tokens the document contains —

```
Σ_V [ present ? log p : log(1−p) ]  =  Σ_V log(1−p)  +  Σ_{d∩V} [ log p − log(1−p) ]
```

— which is the same number, computed in O(document) instead of O(vocabulary). The cached
constant is discarded whenever training or `Alpha` changes.

### Complement (WCNB)

For each candidate category, Complement NB scores how much the document looks like *everything
except* that category, then negates it so the usual "highest wins" convention holds:

```
θ_complement(t | c) = (α + count(t, ¬c)) / (α·|V| + total_tokens(¬c))
score(c)            = ∑ tf(t) · ( log θ_complement(t) / ‖w_c‖₁ )
```

The per-category weight vector is L1-normalized — the "weighted" refinement of Rennie et al.
(2003) — so categories holding very different amounts of text stay comparable. That is what
makes CNB hold up when one category has ten times the training data of another. The
corpus-wide counts and the weight norms depend only on training, so they are cached and
rebuilt only on the next `trainOne()` or `load()`.

## Calibration

A Naive Bayes score is a ranking, not a probability. Calibration fits a small monotone
correction on labeled documents the model was **not** trained on, so the corrected score
matches the observed frequency: of the documents scored 0.8, about 80% are really in that
category. Two methods are provided, one per kind of decision.

**Temperature scaling** (`TTemperatureScaling`) for a classifier's distribution over
categories. The log-posteriors are divided by one fitted constant, the temperature, before
normalization; above one flattens, below one sharpens, and the winner never changes. It is one
parameter, fitted by minimizing the negative log-likelihood of the true labels with a
one-dimensional search, so a few dozen held-out documents suffice and it cannot overfit the way
a per-class correction could.

```php
$classifier->calibrate($heldOut);         // fits and installs; saved with the model
$classifier->score($text);                // calibrated
$classifier->logScores($text);            // raw log-posteriors, always available
```

**Platt scaling** (`TPlattScaling`) for a binary decision value, which is what each label of a
tagger is: a fitted logistic curve `P = σ(slope · value + intercept)`, with Platt's target
smoothing so a set with few positives or few negatives still gives a sane curve.
`TBayesianTagger::calibrate()` fits one per label.

Judge a calibration with `TCalibrationMetrics` on a *third* set of documents, never the ones it
was fitted on: `distributionLogLoss()` and `distributionCalibrationError()` for a classifier,
`logLoss()`, `brierScore()` and `expectedCalibrationError()` for a tagger's per-label
probabilities. Lower is better; a calibration that does not lower them on unseen data is not
worth keeping. Refit after substantial further training — the calibration is a property of the
model as it was when fitted, and it is saved with the model.

## Multi-label tagging

A classifier answers "which one category?"; its scores sum to one, so two labels that both
apply split the mass. Tagging answers "which labels apply?" with an independent probability per
label, and `TBayesianTagger` gets that from one shared Naive Bayes model.

Every document is trained under each of its labels and, once, under a *background* category
(`BackgroundCategory`, default `*`) that every document goes into. For any label `L` the "rest
of the corpus" is then the background minus the label — `count(t, ¬L) = count(t, *) − count(t, L)`,
`docs(¬L) = docs(*) − docs(L)` — so each label gets an exact binary Naive Bayes decision:

```
log P(L | d) − log P(¬L | d) = log (n_L + 1)/(n_¬L + 1) + Σ_t w_t · [ log θ_L(t) − log θ_¬L(t) ]
```

with the same Laplace-smoothed multinomial likelihoods, `Alpha` and TF-IDF weights as the
classifier, turned into a probability by the logistic function. Nothing is stored per label
pair, a document costs one write per label plus one, and every storage backend and layout works
unchanged — including per-token models trained from many processes.

`tag()` returns the labels whose probability reaches `Threshold` (default 0.5), highest first, at
most `MaxTags` of them. A document with no labels is a valid, useful training example: it
teaches the model the words that mean nothing. Untraining mirrors training. The probabilities
are as overconfident as any Naive Bayes output until `calibrate()` fits a Platt scaling per
label on held-out examples; the calibrations are saved with the model.

## TF-IDF re-weighting

With `UseTfidf` enabled, a token's contribution is scaled by `TFIdf::weight()` instead of its
raw count:

```
tf(f)          = 1 + log(f)                       for f > 0
idf(t)         = log((N + 1) / (df(t) + 1)) + 1
weight(t, f)   = tf(f) · idf(t)
```

The `1 + log(f)` term compresses repetition, so a word repeated 100 times is not 100× the
evidence of one occurrence. The smoothed IDF is always ≥ 1, so a term appearing in every
document is damped but never silenced.

`UseTfidf` is part of the saved payload — a model reloads with the weighting it was trained
under.

## Why log space

A document of a few hundred tokens multiplies a few hundred probabilities, each well below 1.
In IEEE 754 doubles that underflows to exactly zero long before the document ends, and every
category ties at zero. `TBayesMath` keeps everything additive in log space instead:

- `logAdd(a, b)` — `log(exp(a) + exp(b))`, computed by factoring out the larger operand so
  neither `exp()` call overflows.
- `logSum(array)` — the same over a list.
- `normalize(array)` — shifts by the maximum before exponentiating, then divides by the total,
  turning relative log-scores into a distribution summing to 1.
- `logComplement(v)` — `log(1 − exp(v))` via `log1p`, for Bernoulli's absence term.

If every category scores `-INF` or `NaN` — a trained model that nonetheless cannot score the
document — the classifier raises `bayesian_classifier_score_undefined` rather than returning an
arbitrary category.

## Tokenizers

| Class | Produces |
| --- | --- |
| `TWordTokenizer` | Lowercased words; configurable `MinLength`, `StopWords`, and `Pattern`. The default. |
| `TNGramTokenizer` | Character or word n-grams. `N` sets the width; `Characters` picks the mode; `Pad` emits partial grams at the edges. |
| `TRegexTokenizer` | Whatever the `Pattern` matches — capturing group 1 when the pattern has one, the whole match otherwise. |
| `TBayesianTokenizerChain` | The concatenated output of several tokenizers. |

A chain gives every member the **original text**, not the previous member's output — so a chain
composes complementary feature sets (words *plus* character trigrams), it does not pipe one
tokenizer into the next. Duplicate tokens are kept, which is exactly what the multinomial model
expects.

Character n-grams are the standard answer to deliberate obfuscation (`v1agra`, `f r e e`) and to
languages that do not delimit words with spaces. They cost vocabulary size: expect many more
distinct features than word tokenization.

`TBayesianTokenizerFactory` handles the serialization seam. `export()` writes the tokenizer's
class and settings into the model payload; `restore()` rebuilds it on load, validating that the
class still exists and implements `IBayesianTokenizer` (`bayesian_tokenizer_class_invalid`) and
that any pattern still compiles (`bayesian_tokenizer_pattern_invalid`). It also scrubs invalid
UTF-8 from input text, so a malformed byte sequence cannot make PCRE fail mid-corpus.

## Recommendation

`TBayesianRecommender` ranks candidates rather than labeling documents, but it is the same
classifier underneath. Train on positive and negative interactions — `"liked"` versus
`"ignored"` — then:

```php
$recommender->recommend(['inception', 'interstellar'], ['tenet', 'dunkirk', 'barbie']);
```

For each candidate, the recommender scores the context and the candidate *together* as one
document and reads off `P(positive)`. The result is a map of candidate to score in descending
order.

Details worth knowing: candidates are a set (a repeated identifier is scored once), blank
identifiers are ignored, and an empty candidate list throws. If `PositiveCategory` is not one of
the classifier's trained categories, every candidate would score 0.0 and the ranking would be
meaningless — so that raises `bayesian_recommender_category_unknown` instead of returning a flat
list.

## Evaluation

`TConfusionMatrix` tallies expected-versus-predicted pairs over a labeled set; `TBayesianMetrics`
reads that tally as metrics.

```php
$matrix = new TConfusionMatrix(['spam', 'ham']);
foreach ($labeled as [$text, $expected]) {
    $matrix->record($expected, $classifier->classify($text));
}
$metrics = new TBayesianMetrics($matrix);
```

The matrix holds a **fixed label set** given at construction; recording an unregistered label
throws `bayesian_confusion_label_unknown` rather than silently growing the matrix. The metrics
object reads the matrix on demand, so counts recorded after construction are included.

Available: `getAccuracy()`, per-label `getPrecision()` / `getRecall()` / `getF1()`, and both
macro (unweighted mean over labels) and micro (pooled over instances) averages. On a balanced
two-class problem they agree; on an imbalanced one, **macro** exposes a classifier that is doing
well only on the majority class, which is usually the number you actually want to see.

Always evaluate on documents the model was not trained on. A Naive Bayes model scores its own
training set very well, and that number tells you nothing.
