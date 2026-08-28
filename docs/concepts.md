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
   probability distribution that sums to 1.

`classify()` returns the highest-scoring category name. `score()` returns the whole
distribution, which is what you want whenever "how confident?" matters.

> **Numeric category names.** `score()` returns a PHP array keyed by category name, and PHP
> coerces a purely numeric string key to an integer — a category named `"2024"` comes back
> under the integer key `2024`. `classify()` always returns a string. The same applies to
> candidate identifiers from `TBayesianRecommender::recommend()`.

## Training

`trainOne(string $category, $document)` adds one document to one category. `train(TBayesianTrainingSet $set)`
adds many. Both do the same thing per document: tokenize it, increment the category's document
count, increment each token's count in that category, and increment each distinct token's
document frequency.

Training is **incremental and additive** — there is no separate "fit" step, and calling
`trainOne()` again later refines the same model. It is also **destructive to nothing**: the
vocabulary only grows.

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
