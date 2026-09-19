# Class reference

Every public type in the extension, by namespace. Signatures are the real ones; for the full
per-method contract read the PHPDoc on the class itself.

Classes extending `TComponent` participate in PRADO's property system, so every `getX()`/`setX()`
pair is settable as an XML attribute (`Alpha="0.5"`) in application configuration. Classes that
do not extend `TComponent` — the training/vocabulary types and the two static math utilities —
are plain PHP objects used from code.

---

## `Belisoful\Prado\Util\Bayesian`

### `TBayesianModule` — *extends `TModule`*

The extension's bootstrap module, named in `extra.prado.bootstrap`. Owns the application's
default classifier and default storage backend, and eagerly loads a named model at startup.
See [Configuration](configuration.md).

```php
init($config)
getClassifier(?string $id = null): IBayesianClassifier    setClassifier(IBayesianClassifier $value): void
getClassifiers(): array                                   addClassifier(string $id, IBayesianClassifier $value): void
hasClassifier(string $id): bool
getStorage(): ?IBayesianStorage                           setStorage(IBayesianStorage $value): void
getDefaultClassifier(): ?string                           setDefaultClassifier(?string $value): void
getDefaultClassifierID(): ?string                         setDefaultClassifierID(?string $value): void
```

One module may own several classifiers over one storage backend — each `<classifier id="...">`
with its own `Model` (the storage key), its own variant, and its own `<tokenizer>`.
`getClassifier($id)` selects one; `getClassifier()` returns the `DefaultClassifierID` one, or
the first configured. A single unnamed `<classifier>` is the default and needs no id. See
[Configuration → Several models in one module](configuration.md#several-models-in-one-module).

`DefaultClassifier` names the model the default classifier loads from storage during `init()`.
Each configured model is loaded then if the storage already holds it; a model that does not
exist yet is simply not loaded. Any other storage failure (unreachable database, unwritable
directory) propagates as a configuration error rather than being swallowed.

### `IBayesianTagger` / `TBayesianTagger` — *`TComponent`*

Multi-label tagging as one-versus-rest Naive Bayes over one shared model. See
[Concepts → Multi-label tagging](concepts.md#multi-label-tagging).

```php
train(array $labels, $document): void          untrain(array $labels, $document): void
probabilities($document): array                // label => P, independent, in label order
tag($document): array                          // labels at or above Threshold, highest first, at most MaxTags
logOdds($document): array                      // raw per-label log-odds
calibrate(array $examples): array              // [[labels, document], ...] => TPlattScaling per label
getCalibrations(): array                       setCalibrations(?array $value): void
getIsCalibrated(): bool                        getIsTrained(): bool
getLabels(): array
getClassifier(): TNaiveBayesClassifier         setClassifier(TNaiveBayesClassifier $value): void
getBackgroundCategory(): string                setBackgroundCategory(string $value): void   // default '*'
getThreshold(): float                          setThreshold(float $value): void             // default 0.5
getMaxTags(): int                              setMaxTags(int $value): void                 // default 0 = unlimited
```

The tagger's calibrations and any other state live in the classifier's extra state
(`TBayesianTagger::EXTRA_STATE_KEY`), so saving and loading the classifier carries them.

### `TBayesianPayload`

Static, typed reads out of decoded JSON and configuration arrays: `int()`, `float()`, `string()`,
`bool()`, `map()`, `list()`, `intMap()`, `floatMap()`, `stringList()`, `formatFloat()`. A value
of the wrong shape becomes the default rather than a notice or a wrong cast.

### `TBayesianTokenHistogram`

Static helpers for the "count of counts" histograms that Bernoulli and Complement Naive Bayes
sum over instead of walking the vocabulary: `families()`, `key()` / `parseKey()`,
`contributions()` (the cells one token counts towards), `delta()` (what a write changes),
`build()`, `fromVocabulary()`, `expand()` / `flatten()`, `complementCounts()`. See
[Storage → Incremental training and the variants](storage.md#incremental-training-and-the-variants).

### `IBayesianRecommender` / `TBayesianRecommender` — *`TComponent`*

Ranks candidate items by the probability of a positive interaction. See
[Concepts → Recommendation](concepts.md#recommendation).

```php
recommend(array $context, array $candidates): array   // candidate => score, highest first
getClassifier(): IBayesianClassifier          setClassifier(IBayesianClassifier $value): void
getPositiveCategory(): string                 setPositiveCategory(string $value): void
```

`PositiveCategory` defaults to `'liked'` and must be one of the classifier's trained categories.

### `IBayesianVocabulary`

The learned statistics a classifier scores against, behind an interface so they need not all be
resident. `getVocabulary()` returns this type.

```php
getCategory(string $name): ?TBayesianCategory   getCategories(): array   getCategoryNames(): array
getIsEmpty(): bool                              getTotalDocuments(): int
getVocabularySize(): int                        hasToken(string $token): bool
getTokenDocumentFrequency(string $token): int   getTokenGlobalCount(string $token): int
getGlobalTokenTotal(): int
prefetch(array $tokens): void                   getSupportsFullScan(): bool
getDocumentFrequency(): array                   getStateSignature(): string
addDocument(string $category, array $tokens): void
removeDocument(string $category, array $tokens): void   // the exact inverse
setStats(array $categories, array $documentFrequency, int $totalDocuments): void
```

The split that makes storage-backed models possible runs through this interface: scalars and
categories are always cheap; per-token reads (`hasToken()`, `getTokenDocumentFrequency()`) are
cheap once `prefetch()` has fetched the document's tokens; and whole-vocabulary reads
(`getDocumentFrequency()`) are available only when `getSupportsFullScan()` is true.

### `TBayesianVocabulary` — *implements `IBayesianVocabulary`*

The resident implementation: the whole model in the process. `getSupportsFullScan()` is true,
`prefetch()` is a no-op, and the vocabulary is mutated in place by training. Adds
`getOrCreateCategory()` and the generation counter behind `getStateSignature()`. This is the
right choice for a model that fits in memory, and the default everywhere except a per-token load.

### `TLazyBayesianVocabulary` — *implements `IBayesianVocabulary`*

The storage-backed implementation. Holds scalars and categories resident and reads per-token
statistics from an `IBayesianTokenStorage` — `prefetch()` issues one batched query per
classification, `getSupportsFullScan()` is false, and `getDocumentFrequency()` throws rather
than return a partial map. `initialize()` loads the scalars; `applyDocument()` is the
incremental-training write path, which sends the document's deltas and then re-reads the
scalars from storage. Bound to a model by `TNaiveBayesClassifier::load()` when the storage is
in per-token mode.

```php
initialize(array $meta, array $categories): void     refresh(): void
applyDocument(string $category, array $tokens, array $meta): array
withdrawDocument(string $category, array $tokens, array $meta): array   // the incremental untrain path
getMaxBatchTokens(): int                             setMaxBatchTokens(int $value): void   // default 50000
getModelName(): string                               getPrefetchedCount(string $category, string $token, string $field): int
```

`refresh()` re-reads the model's totals from storage, for a long-lived process that wants to see
training done elsewhere. `MaxBatchTokens` bounds the cumulative prefetch batch, so a worker that
scores documents indefinitely holds a bounded number of tokens.

### `TLazyBayesianCategory` — *extends `TBayesianCategory`*

A category whose scalar totals are resident but whose per-token counts come from the vocabulary's
last `prefetch()`. `getTokenCount()` / `getTokenDocumentCount()` answer from the batch; the
whole-map accessors (`getTokenCounts()`, `getVocabularySize()`) throw, because returning the
prefetched slice would silently be a fraction of the truth.

### `TBayesianCategory`

One category's training statistics: how many documents it holds, how often each token occurred
in it, and in how many of its documents each token appeared (the last is what Bernoulli needs).

```php
__construct(string $name)
getName(): string                    getDocumentCount(): int
getTokenCounts(): array              getTokenCount(string $token): int
getTotalTokens(): int                getVocabularySize(): int
getTokenDocumentCounts(): array      getTokenDocumentCount(string $token): int
addDocument(): void                  addToken(string $token, int $count = 1): void
addTokenDocument(string $token): void
removeDocument(): void               removeToken(string $token, int $count = 1): void
removeTokenDocument(string $token): void                                // all clamped at zero
setStats(int $documentCount, array $tokenCounts, array $tokenDocumentCounts, int $totalTokens): void
```

### `TBayesianTrainingSet`

An iterable labeled corpus, for training many documents in one call.

```php
add(string $category, $document): void
getCategories(): array               getCategoryDocuments(string $category): array
getIsEmpty(): bool                   getTotalDocuments(): int
each(): \Generator                   // yields each document, keyed by its category
```

### `TBayesianModelConverter`

Rewrites a whole-payload model into a per-token backend without retraining. See
[Storage → Converting a model to per-token](storage.md#converting-a-model-to-per-token).

```php
convert(IBayesianStorage $source, IBayesianTokenStorage $destination, string $name, ?string $destName = null): void
convertAll(IBayesianStorage $source, IBayesianTokenStorage $destination): array
registerKind(string $kind, string $class): void   // teach it a custom classifier variant
```

Loads the payload into a resident vocabulary and re-saves it in the destination's layout,
choosing the classifier variant from the model's stored `kind`. Exact and one-directional — the
per-token layout cannot be enumerated back into a payload.

---

## `Belisoful\Prado\Util\Bayesian\Classifier`

### `IBayesianClassifier`

The classifier contract — the seam the module, the recommender, and the HTTP service all program
against.

```php
train(TBayesianTrainingSet $set): void        trainOne(string $category, $document): void
untrain(TBayesianTrainingSet $set): void      untrainOne(string $category, $document): void
classify($document): string                   score($document): array
save(): void                                  load(string $name): void
getName(): ?string                            setName(?string $value): void
getTokenizer(): IBayesianTokenizer            setTokenizer(IBayesianTokenizer $value): void
getStorage(): ?IBayesianStorage               setStorage(?IBayesianStorage $value): void
getVocabulary(): IBayesianVocabulary          getIsTrained(): bool
```

`$document` is a string, or a pre-tokenized `string[]` when you want to bypass the tokenizer.

### `TNaiveBayesClassifier` — *`TComponent`*

The multinomial Naive Bayes implementation and the base class of the other three. Adds to the
interface:

```php
const FORMAT_VERSION = 1             // written into every payload as formatVersion
isSpam($document): bool
logScores($document): array          // raw log-posteriors, keyed by category
calibrate(TBayesianTrainingSet $heldOut): TTemperatureScaling
getCalibration(): ?TTemperatureScaling   setCalibration(?TTemperatureScaling $value): void
getIsCalibrated(): bool
getExtraState(string $owner): ?array     setExtraState(string $owner, ?array $state): void   // saved with the model
getUseTfidf(): bool                  setUseTfidf(bool $value): void
getAlpha(): float                    setAlpha(float $value): void       // must be positive and finite
getSpamCategory(): string            setSpamCategory(string $value): void
```

`score()` returns normalized Naive Bayes scores until a calibration is installed, then
calibrated probabilities; see [Concepts → Calibration](concepts.md#calibration).

`isSpam()` is a two-category shortcut against `SpamCategory`; in a multi-class setup use
`score()` and read the distribution.

### `TMultinomialNaiveBayes`, `TBernoulliNaiveBayes`, `TComplementNaiveBayes`

The three event models — see [Concepts](concepts.md#the-three-event-models) for which to pick.
Each overrides only the likelihood, and each writes a distinct `kind` marker into its saved
payload so a model cannot be loaded into the wrong variant.

`TComplementNaiveBayes` also overrides `setAlpha()` — not to change the smoothing, but to
invalidate the caches it keeps: its corpus-wide counts and per-category weight norms depend on
alpha, so changing alpha after training must discard them. A category whose weight norm is zero
(no token distinguishes it from the rest, as with two categories trained on the same text) scores
neutrally rather than `-INF`, so a trained model always classifies.

---

## `Belisoful\Prado\Util\Bayesian\Calibration`

### `TTemperatureScaling`

One fitted temperature that turns a classifier's log-scores into calibrated probabilities. See
[Concepts → Calibration](concepts.md#calibration).

```php
__construct(float $temperature = 1.0)
getTemperature(): float                  setTemperature(float $value): void   // > 0, finite
fit(array $logScores, array $labels): float                                  // golden-section on the log-likelihood
apply(array $logScores): array                                               // calibrated distribution
getLogLoss(array $logScores, array $labels): float
getSampleCount(): int
export(): array                          static import(array $state): ?self
const MIN_TEMPERATURE = 0.01             const MAX_TEMPERATURE = 100.0
```

### `TPlattScaling`

A fitted logistic curve over a binary decision value; one per label in a tagger.

```php
__construct(float $slope = 1.0, float $intercept = 0.0)
getSlope(): float    getIntercept(): float    setParameters(float $slope, float $intercept): void   // finite
fit(array $values, array $positives): void   // Newton's method with Platt's target smoothing
apply(float $value): float                    // P(positive) in [0, 1]
getSampleCount(): int
export(): array                              static import(array $state): ?self
```

---

## `Belisoful\Prado\Util\Bayesian\Tokenizer`

### `IBayesianTokenizer`

```php
tokenize(string $text): array        exportConfig(): array        importConfig(array $config): void
```

`exportConfig()`/`importConfig()` are the persistence seam — they let a tokenizer's settings ride
along in the saved model so a reloaded classifier tokenizes identically.

### `TWordTokenizer` — *`TComponent`* — the default

```php
tokenize(string $text): array
getMinLength(): int                  setMinLength(int $value): void
getStopWords(): ?array               setStopWords(?array $value): void
getPattern(): string                 setPattern(string $value): void
```

### `TNGramTokenizer` — *`TComponent`*

Character or word n-grams; holds an inner `TWordTokenizer` for word mode.

```php
tokenize(string $text): array
getN(): int                          setN(int $value): void
getCharacters(): bool                setCharacters(bool $value): void
getPad(): bool                       setPad(bool $value): void
getWordTokenizer(): TWordTokenizer   setWordTokenizer(TWordTokenizer $value): void
```

### `TRegexTokenizer` — *`TComponent`*

```php
tokenize(string $text): array
getPattern(): string                 setPattern(string $value): void
getLowercase(): bool                 setLowercase(bool $value): void
```

Capturing group 1 supplies the tokens when the pattern has one; the whole match otherwise.

### `TBayesianTokenizerChain` — *`TComponent`*

```php
tokenize(string $text): array        // concatenated output of every member, in order
addTokenizer(IBayesianTokenizer $tokenizer): void
removeTokenizer(IBayesianTokenizer $tokenizer): bool
getTokenizers(): array               clear(): void
```

Every member sees the original text — a chain composes feature sets, it does not pipe.

### `TBayesianTokenizerFactory`

Static; the serialization and safety seam for tokenizers.

```php
static export(IBayesianTokenizer $tokenizer): array
static restore(array $state, ?IBayesianTokenizer $current = null): ?IBayesianTokenizer
static scrubText(string $text): string        // removes invalid UTF-8 so PCRE cannot fail on it
static assertPattern(string $pattern): void   // throws if the pattern does not compile
static checkPregError(string $pattern): void  // turns a PCRE runtime failure into an exception
```

### `TBayesianTokenizerTrait`

Shared plumbing for the tokenizers: property-list-driven `exportConfig()`/`importConfig()` and a
`matchAll()` that turns a PCRE backtrack/recursion failure into an exception instead of letting
it masquerade as "no tokens".

---

## `Belisoful\Prado\Util\Bayesian\Math`

### `TBayesMath`

Static log-space arithmetic. See [Concepts → Why log space](concepts.md#why-log-space).

```php
static logAdd(float $a, float $b): float      static logSum(array $values): float
static normalize(array $scores): array        static logComplement(float $value): float
```

### `TFIdf`

Static term weighting.

```php
static termFrequency(int $frequency): float                        // 1 + log(f)
static idf(int $documentFrequency, int $totalDocuments): float
static weight(int $frequency, int $documentFrequency, int $totalDocuments): float
```

`idf()` and `weight()` take document-frequency **counts**, not a map: a classifier scoring
against storage-backed statistics has the one count for the term it is weighting without holding
the whole vocabulary. Both use the smoothed form, so the weight is never zero for a present term.

---

## `Belisoful\Prado\Util\Bayesian\Evaluation`

### `TConfusionMatrix`

```php
__construct(array $labels)           // >= 1 label; duplicates collapsed, order preserved
getLabels(): array                   record(string $expected, string $predicted): void
getCounts(): array                   getCell(string $expected, string $predicted): int
getTotal(): int
```

The label set is fixed at construction; recording an unregistered label throws.

### `TBayesianMetrics`

```php
__construct(TConfusionMatrix $matrix)
getMatrix(): TConfusionMatrix        getAccuracy(): float
getPrecision(string $label): float   getRecall(string $label): float    getF1(string $label): float
getMacroPrecision(): float           getMacroRecall(): float            getMacroF1(): float
getMicroPrecision(): float           getMicroRecall(): float            getMicroF1(): float
```

Reads the matrix on demand — counts recorded after construction are included.

### `TCalibrationMetrics`

Static; measures whether probabilities match outcomes, which a confusion matrix cannot.

```php
static logLoss(array $probabilities, array $outcomes): float
static brierScore(array $probabilities, array $outcomes): float
static expectedCalibrationError(array $probabilities, array $outcomes, int $bins = 10): float
static distributionLogLoss(array $distributions, array $labels): float
static distributionCalibrationError(array $distributions, array $labels, int $bins = 10): float
```

The binary methods take one probability per example; the distribution methods take one
probability map per example (as `score()` returns) and its true category.

---

## `Belisoful\Prado\Util\Bayesian\Storage`

Covered in full on its own page: [Storage backends](storage.md).

`IBayesianStorage`, `IBayesianTokenStorage` and `IBayesianHistogramStorage` (`loadTokenHistograms()`,
`rebuildTokenHistograms()`), `TMemoryBayesianStorage`,
`TFileBayesianStorage` (`FileMode`, `DirectoryMode`), `TSqlBayesianStorage` (whole-payload or
per-token via `Mode`, connection configured through `TDbPropertiesTrait`, `TOKEN_LAYOUT_VERSION`),
`TRedisBayesianStorage` (whole-payload or per-token via `Mode`, `TOKEN_LAYOUT_VERSION`). In
per-token mode both apply training as atomic increments, so several processes may train one model
at once; see [Storage → Concurrency](storage.md#concurrency).

---

## `Belisoful\Prado\Web\Services`

### `TBayesianService` — *extends `TService`*

A read-only JSON HTTP surface over the configured classifier: `classify` and `recommend`. It
exposes no training, saving, or deletion, and enforces no access control unless configured to;
see [Configuration → HTTP service](configuration.md#http-service) before exposing it.

```php
const PERM_CLASSIFY = 'bayesian_classify'     const PERM_RECOMMEND = 'bayesian_recommend'   const PERM_TAG = 'bayesian_tag'
init($config)                                 // reads <authorization> rules
run()                                         runService(array $params)
getPermissions($manager): array               // IPermissions: the three permissions above
getAuthorizationRules(): TAuthorizationRuleCollection
getClassifier(): IBayesianClassifier          setClassifier(IBayesianClassifier $value): void
getRecommender(): IBayesianRecommender        setRecommender(IBayesianRecommender $value): void
getTagger(): IBayesianTagger                  setTagger(IBayesianTagger $value): void
getMaxTextLength(): int                       setMaxTextLength(int $value): void
getMaxCandidates(): int                       setMaxCandidates(int $value): void   // default 100
getTagThreshold(): float                      setTagThreshold(float $value): void  // default 0.5
getMaxTags(): int                             setMaxTags(int $value): void         // default 0 = unlimited
getModuleID(): ?string                        setModuleID(?string $value): void
```
