# PRADO Bayesian Extension

Bayesian classification and recommendation for the [PRADO PHP Framework](https://github.com/pradosoft/prado) (version 4.4+), implemented as a PRADO 4 extension.

> **Pre-release (0.1.0).** This extension targets the PRADO `master` branch (the upcoming 4.4 release), which adds the `extra.prado.bootstrap` / `error-messages` / `class-map` Composer plugin hooks it relies on. It does not work with PRADO 4.3.x, and until PRADO 4.4 is published on Packagist you must point Composer at the framework repository yourself (see [Installation](#installation)). Public APIs may still change before 1.0.0.

The module is designed for two common use cases out of the box:

- **Spam filtering** — train a Naive Bayes classifier on labeled text and classify new documents with calibrated probability scores.
- **Recommendation** — score items for a user from observed item/category interactions, ranking by the posterior probability the user is in the "likes this" class.

The classifier, tokenizer, and storage are decoupled, so swapping in a different model family, token strategy, or persistence layer is a one-line configuration change.

## Documentation

This README is the quick start. Deeper material lives in [`docs/`](docs/README.md):

| Page | What it covers |
|---|---|
| [Concepts](docs/concepts.md) | The pipeline, the three Naive Bayes event models, smoothing, TF-IDF, log-space arithmetic, tokenization, and evaluation |
| [Class reference](docs/classes.md) | Every public class and interface by namespace, with its role and public API |
| [Storage backends](docs/storage.md) | The `IBayesianStorage` contract, the four backends, and how to choose |
| [Configuration](docs/configuration.md) | Module and service wiring, and the full error-code list |

## Requirements

| Requirement | Scope | Purpose |
|---|---|---|
| PHP 8.1 or higher | required | Language runtime |
| `ext-mbstring` | required | Multibyte-safe tokenization (every tokenizer uses `mb_*`) |
| PRADO Framework `^4.4@dev` | required (Composer installs it) | `TComponent`, `TService`, `TModule`, `TDbPropertiesTrait`, and the `extra.prado.*` Composer plugin hooks |
| `ext-pdo` | suggested | Required by `TSqlBayesianStorage` (via Prado's `TDbConnection`) for SQL-backed persistence |
| `ext-redis` | suggested | Required by `TRedisBayesianStorage` for Redis-backed persistence |

SQL and Redis are **opt-in**. Add the extension you need with:

```sh
composer require ext-pdo    # for TSqlBayesianStorage
composer require ext-redis  # for TRedisBayesianStorage
```

`TMemoryBayesianStorage` (default) and `TFileBayesianStorage` need no extension. Configuring `TSqlBayesianStorage` or `TRedisBayesianStorage` without the matching PHP extension is a configuration error (`bayesian_storage_pdo_missing` / `bayesian_storage_redis_missing`); there is no silent fallback.

## Installation

```sh
composer require belisoful/prado-bayesian
```

The framework is a real requirement of this package — `pradosoft/prado` at `^4.4@dev` — so
Composer installs it for you. Until PRADO 4.4 is on Packagist, your application still has to
point Composer at the framework's repository, because **Composer reads `repositories` only from
the root project, never from a dependency**:

```json
{
    "repositories": [
        { "type": "composer", "url": "https://asset-packagist.org" },
        { "type": "vcs", "url": "https://github.com/pradosoft/prado" }
    ],
    "require": {
        "belisoful/prado-bayesian": "^0.1"
    },
    "minimum-stability": "dev",
    "prefer-stable": true
}
```

`minimum-stability: dev` with `prefer-stable: true` is what lets `^4.4@dev` resolve while every
other dependency still comes from its stable release. The constraint matches PRADO's
`dev-master` through the branch alias (`dev-master` → `4.4.x-dev`) declared in the framework's
own `composer.json`.

The asset-packagist repository is the framework's requirement, not this package's — PRADO
depends on `bower-asset/*` packages, and by the same root-only rule, installing without it fails
with `bower-asset/jquery ... could not be found`.

Name `pradosoft/prado` in your own `require` as well if you want to pin the framework version
your application runs; nothing here prevents it.

The package's `config/` folder holds what PRADO's third-party plugin support reads system-wide from `composer.json` `extra.prado`: `errorMessages.txt` (the `bayesian_*` exception codes, registered through `error-messages`) and `prado-bayesian-classes.json` (Prado3-style short class names → PHP FQNs, registered through `class-map`, so `TNaiveBayesClassifier` resolves in Prado3-style configuration). Both load for every installed extension whether or not the bootstrap module is used.

## What it provides

| Class | Namespace | Role |
|---|---|---|
| `TBayesianModule` | `Belisoful\Prado\Util\Bayesian` | The `extra.prado.bootstrap` module; owns the configured default classifier |
| `TBayesianService` | `Belisoful\Prado\Web\Services` | A `TService` exposing classification and recommendation over the PRADO service pipeline (HTTP request) |
| `IBayesianClassifier` | `Belisoful\Prado\Util\Bayesian\Classifier` | The classifier contract: `train()`, `trainOne()`, `classify()`, `score()`, `save()`, `load()` |
| `TNaiveBayesClassifier` | `Belisoful\Prado\Util\Bayesian\Classifier` | The classic Naive Bayes (multinomial event model with Laplace smoothing) — the default spam filter |
| `TMultinomialNaiveBayes` | `Belisoful\Prado\Util\Bayesian\Classifier` | Multinomial Naive Bayes; counts token occurrences per category |
| `TBernoulliNaiveBayes` | `Belisoful\Prado\Util\Bayesian\Classifier` | Bernoulli Naive Bayes; tracks token presence/absence per document |
| `TComplementNaiveBayes` | `Belisoful\Prado\Util\Bayesian\Classifier` | Complement Naive Bayes; well-suited to imbalanced text classification |
| `IBayesianTokenizer` | `Belisoful\Prado\Util\Bayesian\Tokenizer` | The tokenizer seam: input text → list of feature tokens |
| `TWordTokenizer` | `Belisoful\Prado\Util\Bayesian\Tokenizer` | Default word tokenizer; lowercases, strips punctuation, drops short tokens, supports stop words |
| `TNGramTokenizer` | `Belisoful\Prado\Util\Bayesian\Tokenizer` | Character or word n-grams (`n` configurable) |
| `TRegexTokenizer` | `Belisoful\Prado\Util\Bayesian\Tokenizer` | A regex-driven tokenizer for custom patterns |
| `TBayesianTokenizerChain` | `Belisoful\Prado\Util\Bayesian\Tokenizer` | Composes multiple tokenizers (each contributes its tokens) |
| `TBayesianTokenizerTrait` | `Belisoful\Prado\Util\Bayesian\Tokenizer` | Shared tokenizer plumbing: property-driven `exportConfig()`/`importConfig()`, safe `matchAll()`, `normalizeText()` |
| `TBayesianTokenizerFactory` | `Belisoful\Prado\Util\Bayesian\Tokenizer` | Serializes/restores tokenizers into the saved model, UTF-8 scrubbing, regex validation |
| `IBayesianVocabulary` | `Belisoful\Prado\Util\Bayesian` | The vocabulary seam: resident, or read per token from storage. `getVocabulary()` returns this |
| `TBayesianVocabulary` | `Belisoful\Prado\Util\Bayesian` | The resident vocabulary; per-category token counts, totals, smoothing |
| `TLazyBayesianVocabulary` | `Belisoful\Prado\Util\Bayesian` | The storage-backed vocabulary; reads a document's tokens per classification via `IBayesianTokenStorage` |
| `TLazyBayesianCategory` | `Belisoful\Prado\Util\Bayesian` | A category whose per-token counts come from the vocabulary's last prefetch |
| `TBayesianCategory` | `Belisoful\Prado\Util\Bayesian` | One category: its name, document count, and token counts |
| `TBayesianTrainingSet` | `Belisoful\Prado\Util\Bayesian` | An iterable labeled training set: maps categories to tokenized documents |
| `TBayesianModelConverter` | `Belisoful\Prado\Util\Bayesian` | Rewrites a whole-payload model into a per-token backend without retraining |
| `TFIdf` | `Belisoful\Prado\Util\Bayesian\Math` | Term-frequency × inverse-document-frequency weighting |
| `TBayesMath` | `Belisoful\Prado\Util\Bayesian\Math` | Log-space arithmetic helpers used by the classifiers to avoid underflow |
| `TConfusionMatrix` | `Belisoful\Prado\Util\Bayesian\Evaluation` | Confusion matrix for evaluating a classifier against a labeled set |
| `TBayesianMetrics` | `Belisoful\Prado\Util\Bayesian\Evaluation` | Precision, recall, F1, accuracy, macro/micro averages |
| `IBayesianStorage` | `Belisoful\Prado\Util\Bayesian\Storage` | The persistence seam for a trained model |
| `IBayesianTokenStorage` | `Belisoful\Prado\Util\Bayesian\Storage` | A storage backend that also serves a model per token, for models larger than a process |
| `TMemoryBayesianStorage` | `Belisoful\Prado\Util\Bayesian\Storage` | Process-local in-memory storage (default; no I/O) |
| `TFileBayesianStorage` | `Belisoful\Prado\Util\Bayesian\Storage` | JSON file storage (good for development, small models, single host) |
| `TSqlBayesianStorage` | `Belisoful\Prado\Util\Bayesian\Storage` | SQL-backed storage via `TDbConnection` (SQLite, MySQL, PostgreSQL); whole-payload or per-token (`Mode`); connection through `TDbPropertiesTrait` |
| `TRedisBayesianStorage` | `Belisoful\Prado\Util\Bayesian\Storage` | Redis-backed storage for shared hosts; whole-payload or per-token (`Mode`), with atomic `HINCRBY` incremental training (requires `ext-redis`) |
| `IBayesianRecommender` | `Belisoful\Prado\Util\Bayesian` | The recommender contract: `recommend()` for a user/item context |
| `TBayesianRecommender` | `Belisoful\Prado\Util\Bayesian` | A probabilistic recommender built on top of any `IBayesianClassifier` |

## Architecture

```
  entry points          TBayesianModule ────────────► TBayesianService
                     (extra.prado.bootstrap;         (TService; HTTP
                      owns default classifier)        classify/recommend)
                                 │  both resolve
                                 ▼
  seam                   IBayesianClassifier ◄──────► IBayesianStorage
                                 │                    memory / file / SQL / Redis
                                 │ implemented by
                                 ▼
  classifiers          TNaiveBayesClassifier          TBayesianRecommender
                                 ▲  extends           (ranks candidates by
                 ┌───────────────┼───────────────┐     P(positive), reusing
     TMultinomialNaiveBayes  TBernoulli-  TComplement-  any classifier)
                             NaiveBayes    NaiveBayes
                                 │ reads / writes
                                 ▼
  training state    TBayesianVocabulary ─── TBayesianCategory ─── TBayesianTrainingSet
                                 │ scores with
                                 ▼
  math                       TBayesMath  ───  TFIdf
                                 │ features from
                                 ▼
  tokenizers             IBayesianTokenizer
                    TWordTokenizer / TNGramTokenizer / TRegexTokenizer / TBayesianTokenizerChain
                                 │
                          (text in, tokens out)
```

The layers stack cleanly:

- **Math** — `TBayesMath` works in log-space, so the Naive Bayes product of thousands of small probabilities never underflows. `TFIdf` weights token contributions by how discriminating they are across the corpus.
- **Tokenizer** — `IBayesianTokenizer` is the seam between text and features. Default `TWordTokenizer` is good enough for spam filtering; swap in `TNGramTokenizer` for language-agnostic content or `TRegexTokenizer` for structured input.
- **Vocabulary & categories** — `IBayesianVocabulary` is the statistics the classifier scores against, behind an interface so they need not all be resident: `TBayesianVocabulary` holds the whole model, `TLazyBayesianVocabulary` reads a document's tokens from storage per classification. `TBayesianCategory` represents one class. `TBayesianTrainingSet` is the labeled corpus in training-time form.
- **Classifiers** — All implement `IBayesianClassifier` and accept any tokenizer + storage. `TNaiveBayesClassifier` is the canonical spam filter and the base class of the other three; `TMultinomialNaiveBayes`, `TBernoulliNaiveBayes`, and `TComplementNaiveBayes` override only the likelihood, so switching event model is a one-line change. Each writes a distinct `kind` marker into its saved model, so several variants can share one storage backend safely.
- **Storage** — `IBayesianStorage` persists a trained model. `TMemoryBayesianStorage` is the no-I/O default; `TFileBayesianStorage` writes JSON; `TSqlBayesianStorage` uses Prado's `TDbConnection`/`TDbCommand` for SQL-backed persistence (SQLite, MySQL, PostgreSQL), configured through `TDbPropertiesTrait` like any other Prado database component, and can store a model per token (`Mode="token"`) so it is bounded by the database rather than by PHP memory; `TRedisBayesianStorage` scales across processes and hosts via Redis, and like the SQL backend can store a model per token (`Mode="token"`), though there the model lives in Redis's RAM rather than on disk.
- **Recommender** — `TBayesianRecommender` reuses the classifier: train it on user/item interactions with a positive and a negative label (`PositiveCategory` defaults to `liked`), then ask it to rank candidate items.
- **Module & service** — `TBayesianModule` is the `extra.prado.bootstrap` entry point that owns the configured classifiers and storage; one module can hold several models over one backend. `TBayesianService` exposes a classifier and the recommender over the PRADO service pipeline (HTTP), sourcing its classifier from the module.

## Usage

### Spam filter (the default)

```php
use Belisoful\Prado\Util\Bayesian\Classifier\TNaiveBayesClassifier;

$classifier = new TNaiveBayesClassifier();
$classifier->setName('comment-spam');
foreach ([
    'Buy cheap watches now!!!',
    'Limited time offer, click here',
    'Congratulations, you have won a prize',
] as $document) {
    $classifier->trainOne('spam', $document);
}
foreach ([
    'Hey, are we still meeting for lunch tomorrow?',
    'I attached the report you asked for.',
    'Thanks for the help with the bug fix.',
] as $document) {
    $classifier->trainOne('ham', $document);
}

$label = $classifier->classify('FREE VIAGRA!!! Lowest prices online');  // 'spam'
$spam  = $classifier->isSpam('FREE VIAGRA!!! Lowest prices online');    // true
$score = $classifier->score('Buy cheap watches now!!!');                // ['spam' => 0.99..., 'ham' => 0.00...]
```

### Persist a trained model

```php
use Belisoful\Prado\Util\Bayesian\Storage\TFileBayesianStorage;

$storage = new TFileBayesianStorage();
$storage->setDirectory('/var/lib/myapp/bayesian');
$classifier->setStorage($storage);
$classifier->save();                // serialize the trained model
$classifier->load('comment-spam');  // restore on a future request
```

The saved state carries the tokenizer class and its settings, so a model trained with a `TNGramTokenizer` (or a `TBayesianTokenizerChain`) tokenizes identically after `load()` into a fresh classifier. Each classifier variant writes a `kind` marker and refuses to load a payload saved by a different variant (`bayesian_classifier_kind_mismatch`); load a model with the class that saved it. `TSqlBayesianStorage` creates its table on first use with driver-aware DDL (`VARCHAR(191)`/`LONGTEXT` on MySQL); set `AutoCreateTable="false"` to manage the schema yourself.

### As a PRADO module

PRADO reads either XML or PHP application configuration; both forms are shown throughout.

**`protected/application.xml`**

```xml
<modules>
    <module id="bayesian" class="Belisoful\Prado\Util\Bayesian\TBayesianModule" DefaultClassifier="comment-spam">
        <!-- optional: pick the classifier class and set its properties (default: TNaiveBayesClassifier) -->
        <classifier class="Belisoful\Prado\Util\Bayesian\Classifier\TComplementNaiveBayes" Alpha="0.5" />
        <storage class="Belisoful\Prado\Util\Bayesian\Storage\TFileBayesianStorage" Directory="/var/lib/myapp/bayesian" />
    </module>
</modules>
<services>
    <service id="bayesian" class="TBayesianService" ModuleID="bayesian" MaxTextLength="65536" />
</services>
```

**`protected/application.php`**

```php
<?php
return [
    'modules' => [
        'bayesian' => [
            'class' => 'Belisoful\Prado\Util\Bayesian\TBayesianModule',
            // Module properties go under 'properties'; the <classifier>/<storage> child
            // elements become sibling keys of 'class'.
            'properties' => ['DefaultClassifier' => 'comment-spam'],
            'classifier' => ['class' => 'TComplementNaiveBayes', 'Alpha' => 0.5],
            'storage'    => ['class' => 'TFileBayesianStorage', 'Directory' => '/var/lib/myapp/bayesian'],
        ],
    ],
    'services' => [
        'bayesian' => [
            'class' => 'TBayesianService',
            'properties' => ['ModuleID' => 'bayesian', 'MaxTextLength' => 65536],
        ],
    ],
];
```

The two are equivalent. Note the shape difference: an XML attribute on `<module>` is a *module
property* and lives under `'properties'`, while `<classifier>` and `<storage>` are *child
elements* and become their own keys alongside `'class'`. Within those two, the class and its
properties sit side by side, exactly as the XML attributes do.

Registering by package name works in PHP too — but then omit `'class'`, because the class comes
from `extra.prado.bootstrap` and supplying both is a configuration error:

```php
'modules' => [
    'belisoful/prado-bayesian' => [
        'properties' => ['DefaultClassifier' => 'comment-spam'],
        'storage'    => ['class' => 'TFileBayesianStorage', 'Directory' => '/var/lib/myapp/bayesian'],
    ],
],
```

Both forms of module registration work: `<module id="belisoful/prado-bayesian">` (the package name; the class comes from `extra.prado.bootstrap`) or `<module id="bayesian" class="Belisoful\Prado\Util\Bayesian\TBayesianModule">`. Register the service by its class-map short name `TBayesianService`: PRADO `master` resolves `<service class="…">` through `Prado::usingClass()`, which currently does not fall back to the Composer autoloader for a not-yet-loaded fully-qualified name outside the framework directory.

`DefaultClassifier` names the model: the classifier takes that name, and if the storage already holds a saved model of that name it is loaded when the module initializes. A model that has not been trained/saved yet is simply empty until you train and `save()` it; a storage that cannot be reached is a configuration error at startup, not a silent empty model.

Application code reaches the configured default classifier through the module:

```php
$module = Prado::getApplication()->getModule('bayesian');
$label  = $module->getClassifier()->classify($text);
```

`TBayesianService` is read-only over HTTP (no training or deletion) and answers with JSON. With the service id `bayesian`:

| Request | Response |
|---|---|
| `?bayesian&text=Free+pills` (or `&action=classify`) | `{"category":"spam","scores":{"spam":0.98,"ham":0.02}}` |
| `?bayesian&text=Free+pills&category=spam` | adds `"isSpam": true` |
| `?bayesian&action=recommend&context[]=red+shoes&candidates[]=red+hat&candidates[]=blue+hat` | `{"scores":{"red hat":0.7,"blue hat":0.4}}` (always a JSON object, highest first) |

Errors are JSON with an HTTP status: `400` `{"error":"bayesian_service_text_required",...}` for a missing/malformed parameter or unknown action, `413` when `text` exceeds `MaxTextLength` (default 65536 bytes, `0` = unlimited), and `503` `bayesian_classifier_not_trained` when no model has been trained yet. `ModuleID` selects which `TBayesianModule` supplies the classifier (default: the first one registered).

### Multiple models

A model is identified by its **name**, and storage is keyed by that name — so one storage
backend holds as many models as you like. There is no registry to configure; naming and saving
is all it takes.

```php
$storage = new TFileBayesianStorage();
$storage->setDirectory('/var/lib/myapp/bayesian');

$spam = new TNaiveBayesClassifier();
$spam->setStorage($storage);
$spam->setName('comment-spam');
$spam->trainOne('spam', 'cheap pills buy now');
$spam->trainOne('ham', 'project meeting tomorrow');
$spam->save();

$language = new TBernoulliNaiveBayes();       // a different variant, same storage
$language->setStorage($storage);
$language->setName('language-id');
$language->trainOne('en', 'the quick brown fox');
$language->trainOne('fr', 'le renard brun rapide');
$language->save();

$storage->list();   // ['comment-spam', 'language-id'] — sorted ascending
```

Two things make this safe to do in one store:

- **`load()` replaces the classifier's state entirely.** You can reuse a single instance across
  models — `$c->load('comment-spam')` then `$c->load('language-id')` — and no categories,
  counts, or tokenizer settings survive from the previous model.
- **Variants cannot be crossed.** Each classifier class writes a `kind` marker, so loading a
  Bernoulli model into a `TComplementNaiveBayes` throws `bayesian_classifier_kind_mismatch`
  rather than silently scoring with the wrong math. (`TNaiveBayesClassifier` and
  `TMultinomialNaiveBayes` are interchangeable — they are the same model.)

Whether to reuse one instance or hold several depends on access pattern. Holding several keeps
every model resident in memory at once; reusing one re-reads and re-decodes the payload on each
`load()`. See [Model size and storage limits](#model-size-and-storage-limits) for what that costs.

The module's `DefaultClassifier` covers only the single model the application boots with. For
the rest, construct classifiers yourself and hand them the module's storage:

```php
$storage = $this->getApplication()->getModule('bayesian')->getStorage();
```

### Model size and storage limits

**In the default `payload` mode, a backend loads the whole model.** The model is one JSON unit —
one file, one row, one Redis key — and `load()` decodes all of it into PHP arrays before the
first classification. The figures below are for that mode, which is the right choice whenever the
model fits comfortably in a request.

`TSqlBayesianStorage` and `TRedisBayesianStorage` also offer `Mode="token"`, where the model is
stored per token and a classification reads only the document's own tokens — so a loaded model
costs kilobytes of PHP memory regardless of its size (a 100,000-token SQL model loads in 0.7 ms
and 0.2 MB against 106 ms and 44 MB for the payload form). See
[Storage backends](docs/storage.md#model-size-and-memory) for when that trade is worth it. The
sizing below is what the whole model costs when it is resident.

Measured on this codebase, with every category having seen the whole vocabulary:

| Vocabulary | Categories | JSON payload | Loaded in PHP |
|---:|---:|---:|---:|
| 5,000 | 2 | 361 KB | 1.6 MB |
| 5,000 | 10 | 1.5 MB | 6.6 MB |
| 20,000 | 2 | 1.5 MB | 6.3 MB |
| 20,000 | 10 | 6.2 MB | 26.3 MB |
| 100,000 | 2 | 7.6 MB | 25.0 MB |

Two rules of thumb follow. The payload runs **30–40 bytes per token-per-category**, because each
category stores its own occurrence and document counts for every token it has seen, plus one
corpus-wide document-frequency map. And the decoded PHP structure is **roughly 3–4× the JSON**,
since PHP's hash tables cost far more per entry than the text does. Budget the sum of both:
`json_decode()` holds the string and the growing array at the same time.

Size therefore scales with vocabulary **times** categories, not vocabulary alone — ten categories
over the same words is five times the model of two. Trimming the vocabulary is the effective
lever: raise `MinLength`, supply `StopWords`, or prefer word tokens over character n-grams, which
generate far more distinct features.

| Backend | Ceiling on one model | Practical limit |
|---|---|---|
| `TMemoryBayesianStorage` | PHP `memory_limit` | Also gone at end of process |
| `TFileBayesianStorage` | Filesystem | One `.json` file per model, read whole via `file_get_contents()` |
| `TSqlBayesianStorage` | `LONGTEXT` 4 GB (MySQL), `TEXT` ~1 GB (PostgreSQL/SQLite) | MySQL `max_allowed_packet` (64 MB default) usually binds first |
| `TRedisBayesianStorage` | 512 MB per value (payload); the Redis instance's RAM (token) | Redis holds the whole model in RAM either way |

In every case PHP's `memory_limit` binds long before the backend's own ceiling: a 64 MB payload
needs roughly 200–250 MB of PHP memory to decode and hold. If you need models larger than a
process can hold, the fix is a smaller feature space — not a different backend.

### Recommendation

Train a `TBayesianRecommender`'s underlying classifier on user behavior: the "positive" category is the items the user engaged with; the "negative" category is items they ignored. Then ask it to rank candidates for a new user context.

```php
use Belisoful\Prado\Util\Bayesian\TBayesianRecommender;

$rec = new TBayesianRecommender();
$rec->setPositiveCategory('liked');
$classifier = $rec->getClassifier();
foreach (['red shoes', 'blue sneakers', 'leather boots'] as $item) {
    $classifier->trainOne('liked', $item);
}
foreach (['red hat', 'blue scarf', 'leather belt'] as $item) {
    $classifier->trainOne('ignored', $item);
}

$top = $rec->recommend(['red shoes', 'leather belt'], ['red sneakers', 'blue sneakers', 'red hat', 'leather wallet']);
// ['red sneakers' => 0.83, 'blue sneakers' => 0.83, 'leather wallet' => 0.5, 'red hat' => 0.17]
// (values from this exact toy corpus; ties keep candidate order)
```

Note that unseen words are ignored at classification time (they carry no learned evidence), so a candidate made only of tokens the classifier has never seen scores at the class prior.

```php
// A purely numeric identifier such as '123' comes back as the integer key 123 (PHP array
// semantics); json_encode it with JSON_FORCE_OBJECT if the result must stay a map.
```

### Evaluate a model

```php
use Belisoful\Prado\Util\Bayesian\Evaluation\TConfusionMatrix;
use Belisoful\Prado\Util\Bayesian\Evaluation\TBayesianMetrics;

$matrix = new TConfusionMatrix(['spam', 'ham']);
foreach ($labeledTestSet as $document => $expectedLabel) {
    $predicted = $classifier->classify($document);
    $matrix->record($expectedLabel, $predicted);
}

$metrics  = new TBayesianMetrics($matrix);
$accuracy = $metrics->getAccuracy();
$spamF1   = $metrics->getF1('spam');
echo "accuracy={$accuracy}, spam-F1={$spamF1}\n";
```

## Full lifecycle example

Every part of the extension in one script: a composed tokenizer, a persistent backend, a
Complement Naive Bayes classifier, a labeled training set, save/reload, classification,
evaluation on held-out documents, and incremental retraining.

```php
use Belisoful\Prado\Util\Bayesian\Classifier\TComplementNaiveBayes;
use Belisoful\Prado\Util\Bayesian\Evaluation\TBayesianMetrics;
use Belisoful\Prado\Util\Bayesian\Evaluation\TConfusionMatrix;
use Belisoful\Prado\Util\Bayesian\Storage\TFileBayesianStorage;
use Belisoful\Prado\Util\Bayesian\TBayesianTrainingSet;
use Belisoful\Prado\Util\Bayesian\Tokenizer\TBayesianTokenizerChain;
use Belisoful\Prado\Util\Bayesian\Tokenizer\TNGramTokenizer;
use Belisoful\Prado\Util\Bayesian\Tokenizer\TWordTokenizer;

// 1. Tokenizer — words for meaning, character trigrams to survive obfuscation ("ch34p").
//    Chain members each see the original text; their outputs are concatenated.
$words = new TWordTokenizer();
$words->setMinLength(3);
$words->setStopWords(['the', 'and', 'for', 'you', 'your']);

$trigrams = new TNGramTokenizer();
$trigrams->setN(3);
$trigrams->setCharacters(true);

$tokenizer = new TBayesianTokenizerChain();
$tokenizer->addTokenizer($words);
$tokenizer->addTokenizer($trigrams);

// 2. Storage — one JSON file per model, written atomically.
$storage = new TFileBayesianStorage();
$storage->setDirectory('/var/lib/myapp/bayesian');

// 3. Classifier — Complement NB handles the class imbalance real spam corpora have.
$classifier = new TComplementNaiveBayes();
$classifier->setName('comment-spam');       // the storage key
$classifier->setTokenizer($tokenizer);
$classifier->setStorage($storage);
$classifier->setAlpha(0.5);                 // must be > 0
$classifier->setUseTfidf(true);

// 4. Train from a labeled set.
$training = new TBayesianTrainingSet();
foreach ([
    'Buy cheap watches now, limited time offer',
    'Congratulations! You have won a free prize',
    'Lowest prices online, click here to order',
    'Cheap pills delivered discreetly, order now',
    'Make money fast working from home',
] as $doc) {
    $training->add('spam', $doc);
}
foreach ([
    'Are we still meeting for lunch tomorrow?',
    'I attached the quarterly report you asked for',
    'Thanks for the help with that bug fix',
    'The deployment finished, staging looks healthy',
    'Can you review my pull request this week?',
] as $doc) {
    $training->add('ham', $doc);
}
$classifier->train($training);
// getIsTrained() === true, categories ["spam","ham"], 10 documents

// 5. Persist. The payload carries the tokenizer class and its settings too.
$classifier->save();
$storage->list();   // ['comment-spam']  — the file is ~30 KB for this toy corpus

// 6. Reload in a later request, into a fresh instance that was never configured.
$loaded = new TComplementNaiveBayes();
$loaded->setStorage($storage);
$loaded->load('comment-spam');
// The chain and both its members come back, and Alpha is 0.5 again — nothing to re-set.

// 7. Classify.
$probe  = 'Cheap watches, lowest prices, order now';
$label  = $loaded->classify($probe);        // 'spam'
$scores = $loaded->score($probe);           // ['spam' => 0.512, 'ham' => 0.488]

// 8. Evaluate on documents the model was NOT trained on.
$heldOut = [
    ['Free prize waiting, claim it today', 'spam'],
    ['Order cheap pills online now',       'spam'],
    ['Lunch tomorrow at the usual place?', 'ham'],
    ['Please review the attached report',  'ham'],
];
$matrix = new TConfusionMatrix(['spam', 'ham']);
foreach ($heldOut as [$text, $expected]) {
    $matrix->record($expected, $loaded->classify($text));
}
$metrics = new TBayesianMetrics($matrix);
$metrics->getAccuracy();    // 1.00
$metrics->getF1('spam');    // 1.00
$metrics->getMacroF1();     // 1.00  — prefer macro when classes are imbalanced

// 9. Training is incremental: correct a mistake and re-save, no full retrain.
$loaded->trainOne('spam', 'Exclusive offer just for you, act now');
$loaded->save();
```

The commented values are this script's actual output. Two things they illustrate honestly: a
ten-document corpus produces a **narrow margin** (0.512 vs 0.488) even when the label is right,
because Complement NB normalizes each category's weight vector and there is very little evidence
to separate them — real corpora run to thousands of documents and separate far more sharply. And
perfect scores on four held-out documents mean nothing statistically; they show the evaluation
wiring works, not that the model is good.

## Development

PRADO 4.4 is not on Packagist yet, so `composer.json` declares the framework's GitHub
repository and requires `pradosoft/prado` at `^4.4@dev`, which resolves through the branch alias
in PRADO's own `composer.json` (`dev-master` → `4.4.x-dev`) — `composer install` needs no
further setup. To develop against a local PRADO checkout instead, add a path repository to your
working copy and leave it uncommitted:

```sh
composer config repositories.prado --json \
  '{"type":"path","url":"../prado.master","options":{"versions":{"pradosoft/prado":"4.4.x-dev"}}}'
composer update pradosoft/prado
```

```sh
composer install
vendor/bin/phpunit --testsuite unit                  # tests
vendor/bin/php-cs-fixer fix --dry-run src/           # code style
vendor/bin/phpstan analyse src/ --memory-limit=512M  # static analysis
composer coverage                                    # tests with a coverage report
composer integration                                 # Composer-extension install check
```

`composer integration` builds a throwaway consumer project that requires this package through
Composer and asserts what `extra.prado` promises: the error-message file resolves `bayesian_*`
codes, the class map resolves the Prado3 short names, the bootstrap module boots under its
package id, and a real request reaches `TBayesianService`. It expects the framework at
`../prado.master`; pass another path as the first argument.

The consumer project is removed when the run passes. A failing run keeps it — and prints where —
so the half-built install can be inspected; `KEEP_WORK_DIR=1` keeps it after a passing run too. A
work directory passed as the second argument belongs to the caller and is never removed.

The committed `composer.json` carries no machine-specific paths: it points at the framework's public repository, which is what CI installs from too. A full check before committing is, in order: `php -l`, php-cs-fixer, phpstan, phpunit.

Tests cover the math (log-space arithmetic, TF-IDF), tokenizers (word, n-gram, regex, chain, factory round-trips), the classifiers (Naive Bayes and the three variants over tokenized corpora, save/load including the tokenizer), the recommender, the storage backends (including the ascending-order `list()` contract exercised with out-of-order saves, and the framework database-connection contract of `TSqlBayesianStorage`), the module (one classifier and several sharing one backend, each with its own tokenizer), the service, and the metrics. Per-token storage is tested for score-equivalence with the whole-payload layout across every variant, in both SQL and Redis, along with incremental training and the `TBayesianModelConverter`; the Redis field encoding is additionally proven in isolation so it does not rest on a live server being present.

SQL tests skip cleanly when `ext-pdo`/`pdo_sqlite` is unavailable; Redis tests skip when `ext-redis` is absent or no server listens on `127.0.0.1:6379`; the MySQL and PostgreSQL round-trips run only when `BAYESIAN_MYSQL_DSN` / `BAYESIAN_PGSQL_DSN` name a reachable server:

```sh
BAYESIAN_PGSQL_DSN="pgsql:host=127.0.0.1;port=5432;dbname=bayesian_test" BAYESIAN_PGSQL_USER=postgres vendor/bin/phpunit --testsuite unit
```

CI provides Redis, MySQL, and PostgreSQL service containers and sets `BAYESIAN_REQUIRE_BACKENDS=1`, which turns each of those skips into a failure — so a green build means every backend really ran, rather than quietly skipping.

Line coverage is ~87% locally with only SQLite available, and higher in CI where Redis, MySQL, and PostgreSQL all run (`composer coverage`, needs Xdebug or PCOV; CI enforces a 93% floor). The gap between the two figures is almost entirely the Redis backend, which cannot run a line without `ext-redis`.

Locally the uncovered remainder is, in order of size:

- **`TRedisBayesianStorage`** (the largest block) — its constructor throws without `ext-redis`, so no instance can exist on a machine that lacks the extension: the whole class, both payload and per-token, is unreachable locally. The one branch that *is* reachable there — the missing-extension guard — is covered by `testConstructorThrowsWhenRedisExtensionMissing`, and the per-token field encoding is covered by `TRedisTokenEncodingTest`, which reaches the pure helpers by reflection without constructing the class. Everything else runs in CI.
- **The MySQL-only branches of `TSqlBayesianStorage`** — the `ON DUPLICATE KEY UPDATE` upserts in both the payload and per-token paths, reachable only against a real MySQL server.
- **IO- and DB-failure guards** — a `tempnam`/`file_put_contents` failure in `TFileBayesianStorage`, a connect failure in `TSqlBayesianStorage`. Reaching these needs the filesystem or server to fail between the check and the write.
- **Defensive guards the public API cannot reach at all** — a smoothing denominator of zero (a positive `Alpha` rules it out), a zero TF-IDF weight (the smoothed IDF is always ≥ 1), an empty-string split behind a caller that already returned early, and a pad whose target width is never below the input. These are unreachable by construction, not merely untested.

See [CHANGELOG.md](CHANGELOG.md) for release notes.

## License

BSD-3-Clause. See [LICENSE](LICENSE).
