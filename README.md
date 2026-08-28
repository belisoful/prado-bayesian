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
| PRADO Framework `^4.4` (or `4.4.x-dev`) | provided by the application | `TComponent`, `TService`, `TModule`, and the `extra.prado.*` Composer plugin hooks |
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

Until PRADO 4.4 is released, add the framework's `master` branch as a repository in your application's `composer.json` (a `path` repository to a local checkout also works):

```json
{
    "repositories": [
        { "type": "composer", "url": "https://asset-packagist.org" },
        { "type": "vcs", "url": "https://github.com/pradosoft/prado" }
    ],
    "require": {
        "pradosoft/prado": "4.4.x-dev",
        "belisoful/prado-bayesian": "^0.1"
    },
    "minimum-stability": "dev",
    "prefer-stable": true
}
```

Both requirements are listed on purpose: like the other PRADO extensions, this package treats
the framework as a development dependency, so your application declares the PRADO version it
runs. The asset-packagist repository is required by the framework itself — PRADO depends on
`bower-asset/*` packages, and Composer only reads repositories from the root project, never
from a dependency, so installing without it fails with `bower-asset/jquery ... could not be found`.

The package's `config/` folder holds what PRADO's third-party plugin support reads system-wide from `composer.json` `extra.prado`: `errorMessages.txt` (the `bayesian_*` exception codes, registered through `error-messages`) and `prado-bayesian-classes.json` (Prado3-style short class names → PHP FQNs, registered through `class-map`, so `TNaiveBayesClassifier` resolves in Prado3-style configuration). Both load for every installed extension whether or not the bootstrap module is used.

## What it provides

| Class | Namespace | Role |
|---|---|---|
| `TBayesianModule` | `Prado\Util\Bayesian` | The `extra.prado.bootstrap` module; owns the configured default classifier |
| `TBayesianService` | `Prado\Web\Services` | A `TService` exposing classification and recommendation over the PRADO service pipeline (HTTP request) |
| `IBayesianClassifier` | `Prado\Util\Bayesian\Classifier` | The classifier contract: `train()`, `trainOne()`, `classify()`, `score()`, `save()`, `load()` |
| `TNaiveBayesClassifier` | `Prado\Util\Bayesian\Classifier` | The classic Naive Bayes (multinomial event model with Laplace smoothing) — the default spam filter |
| `TMultinomialNaiveBayes` | `Prado\Util\Bayesian\Classifier` | Multinomial Naive Bayes; counts token occurrences per category |
| `TBernoulliNaiveBayes` | `Prado\Util\Bayesian\Classifier` | Bernoulli Naive Bayes; tracks token presence/absence per document |
| `TComplementNaiveBayes` | `Prado\Util\Bayesian\Classifier` | Complement Naive Bayes; well-suited to imbalanced text classification |
| `IBayesianTokenizer` | `Prado\Util\Bayesian\Tokenizer` | The tokenizer seam: input text → list of feature tokens |
| `TWordTokenizer` | `Prado\Util\Bayesian\Tokenizer` | Default word tokenizer; lowercases, strips punctuation, drops short tokens, supports stop words |
| `TNGramTokenizer` | `Prado\Util\Bayesian\Tokenizer` | Character or word n-grams (`n` configurable) |
| `TRegexTokenizer` | `Prado\Util\Bayesian\Tokenizer` | A regex-driven tokenizer for custom patterns |
| `TBayesianTokenizerChain` | `Prado\Util\Bayesian\Tokenizer` | Composes multiple tokenizers (each contributes its tokens) |
| `TBayesianTokenizerTrait` | `Prado\Util\Bayesian\Tokenizer` | Shared tokenizer plumbing: property-driven `exportConfig()`/`importConfig()`, safe `matchAll()`, `normalizeText()` |
| `TBayesianTokenizerFactory` | `Prado\Util\Bayesian\Tokenizer` | Serializes/restores tokenizers into the saved model, UTF-8 scrubbing, regex validation |
| `TBayesianVocabulary` | `Prado\Util\Bayesian` | The feature vocabulary; per-category token counts, totals, smoothing |
| `TBayesianCategory` | `Prado\Util\Bayesian` | One category: its name, document count, and token counts |
| `TBayesianTrainingSet` | `Prado\Util\Bayesian` | An iterable labeled training set: maps categories to tokenized documents |
| `TFIdf` | `Prado\Util\Bayesian\Math` | Term-frequency × inverse-document-frequency weighting |
| `TBayesMath` | `Prado\Util\Bayesian\Math` | Log-space arithmetic helpers used by the classifiers to avoid underflow |
| `TConfusionMatrix` | `Prado\Util\Bayesian\Evaluation` | Confusion matrix for evaluating a classifier against a labeled set |
| `TBayesianMetrics` | `Prado\Util\Bayesian\Evaluation` | Precision, recall, F1, accuracy, macro/micro averages |
| `IBayesianStorage` | `Prado\Util\Bayesian\Storage` | The persistence seam for a trained model |
| `TMemoryBayesianStorage` | `Prado\Util\Bayesian\Storage` | Process-local in-memory storage (default; no I/O) |
| `TFileBayesianStorage` | `Prado\Util\Bayesian\Storage` | JSON file storage (good for development, small models, single host) |
| `TSqlBayesianStorage` | `Prado\Util\Bayesian\Storage` | SQL-backed storage using Prado's `TDbConnection` (SQLite, MySQL, PostgreSQL) |
| `TRedisBayesianStorage` | `Prado\Util\Bayesian\Storage` | Redis-backed storage for shared hosts (requires `ext-redis`) |
| `IBayesianRecommender` | `Prado\Util\Bayesian` | The recommender contract: `recommend()` for a user/item context |
| `TBayesianRecommender` | `Prado\Util\Bayesian` | A probabilistic recommender built on top of any `IBayesianClassifier` |

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
- **Vocabulary & categories** — `TBayesianVocabulary` is the per-category token-count map that backs the classifier. `TBayesianCategory` represents one class. `TBayesianTrainingSet` is the labeled corpus in training-time form.
- **Classifiers** — All implement `IBayesianClassifier` and accept any tokenizer + storage. `TNaiveBayesClassifier` is the canonical spam filter and the base class of the other three; `TMultinomialNaiveBayes`, `TBernoulliNaiveBayes`, and `TComplementNaiveBayes` override only the likelihood, so switching event model is a one-line change. Each writes a distinct `kind` marker into its saved model, so several variants can share one storage backend safely.
- **Storage** — `IBayesianStorage` persists a trained model. `TMemoryBayesianStorage` is the no-I/O default; `TFileBayesianStorage` writes JSON; `TSqlBayesianStorage` uses Prado's `TDbConnection`/`TDbCommand` for SQL-backed persistence (SQLite, MySQL, PostgreSQL); `TRedisBayesianStorage` scales across processes and hosts via Redis.
- **Recommender** — `TBayesianRecommender` reuses the classifier: train it on user/item interactions with a positive and a negative label (`PositiveCategory` defaults to `liked`), then ask it to rank candidate items.
- **Module & service** — `TBayesianModule` is the `extra.prado.bootstrap` entry point that registers the error message file and the default classifier. `TBayesianService` exposes the classifier and recommender over the PRADO service pipeline (HTTP), sourcing its classifier from the module.

## Usage

### Spam filter (the default)

```php
use Prado\Util\Bayesian\Classifier\TNaiveBayesClassifier;

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
use Prado\Util\Bayesian\Storage\TFileBayesianStorage;

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
    <module id="bayesian" class="Prado\Util\Bayesian\TBayesianModule" DefaultClassifier="comment-spam">
        <!-- optional: pick the classifier class and set its properties (default: TNaiveBayesClassifier) -->
        <classifier class="Prado\Util\Bayesian\Classifier\TComplementNaiveBayes" Alpha="0.5" />
        <storage class="Prado\Util\Bayesian\Storage\TFileBayesianStorage" Directory="/var/lib/myapp/bayesian" />
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
            'class' => 'Prado\Util\Bayesian\TBayesianModule',
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

Both forms of module registration work: `<module id="belisoful/prado-bayesian">` (the package name; the class comes from `extra.prado.bootstrap`) or `<module id="bayesian" class="Prado\Util\Bayesian\TBayesianModule">`. Register the service by its class-map short name `TBayesianService`: PRADO `master` resolves `<service class="…">` through `Prado::usingClass()`, which currently does not fall back to the Composer autoloader for a not-yet-loaded fully-qualified name outside the framework directory.

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

**Every backend loads the whole model.** A model is serialized to one JSON payload and stored as
a single unit — one file, one row, one Redis key — and `load()` decodes all of it into PHP
arrays before the first classification. No backend looks up individual tokens on demand, so the
choice of backend does not change the memory a loaded model costs. It changes only where the
model lives, who can reach it, and how large a payload the backend will physically accept.

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
| `TRedisBayesianStorage` | 512 MB per string value | — |

In every case PHP's `memory_limit` binds long before the backend's own ceiling: a 64 MB payload
needs roughly 200–250 MB of PHP memory to decode and hold. If you need models larger than a
process can hold, the fix is a smaller feature space — not a different backend.

### Recommendation

Train a `TBayesianRecommender`'s underlying classifier on user behavior: the "positive" category is the items the user engaged with; the "negative" category is items they ignored. Then ask it to rank candidates for a new user context.

```php
use Prado\Util\Bayesian\TBayesianRecommender;

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
use Prado\Util\Bayesian\Evaluation\TConfusionMatrix;
use Prado\Util\Bayesian\Evaluation\TBayesianMetrics;

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
use Prado\Util\Bayesian\Classifier\TComplementNaiveBayes;
use Prado\Util\Bayesian\Evaluation\TBayesianMetrics;
use Prado\Util\Bayesian\Evaluation\TConfusionMatrix;
use Prado\Util\Bayesian\Storage\TFileBayesianStorage;
use Prado\Util\Bayesian\TBayesianTrainingSet;
use Prado\Util\Bayesian\Tokenizer\TBayesianTokenizerChain;
use Prado\Util\Bayesian\Tokenizer\TNGramTokenizer;
use Prado\Util\Bayesian\Tokenizer\TWordTokenizer;

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
repository and takes `pradosoft/prado` from its `master` branch as a development dependency —
`composer install` needs no further setup. To develop against a local PRADO checkout instead,
add a path repository to your working copy and leave it uncommitted:

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

Tests cover the math (log-space arithmetic, TF-IDF), tokenizers (word, n-gram, regex, chain, factory round-trips), the classifiers (Naive Bayes and the three variants over tokenized corpora, save/load including the tokenizer), the recommender, the storage backends (including the ascending-order `list()` contract, exercised with out-of-order saves so a sort cannot be confused with insertion order), the module, the service, and the metrics.

SQL tests skip cleanly when `ext-pdo`/`pdo_sqlite` is unavailable; Redis tests skip when `ext-redis` is absent or no server listens on `127.0.0.1:6379`; the MySQL and PostgreSQL round-trips run only when `BAYESIAN_MYSQL_DSN` / `BAYESIAN_PGSQL_DSN` name a reachable server:

```sh
BAYESIAN_PGSQL_DSN="pgsql:host=127.0.0.1;port=5432;dbname=bayesian_test" BAYESIAN_PGSQL_USER=postgres vendor/bin/phpunit --testsuite unit
```

CI provides Redis, MySQL, and PostgreSQL service containers and sets `BAYESIAN_REQUIRE_BACKENDS=1`, which turns each of those skips into a failure — so a green build means every backend really ran, rather than quietly skipping.

Line coverage is ~92% locally with SQLite available, and higher in CI where Redis, MySQL, and PostgreSQL all run (`composer coverage`, needs Xdebug or PCOV; CI enforces a 93% floor).

Locally the uncovered remainder is, in order of size:

- **`TRedisBayesianStorage`** — its constructor throws without `ext-redis`, so no instance can exist on a machine that lacks the extension and even the property accessors are unreachable. The one branch that *is* reachable there — the missing-extension guard itself — is covered by `testConstructorThrowsWhenRedisExtensionMissing`, which is the inverse case: it skips when `ext-redis` is present.
- **The MySQL-only branches of `TSqlBayesianStorage`** — the `ON DUPLICATE KEY UPDATE` upsert, reachable only against a real MySQL server.
- **IO- and DB-failure guards** — a `tempnam`/`file_put_contents` failure in `TFileBayesianStorage`, a connect failure in `TSqlBayesianStorage`. Reaching these needs the filesystem or server to fail between the check and the write.
- **Defensive guards the public API cannot reach at all** — a smoothing denominator of zero (a positive `Alpha` rules it out), a zero TF-IDF weight (the smoothed IDF is always ≥ 1), an empty-string split behind a caller that already returned early, and a pad whose target width is never below the input. These are unreachable by construction, not merely untested.

See [CHANGELOG.md](CHANGELOG.md) for release notes.

## License

BSD-3-Clause. See [LICENSE](LICENSE).
