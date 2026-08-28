# Changelog

All notable changes to `belisoful/prado-bayesian` are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow [Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-08-25

First public pre-release. Targets the PRADO `master` branch (upcoming 4.4): the extension is
registered through `composer.json` `extra.prado.bootstrap`, and its `config/errorMessages.txt`
and `config/prado-bayesian-classes.json` load system-wide via `extra.prado.error-messages` and
`extra.prado.class-map`. Not compatible with PRADO 4.3.x. APIs may change before 1.0.0.

Classes are namespaced under `Belisoful\Prado\Util\Bayesian` (`Classifier`, `Tokenizer`, `Math`,
`Evaluation`, and `Storage` sub-namespaces), except `TBayesianService`, which lives at
`Belisoful\Prado\Web\Services\TBayesianService`. The PSR-4 root is `Belisoful\Prado\` → `src/`:
the tree mirrors the framework's own layout so a class sits where a PRADO developer expects it,
under a vendor prefix that keeps it out of `Prado\` itself. Every class is also registered under
its Prado3-style short name, so application configuration can use either form.

### Added
- `TNaiveBayesClassifier` (multinomial Naive Bayes with Laplace smoothing and optional TF-IDF), plus `TMultinomialNaiveBayes`, `TBernoulliNaiveBayes`, and `TComplementNaiveBayes` (WCNB) variants behind one `IBayesianClassifier` contract.
- `TBayesianVocabulary` / `TBayesianCategory` / `TBayesianTrainingSet` training statistics; `TBayesMath` (log-space arithmetic) and `TFIdf`.
- Tokenizers: `TWordTokenizer`, `TNGramTokenizer` (character/word), `TRegexTokenizer`, `TBayesianTokenizerChain`, all behind `IBayesianTokenizer` and sharing `TBayesianTokenizerTrait` (property-list driven `exportConfig()`/`importConfig()`, safe `matchAll()`); `TBayesianTokenizerFactory` serializes/restores tokenizers, scrubs invalid UTF-8, and validates regex patterns.
- `TSqlBayesianStorage` configures its connection through `Prado\Data\TDbPropertiesTrait`, the same way the framework's other database-backed components do: `ConnectionID` resolves a `TDataSourceConfig` module and shares its connection, falling back to the storage's own DSN properties or an injected connection, and `getHasDbConnection()` / `deactivateDbConnection()` / `getTableGateway()` come with it. An unset `ConnectionID` reads as `''`, per the framework convention.
- Storage backends behind `IBayesianStorage`: `TMemoryBayesianStorage`, `TFileBayesianStorage` (atomic JSON files), `TSqlBayesianStorage` (SQLite/MySQL/PostgreSQL via `TDbConnection`, driver-aware DDL, `AutoCreateTable`), `TRedisBayesianStorage` (ext-redis).
- `TBayesianRecommender` / `IBayesianRecommender` ranking candidates by P(positive | context + candidate).
- `TConfusionMatrix` and `TBayesianMetrics` (accuracy, precision, recall, F1, macro/micro).
- `TBayesianModule` (`TModule` bootstrap named in `extra.prado.bootstrap`; `<classifier>`/`<storage>` child elements, `DefaultClassifier` eager load). Several `<classifier id="..." Model="...">` elements may share one storage backend, each with its own variant and its own `<tokenizer>`; `getClassifier($id)` selects between them and `DefaultClassifierID` picks the default.
- `TBayesianService` (read-only JSON `classify`/`recommend` HTTP actions with JSON error responses, `MaxTextLength`, `ModuleID`).
- Per-token storage for `TSqlBayesianStorage` and `TRedisBayesianStorage`. With `Mode="token"` a model is stored per (token, category) — SQL rows, or a Redis hash per token — rather than one JSON payload, and a classifier scores a document by reading only that document's tokens through `TLazyBayesianVocabulary`. On a 100,000-token SQL model that is 0.7 ms and 0.2 MB to load against 106 ms and 44.1 MB; the SQL model is bounded by the database, the Redis model by the Redis instance's RAM (which still holds it whole). Scores are identical to the whole-payload layout in both backends.
- Training against per-token storage is incremental: `trainOne()` writes only the document's rows instead of re-serializing the model, about 11x faster on a 100,000-token SQL model and independent of its size. The Redis backend applies the deltas with `HINCRBY`, so a document's counts accumulate atomically with no read-modify-write.
- `IBayesianVocabulary` (implemented by `TBayesianVocabulary` and `TLazyBayesianVocabulary`) and `IBayesianTokenStorage`, the two seams the above runs through. `TFIdf::idf()` and `weight()` take document-frequency counts rather than the whole map.
- `TBayesianModelConverter` rewrites a whole-payload model into a per-token backend without retraining — one model or every model a backend holds, across backends or in place within one database. It reads the classifier variant from the model's stored `kind`, so a caller need not know which variant each model is. Conversion is exact; the reverse direction is not offered, since the per-token layout deliberately cannot enumerate its vocabulary.
- `docs/`: concepts (the pipeline, the three event models, smoothing, TF-IDF, log-space math, tokenization, evaluation), a class reference, the storage-backend guide (including measured model sizes and the memory each backend costs), and configuration in both XML and PHP form, including every error code.

### Behavior
- Out-of-vocabulary tokens are skipped at classification time in every variant, rather than contributing a smoothed penalty that would bias novel documents toward the smallest category.
- The saved model state includes the tokenizer class and configuration, so `load()` restores the tokenizer a model was trained with.
- Each classifier variant writes a `kind` marker and refuses payloads of another kind (`bayesian_classifier_kind_mismatch`); `naive-bayes` and `multinomial-naive-bayes` are interchangeable.
- `setAlpha()` requires a positive finite value (`bayesian_alpha_invalid`); a trained classifier whose every category scores −INF/NaN throws `bayesian_classifier_score_undefined` rather than `bayesian_classifier_not_trained`.
- Tokenizers throw `bayesian_tokenizer_pattern_invalid` on a non-compiling pattern and `bayesian_tokenizer_pattern_failed` when PCRE fails at match time (backtrack limit); invalid UTF-8 input is scrubbed rather than emptying the token list. `TRegexTokenizer` uses the first capturing group consistently (an unmatched optional group yields no token). `TNGramTokenizer` normalizes whitespace in character mode.
- Every storage backend returns `list()` sorted ascending by name; the persistent backends report a payload that is present but unparseable as a missing model rather than throwing.
- Bernoulli classification is O(document x categories), not O(vocabulary x categories). Absence is evidence in the Bernoulli model, but the absent-token mass is a constant per category, so it is summed once and cached and each document only corrects it for the tokens it contains — arithmetically identical to the literal sum over the vocabulary.
- The derived caches that Bernoulli and Complement keep are keyed on a vocabulary state signature that changes on any mutation, including one made directly on a category obtained from `getVocabulary()`. A stale aggregate would shift scores silently rather than raise, so the key covers more than the document and category totals.
- `TFileBayesianStorage` writes through a unique temp file with `LOCK_EX`; an empty `Directory` is treated as unset. `TSqlBayesianStorage` uses `VARCHAR(191)`/`LONGTEXT`/`BIGINT` on MySQL and creates the table once per connection; JSON encoding failures throw in every backend. `TRedisBayesianStorage` checks `AUTH`/`SELECT`/write results.
- `TBayesianModule::init()` applies `DefaultClassifier` even without a `<classifier>` element and only loads a model that `exists()`; storage errors propagate instead of being swallowed. Configured classes are resolved and type-checked before instantiation.
- `TBayesianService` returns JSON errors (`400`/`413`/`503`) for bad parameters, oversize text, or an untrained model; malformed (array-valued) parameters are rejected; recommend `scores` are always a JSON object; responses carry `X-Content-Type-Options: nosniff`.
- `TBayesianRecommender` ignores blank candidates, throws when the classifier is untrained, and throws `bayesian_recommender_category_unknown` when its `PositiveCategory` is not one of the classifier's trained categories. `TBayesianMetrics::getF1()` returns `0.0` (not NaN) when precision and recall are both defined and zero, and per-label metrics reject unregistered labels.

### Packaging
- `pradosoft/prado` is a runtime requirement at `^4.4@dev`, not a development one, so Composer installs the framework with the package. It resolves straight from Packagist through the branch alias PRADO declares (`dev-master` → `4.4.x-dev`), so `composer.json` declares no framework repository — only `minimum-stability: dev` with `prefer-stable: true`. An application still lists the asset-packagist repository itself, since Composer reads `repositories` only from the root project and PRADO depends on `bower-asset/*`. Once PRADO 4.4 reaches a stable Packagist release the constraint can drop `@dev`.

### Testing
- CI runs the matrix (PHP 8.1/8.2/8.3) against Redis, MySQL 8, and PostgreSQL 16 service containers with `BAYESIAN_REQUIRE_BACKENDS=1`, so a backend that is unavailable fails the build instead of silently skipping its tests. The MySQL and PostgreSQL round-trips cover the driver-aware DDL, the upsert, and a payload larger than MySQL's 64 KB `TEXT` limit.
- `composer coverage` reports line coverage and CI enforces a floor (`tests/test_tools/check-coverage.php`).
- `composer integration` (`tests/integration/`) installs the extension into a throwaway consumer project through Composer and verifies the `extra.prado` error-messages, class-map, and bootstrap wiring plus a served request; it runs as its own CI job.

[0.1.0]: https://github.com/belisoful/prado-bayesian/releases/tag/v0.1.0
