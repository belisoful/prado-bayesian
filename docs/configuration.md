# Configuration

Wiring the extension into a PRADO application, and the full list of error codes it can raise.

## Installation

```bash
composer require belisoful/prado-bayesian
```

The framework comes with it: `pradosoft/prado` is a requirement at `^4.4@dev`, which resolves
straight from Packagist through the branch alias PRADO declares (`dev-master` → `4.4.x-dev`).
Your application needs `minimum-stability: dev`, `prefer-stable: true`, and the asset-packagist
repository listed in its own `composer.json` (PRADO depends on `bower-asset/*`, and Composer
reads `repositories` only from the root project). See the
[README](../README.md#installation) for the exact block.

The package is a `prado4-extension`. Its `composer.json` `extra.prado` block registers three
things system-wide, so an installed application needs no further wiring:

| Field | Registers |
| --- | --- |
| `bootstrap` | `Belisoful\Prado\Util\Bayesian\TBayesianModule` as the extension's bootstrap module |
| `error-messages` | `config/errorMessages.txt`, so `bayesian_*` codes resolve to messages |
| `class-map` | `config/prado-bayesian-classes.json`, the short-name → FQN map |

Optional PHP extensions: `ext-pdo` for `TSqlBayesianStorage`, `ext-redis` for
`TRedisBayesianStorage`. Neither is needed for the in-process or file backends.

## Short class names

Every class in the extension is registered under its short name, so configuration can say:

```xml
<classifier class="TComplementNaiveBayes" Alpha="0.5" />
```

instead of the fully-qualified `Belisoful\Prado\Util\Bayesian\Classifier\TComplementNaiveBayes`. Both
forms work for modules and for the `<classifier>`/`<storage>` child elements.

**Services are the exception, and must use the short name.** PRADO resolves
`<service class="...">` through `Prado::usingClass()`, which does not fall back to the Composer
autoloader for a not-yet-loaded fully-qualified name outside the framework directory. So:

```xml
<service id="bayesian" class="TBayesianService" />   <!-- correct -->
```

## The module

`TBayesianModule` owns the application's default classifier and default storage.

**XML** — `protected/application.xml`

```xml
<modules>
    <module id="bayesian" class="Belisoful\Prado\Util\Bayesian\TBayesianModule" DefaultClassifier="comment-spam">
        <classifier class="TComplementNaiveBayes" Alpha="0.5" UseTfidf="true" />
        <storage class="TFileBayesianStorage" Directory="/var/lib/myapp/bayesian" />
    </module>
</modules>
```

**PHP** — `protected/application.php`

```php
<?php
return [
    'modules' => [
        'bayesian' => [
            'class' => 'Belisoful\Prado\Util\Bayesian\TBayesianModule',
            'properties' => ['DefaultClassifier' => 'comment-spam'],
            'classifier' => ['class' => 'TComplementNaiveBayes', 'Alpha' => 0.5, 'UseTfidf' => true],
            'storage'    => ['class' => 'TFileBayesianStorage', 'Directory' => '/var/lib/myapp/bayesian'],
        ],
    ],
];
```

The mapping is mechanical, with one wrinkle worth stating plainly: an **attribute** on
`<module>` is a module property and belongs under `'properties'`, whereas `<classifier>` and
`<storage>` are **child elements** and become their own keys next to `'class'`. Inside those
two, class and properties sit at the same level — there is no nested `'properties'` key.

| Element / attribute | Effect |
| --- | --- |
| `<classifier>` | The classifier class and its properties. A `TNaiveBayesClassifier` is used when omitted. |
| `<storage>` | The storage backend and its properties. Without it, models do not persist across requests. |
| `DefaultClassifier` | The model name to take, and to load from storage during `init()`. |

`DefaultClassifier` makes the module ready to classify as soon as the application starts. A model
that does not exist in storage yet is simply not loaded — the classifier stays empty until it is
trained and saved. Any other storage failure (unreachable database, unwritable directory) is a
configuration error and propagates rather than being swallowed.

The module can equally be registered by package name, taking its class from
`extra.prado.bootstrap`:

```xml
<module id="belisoful/prado-bayesian" DefaultClassifier="comment-spam" />
```

```php
'modules' => [
    'belisoful/prado-bayesian' => [
        // No 'class' key: it comes from extra.prado.bootstrap, and supplying both raises
        // appconfig_moduletype_inapplicable.
        'properties' => ['DefaultClassifier' => 'comment-spam'],
        'storage'    => ['class' => 'TFileBayesianStorage', 'Directory' => '/var/lib/myapp/bayesian'],
    ],
],
```

Reach the configured classifier from application code:

```php
$classifier = $this->getApplication()->getModule('bayesian')->getClassifier();
```

### Several models in one module

Give each `<classifier>` an `id` and its own `Model` (the storage key), and they share the
module's storage — one database holding a spam filter, a language detector, and a topic router,
each with its own classifier variant and its own tokenizer:

```xml
<module id="bayesian" class="Belisoful\Prado\Util\Bayesian\TBayesianModule" DefaultClassifierID="spam">
    <storage class="TSqlBayesianStorage" ConnectionString="sqlite:/var/lib/myapp/bayesian.db" Mode="token" />
    <classifier id="spam" class="TComplementNaiveBayes" Model="comment-spam" Alpha="0.5">
        <tokenizer class="TWordTokenizer" MinLength="3" />
    </classifier>
    <classifier id="lang" class="TBernoulliNaiveBayes" Model="language-id">
        <tokenizer class="TNGramTokenizer" N="3" Characters="true" />
    </classifier>
</module>
```

```php
'modules' => [
    'bayesian' => [
        'class' => 'Belisoful\Prado\Util\Bayesian\TBayesianModule',
        'properties' => ['DefaultClassifierID' => 'spam'],
        'storage' => ['class' => 'TSqlBayesianStorage', 'ConnectionString' => 'sqlite:/var/lib/myapp/bayesian.db', 'Mode' => 'token'],
        'classifier' => [
            'spam' => ['class' => 'TComplementNaiveBayes', 'Model' => 'comment-spam', 'Alpha' => 0.5],
            'lang' => ['class' => 'TBernoulliNaiveBayes', 'Model' => 'language-id'],
        ],
    ],
],
```

```php
$module = $this->getApplication()->getModule('bayesian');
$module->getClassifier('spam')->classify($comment);
$module->getClassifier('lang')->classify($text);
$module->getClassifier();            // the DefaultClassifierID one, or the first configured
$module->getClassifiers();           // ['spam' => ..., 'lang' => ...]
```

| Attribute | On | Meaning |
|---|---|---|
| `id` | `<classifier>` | The key `getClassifier($id)` selects it by |
| `Model` | `<classifier>` | The storage key this classifier reads and writes |
| `<tokenizer>` | inside `<classifier>` | The tokenizer to train with; a loaded model brings back its own |
| `DefaultClassifierID` | `<module>` | Which one `getClassifier()` returns; defaults to the first |

Each configured model is loaded eagerly at boot if the storage already holds it, exactly as the
single-model case is. Models are keyed by name throughout, so they stay fully isolated even when
their vocabularies overlap completely — a token's statistics are stored per `(model, token,
category)`, never per token alone.

## HTTP service

`TBayesianService` exposes the module's classifier over HTTP as JSON. It is **read-only** — it
classifies and recommends, and offers no training, saving, or deletion.

```xml
<services>
    <service id="bayesian" class="TBayesianService" ModuleID="bayesian" MaxTextLength="65536" />
</services>
```

```php
'services' => [
    'bayesian' => [
        'class' => 'TBayesianService',
        'properties' => ['ModuleID' => 'bayesian', 'MaxTextLength' => 65536],
    ],
],
```

`ModuleID` names the `TBayesianModule` to source the classifier from; without it the service uses
the first `TBayesianModule` registered in the application. `MaxTextLength` caps the `text`
parameter, the joined `context` list and each `candidates` entry, in bytes (default 65536);
`MaxCandidates` caps the number of candidates a `recommend` request may send (default 100). Set
either to `0` to disable it. Classification cost grows with input size and every candidate costs a
classification, so the caps bound what one request can demand. `TagThreshold` (default 0.5) and
`MaxTags` (default 0, unlimited) shape the `tag` action, which runs a `TBayesianTagger` over the
service's classifier — a `TNaiveBayesClassifier` trained through a tagger.

### Access control

> **The service enforces nothing by default.** Every request that reaches it is answered, each
> one costs a classification, and the scores describe your model. Do not expose it publicly
> without one of the two restrictions below, or a reverse proxy that restricts it for you.

Both mechanisms are PRADO's own and both are opt-in.

**Authorization rules** are the `<allow>`/`<deny>` rules a page uses, declared inside an
`<authorization>` child of the service element (or an `authorization` list in PHP configuration),
with the usual `users`, `roles`, `verb`, `ips` and `priority` attributes. The service evaluates
them in `run()` against the application user before doing any work. Code can add rules too,
through `getAuthorizationRules()`.

```xml
<services>
    <service id="bayesian" class="TBayesianService" ModuleID="bayesian">
        <authorization>
            <allow roles="editor" />
            <deny users="*" />
        </authorization>
    </service>
</services>
```

```php
'services' => [
    'bayesian' => [
        'class' => 'TBayesianService',
        'properties' => ['ModuleID' => 'bayesian'],
        'authorization' => [
            ['action' => 'allow', 'roles' => 'editor'],
            ['action' => 'deny', 'users' => '*'],
        ],
    ],
],
```

A refused guest gets `401` and a refused authenticated user `403`, both as JSON
(`bayesian_service_unauthorized`). Rules need a user to evaluate against, which means an
authentication module (`TAuthManager` with a user manager) must be configured. When rules exist
but no user is on the application, the request is refused with `401`
`bayesian_service_user_required` rather than let through: a service meant to be restricted must
never fall open because of a second configuration mistake.

**Permissions**: the service implements `IPermissions` and registers three permissions,
`bayesian_classify`, `bayesian_recommend` and `bayesian_tag`, with the application's
`TPermissionsManager` when one is configured. Each action then first raises its dynamic event
(`dyClassify` / `dyRecommend` / `dyTag`), which the permissions behavior answers from the current user's roles and the
manager's rules; a user without the permission gets `401`/`403` `bayesian_service_permission_denied`.
With the manager's default `AutoDenyAll`, configuring it locks the service down until a rule or a
role grants the permission. Without a permissions manager the events are inert.

### Requests

| Parameter | Action | Meaning |
| --- | --- | --- |
| `action` | all | `classify` (the default), `recommend` or `tag` |
| `text` | classify, tag | The document to classify or tag |
| `category` | classify | Optional; adds `isSpam` to the response — whether the prediction equals this category |
| `context[]` | recommend | The items the user has already interacted with |
| `candidates[]` | recommend | The items to rank |

### Responses

`classify` returns the predicted category and the full distribution, plus `isSpam` when
`category` was given. `calibrated` says whether the scores are calibrated probabilities (the
classifier has a fitted calibration) or the plain normalized Naive Bayes scores:

```json
{"category": "ham", "scores": {"spam": 0.13, "ham": 0.87}, "calibrated": false, "isSpam": false}
```

`tag` returns the labels whose probability reaches `TagThreshold`, highest first, at most
`MaxTags` of them, and whether the tagger's probabilities are calibrated. The map is always a
JSON object, even when empty:

```json
{"tags": {"php": 0.91, "security": 0.78}, "calibrated": true}
```

`recommend` returns candidates ordered by P(positive). The map is always encoded as a JSON
object, even when every identifier is numeric:

```json
{"scores": {"tenet": 0.91, "dunkirk": 0.74, "barbie": 0.11}}
```

Errors are JSON too, with a matching HTTP status:

```json
{"error": "bayesian_service_text_required", "message": "..."}
```

| Status | When |
| --- | --- |
| 400 | Missing or non-string `text`, an array where a scalar was expected, no candidates, unknown `action` |
| 401 | Access refused for a guest, or rules are configured but the application has no user; also a denied permission for an anonymous caller |
| 403 | Access refused for an authenticated user, by a rule or a missing permission |
| 413 | `text`, the joined `context` or a candidate exceeds `MaxTextLength`, or more than `MaxCandidates` candidates were sent |
| 503 | The classifier has not been trained yet |

A server-side misconfiguration — no classifier resolvable at all — propagates to the framework's
error handler rather than being reported as a client error. Responses carry
`X-Content-Type-Options: nosniff` and never contain invalid UTF-8.

## Error codes

All codes live in `config/errorMessages.txt` and are registered system-wide via
`extra.prado.error-messages`. The framework's own `messages.txt` is not used.

### Training and classification

| Code | Raised when |
| --- | --- |
| `bayesian_category_required` | A training call passed an empty category name |
| `bayesian_training_set_empty` | `train()` was given a training set with no documents |
| `bayesian_classifier_not_trained` | Classifying or recommending before any training |
| `bayesian_classifier_score_undefined` | Trained, but every category scored `-INF` or `NaN` |
| `bayesian_alpha_invalid` | `Alpha` is not a positive finite number |
| `bayesian_classifier_class_invalid` | A configured classifier class does not implement `IBayesianClassifier` |
| `bayesian_classifier_id_unknown` | `getClassifier($id)` named a classifier the module was not configured with |

### Calibration and tagging

| Code | Raised when |
| --- | --- |
| `bayesian_calibration_temperature_invalid` | A `TTemperatureScaling` temperature is not a positive finite number |
| `bayesian_calibration_parameters_invalid` | A `TPlattScaling` slope or intercept is not finite |
| `bayesian_calibration_samples_mismatch` | The predictions and labels given to a calibration or metric differ in length |
| `bayesian_calibration_set_empty` | A calibration or metric received no usable labeled example |
| `bayesian_tagger_label_reserved` | A tag equals the tagger's `BackgroundCategory` |
| `bayesian_tagger_classifier_invalid` | The service's classifier is not a `TNaiveBayesClassifier`, so no tagger can be built over it |

### Persistence

| Code | Raised when |
| --- | --- |
| `bayesian_storage_required` | `save()` or `load()` with no storage backend set |
| `bayesian_classifier_name_required` | `save()` with no model name set |
| `bayesian_classifier_model_missing` | `load()` named a model the storage does not hold |
| `bayesian_classifier_kind_mismatch` | Loading a payload saved by a different classifier variant |
| `bayesian_storage_class_invalid` | A configured storage class does not implement `IBayesianStorage` |
| `bayesian_storage_name_invalid` | A model name is empty, starts with a dot, holds a path separator or null byte, or (Redis) contains the reserved sequence `:__` |
| `bayesian_storage_encode_failed` | The payload could not be encoded to JSON |
| `bayesian_model_format_unsupported` | A payload's `formatVersion` is newer than this release reads |

### File storage

| Code | Raised when |
| --- | --- |
| `bayesian_storage_directory_required` | `Directory` is unset or empty |
| `bayesian_storage_directory_unwritable` | The directory cannot be created or written |
| `bayesian_storage_save_failed` | The temp-file write or the atomic rename failed |

### SQL storage

| Code | Raised when |
| --- | --- |
| `bayesian_storage_pdo_missing` | `ext-pdo` is not loaded |
| `bayesian_storage_pdo_dsn_required` | No DSN, connection, or `ConnectionID` was configured |
| `bayesian_storage_pdo_connect_failed` | The connection could not be opened, or `ConnectionID` names no module |
| `bayesian_storage_table_invalid` | `Table` is not a plain SQL identifier of at most 48 characters |
| `bayesian_storage_mode_invalid` | `Mode` is neither `payload` nor `token` |

### Per-token storage

| Code | Raised when |
| --- | --- |
| `bayesian_storage_token_mode_required` | A per-token operation was attempted on storage in `payload` mode |
| `bayesian_classifier_token_mode_payload` | A model stored per token was read through the whole-payload path |
| `bayesian_vocabulary_full_scan_unavailable` | The whole vocabulary was requested from a storage-backed model, or such a model was asked to save itself |
| `bayesian_vocabulary_readonly` | A storage-backed vocabulary was mutated directly instead of through the classifier |
| `bayesian_classifier_aggregate_missing` | A Bernoulli or Complement model is storage-backed and its full-scan aggregate was neither stored nor recomputable (see [Storage → Incremental training and the variants](storage.md#incremental-training-and-the-variants)) |
| `bayesian_storage_layout_unsupported` | A per-token model's `layoutVersion` is newer than this release reads |

### Model conversion

| Code | Raised when |
| --- | --- |
| `bayesian_convert_destination_not_token` | The conversion destination is not in per-token mode |
| `bayesian_convert_source_already_token` | The source model is already stored per token |
| `bayesian_convert_kind_unknown` | The source model's `kind` marks a variant the converter has no class for |

### Redis storage

| Code | Raised when |
| --- | --- |
| `bayesian_storage_redis_missing` | `ext-redis` is not loaded |
| `bayesian_storage_redis_connect_failed` | Connect, `AUTH`, or `SELECT` failed |
| `bayesian_storage_redis_write_failed` | Redis rejected a write command |

### Tokenizers

| Code | Raised when |
| --- | --- |
| `bayesian_tokenizer_class_invalid` | A saved model names a tokenizer class that is missing or wrong |
| `bayesian_tokenizer_pattern_invalid` | A pattern does not compile |
| `bayesian_tokenizer_pattern_failed` | A pattern failed at runtime (backtrack or recursion limit) |

### Recommendation and evaluation

| Code | Raised when |
| --- | --- |
| `bayesian_recommendation_candidates_empty` | No candidates, or only blank identifiers |
| `bayesian_recommender_category_unknown` | `PositiveCategory` is not a trained category |
| `bayesian_metric_label_required` | A metric was asked for an empty label |
| `bayesian_confusion_label_required` | A confusion matrix was constructed with no labels |
| `bayesian_confusion_label_unknown` | `record()` was given a label the matrix does not hold |

### Service

| Code | Raised when |
| --- | --- |
| `bayesian_service_text_required` | The `text` parameter is missing |
| `bayesian_service_text_too_long` | `text`, the joined `context` or a candidate exceeds `MaxTextLength` |
| `bayesian_service_candidates_too_many` | More than `MaxCandidates` candidates were sent |
| `bayesian_service_parameter_invalid` | A parameter has the wrong shape |
| `bayesian_service_action_unknown` | `action` is none of `classify`, `recommend` and `tag` |
| `bayesian_service_classifier_missing` | No default classifier could be resolved |
| `bayesian_service_unauthorized` | The authorization rules refused the current user |
| `bayesian_service_user_required` | Authorization rules are configured but the application has no user |
| `bayesian_service_permission_denied` | The current user lacks the action's permission |
| `bayesian_service_rule_invalid` | A configured rule is neither `allow` nor `deny` |
