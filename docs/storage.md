# Storage backends

A trained model is worth persisting: training is the expensive part, and a classifier that has
to relearn on every request is not useful. `IBayesianStorage` is the seam that makes the
persistence choice a configuration detail rather than a code change.

## The contract

```php
interface IBayesianStorage
{
    public function save(string $name, array $payload): void;
    public function load(string $name): ?array;
    public function exists(string $name): bool;
    public function delete(string $name): void;
    public function list(): array;
}
```

The **payload is opaque** to the storage. It is a JSON-serializable array the classifier
produced and is responsible for parsing back, so each backend picks the encoding that suits it:
a single JSON blob for the file and in-process backends, a text column for SQL, a string value
for Redis.

Behavior every backend shares:

- `load()` returns `null` for an unknown name — and also for a payload that is present but not
  valid JSON. A corrupt model reads as a missing one; it does not throw.
- `delete()` on a name that is not stored is a no-op.
- `list()` returns names **sorted ascending**. The sort is byte order for the in-process, file,
  and Redis backends and the column collation for SQL, and those can disagree on case and
  accents — re-sort yourself if you need one exact ordering across backends.
- A payload that cannot be JSON-encoded throws `bayesian_storage_encode_failed` rather than
  storing an empty record. (The in-process backend is the exception: it stores the array
  as-is and never encodes.)
- Every payload carries a `formatVersion` (currently `1`). A payload without one was written by
  0.1.0 and reads as version 1; a payload from a newer format than the installed release is
  refused with `bayesian_model_format_unsupported` rather than misread.

## Concurrency

Which backend and mode you choose decides whether more than one process may train a model.

**Payload mode is single-writer.** In the in-process, file, SQL-payload and Redis-payload
layouts a save replaces the whole model, so two processes that each load, train and save the
same model lose one another's documents: the last writer wins, silently. Reading is always safe
(the file backend renames a complete file into place; SQL and Redis replace one value), and one
writer with any number of readers is fine. Use payload mode when training happens in one place
— a cron job, a console command, an admin page.

**Per-token mode is multi-writer.** With `Mode="token"` on `TSqlBayesianStorage` or
`TRedisBayesianStorage`, every training write is an atomic increment in the store — `cnt = cnt
+ delta` in SQL, `HINCRBY` inside a Lua script in Redis — and the model's totals are derived by
the store from what it holds rather than written from any process's snapshot. Any number of web
requests and background workers may train one model at once without losing a count; the test
suite proves it with parallel processes against SQLite, MySQL, PostgreSQL and Redis, for every
classifier variant — including the vocabulary-wide histograms Bernoulli and Complement score
from. This is the mode to use when training happens from concurrent requests.

Two things follow for a process holding a per-token model:

- Its resident scalars (document total, vocabulary size, category totals) are a snapshot. They
  are re-read from the store after every `trainOne()` on that process, and
  `TLazyBayesianVocabulary::refresh()` re-reads them on demand; between those points another
  process may have moved them. Scores use the snapshot; the per-token statistics are always read
  live.
- The prefetched token batch is bounded by `MaxBatchTokens` (default 50,000) so a worker that
  scores documents indefinitely holds a bounded amount of memory.

Deltas are increments in both directions: `untrainOne()` sends negative ones. Every count is
clamped at zero, and a token that no document contains any more after a withdrawal leaves the
vocabulary (its vocabulary row is dropped in SQL, its hash and set membership in Redis), so a
model that trained and then untrained a document ends as it would have without it.

### Incremental training and the variants

Every classifier trains incrementally against a per-token model: `TNaiveBayesClassifier`,
`TMultinomialNaiveBayes`, `TBernoulliNaiveBayes` and `TComplementNaiveBayes` alike, from any
number of processes at once, and a storage-backed model scores exactly — bit for bit — as a
resident model trained on the same documents.

Bernoulli and Complement need more than the document's own tokens: each keeps a per-category
aggregate that is a sum over the whole vocabulary (Bernoulli's absent-token mass, Complement's
weight norm), which a storage-backed vocabulary cannot walk. They do not have to. Each term of
those sums depends on a token only through a small integer — how many of the category's
documents contain it, how often it occurs outside the category — so the sum over tokens is a sum
over the distinct integers, weighted by how many tokens share each one. Those weights are the
`TBayesianTokenHistogram` histograms, and a storage implementing `IBayesianHistogramStorage`
(both `TSqlBayesianStorage` and `TRedisBayesianStorage`) keeps them:

- The classifier names the families it needs in the model's metadata (`histograms`: `["d"]` for
  Bernoulli, `["g","a","k"]` for Complement, none for the multinomial classifiers, whose training
  writes therefore pay for none).
- `saveTokenModel()` writes them; `applyDeltas()` moves them in the same atomic unit as the
  counts. The cells the document's tokens counted towards lose one each, the cells they count
  towards afterwards gain one. The cost is proportional to the document.
- A loaded classifier reads them with the model (one small read) and again after each of its own
  training writes. They are integers and do not depend on alpha.

SQL keeps them in the `<Table>_hist` table. Bernoulli's histogram depends only on the row a
training write already locks, so it moves with no further locking and a Bernoulli model is
written as concurrently as a multinomial one. Complement's depends on a token's counts in every
category, so a write to a Complement model first locks the model's row in `<Table>_counters` and
those writes run one after another (each is a handful of statements); different models never
wait for each other. With `AutoCreateTable="false"`, create the table from
`getCreateTokenTableSql()`. Redis keeps them in the `<KeyPrefix><name>:__hist` hash and moves
them inside the same Lua script that moves the token's counts.

#### Histogram modes (SQL, Complement)

The per-model lock is the price of histograms that are exact after every write. When a
Complement model is trained from many processes and a slightly stale score is acceptable,
`HistogramMode` on `TSqlBayesianStorage` trades freshness for concurrency:

| `HistogramMode` | A training write | The histograms | Writers to one model |
| --- | --- | --- | --- |
| `immediate` (default) | moves them, under the model lock | exact after every write | one after another |
| `deferred` | journals the tokens it touched (`<Table>_journal`) | trail training until a worker folds the journal; exact once it is empty | side by side |
| `periodic` | does nothing extra | trail training until the next scheduled recount | side by side, at the cost of a multinomial write |

Both relaxed modes are served by one call, `maintainTokenHistograms($name)`: it folds a deferred
model's journal until it is empty and recounts a periodic model. Run it from **one** worker — a
cron job, a queue consumer, a loop in a console command:

```php
$storage = $module->getStorage();                 // the configured TSqlBayesianStorage
foreach (['comments', 'submissions'] as $model) {
    $storage->maintainTokenHistograms($model);    // returns the tokens folded / cells written
}
```

A fold costs in proportion to the tokens written since the last one and holds their rows only
for one small transaction (`foldTokenHistograms($name, $limit)` folds one batch), so a deferred
worker can run every few seconds or continuously. A recount costs in proportion to the model
and runs inside the database, so a periodic worker suits minutes or hours. Two workers at once
are safe — they take turns on the model's fold lock — just pointless. `getPendingTokenCount()`
reports the journal's length, which is the thing to monitor.

Between maintenance runs a Complement model scores with the histograms of the last run and the
counts of now, which shifts its weight norms slightly; rankings of clearly different documents
are unaffected, and scores are exact again after the next run. Do not use a relaxed mode where
every score must reflect every write the instant it lands.

The mode belongs to the model, not to the writer: it is recorded in the model's metadata
(`histogramMode`) when the model is saved through a storage configured with it, and from then on
every process follows the model whatever its own `HistogramMode` says. Change an existing model
with `setTokenHistogramMode($name, $mode)`, which recounts on the way; do it while training is
paused, since a writer caught mid-switch may have its histogram change missed until the next
`maintainTokenHistograms()`, which notices and recounts. Bernoulli and the multinomial
classifiers ignore the mode: they have no lock to avoid. The deferred mode keeps a copy of the
token statistics as last folded (`<Table>_folded`), so a deferred Complement model takes roughly
twice the rows of an immediate one. Redis has no modes; its writes are atomic scripts already.

A per-token Bernoulli or Complement model stored before histograms existed has none. They are
built in the store the first time such a model is loaded or trained — grouped queries in SQL, one
Lua script in Redis — and recorded in the metadata. That rebuild is the one step proportional to
the model: on Redis it blocks the server while it walks the model's tokens, so migrate a large
model off-peak by calling
`rebuildTokenHistograms($name, $families)` yourself. The same call repairs a model whose rows
were edited by hand.

A third-party `IBayesianTokenStorage` that does not implement `IBayesianHistogramStorage` still
serves Bernoulli and Complement models saved from a resident vocabulary (the aggregates travel in
the metadata), but training one incrementally clears them, and the next `classify()` raises
`bayesian_classifier_aggregate_missing` rather than score with a stale constant.

## What the classifier stores

`save()` on a classifier writes the payload built by `exportState()`:

| Key | Purpose |
| --- | --- |
| `kind` | The classifier variant marker; loading into a different variant throws `bayesian_classifier_kind_mismatch` |
| `name` | The model name |
| `alpha`, `useTfidf`, `spamCategory` | The configuration that affects the math |
| tokenizer class + settings | So a reloaded model tokenizes exactly as it was trained |
| per-category statistics | Document counts, token counts, per-token document counts, totals |
| document frequency map | Corpus-wide, for TF-IDF and for the out-of-vocabulary check |
| `calibration` | The fitted `TTemperatureScaling`, when the model has been calibrated |
| `extra` | State other components keep with the model, e.g. a tagger's per-label calibrations |
| `formatVersion` | The payload format (currently 1); see below |

Because the tokenizer travels with the model, a classifier trained with an n-gram tokenizer or a
chain tokenizes identically after `load()` into a fresh instance.

## Model size and memory

**In `payload` mode, a backend loads the whole model.** The payload is stored as a single unit —
one file, one row, one Redis key — and `load()` decodes all of it into PHP arrays before the
first classification. The figures here are for that mode; it is the right choice whenever the
model fits comfortably in a request, and the only mode the in-process and file backends offer.

The SQL and Redis backends also offer **`Mode="token"`** (below), where a classification reads
only the document's own tokens, so a loaded model costs kilobytes regardless of size. That is a
different trade — one indexed query per classification instead of a one-time decode — covered
after the sizing here, which is what a resident model costs.

Measured with `composer benchmark` (`tests/benchmark/benchmark-storage.php`) on an Apple M-series
laptop, PHP 8.1, SQLite for the per-token layout, two categories that have both seen the whole
vocabulary. Rerun it on your own hardware; the ratios matter more than the milliseconds.

| Vocabulary | JSON payload | Decoded in PHP | Payload `load()` | Per-token `load()` | Payload `trainOne()`+`save()` | Per-token `trainOne()` |
|---:|---:|---:|---:|---:|---:|---:|
| 5,000 | 0.7 MB | 2.3 MB | 5 ms, 2.3 MB | 2 ms, 0.1 MB | 2 ms | 1 ms |
| 20,000 | 2.8 MB | 9.7 MB | 21 ms, 9.7 MB | 0.3 ms, < 0.1 MB | 4 ms | 1 ms |
| 100,000 | 14.0 MB | 43.7 MB | 106 ms, 43.7 MB | 0.3 ms, < 0.1 MB | 17 ms | 1 ms |

The payload runs **30–40 bytes per token-per-category**: each category keeps its own occurrence
and document counts for every token it has seen, plus one corpus-wide document-frequency map.
The decoded PHP structure is **3–4× the JSON**, because hash-table entries cost far more than
text. Budget both at once — `json_decode()` holds the string and the growing array
simultaneously. Ten categories over the same words cost roughly five times the model of two.

The per-token columns are what `Mode="token"` buys: loading a model costs a metadata read and a
category read whatever its size, and training one document writes that document's rows instead
of re-serializing the model — about 12× faster at 100,000 tokens, and independent of the
vocabulary. Each classification then costs one indexed query for the document's tokens.

So size scales with vocabulary **times** categories. The effective lever is the feature space,
not the backend: raise `MinLength`, supply `StopWords`, or prefer word tokens over character
n-grams, which produce far more distinct features.

| Backend | Ceiling on one model | What usually binds first |
|---|---|---|
| `TMemoryBayesianStorage` | PHP `memory_limit` | The limit itself; the model is also gone at end of process |
| `TFileBayesianStorage` | Filesystem | PHP memory — one `.json` per model, read whole via `file_get_contents()` |
| `TSqlBayesianStorage` | `LONGTEXT` 4 GB (MySQL), `TEXT` ~1 GB (PostgreSQL/SQLite) | MySQL `max_allowed_packet`, 64 MB by default |
| `TRedisBayesianStorage` | 512 MB per string value (payload), or the Redis instance's RAM (token) | PHP memory in payload mode; the Redis instance in token mode |

In practice PHP's `memory_limit` binds long before any backend ceiling: a 64 MB payload needs
roughly 200–250 MB of PHP memory to decode and hold. A model too large for one process is a
signal to shrink the feature space, not to change backend.

## Choosing a backend

| Backend | Namespace suffix | Survives the request? | Shared across processes? | Needs |
| --- | --- | --- | --- | --- |
| `TMemoryBayesianStorage` | `Storage` | No | No | — |
| `TFileBayesianStorage` | `Storage` | Yes | Same host only | A writable directory |
| `TSqlBayesianStorage` | `Storage` | Yes | Yes | `ext-pdo` |
| `TRedisBayesianStorage` | `Storage` | Yes | Yes | `ext-redis` |

All four live in `Belisoful\Prado\Util\Bayesian\Storage`.

Configuring the SQL or Redis backend without its extension is a **configuration error**
(`bayesian_storage_pdo_missing` / `bayesian_storage_redis_missing`) — there is deliberately no
silent fallback to a backend that would quietly lose your models.

### `TMemoryBayesianStorage`

Process-local, no I/O. The default when nothing is configured. Right for unit tests,
request-scoped classifiers, and any short-lived classification. Models are lost when the process
exits.

### `TFileBayesianStorage`

JSON files in a directory, one per model. Good for development, small models, and single-host
deployments.

```php
$storage->setDirectory('/var/lib/myapp/bayesian');
```

Writes are **atomic**: the payload goes to a per-call unique temp file in the same directory
with `LOCK_EX`, then `rename()`s into place. `rename()` is atomic within a filesystem, so a
reader never sees a partial file and two concurrent saves of the same model cannot interleave
— but each save still replaces the whole model, so this is a single-writer backend (see
[Concurrency](#concurrency)). The directory is created on demand; an unset or empty `Directory`
throws `bayesian_storage_directory_required`, and one that cannot be created or written throws
`bayesian_storage_directory_unwritable`.

Files are created with `FileMode` (default `0644`) and a created directory with `DirectoryMode`
(default `0755`); both accept an integer or the octal string a configuration file carries. A
model file holds the tokens of its training data, so tighten them (`FileMode="0600"`) when other
accounts on the host must not read it.

Model names are validated: a name containing a path separator or a null byte is rejected
(`bayesian_storage_name_invalid`) rather than resolved, so a name like `../../etc/passwd` cannot
escape the directory, and a name starting with a dot is rejected because a dotfile would be
saved but never listed.

### `TSqlBayesianStorage`

SQLite, MySQL/MariaDB, or PostgreSQL through PRADO's `TDbConnection` — the class uses
`TDbConnection`/`TDbCommand` throughout and never holds a raw PDO handle.

The connection is configured exactly as it is for every other database-backed component in the
framework, through `Prado\Data\TDbPropertiesTrait`. `ConnectionID` names a
`Prado\Data\TDataSourceConfig` module and the storage shares that module's connection — the
route to use when the application already has a database configured. `getHasDbConnection()`,
`deactivateDbConnection()`, and `getTableGateway()` come with the trait and behave as they do on
`TDbCache` or `TDbLogRoute`. Note the framework convention that an unset `ConnectionID` reads as
`''`, not `null`.

Three ways to configure the connection, in the order the storage tries them:

```xml
<!-- 1. A DSN on the storage itself -->
<storage class="TSqlBayesianStorage" ConnectionString="sqlite:/var/lib/myapp/bayesian.db" />

<!-- 2. A shared TDataSourceConfig module -->
<module id="db" class="Prado\Data\TDataSourceConfig">
    <database ConnectionString="mysql:host=localhost;dbname=mydb" Username="user" Password="pass" />
</module>
<storage class="TSqlBayesianStorage" ConnectionID="db" />
```

The same two, in PHP configuration:

```php
// 1. A DSN on the storage itself
'storage' => ['class' => 'TSqlBayesianStorage', 'ConnectionString' => 'sqlite:/var/lib/myapp/bayesian.db'],

// 2. A shared TDataSourceConfig module, referenced by id.
//    Note 'database', not 'properties': ConnectionString belongs to the connection the
//    module wraps, mirroring the <database> child element in the XML form.
'modules' => [
    'db' => [
        'class' => 'Prado\Data\TDataSourceConfig',
        'database' => [
            'ConnectionString' => 'mysql:host=localhost;dbname=mydb',
            'Username' => 'user',
            'Password' => 'pass',
        ],
    ],
    'bayesian' => [
        'class' => 'Belisoful\Prado\Util\Bayesian\TBayesianModule',
        'storage' => ['class' => 'TSqlBayesianStorage', 'ConnectionID' => 'db'],
    ],
],
```

```php
// 3. Inject an already-configured connection from code
$storage->setDbConnection($connection);
```

The table (`bayesian_models` by default) is created on first use with driver-aware DDL —
`VARCHAR(191)` key and `LONGTEXT` payload on MySQL, `TEXT` on SQLite and PostgreSQL, 64-bit
`updated_at` everywhere. MySQL cannot index a `TEXT` column without a prefix length and caps
`TEXT` at 64 KB, which is why the key and payload types differ there.

Set `AutoCreateTable="false"` when migrations own the schema or the database user has no DDL
rights. The DDL runs at most once per connection, because it forces an implicit commit on MySQL.

The upsert is driver-aware too: `ON DUPLICATE KEY UPDATE` on MySQL, `ON CONFLICT … DO UPDATE`
on SQLite and PostgreSQL.

When nothing at all is configured the storage raises `bayesian_storage_pdo_dsn_required` rather
than creating a SQLite file in the runtime path the way a cache does — a trained model is not
scratch data, and a runtime directory that gets cleared is the wrong place for one.

`Table` is interpolated into the SQL — an identifier cannot be a bound parameter — so it is
validated against `[A-Za-z_][A-Za-z0-9_]*` and at most 48 characters (the per-token tables add
suffixes, and PostgreSQL and MySQL cap identifiers at 63/64) and throws
`bayesian_storage_table_invalid` otherwise, which keeps it from becoming an injection vector.

#### Per-token mode

With `Mode="token"` the storage uses eight tables: `<table>` for the metadata row,
`<table>_tokens` (one row per model, token and category), `<table>_categories`,
`<table>_vocab` (one row per distinct token of a model), `<table>_counters`, `<table>_hist`
(the token histograms of Bernoulli and Complement models; see
[Incremental training and the variants](#incremental-training-and-the-variants)), and
`<table>_journal` and `<table>_folded`, used only by models in the deferred
[histogram mode](#histogram-modes-sql-complement). They are
created on first use like the main table; with `AutoCreateTable="false"`, take the DDL from
`getCreateTokenTableSql($driver)`. Training is a set of atomic upserts inside one transaction,
so many processes may train at once ([Concurrency](#concurrency)). Every write takes its locks in
one order — a token's `<table>_vocab` row first (shared by writes that add documents, exclusive
for writes that withdraw them, since whether a token leaves the vocabulary depends on its rows in
every category), then the token rows in sorted order — so writers queue rather than deadlock, and
a write the database still picks as a deadlock victim is retried, which is safe because it is
made of increments.

Tokens are stored in a `VARCHAR(191)` column on MySQL and PostgreSQL. A token longer than 191
characters, or one that is not valid UTF-8, is stored under a fixed-width surrogate key (its
first 150 characters, a `~`, and 40 hex digits of its SHA-1) on every driver, so a URL from a
regex tokenizer or a run of characters trains and scores like any other token; the encoding is
invisible to callers. Model and category names are limited to 191 characters by the same
columns.

The metadata row of a per-token model carries a `layoutVersion` (currently `2`). A model written
by 0.1.0 (layout 1, which kept the totals in the metadata and had no vocabulary or counters
tables) is upgraded in place, inside one transaction, the first time it is read — the two extra
tables are created automatically when `AutoCreateTable` is on; add them yourself otherwise. A
layout newer than the installed release is refused with `bayesian_storage_layout_unsupported`.

### `TRedisBayesianStorage`

One Redis key per model, plus a Redis set holding the index of model names (which is what
`list()` reads). Right for shared hosting and for multiple application servers that should see
the same trained model.

```xml
<storage class="TRedisBayesianStorage" Host="127.0.0.1" Port="6379" KeyPrefix="bayesian:" />
```

```php
'storage' => [
    'class' => 'TRedisBayesianStorage',
    'Host' => '127.0.0.1', 'Port' => 6379, 'KeyPrefix' => 'bayesian:',
],
```

Configurable: `Host`, `Port`, `Timeout`, `Password`, `Database`, `KeyPrefix`, `IndexKey`. Or
inject a fully configured client with `setRedis()`.

The connection opens lazily on the first save or load, and `AUTH`, `SELECT`, and write results
are all checked — a failed auth or a rejected write raises rather than being mistaken for
success (`bayesian_storage_redis_connect_failed`, `bayesian_storage_redis_write_failed`).

Unlike the file backend, a path separator in a model name is harmless here. A name is rejected
when it is empty, contains a null byte, or contains the sequence `:__`, which the per-token
sub-keys start with (a model named `m:__t:x` would share a key with token `x` of model `m`).
Two applications sharing one Redis must set both `KeyPrefix` and `IndexKey`: the index of model
names is a key of its own and does not follow the prefix.

Like the SQL backend, it can also store a model **per token** (`Mode="token"`): a metadata
string, two category hashes, a token set, one hash per token, and for Bernoulli and Complement
models a histogram hash, with the document's tokens
read back in a single pipelined round trip. Training is one `MULTI` of Lua scripts applying
`HINCRBY` increments, so a document's counts land atomically and any number of processes may
train at once ([Concurrency](#concurrency)); the vocabulary size is the cardinality of the token
set and the document total the sum of the category counts, so neither can drift. Deleting a
per-token model is one script too, so a trainer racing the delete cannot leave orphan keys. The
scripts touch keys derived from the model name, which a Redis Cluster does not allow: use a
single instance or Sentinel-managed replicas.

The metadata carries a `layoutVersion` (currently `2`); a model written by 0.1.0 is upgraded in
place on first read, and a newer layout is refused with `bayesian_storage_layout_unsupported`.

The important caveat is that per-token mode raises the *per-process* ceiling, not the
machine's — Redis still holds the whole model in RAM. It solves the cost of loading a large
model into every PHP request and the `memory_limit` wall; it does not give disk-bound models the
way SQL does.

```php
'storage' => [
    'class' => 'TRedisBayesianStorage',
    'Host' => '127.0.0.1', 'Port' => 6379, 'Mode' => 'token',
],
```

## Converting a model to per-token

A model trained and saved in `payload` mode can be moved to a per-token backend without
retraining, with `TBayesianModelConverter`. It loads the whole-payload model into a resident
vocabulary and re-saves it in the destination's per-token layout, reading the classifier variant
from the model's stored `kind` so a caller need not know which variant each model is.

```php
use Belisoful\Prado\Util\Bayesian\TBayesianModelConverter;

$converter = new TBayesianModelConverter();
$converter->convert($fileStorage, $sqlTokenStorage, 'comment-spam');   // one model
$converter->convertAll($fileStorage, $sqlTokenStorage);                // every model the source holds
```

To promote a model to per-token **within one database**, point a payload-mode and a token-mode
storage at the same connection and convert between them:

```php
$payload = new TSqlBayesianStorage();
$payload->setConnectionString('sqlite:/var/lib/myapp/bayesian.db');
$token = new TSqlBayesianStorage();
$token->setConnectionString('sqlite:/var/lib/myapp/bayesian.db');
$token->setMode('token');
$converter->convert($payload, $token, 'comment-spam');
```

The conversion is exact — the per-token model scores identically to the payload one. Only this
direction is supported: per-token to payload would mean enumerating the whole vocabulary, which
the per-token layout deliberately does not expose. Keep the payload copy, or retrain, if you need
to go back.

## Writing your own

Implement `IBayesianStorage` and honor the contract above — most importantly `load()` returning
`null` for both "unknown" and "unparseable", and `list()` returning sorted names. Nothing else in
the extension needs to know the backend exists:

```php
$classifier->setStorage(new MyStorage());
$classifier->setName('comment-spam');
$classifier->save();
```
