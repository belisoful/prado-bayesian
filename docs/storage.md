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

Measured on this codebase, with every category having seen the whole vocabulary:

| Vocabulary | Categories | JSON payload | Loaded in PHP |
|---:|---:|---:|---:|
| 5,000 | 2 | 361 KB | 1.6 MB |
| 5,000 | 10 | 1.5 MB | 6.6 MB |
| 20,000 | 2 | 1.5 MB | 6.3 MB |
| 20,000 | 10 | 6.2 MB | 26.3 MB |
| 100,000 | 2 | 7.6 MB | 25.0 MB |

The payload runs **30–40 bytes per token-per-category**: each category keeps its own occurrence
and document counts for every token it has seen, plus one corpus-wide document-frequency map.
The decoded PHP structure is **3–4× the JSON**, because hash-table entries cost far more than
text. Budget both at once — `json_decode()` holds the string and the growing array
simultaneously.

So size scales with vocabulary **times** categories. Ten categories over the same words is five
times the model of two. The effective lever is the feature space, not the backend: raise
`MinLength`, supply `StopWords`, or prefer word tokens over character n-grams, which produce far
more distinct features.

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
reader never sees a partial file and two concurrent saves of the same model cannot interleave.
The directory is created on demand; an unset or empty `Directory` throws
`bayesian_storage_directory_required`, and one that cannot be created or written throws
`bayesian_storage_directory_unwritable`.

Model names are validated: a name containing a path separator or a null byte is rejected
(`bayesian_storage_name_invalid`) rather than resolved, so a name like `../../etc/passwd` cannot
escape the directory.

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
validated against `[A-Za-z_][A-Za-z0-9_]*` and throws `bayesian_storage_table_invalid`
otherwise, which keeps it from becoming an injection vector.

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

Unlike the file backend, a path separator in a model name is harmless here, so only empty names
and null bytes are rejected.

Like the SQL backend, it can also store a model **per token** (`Mode="token"`): a metadata
string, a categories hash, and one hash per token, with the document's tokens read back in a
single pipelined round trip. Incremental training uses `HINCRBY`, so a document's counts land
atomically without a read-modify-write. The important caveat is that this raises the *per-process*
ceiling, not the machine's — Redis still holds the whole model in RAM. It solves the cost of
loading a large model into every PHP request and the `memory_limit` wall; it does not give
disk-bound models the way SQL does.

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
