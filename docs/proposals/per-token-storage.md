# Proposal: per-token storage lookup

**Status:** implemented for `TSqlBayesianStorage` and `TRedisBayesianStorage` (both `Mode="token"`),
including the Bernoulli/Complement optimizations of Phase 1 and incremental training. This
document is kept as the design record; the shipped behaviour is documented in
[storage.md](../storage.md). Still open: the file backend deliberately does not get a per-token
mode (use SQLite through `TSqlBayesianStorage` instead), and there is no blob-to-per-token
converter yet.

Today every backend serializes a model to one JSON payload and `load()` decodes all of it before
the first classification. This proposal adds an *optional* second storage mode in which a
classifier reads only the statistics for the tokens in the document being classified.

---

## 1. The case, measured

A 100,000-token vocabulary over 2 categories, compared against a prototype SQLite schema holding
the same statistics as 200,000 rows:

| | Blob (today) | Per-token (prototype) |
|---|---:|---:|
| On disk | 15.4 MB JSON | 15.8 MB SQLite |
| Cost before first classification | **116.5 ms** to load and decode | 0 |
| Cost per classification | ~0 once loaded | **0.13 ms** (one query, 162 rows) |
| PHP memory held | **44.1 MB** | **83 KB** |

The decisive detail is PHP's execution model. A PRADO application is share-nothing per request,
so a web deployment pays that 116.5 ms **and** 44.1 MB on *every request* that classifies
anything. Per-token lookup replaces a fixed six-figure cost with one indexed query proportional
to the document.

The enabling fact is that documents are small even when verbose. Distinct tokens per document,
measured with this extension's own tokenizers:

| Document | Word | Char 3-gram | Word + 3-gram chain |
|---|---:|---:|---:|
| Short comment (39 chars) | 6 | 38 | 43 |
| Email (~900 chars) | 9 | 73 | 81 |

Repetition collapses under `array_unique`, so a batched lookup needs tens of bind parameters, not
thousands. (SQLite on this machine accepted 28,800; the historical floor is 999. Chunk at ~500
for portability.)

## 2. What the classifiers actually need

This is where the proposal gets interesting: the three models have materially different
requirements, and one of them cannot be made per-token without an algebraic change.

| | Per-document-token lookups | Corpus-wide requirement |
|---|---|---|
| **Multinomial** (`TNaiveBayesClassifier`, `TMultinomialNaiveBayes`) | `df(t)`, `count(t,c)` | Scalars only: `N`, `\|V\|`, and per-category `documentCount`, `totalTokens` |
| **Complement** (`TComplementNaiveBayes`) | `df(t)`, `globalCount(t)`, `count(t,c)` | Scalars, **plus a per-category L1 norm over the full vocabulary** |
| **Bernoulli** (`TBernoulliNaiveBayes`) | `docCount(t,c)` | **Iterates the entire vocabulary, per category, per classification** |

**Multinomial is ready as-is.** `logLikelihood()` loops over the document's tokens and needs
nothing else but scalars. Note `|V|` is currently obtained as `count($docFreq)` — it must become
a stored scalar rather than a map to be counted.

**Complement needs one stored aggregate.** Its document loop is already per-token, but
`categoryNorm()` sums `abs(log(…))` across the whole vocabulary for each category. It is already
cached in memory and invalidated on training or alpha change; the change is to persist it as a
per-category scalar under the same invalidation rule.

**Bernoulli needs the standard identity rewrite.** Absence is evidence, so it deliberately walks
the global feature set:

```php
foreach ($vocabulary as $token => $_) {
    $logSum += isset($present[$token]) ? log($p) : log(1.0 - $p);
}
```

Per-token lookup cannot serve that loop — it would turn one bulk read into `|V| × K` lookups,
which is far worse than today. The fix is to split the sum into a constant and a correction:

```
logLikelihood(d, c) = S_c + Σ_{t ∈ d ∩ V} [ log p(t|c) − log(1 − p(t|c)) ]
where  S_c = Σ_{t ∈ V} log(1 − p(t|c))
```

`S_c` depends only on the training statistics and alpha, so it is computed once and stored per
category exactly like Complement's norm. The document loop then touches only the document's
tokens. This is worth doing **regardless of this proposal** — it makes in-memory Bernoulli
O(document) instead of O(vocabulary) per classification.

## 3. The seam

The blocker is not storage, it is that `TNaiveBayesClassifier` owns an eager
`TBayesianVocabulary` and `IBayesianClassifier::getVocabulary(): TBayesianVocabulary` exposes the
concrete class publicly. `TBayesianCategory::getTokenCounts()` returns the entire map — a lazy
implementation cannot honor that cheaply.

Two decisions follow, and they are the substance of the work:

**a) A second, optional storage interface.** Do not widen `IBayesianStorage`; that would break
every implementer. Add a capability interface that backends may also implement, and have
classifiers feature-detect with `instanceof`:

```php
interface IBayesianTokenStorage extends IBayesianStorage
{
    /** Model-level scalars: totalDocuments, vocabularySize, kind, tokenizer config. */
    public function loadMeta(string $name): ?array;

    /** Per-category scalars: documentCount, totalTokens, and the cached aggregates. */
    public function loadCategories(string $name): array;

    /**
     * THE batch call. One round trip for the whole document.
     * @return array<string, array{df:int, global:int, categories:array<string, array{count:int, docCount:int}>}>
     */
    public function loadTokens(string $name, array $tokens): array;

    /** Incremental write: apply one document's deltas without rewriting the model. */
    public function applyDeltas(string $name, string $category, array $tokenDeltas, array $meta): void;
}
```

`loadTokens()` taking an array is the single most important design point. A per-token method
would be a textbook N+1 — `D × K` queries per classification instead of one.

**b) `getVocabulary()` must return an interface.** Introduce `IBayesianVocabulary` with the
methods the classifiers actually call, keep `TBayesianVocabulary` as the eager implementation, and
add a lazy one backed by `IBayesianTokenStorage`. This is a **breaking change to
`IBayesianClassifier`**, so it belongs in 0.2.0, not a patch. The lazy implementation should
throw a clear exception from whole-map accessors like `getDocumentFrequency()` rather than
silently materializing a 100k-entry array.

## 4. Backend viability

### SQL — yes, and it is the natural fit

```sql
CREATE TABLE bayesian_tokens (
    model     VARCHAR(191) NOT NULL,
    token     VARCHAR(191) NOT NULL,
    category  VARCHAR(191) NOT NULL,
    cnt       BIGINT NOT NULL,
    doccnt    BIGINT NOT NULL,
    PRIMARY KEY (model, token, category)
);
CREATE INDEX idx_bayesian_tokens_lookup ON bayesian_tokens (model, token);

CREATE TABLE bayesian_categories (         -- per-category scalars and cached aggregates
    model VARCHAR(191), category VARCHAR(191),
    doc_count BIGINT, total_tokens BIGINT,
    bernoulli_s DOUBLE NULL, complement_norm DOUBLE NULL,
    PRIMARY KEY (model, category)
);
-- bayesian_models keeps its existing row for kind, tokenizer config, and model scalars.
```

One `WHERE model = ? AND token IN (…)` returns every category's row for every document token in
a single round trip. Verified at 0.13 ms for 81 tokens against 200k rows. Model size becomes
bounded by disk, not by PHP memory — the actual answer to the original question.

Driver notes: the existing driver-aware DDL and upsert logic extends directly. Chunk the `IN`
list at ~500. `doccnt` must stay a separate column, not a second query.

### Redis — yes, via hashes and pipelining

```
{prefix}{model}:t:{token}   HASH   field = category, value = "cnt:doccnt"
{prefix}{model}:cat         HASH   field = category, value = packed scalars
{prefix}{model}:meta        HASH   totalDocuments, vocabularySize, kind, tokenizer
{prefix}{model}:index       SET    (existing) the model-name index
```

Batch with `multi(Redis::PIPELINE)` — `D` × `HGETALL` in one round trip. Training uses `HINCRBY`,
which is atomic per field and removes the read-modify-write race the blob backend has today.
Small hashes use Redis's listpack encoding, so per-token overhead is modest.

**One caveat to state plainly:** this moves the model out of *PHP's* memory, not out of memory
altogether. Redis still holds it all in RAM, and the 512 MB-per-value ceiling is replaced by the
Redis instance's own capacity. It solves the per-request PHP cost and the `memory_limit` wall; it
does not give unbounded models the way SQL does.

### File — no; recommend SQLite instead

Three designs, none good. One file per token destroys the filesystem at 100k tokens (inode and
dentry pressure, and `readdir` for `list()`). Sharding by token-hash prefix still decodes a whole
shard per lookup, so it only divides the constant. A DBM handle via `dba_*` depends on an
extension that is off by default in most builds.

The correct file-backed per-token store already exists and is already in this package: SQLite
through `TSqlBayesianStorage` with `sqlite:/path/to/models.db`. **Recommendation: do not build
per-token file storage.** Document the SQLite route in `docs/storage.md`, and keep
`TFileBayesianStorage` as the simple, dependency-free blob option it is good at being.

### Memory — out of scope, correctly

`TMemoryBayesianStorage` holds a PHP array; per-token lookup is what a PHP array already is.

## 5. Training

The write path is the quiet win. Today `save()` re-serializes and rewrites the entire model —
`trainOne()` on a 100k-token model rewrites 15.4 MB. With `applyDeltas()`, training one document
writes one row per distinct token (tens), plus one category row and one meta row.

Requirements: batch the upserts into one multi-row statement; wrap in a transaction so a
half-applied document cannot corrupt the model; recompute and store `bernoulli_s` /
`complement_norm` after training, since both are `Θ(|V|)` and must not be recomputed per
classification. That last point is the main correctness risk in the whole proposal — a stale
aggregate produces silently wrong scores rather than an error. It needs an explicit generation
counter or a hash of `(totalDocuments, categoryCount, alpha)` stored alongside.

## 6. Compatibility and phasing

Existing models keep working: the blob format stays, `IBayesianStorage` is untouched, and a
classifier with a blob-only backend behaves exactly as it does now. A model must be written in
one mode or the other; provide a converter rather than trying to read both.

Suggested phasing:

1. **0.1.x** — Bernoulli's `S_c` identity rewrite and Complement's persisted norm as pure
   in-memory optimizations. No API change, immediately useful, and it de-risks the hard part.
2. **0.2.0** — `IBayesianVocabulary`, the lazy implementation, `IBayesianTokenStorage`, and
   `TSqlBayesianStorage` per-token mode behind a `Mode` property defaulting to blob.
3. **0.2.x** — Redis per-token mode; a `bayesian:convert` style tool for blob → per-token.

Rough effort, excluding review: (1) is 1–2 days with tests. (2) is the bulk — 1–2 weeks, most of
it in the vocabulary abstraction and in test coverage for the three models × two storage modes.
(3) is a few days once the seam exists.

## 7. Open questions

- Should `getVocabulary()` on a lazy model throw, or return a bounded view? Throwing is honest but
  breaks any caller doing introspection.
- Does per-token mode need a read-through cache for hot tokens within one request, or is one
  batched query per classification enough? (Probably enough; measure before adding.)
- `TFIdf` needs `df(t)` per token, already covered — but confirm no other call site expects the
  whole `documentFrequency` map.
- Concurrent training from multiple processes: SQL upserts and Redis `HINCRBY` are safe per row,
  but the cached aggregates need a locking or regeneration story.
