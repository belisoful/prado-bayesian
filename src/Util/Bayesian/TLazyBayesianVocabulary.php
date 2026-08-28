<?php

/**
 * TLazyBayesianVocabulary class file.
 *
 * @author Brad Anderson <belisoful@icloud.com>
 * @link https://github.com/belisoful/prado-bayesian
 * @license https://github.com/pradosoft/prado/blob/master/LICENSE
 */

namespace Prado\Util\Bayesian;

use Prado\Exceptions\TInvalidOperationException;
use Prado\Util\Bayesian\Storage\IBayesianTokenStorage;

/**
 * TLazyBayesianVocabulary class.
 *
 * A vocabulary that keeps its scalars and categories resident and reads per-token statistics
 * from an {@see IBayesianTokenStorage} on demand.  It is what lets a model outgrow the process
 * that classifies against it: a 100,000-token model costs tens of megabytes as a
 * {@see TBayesianVocabulary} and a single indexed query here.
 *
 * The read pattern is a batch, not a stream.  {@see prefetch()} issues one call for every token
 * of the document about to be scored and holds the result for the scoring pass; the per-token
 * accessors then answer from that batch without touching storage.  Scoring without prefetching
 * still produces correct scores, but every unfetched token reads as out-of-vocabulary — so the
 * classifiers call it, and anything else reading a token directly should too.
 *
 * What this class will not do is pretend to a completeness it does not have.  There is no way
 * to enumerate the vocabulary, so {@see getDocumentFrequency()} throws rather than returning
 * the prefetched slice, {@see getSupportsFullScan()} answers false, and the classifier
 * aggregates that need a full pass — Bernoulli's absent-token mass, Complement's weight norms —
 * are read from the model's stored metadata rather than recomputed.  A caller that needs the
 * whole model should load it into a {@see TBayesianVocabulary} instead.
 *
 * @author Brad Anderson <belisoful@icloud.com>
 * @since 0.1.0
 */
class TLazyBayesianVocabulary implements IBayesianVocabulary
{
	/** @var IBayesianTokenStorage The backend serving the per-token statistics. */
	private IBayesianTokenStorage $_storage;

	/** @var string The model name to read under. */
	private string $_model;

	/** @var array<string, TLazyBayesianCategory> The categories, keyed by name. */
	private array $_categories = [];

	/** @var int The total number of training documents. */
	private int $_totalDocuments = 0;

	/** @var int The number of distinct tokens in the corpus, from the stored metadata. */
	private int $_vocabularySize = 0;

	/** @var int The corpus-wide token total, summed from the category scalars. */
	private int $_globalTokenTotal = 0;

	/**
	 * @var array<string, array<string, array{count:int, docCount:int}>> The last prefetched
	 * batch, keyed by token then category.
	 */
	private array $_batch = [];

	/** @var array<string, bool> The tokens the current batch covers, whether or not they exist. */
	private array $_fetched = [];

	/** @var int Bumped whenever the model is re-read, for {@see getStateSignature()}. */
	private int $_generation = 0;

	/**
	 * Binds a vocabulary to a model in a per-token backend.
	 * @param IBayesianTokenStorage $storage The backend.
	 * @param string $model The model name.
	 */
	public function __construct(IBayesianTokenStorage $storage, string $model)
	{
		$this->_storage = $storage;
		$this->_model = $model;
	}

	/**
	 * Loads the model's scalars and categories, leaving the per-token statistics in storage.
	 * @param array<string, mixed> $meta The model metadata.
	 * @param array<string, array{documentCount:int, totalTokens:int}> $categories The scalars.
	 */
	public function initialize(array $meta, array $categories): void
	{
		$this->_totalDocuments = (int) ($meta['totalDocuments'] ?? 0);
		$this->_vocabularySize = (int) ($meta['vocabularySize'] ?? 0);
		$this->_categories = [];
		$this->_globalTokenTotal = 0;
		// Restore the training order the model was saved with; anything the metadata does not
		// name (a category added by a writer that did not update it) follows, so a category can
		// never be dropped by an out-of-date order list.
		$ordered = [];
		foreach (($meta['categoryOrder'] ?? []) as $name) {
			$name = (string) $name;
			if (isset($categories[$name])) {
				$ordered[$name] = $categories[$name];
			}
		}
		foreach ($categories as $name => $stats) {
			$ordered[(string) $name] = $stats;
		}
		$categories = $ordered;
		foreach ($categories as $name => $stats) {
			$documentCount = (int) ($stats['documentCount'] ?? 0);
			$totalTokens = (int) ($stats['totalTokens'] ?? 0);
			$this->_categories[(string) $name] = new TLazyBayesianCategory((string) $name, $this, $documentCount, $totalTokens);
			$this->_globalTokenTotal += $totalTokens;
		}
		$this->_batch = [];
		$this->_fetched = [];
		$this->_generation++;
	}

	/**
	 * Returns the model name this vocabulary reads under.
	 * @return string The model name.
	 */
	public function getModelName(): string
	{
		return $this->_model;
	}

	/**
	 * Returns a statistic of one token in one category from the current batch.  Used by
	 * {@see TLazyBayesianCategory}; a token outside the batch reads as 0.
	 * @param string $category The category name.
	 * @param string $token The token.
	 * @param string $field Either `count` or `docCount`.
	 * @return int The statistic.
	 */
	public function getPrefetchedCount(string $category, string $token, string $field): int
	{
		return (int) ($this->_batch[$token][$category][$field] ?? 0);
	}

	/**
	 * Fetches the statistics of the given tokens in one batched read.
	 *
	 * Tokens already covered by the current batch are not re-requested, so scoring the same
	 * document against several classifiers costs one query, not several.  The batch is
	 * cumulative within a model and is dropped whenever {@see initialize()} re-reads it.
	 * @param string[] $tokens The tokens about to be read.
	 */
	public function prefetch(array $tokens): void
	{
		$wanted = [];
		foreach ($tokens as $token) {
			$token = (string) $token;
			if (!isset($this->_fetched[$token])) {
				$wanted[$token] = true;
			}
		}
		if ($wanted === []) {
			return;
		}
		$rows = $this->_storage->loadTokens($this->_model, array_keys($wanted));
		foreach ($rows as $token => $categories) {
			$this->_batch[(string) $token] = $categories;
		}
		// Record every token asked for, not only those that came back: a token with no row is
		// out of vocabulary, and remembering that keeps it from being re-queried.
		foreach ($wanted as $token => $_) {
			$this->_fetched[(string) $token] = true;
		}
	}

	/**
	 * Returns the named category, or null if it does not exist.
	 * @param string $name The category name.
	 * @return ?TBayesianCategory The category, or null.
	 */
	public function getCategory(string $name): ?TBayesianCategory
	{
		return $this->_categories[$name] ?? null;
	}

	/**
	 * Returns the categories in the order they were loaded.
	 * @return TBayesianCategory[] The categories.
	 */
	public function getCategories(): array
	{
		return array_values($this->_categories);
	}

	/**
	 * Returns the category names in the order they were loaded.
	 * @return string[] The category names.
	 */
	public function getCategoryNames(): array
	{
		return array_keys($this->_categories);
	}

	/**
	 * Returns whether any category was loaded.
	 * @return bool Whether the vocabulary is empty.
	 */
	public function getIsEmpty(): bool
	{
		return $this->_categories === [];
	}

	/**
	 * Returns the total number of training documents across all categories.
	 * @return int The total.
	 */
	public function getTotalDocuments(): int
	{
		return $this->_totalDocuments;
	}

	/**
	 * Returns the number of distinct tokens in the corpus, from the stored metadata rather than
	 * by counting rows.
	 * @return int The vocabulary size.
	 */
	public function getVocabularySize(): int
	{
		return $this->_vocabularySize;
	}

	/**
	 * Returns whether the token was seen anywhere in the corpus, from the current batch.
	 * @param string $token The token.
	 * @return bool Whether the token is in the vocabulary.
	 */
	public function hasToken(string $token): bool
	{
		return isset($this->_batch[$token]);
	}

	/**
	 * Returns the number of documents containing the token, across all categories.
	 *
	 * A training document belongs to exactly one category, so the corpus-wide document
	 * frequency is the sum of the per-category document counts already in the batch — it needs
	 * no separate stored value, and so cannot disagree with one.
	 * @param string $token The token.
	 * @return int The document frequency; 0 when the token is unknown.
	 */
	public function getTokenDocumentFrequency(string $token): int
	{
		$total = 0;
		foreach ($this->_batch[$token] ?? [] as $stats) {
			$total += (int) ($stats['docCount'] ?? 0);
		}
		return $total;
	}

	/**
	 * Returns the total occurrences of the token across every category, summed from the batch.
	 * @param string $token The token.
	 * @return int The corpus-wide occurrence count.
	 */
	public function getTokenGlobalCount(string $token): int
	{
		$total = 0;
		foreach ($this->_batch[$token] ?? [] as $stats) {
			$total += (int) ($stats['count'] ?? 0);
		}
		return $total;
	}

	/**
	 * Returns the corpus-wide token total, summed from the resident category scalars.
	 * @return int The corpus-wide token total.
	 */
	public function getGlobalTokenTotal(): int
	{
		return $this->_globalTokenTotal;
	}

	/**
	 * Returns false: the vocabulary lives in storage and is not enumerable from here.
	 * @return bool Always false.
	 */
	public function getSupportsFullScan(): bool
	{
		return false;
	}

	/**
	 * Not available.  Returning the prefetched slice would answer "every token in the corpus"
	 * with a fraction of it, and nothing downstream could tell the difference.
	 * @throws TInvalidOperationException Always.
	 * @return array<string, int> Never returns.
	 */
	public function getDocumentFrequency(): array
	{
		throw new TInvalidOperationException('bayesian_vocabulary_full_scan_unavailable', $this->_model);
	}

	/**
	 * Returns a cache key covering the loaded state.  The per-token statistics are not resident,
	 * so the signature tracks the model identity and the scalars that change when it is
	 * re-read or trained.
	 * @return string The signature.
	 */
	public function getStateSignature(): string
	{
		return 'lazy:' . $this->_model . ':' . $this->_generation . ':' . $this->_totalDocuments
			. ':' . $this->_vocabularySize . ':' . count($this->_categories);
	}

	/**
	 * Not available: a storage-backed vocabulary is trained through
	 * {@see applyDocument()}, which writes the deltas out rather than accumulating them here.
	 * @param string $category The category name.
	 * @param string[] $tokens The document's tokens.
	 * @throws TInvalidOperationException Always.
	 */
	public function addDocument(string $category, array $tokens): void
	{
		throw new TInvalidOperationException('bayesian_vocabulary_readonly', $this->_model);
	}

	/**
	 * Records one training document by writing its deltas to storage and advancing the resident
	 * scalars to match.
	 *
	 * Only the document's own tokens are read and written, so the cost is proportional to the
	 * document rather than to the model — the reason a per-token model can be trained
	 * incrementally at all, where a whole-payload model has to be re-serialized in full.
	 * @param string $category The category name.
	 * @param string[] $tokens The document's tokens, with multiplicity.
	 * @param array<string, mixed> $meta The model metadata to write alongside the deltas.
	 * @return array<string, mixed> The metadata actually written, with the advanced scalars.
	 */
	public function applyDocument(string $category, array $tokens, array $meta): array
	{
		$counts = [];
		foreach ($tokens as $token) {
			$token = (string) $token;
			$counts[$token] = ($counts[$token] ?? 0) + 1;
		}
		// Read the document's tokens first: a token with no row anywhere is new to the corpus
		// and grows |V|, which nothing else could tell us without counting the whole table.
		$this->prefetch(array_keys($counts));
		$newTokens = 0;
		$deltas = [];
		foreach ($counts as $token => $count) {
			$token = (string) $token;
			if (!isset($this->_batch[$token])) {
				$newTokens++;
			}
			$deltas[$token] = ['count' => $count, 'docCount' => 1];
		}

		$existing = $this->_categories[$category] ?? null;
		$documentCount = ($existing?->getDocumentCount() ?? 0) + 1;
		$totalTokens = ($existing?->getTotalTokens() ?? 0) + array_sum($counts);

		$this->_totalDocuments++;
		$this->_vocabularySize += $newTokens;
		$meta['totalDocuments'] = $this->_totalDocuments;
		$meta['vocabularySize'] = $this->_vocabularySize;

		$this->_storage->applyDeltas($this->_model, $category, $deltas, $meta, [
			'documentCount' => $documentCount,
			'totalTokens' => $totalTokens,
		]);

		// Re-read the affected tokens so the batch reflects the write, and refresh the
		// category scalars from the values just persisted.
		foreach (array_keys($deltas) as $token) {
			unset($this->_fetched[(string) $token]);
		}
		$this->prefetch(array_keys($deltas));
		$this->_categories[$category] = new TLazyBayesianCategory($category, $this, $documentCount, $totalTokens);
		$this->_globalTokenTotal = 0;
		foreach ($this->_categories as $resident) {
			$this->_globalTokenTotal += $resident->getTotalTokens();
		}
		$this->_generation++;
		return $meta;
	}

	/**
	 * Not available: use {@see initialize()}, which takes the scalar form this class holds.
	 * @param TBayesianCategory[] $categories The categories.
	 * @param array<string, int> $documentFrequency The document frequencies.
	 * @param int $totalDocuments The total document count.
	 * @throws TInvalidOperationException Always.
	 */
	public function setStats(array $categories, array $documentFrequency, int $totalDocuments): void
	{
		throw new TInvalidOperationException('bayesian_vocabulary_readonly', $this->_model);
	}
}
