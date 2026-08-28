<?php

/**
 * TBayesianVocabulary class file.
 *
 * @author Brad Anderson <belisoful@icloud.com>
 * @link https://github.com/belisoful/prado-bayesian
 * @license https://github.com/pradosoft/prado/blob/master/LICENSE
 */

namespace Prado\Util\Bayesian;

/**
 * TBayesianVocabulary class.
 *
 * The vocabulary owned by a classifier: one {@see TBayesianCategory} per label, plus the
 * cross-category totals the classifiers need (the document-frequency map and the total
 * document count).  The vocabulary owns the categories; the classifier owns the vocabulary.
 *
 * Operations are training-time mutations and read-only accessors for the classification path.
 * The vocabulary does not compute probabilities itself — that is the classifier's job — but
 * it does keep the document-frequency map (the number of documents a token appeared in
 * across all categories), which {@see TFIdf} uses to weight the terms of a document being
 * classified.  Category names are array keys, so a purely numeric name is exposed as an
 * integer key by {@see getCategoryNames()}.
 *
 * @author Brad Anderson <belisoful@icloud.com>
 * @since 0.1.0
 */
class TBayesianVocabulary implements IBayesianVocabulary
{
	/** @var array<string, TBayesianCategory> The categories, keyed by name. */
	private array $_categories = [];

	/** @var array<string, int> The document frequency of each token (across all categories). */
	private array $_documentFrequency = [];

	/** @var int The total number of training documents across all categories. */
	private int $_totalDocuments = 0;

	/** @var int Bumped by every mutation of the vocabulary itself. */
	private int $_generation = 0;

	/**
	 * Returns the category with the given name, creating it on first access.
	 * @param string $name The category name.
	 * @return TBayesianCategory The category.
	 */
	public function getOrCreateCategory(string $name): TBayesianCategory
	{
		if (!isset($this->_categories[$name])) {
			$this->_categories[$name] = new TBayesianCategory($name);
			$this->_generation++;
		}
		return $this->_categories[$name];
	}

	/**
	 * Returns the mutation counter of the vocabulary itself.  It does not cover changes made
	 * directly to a {@see TBayesianCategory} this vocabulary holds — use
	 * {@see getStateSignature()} for a key that does.
	 * @return int The generation.
	 */
	public function getGeneration(): int
	{
		return $this->_generation;
	}

	/**
	 * Returns a token identifying the current state of the vocabulary and every category in
	 * it, for use as a cache key.
	 *
	 * Classifiers cache corpus-wide aggregates that cost O(|V|) to derive — Bernoulli's
	 * absent-token constant and Complement's per-category weight norm.  Those caches must be
	 * discarded whenever any underlying count moves, and {@see getCategories()} hands out live
	 * objects a caller can mutate directly, so a signature over the vocabulary's own counters
	 * would not be enough.  This combines the vocabulary's generation with each category's,
	 * which is O(categories) — a handful of integers — not O(|V|).
	 * @return string The signature.
	 */
	public function getStateSignature(): string
	{
		$parts = [$this->_generation, $this->_totalDocuments, count($this->_categories)];
		foreach ($this->_categories as $name => $category) {
			$parts[] = $name . '#' . $category->getGeneration();
		}
		return implode(':', $parts);
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
	 * Returns the categories in the order they were first added.
	 * @return TBayesianCategory[] The categories.
	 */
	public function getCategories(): array
	{
		return array_values($this->_categories);
	}

	/**
	 * Returns the category names in the order they were first added.
	 * @return string[] The category names.
	 */
	public function getCategoryNames(): array
	{
		return array_keys($this->_categories);
	}

	/**
	 * Returns whether any category has been added.
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
	 * Returns the number of distinct tokens seen anywhere in the corpus.  Every distinct token
	 * has exactly one entry in the document-frequency map, so its size is |V|.
	 * @return int The vocabulary size.
	 */
	public function getVocabularySize(): int
	{
		return count($this->_documentFrequency);
	}

	/**
	 * Returns whether a token was seen anywhere in the corpus.
	 * @param string $token The token.
	 * @return bool Whether the token is in the vocabulary.
	 */
	public function hasToken(string $token): bool
	{
		return isset($this->_documentFrequency[$token]);
	}

	/**
	 * Returns the number of training documents containing the token, across all categories.
	 * @param string $token The token.
	 * @return int The document frequency; 0 when the token is unknown.
	 */
	public function getTokenDocumentFrequency(string $token): int
	{
		return $this->_documentFrequency[$token] ?? 0;
	}

	/**
	 * Returns the total number of occurrences of the token across every category.  Summed on
	 * demand over the categories, which is O(categories) — a handful — not O(|V|).
	 * @param string $token The token.
	 * @return int The corpus-wide occurrence count.
	 */
	public function getTokenGlobalCount(string $token): int
	{
		$total = 0;
		foreach ($this->_categories as $category) {
			$total += $category->getTokenCount($token);
		}
		return $total;
	}

	/**
	 * Returns the total number of token occurrences across every category.
	 * @return int The corpus-wide token total.
	 */
	public function getGlobalTokenTotal(): int
	{
		$total = 0;
		foreach ($this->_categories as $category) {
			$total += $category->getTotalTokens();
		}
		return $total;
	}

	/**
	 * No-op: the whole vocabulary is already resident, so there is nothing to fetch.
	 * @param string[] $tokens The tokens that are about to be read.
	 */
	public function prefetch(array $tokens): void
	{
	}

	/**
	 * Returns true: this implementation holds the entire vocabulary in the process.
	 * @return bool Always true.
	 */
	public function getSupportsFullScan(): bool
	{
		return true;
	}

	/**
	 * Returns the document-frequency map (token -> number of documents it appeared in).
	 * @return array<string, int> The document frequencies.
	 */
	public function getDocumentFrequency(): array
	{
		return $this->_documentFrequency;
	}

	/**
	 * Records one training document in the named category, adding its token set to the
	 * category's per-token counts and the document-frequency map.  Duplicate tokens in the
	 * document are all counted (so a 5-token document contributes 5 to the category total),
	 * but a single document only contributes 1 to the document frequency of each unique
	 * token it contains.  The category's per-token document count is also incremented once
	 * per unique token (used by Bernoulli NB).
	 * @param string $category The category name.
	 * @param string[] $tokens The document's tokens (with multiplicity).
	 */
	public function addDocument(string $category, array $tokens): void
	{
		$cat = $this->getOrCreateCategory($category);
		$cat->addDocument();
		$this->_totalDocuments++;
		$this->_generation++;
		$seen = [];
		foreach ($tokens as $token) {
			$cat->addToken($token);
			$seen[$token] = true;
		}
		foreach ($seen as $token => $_) {
			$this->_documentFrequency[$token] = ($this->_documentFrequency[$token] ?? 0) + 1;
			$cat->addTokenDocument($token);
		}
	}

	/**
	 * Replaces the entire vocabulary (used when restoring from storage).
	 * @param TBayesianCategory[] $categories The categories.
	 * @param array<string, int> $documentFrequency The document frequencies.
	 * @param int $totalDocuments The total document count.
	 */
	public function setStats(array $categories, array $documentFrequency, int $totalDocuments): void
	{
		$this->_categories = [];
		foreach ($categories as $category) {
			$this->_categories[$category->getName()] = $category;
		}
		$this->_documentFrequency = $documentFrequency;
		$this->_totalDocuments = $totalDocuments < 0 ? 0 : $totalDocuments;
		$this->_generation++;
	}
}
