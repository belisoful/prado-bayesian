<?php

/**
 * TBayesianCategory class file.
 *
 * @author Brad Anderson <belisoful@icloud.com>
 * @link https://github.com/belisoful/prado-bayesian
 * @license https://github.com/pradosoft/prado/blob/master/LICENSE
 */

namespace Prado\Util\Bayesian;

/**
 * TBayesianCategory class.
 *
 * One category in a Naive Bayes classifier: a name, the count of documents that belong to it,
 * and the per-token occurrence counts (the "category profile").  The classifier's training
 * path increments these counts; the classification path reads them through log-probabilities
 * with Laplace smoothing.
 *
 * The category is mutated by {@see addDocument()} and {@see addToken()} during training; the
 * read-only views are {@see getDocumentCount()}, {@see getTokenCount()}, {@see getTotalTokens()},
 * and {@see getVocabularySize()}.  Callers should not assume the underlying maps are stable
 * across training.
 *
 * @author Brad Anderson <belisoful@icloud.com>
 * @since 0.1.0
 */
class TBayesianCategory
{
	/** @var string The category name. */
	private string $_name;

	/** @var int The number of training documents in this category. */
	private int $_documentCount = 0;

	/** @var array<string, int> The per-token occurrence counts. */
	private array $_tokenCounts = [];

	/** @var array<string, int> The per-token document count (number of documents containing the token). */
	private array $_documentTokenCounts = [];

	/** @var int The total number of token occurrences across all documents in this category. */
	private int $_totalTokens = 0;

	/**
	 * @var int Bumped by every mutation.  Classifiers cache aggregates that are expensive to
	 * derive from these counts (see {@see \Prado\Util\Bayesian\Classifier\TBernoulliNaiveBayes}),
	 * and a stale aggregate produces a silently wrong score rather than an error — so the
	 * cache key has to notice a change the category's own totals would not reveal, such as
	 * one token's count moving to another token.
	 */
	private int $_generation = 0;

	/**
	 * Creates an empty category.  The name is the label the classifier returns for it and is
	 * fixed for the lifetime of the object; counts start at zero and accumulate through
	 * training.
	 * @param string $name The category name.
	 */
	public function __construct(string $name)
	{
		$this->_name = $name;
	}

	/**
	 * Returns the category name.
	 * @return string The name.
	 */
	public function getName(): string
	{
		return $this->_name;
	}

	/**
	 * Returns the number of training documents in this category.
	 * @return int The document count.
	 */
	public function getDocumentCount(): int
	{
		return $this->_documentCount;
	}

	/**
	 * Returns the per-token occurrence counts (live map; do not mutate).
	 * @return array<string, int> The token counts.
	 */
	public function getTokenCounts(): array
	{
		return $this->_tokenCounts;
	}

	/**
	 * Returns the count for one token, or 0 when the token is unseen in this category.
	 * @param string $token The token.
	 * @return int The count.
	 */
	public function getTokenCount(string $token): int
	{
		return $this->_tokenCounts[$token] ?? 0;
	}

	/**
	 * Returns the total number of token occurrences in this category.
	 * @return int The total.
	 */
	public function getTotalTokens(): int
	{
		return $this->_totalTokens;
	}

	/**
	 * Returns the number of distinct tokens seen in this category.
	 * @return int The vocabulary size.
	 */
	public function getVocabularySize(): int
	{
		return count($this->_tokenCounts);
	}

	/**
	 * Returns the mutation counter of this category.  It increases on every change and is
	 * never reset, so a cache keyed on it cannot survive a mutation it did not see.
	 * @return int The generation.
	 */
	public function getGeneration(): int
	{
		return $this->_generation;
	}

	/**
	 * Increments the document count by one.
	 */
	public function addDocument(): void
	{
		$this->_documentCount++;
		$this->_generation++;
	}

	/**
	 * Increments the count of one token by the supplied amount (default 1) and adds the same
	 * amount to the total token count.
	 * @param string $token The token.
	 * @param int $count The increment; clamped to >= 1.
	 */
	public function addToken(string $token, int $count = 1): void
	{
		if ($count < 1) {
			$count = 1;
		}
		$this->_tokenCounts[$token] = ($this->_tokenCounts[$token] ?? 0) + $count;
		$this->_totalTokens += $count;
		$this->_generation++;
	}

	/**
	 * Increments the document count for a token (used by Bernoulli NB to track the number of
	 * documents in this category that contain the token, regardless of repetition).
	 * @param string $token The token.
	 */
	public function addTokenDocument(string $token): void
	{
		$this->_documentTokenCounts[$token] = ($this->_documentTokenCounts[$token] ?? 0) + 1;
		$this->_generation++;
	}

	/**
	 * Returns the number of documents in this category that contain the given token.
	 * @param string $token The token.
	 * @return int The per-token document count.
	 */
	public function getTokenDocumentCount(string $token): int
	{
		return $this->_documentTokenCounts[$token] ?? 0;
	}

	/**
	 * Returns the per-token document count map (live; do not mutate).
	 * @return array<string, int> The map.
	 */
	public function getTokenDocumentCounts(): array
	{
		return $this->_documentTokenCounts;
	}

	/**
	 * Replaces the underlying counts (used when restoring from storage).
	 * @param int $documentCount The document count.
	 * @param array<string, int> $tokenCounts The per-token occurrence counts.
	 * @param array<string, int> $tokenDocumentCounts The per-token document counts.
	 * @param int $totalTokens The total token occurrences.
	 */
	public function setStats(int $documentCount, array $tokenCounts, array $tokenDocumentCounts, int $totalTokens): void
	{
		$this->_documentCount = $documentCount < 0 ? 0 : $documentCount;
		$this->_tokenCounts = $tokenCounts;
		$this->_documentTokenCounts = $tokenDocumentCounts;
		$this->_totalTokens = $totalTokens < 0 ? 0 : $totalTokens;
		$this->_generation++;
	}
}
