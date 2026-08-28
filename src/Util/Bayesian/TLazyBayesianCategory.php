<?php

/**
 * TLazyBayesianCategory class file.
 *
 * @author Brad Anderson <belisoful@icloud.com>
 * @link https://github.com/belisoful/prado-bayesian
 * @license https://github.com/pradosoft/prado/blob/master/LICENSE
 */

namespace Prado\Util\Bayesian;

use Prado\Exceptions\TInvalidOperationException;

/**
 * TLazyBayesianCategory class.
 *
 * A category whose scalar totals are resident but whose per-token counts come from whatever
 * {@see TLazyBayesianVocabulary::prefetch()} last fetched.
 *
 * It is a {@see TBayesianCategory} so the classifiers need no separate code path: they call
 * {@see getTokenCount()} and {@see getTokenDocumentCount()} exactly as they do against a
 * resident model, and get the prefetched value.  A token that was never fetched reads as 0,
 * which is the same answer an unseen token gives — correct on the classification path, where
 * the tokens read are precisely the tokens prefetched.
 *
 * The whole-map accessors are the exception and deliberately throw.  Returning the prefetched
 * subset would be a plausible-looking answer to "give me every token in this category" that is
 * silently a fraction of the truth, and any total computed from it would be wrong without ever
 * raising.  See {@see getTokenCounts()}.
 *
 * @author Brad Anderson <belisoful@icloud.com>
 * @since 0.1.0
 */
class TLazyBayesianCategory extends TBayesianCategory
{
	/** @var TLazyBayesianVocabulary The vocabulary holding the prefetched statistics. */
	private TLazyBayesianVocabulary $_source;

	/**
	 * Creates a category backed by a lazily-populated vocabulary.
	 * @param string $name The category name.
	 * @param TLazyBayesianVocabulary $source The vocabulary that fetches the token statistics.
	 * @param int $documentCount The number of training documents in this category.
	 * @param int $totalTokens The total token occurrences in this category.
	 */
	public function __construct(string $name, TLazyBayesianVocabulary $source, int $documentCount, int $totalTokens)
	{
		parent::__construct($name);
		$this->_source = $source;
		parent::setStats($documentCount, [], [], $totalTokens);
	}

	/**
	 * Returns the occurrence count of one token in this category, from the prefetched batch.
	 * @param string $token The token.
	 * @return int The count; 0 when the token is unseen or was not prefetched.
	 */
	public function getTokenCount(string $token): int
	{
		return $this->_source->getPrefetchedCount($this->getName(), $token, 'count');
	}

	/**
	 * Returns the number of this category's documents containing one token, from the
	 * prefetched batch.
	 * @param string $token The token.
	 * @return int The document count; 0 when the token is unseen or was not prefetched.
	 */
	public function getTokenDocumentCount(string $token): int
	{
		return $this->_source->getPrefetchedCount($this->getName(), $token, 'docCount');
	}

	/**
	 * Not available: the point of this class is that the category's tokens are not resident.
	 * @throws TInvalidOperationException Always.
	 * @return array<string, int> Never returns.
	 */
	public function getTokenCounts(): array
	{
		throw new TInvalidOperationException('bayesian_vocabulary_full_scan_unavailable', $this->getName());
	}

	/**
	 * Not available: the point of this class is that the category's tokens are not resident.
	 * @throws TInvalidOperationException Always.
	 * @return array<string, int> Never returns.
	 */
	public function getTokenDocumentCounts(): array
	{
		throw new TInvalidOperationException('bayesian_vocabulary_full_scan_unavailable', $this->getName());
	}

	/**
	 * Not available: the number of distinct tokens in this category cannot be counted without
	 * enumerating them.
	 * @throws TInvalidOperationException Always.
	 * @return int Never returns.
	 */
	public function getVocabularySize(): int
	{
		throw new TInvalidOperationException('bayesian_vocabulary_full_scan_unavailable', $this->getName());
	}
}
