<?php

/**
 * TBayesianTrainingSet class file.
 *
 * @author Brad Anderson <belisoful@icloud.com>
 * @link https://github.com/belisoful/prado-bayesian
 * @license https://github.com/pradosoft/prado/blob/master/LICENSE
 */

namespace Prado\Util\Bayesian;

/**
 * TBayesianTrainingSet class.
 *
 * An iterable labeled training set: a category name and a list of documents per category.
 * Each document is either a string (which the classifier will tokenize through its tokenizer
 * at training time) or an array of pre-tokenized strings (which it will use as-is).
 *
 * The training set is the input the classifier consumes from {@see IBayesianClassifier::train()};
 * keeping it as a distinct object lets a classifier accept an in-memory list, a stream of
 * documents from a CSV, or a pre-tokenized artifact with the same call shape.
 *
 * @author Brad Anderson <belisoful@icloud.com>
 * @since 0.1.0
 */
class TBayesianTrainingSet
{
	/** @var array<string, array<int, string|string[]>> The documents, keyed by category. */
	private array $_entries = [];

	/**
	 * Adds a document to a category.  The document can be a string (raw text) or a string
	 * array (pre-tokenized).
	 * @param string $category The category name.
	 * @param string|string[] $document The document.
	 */
	public function add(string $category, $document): void
	{
		$this->_entries[$category][] = $document;
	}

	/**
	 * Returns the documents for a category, in insertion order.
	 * @param string $category The category name.
	 * @return array<int, string|string[]> The documents.
	 */
	public function getCategoryDocuments(string $category): array
	{
		return $this->_entries[$category] ?? [];
	}

	/**
	 * Returns the categories in insertion order.
	 * @return string[] The category names.
	 */
	public function getCategories(): array
	{
		return array_keys($this->_entries);
	}

	/**
	 * Returns whether any documents have been added.
	 * @return bool Whether the set is empty.
	 */
	public function getIsEmpty(): bool
	{
		return $this->_entries === [];
	}

	/**
	 * Returns the total number of documents across all categories.
	 * @return int The document count.
	 */
	public function getTotalDocuments(): int
	{
		$total = 0;
		foreach ($this->_entries as $documents) {
			$total += count($documents);
		}
		return $total;
	}

	/**
	 * Iterates the (category, document) pairs, in insertion order.
	 * @return \Generator<string, string|string[]> The generator.
	 */
	public function each(): \Generator
	{
		foreach ($this->_entries as $category => $documents) {
			foreach ($documents as $document) {
				yield $category => $document;
			}
		}
	}
}
