<?php

/**
 * IBayesianClassifier interface file.
 *
 * @author Brad Anderson <belisoful@icloud.com>
 * @link https://github.com/belisoful/prado-bayesian
 * @license https://github.com/pradosoft/prado/blob/master/LICENSE
 */

namespace Belisoful\Prado\Util\Bayesian\Classifier;

use Belisoful\Prado\Util\Bayesian\IBayesianVocabulary;
use Belisoful\Prado\Util\Bayesian\Storage\IBayesianStorage;
use Belisoful\Prado\Util\Bayesian\TBayesianTrainingSet;
use Belisoful\Prado\Util\Bayesian\Tokenizer\IBayesianTokenizer;

/**
 * IBayesianClassifier interface.
 *
 * The classifier contract: a model that takes labeled documents during training and produces
 * probability distributions over categories at classification time.  Every variant
 * (multinomial, Bernoulli, complement) implements this same surface so the
 * {@see \Belisoful\Prado\Web\Services\TBayesianService} and {@see TBayesianRecommender} can swap them without code changes.
 *
 * Training accepts either a {@see TBayesianTrainingSet} or a quick-look shorthand
 * ({@see trainOne()} adds a single document to a category).  Classification accepts raw text
 * (which the configured tokenizer turns into features) or pre-tokenized arrays.  The
 * {@see score()} method returns a normalized probability distribution, the most flexible
 * output for downstream consumers like the recommender.
 *
 * Persistence is optional: setting an {@see IBayesianStorage} backend lets {@see save()} and
 * {@see load()} round-trip the trained state across processes and hosts.
 *
 * @author Brad Anderson <belisoful@icloud.com>
 * @since 0.1.0
 */
interface IBayesianClassifier
{
	/**
	 * Trains the classifier on a labeled training set.
	 *
	 * The training set's documents can be strings (tokenized by the configured tokenizer) or
	 * string arrays (used as pre-tokenized features).  Training is incremental: existing
	 * statistics are preserved and the new counts are added.
	 * @param TBayesianTrainingSet $set The training set.
	 */
	public function train(TBayesianTrainingSet $set): void;

	/**
	 * Adds a single training document to a category.  Convenience for the common
	 * "train per category" pattern; equivalent to building a one-document training set and
	 * passing it to {@see train()}.
	 * @param string $category The category name.
	 * @param string|string[] $document The document (text or pre-tokenized).
	 */
	public function trainOne(string $category, $document): void;

	/**
	 * Classifies a document and returns the most likely category.
	 * @param string|string[] $document The document (text or pre-tokenized).
	 * @return string The predicted category.
	 */
	public function classify($document): string;

	/**
	 * Returns the normalized posterior probability of every category for a document.
	 * @param string|string[] $document The document (text or pre-tokenized).
	 * @return array<string, float> The probabilities, keyed by category, summing to 1.0.
	 */
	public function score($document): array;

	/**
	 * Persists the trained state under the configured name.  Requires a storage backend.
	 */
	public function save(): void;

	/**
	 * Loads a previously-saved trained state by name.  Requires a storage backend.
	 * @param string $name The model name to load.
	 */
	public function load(string $name): void;

	/**
	 * Returns the configured name; the key under which the model is persisted.
	 * @return ?string The name, or null.
	 */
	public function getName(): ?string;

	/**
	 * Sets the name under which the model is persisted.
	 * @param ?string $value The name, or null to clear.
	 */
	public function setName(?string $value): void;

	/**
	 * Returns the configured tokenizer.
	 * @return IBayesianTokenizer The tokenizer.
	 */
	public function getTokenizer(): IBayesianTokenizer;

	/**
	 * Sets the tokenizer.
	 * @param IBayesianTokenizer $value The tokenizer.
	 */
	public function setTokenizer(IBayesianTokenizer $value): void;

	/**
	 * Returns the configured storage backend.
	 * @return ?IBayesianStorage The storage, or null.
	 */
	public function getStorage(): ?IBayesianStorage;

	/**
	 * Sets the storage backend.
	 * @param ?IBayesianStorage $value The storage, or null.
	 */
	public function setStorage(?IBayesianStorage $value): void;

	/**
	 * Returns the vocabulary owned by the classifier.
	 * @return IBayesianVocabulary The vocabulary.
	 */
	public function getVocabulary(): IBayesianVocabulary;

	/**
	 * Returns whether the classifier has been trained.
	 * @return bool Whether any training has been recorded.
	 */
	public function getIsTrained(): bool;
}
