<?php

/**
 * IBayesianRecommender interface file.
 *
 * @author Brad Anderson <belisoful@icloud.com>
 * @link https://github.com/belisoful/prado-bayesian
 * @license https://github.com/pradosoft/prado/blob/master/LICENSE
 */

namespace Belisoful\Prado\Util\Bayesian;

use Belisoful\Prado\Util\Bayesian\Classifier\IBayesianClassifier;

/**
 * IBayesianRecommender interface.
 *
 * The recommender contract: given the items a user has interacted with (the "context") and a
 * set of candidate items, return the candidates ranked by predicted affinity.  The default
 * implementation is built on top of {@see IBayesianClassifier} — train the classifier on
 * positive/negative interactions, then ask it to score each candidate against the context.
 *
 * Implementations may swap the underlying classifier family (multinomial vs Bernoulli) and
 * the scoring strategy (raw P(like) vs log-odds) without changing callers' code.
 *
 * @author Brad Anderson <belisoful@icloud.com>
 * @since 0.1.0
 */
interface IBayesianRecommender
{
	/**
	 * Scores a list of candidate items against a user context and returns them ranked
	 * from best to worst.
	 *
	 * The result is a map of candidate identifier to score; the keys appear in descending
	 * score order.  Candidates are treated as a unique set — a repeated identifier yields a
	 * single entry.  An empty candidate list throws.
	 * @param string[] $context The items the user has interacted with (positive or negative).
	 * @param string[] $candidates The items to rank.
	 * @throws \Prado\Exceptions\TInvalidDataValueException When the candidate list is empty.
	 * @return array<string, float> The ranked scores, highest first.
	 */
	public function recommend(array $context, array $candidates): array;

	/**
	 * Returns the underlying classifier that drives the recommender.
	 * @return IBayesianClassifier The classifier.
	 */
	public function getClassifier(): IBayesianClassifier;

	/**
	 * Sets the underlying classifier.
	 * @param IBayesianClassifier $value The classifier.
	 */
	public function setClassifier(IBayesianClassifier $value): void;

	/**
	 * Returns the category treated as a positive interaction.
	 * @return string The positive category.
	 */
	public function getPositiveCategory(): string;

	/**
	 * Sets the positive category.
	 * @param string $value The positive category.
	 */
	public function setPositiveCategory(string $value): void;
}
