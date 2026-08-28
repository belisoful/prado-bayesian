<?php

/**
 * TMultinomialNaiveBayes class file.
 *
 * @author Brad Anderson <belisoful@icloud.com>
 * @link https://github.com/belisoful/prado-bayesian
 * @license https://github.com/pradosoft/prado/blob/master/LICENSE
 */

namespace Belisoful\Prado\Util\Bayesian\Classifier;

/**
 * TMultinomialNaiveBayes class.
 *
 * The multinomial event model: token occurrences (not presence/absence) drive the
 * likelihood, and a document's classification score is the sum of per-token log-probabilities
 * weighted by occurrence count.  This is the most common Naive Bayes formulation for text
 * and is well-suited to long documents where word frequency carries information.
 *
 * Mathematically equivalent to {@see TNaiveBayesClassifier} — provided as a distinct class so
 * the saved state carries a "multinomial" kind marker, letting you mix classifier variants
 * in the same storage backend and load the right one by name.
 *
 * @author Brad Anderson <belisoful@icloud.com>
 * @since 0.1.0
 */
class TMultinomialNaiveBayes extends TNaiveBayesClassifier
{
	/**
	 * {@inheritDoc}
	 *
	 * The multinomial model uses the base implementation unchanged; the only override is the
	 * kind marker in the saved state.
	 * @return string The kind marker `multinomial-naive-bayes`.
	 */
	protected function getKind(): string
	{
		return 'multinomial-naive-bayes';
	}
}
