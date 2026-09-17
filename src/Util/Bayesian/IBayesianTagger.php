<?php

/**
 * IBayesianTagger interface file.
 *
 * @author Brad Anderson <belisoful@icloud.com>
 * @link https://github.com/belisoful/prado-bayesian
 * @license https://github.com/pradosoft/prado/blob/master/LICENSE
 */

namespace Belisoful\Prado\Util\Bayesian;

use Belisoful\Prado\Util\Bayesian\Classifier\TNaiveBayesClassifier;

/**
 * IBayesianTagger interface.
 *
 * The multi-label contract: a document carries any number of labels (tags), and tagging a
 * new document returns the probability of each label independently — a document can be
 * `php` and `security` at once, or nothing at all.  This is what a classifier cannot say: its
 * scores are a distribution over categories that sum to one, so two applicable labels split
 * the mass between them.
 *
 * {@see TBayesianTagger} implements it as one-versus-rest Naive Bayes over a single shared
 * model, so training a document costs one write per label plus one, and every storage backend
 * works unchanged.
 *
 * @author Brad Anderson <belisoful@icloud.com>
 * @since 0.2.0
 */
interface IBayesianTagger
{
	/**
	 * Records a training document under each of its labels.  An empty label list is a valid
	 * document: it is a negative example for every label.
	 * @param string[] $labels The document's labels.
	 * @param string|string[] $document The document (text or pre-tokenized).
	 */
	public function train(array $labels, $document): void;

	/**
	 * Withdraws a training document from each of its labels; the inverse of {@see train()}.
	 * @param string[] $labels The labels the document was trained under.
	 * @param string|string[] $document The document, as it was trained.
	 */
	public function untrain(array $labels, $document): void;

	/**
	 * Returns the probability of every known label for a document, independently of one
	 * another, in label order.
	 * @param string|string[] $document The document.
	 * @return array<string, float> The probability of each label, in [0, 1].
	 */
	public function probabilities($document): array;

	/**
	 * Returns the labels whose probability reaches the threshold, highest first.
	 * @param string|string[] $document The document.
	 * @return array<string, float> The tags and their probabilities, highest first.
	 */
	public function tag($document): array;

	/**
	 * Returns every label the tagger has seen a document for.
	 * @return string[] The labels.
	 */
	public function getLabels(): array;

	/**
	 * Returns the classifier holding the label statistics.
	 * @return TNaiveBayesClassifier The classifier.
	 */
	public function getClassifier(): TNaiveBayesClassifier;

	/**
	 * Sets the classifier holding the label statistics (and, through it, the tokenizer, the
	 * storage and the model name).
	 * @param TNaiveBayesClassifier $value The classifier.
	 */
	public function setClassifier(TNaiveBayesClassifier $value): void;

	/**
	 * Returns the probability a label needs to be returned by {@see tag()}.
	 * @return float The threshold, in [0, 1].
	 */
	public function getThreshold(): float;

	/**
	 * Sets the probability a label needs to be returned by {@see tag()}.
	 * @param float $value The threshold, in [0, 1].
	 */
	public function setThreshold(float $value): void;

	/**
	 * Returns the most tags {@see tag()} returns; 0 for no limit.
	 * @return int The limit.
	 */
	public function getMaxTags(): int;

	/**
	 * Sets the most tags {@see tag()} returns; 0 for no limit.
	 * @param int $value The limit.
	 */
	public function setMaxTags(int $value): void;
}
