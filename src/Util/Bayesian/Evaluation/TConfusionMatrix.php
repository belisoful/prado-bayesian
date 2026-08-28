<?php

/**
 * TConfusionMatrix class file.
 *
 * @author Brad Anderson <belisoful@icloud.com>
 * @link https://github.com/belisoful/prado-bayesian
 * @license https://github.com/pradosoft/prado/blob/master/LICENSE
 */

namespace Prado\Util\Bayesian\Evaluation;

use Prado\Exceptions\TInvalidDataValueException;

/**
 * TConfusionMatrix class.
 *
 * A confusion matrix for evaluating a classifier: the count of (expected, predicted) pairs
 * per label, registered up front.  The matrix is built incrementally by {@see record()} from
 * a labeled test set, and {@see getCounts()} exposes the raw counts for
 * {@see TBayesianMetrics} to derive precision, recall, F1, and accuracy.
 *
 * Labels are fixed in the constructor (duplicates are collapsed, order preserved); every
 * (expected, predicted) cell over those labels starts at zero.  Recording an unknown label
 * throws — silently dropping it would distort the metrics — so misconfiguration is caught at
 * evaluation time.  Labels are strings; a purely numeric label such as `"1"` is normalized to
 * a string but PHP still exposes it as an integer array key in {@see getCounts()}.
 *
 * @author Brad Anderson <belisoful@icloud.com>
 * @since 0.1.0
 */
class TConfusionMatrix
{
	/** @var string[] The registered labels, in declaration order. */
	private array $_labels;

	/** @var array<string, array<string, int>> The raw counts, keyed by [expected][predicted]. */
	private array $_counts = [];

	/** @var int The total number of recorded predictions. */
	private int $_total = 0;

	/**
	 * Builds an empty matrix over a fixed label set.  Duplicate labels are collapsed, and the
	 * given order is preserved so row/column positions stay stable for display.
	 * @param string[] $labels The labels (>= 1); the order is preserved for matrix access.
	 * @throws TInvalidDataValueException When fewer than one label is provided.
	 */
	public function __construct(array $labels)
	{
		if ($labels === []) {
			throw new TInvalidDataValueException('bayesian_confusion_label_required');
		}
		$normalized = [];
		foreach ($labels as $label) {
			$name = (string) $label;
			if (!in_array($name, $normalized, true)) {
				$normalized[] = $name;
			}
		}
		foreach ($normalized as $row) {
			$this->_counts[$row] = [];
			foreach ($normalized as $col) {
				$this->_counts[$row][$col] = 0;
			}
		}
		$this->_labels = $normalized;
	}

	/**
	 * Returns the registered labels, in declaration order.
	 * @return string[] The labels.
	 */
	public function getLabels(): array
	{
		return $this->_labels;
	}

	/**
	 * Records one (expected, predicted) pair.
	 * @param string $expected The expected label.
	 * @param string $predicted The predicted label.
	 * @throws TInvalidDataValueException When either label is not registered.
	 */
	public function record(string $expected, string $predicted): void
	{
		if (!isset($this->_counts[$expected])) {
			throw new TInvalidDataValueException('bayesian_confusion_label_unknown', $expected);
		}
		if (!isset($this->_counts[$expected][$predicted])) {
			throw new TInvalidDataValueException('bayesian_confusion_label_unknown', $predicted);
		}
		$this->_counts[$expected][$predicted]++;
		$this->_total++;
	}

	/**
	 * Returns the raw counts matrix, keyed by [expected][predicted].
	 * @return array<string, array<string, int>> The counts.
	 */
	public function getCounts(): array
	{
		return $this->_counts;
	}

	/**
	 * Returns the count for one (expected, predicted) cell; 0 when no predictions have been recorded.
	 * @param string $expected The expected label.
	 * @param string $predicted The predicted label.
	 * @return int The cell count.
	 */
	public function getCell(string $expected, string $predicted): int
	{
		return $this->_counts[$expected][$predicted] ?? 0;
	}

	/**
	 * Returns the total number of recorded predictions.
	 * @return int The total.
	 */
	public function getTotal(): int
	{
		return $this->_total;
	}
}
