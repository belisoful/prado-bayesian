<?php

/**
 * TBayesianMetrics class file.
 *
 * @author Brad Anderson <belisoful@icloud.com>
 * @link https://github.com/belisoful/prado-bayesian
 * @license https://github.com/pradosoft/prado/blob/master/LICENSE
 */

namespace Belisoful\Prado\Util\Bayesian\Evaluation;

use Prado\Exceptions\TInvalidDataValueException;

/**
 * TBayesianMetrics class.
 *
 * Derives precision, recall, F1, and accuracy from a {@see TConfusionMatrix}.  The class is
 * stateless after construction — the matrix is the input, and each method returns one number
 * (or NaN where the math is undefined, e.g. precision of a label that was never predicted).
 *
 * Macro averages weight every label equally, useful for class-imbalanced data; micro averages
 * weight every prediction equally (and equal accuracy in the single-label case).
 *
 * @author Brad Anderson <belisoful@icloud.com>
 * @since 0.1.0
 */
class TBayesianMetrics
{
	/** @var TConfusionMatrix The matrix the metrics are computed from. */
	private TConfusionMatrix $_matrix;

	/**
	 * Wraps a confusion matrix so its counts can be read as evaluation metrics.  The matrix is
	 * held by reference and read on demand, so counts tallied after construction are included.
	 * @param TConfusionMatrix $matrix The confusion matrix.
	 */
	public function __construct(TConfusionMatrix $matrix)
	{
		$this->_matrix = $matrix;
	}

	/**
	 * Returns the underlying confusion matrix.
	 * @return TConfusionMatrix The matrix.
	 */
	public function getMatrix(): TConfusionMatrix
	{
		return $this->_matrix;
	}

	/**
	 * Validates a per-label metric argument.
	 * @param string $label The label.
	 * @throws TInvalidDataValueException When the label is empty (`bayesian_metric_label_required`)
	 * or not registered in the matrix (`bayesian_confusion_label_unknown`).
	 */
	private function assertLabel(string $label): void
	{
		if ($label === '') {
			throw new TInvalidDataValueException('bayesian_metric_label_required');
		}
		if (!in_array($label, $this->_matrix->getLabels(), true)) {
			throw new TInvalidDataValueException('bayesian_confusion_label_unknown', $label);
		}
	}

	/**
	 * Returns the accuracy (the fraction of correct predictions); 0.0 when the matrix is empty.
	 * @return float The accuracy in [0, 1].
	 */
	public function getAccuracy(): float
	{
		$total = $this->_matrix->getTotal();
		if ($total <= 0) {
			return 0.0;
		}
		$correct = 0;
		foreach ($this->_matrix->getLabels() as $label) {
			$correct += $this->_matrix->getCell($label, $label);
		}
		return $correct / $total;
	}

	/**
	 * Returns the precision for a label: TP / (TP + FP); NaN when the label was never predicted.
	 * @param string $label The label.
	 * @throws TInvalidDataValueException When the label is empty or not registered in the matrix.
	 * @return float The precision.
	 */
	public function getPrecision(string $label): float
	{
		$this->assertLabel($label);
		$predicted = 0;
		foreach ($this->_matrix->getLabels() as $expected) {
			$predicted += $this->_matrix->getCell($expected, $label);
		}
		if ($predicted === 0) {
			return NAN;
		}
		$tp = $this->_matrix->getCell($label, $label);
		return $tp / $predicted;
	}

	/**
	 * Returns the recall for a label: TP / (TP + FN); NaN when the label was never expected.
	 * @param string $label The label.
	 * @throws TInvalidDataValueException When the label is empty or not registered in the matrix.
	 * @return float The recall.
	 */
	public function getRecall(string $label): float
	{
		$this->assertLabel($label);
		$expected = 0;
		foreach ($this->_matrix->getLabels() as $predicted) {
			$expected += $this->_matrix->getCell($label, $predicted);
		}
		if ($expected === 0) {
			return NAN;
		}
		$tp = $this->_matrix->getCell($label, $label);
		return $tp / $expected;
	}

	/**
	 * Returns the F1 score for a label: 2 · precision · recall / (precision + recall).
	 * NaN when precision or recall is undefined; 0.0 when both are defined but zero (the
	 * scikit-learn convention).
	 * @param string $label The label.
	 * @throws TInvalidDataValueException When the label is empty.
	 * @return float The F1 score.
	 */
	public function getF1(string $label): float
	{
		if ($label === '') {
			throw new TInvalidDataValueException('bayesian_metric_label_required');
		}
		$precision = $this->getPrecision($label);
		$recall = $this->getRecall($label);
		if (is_nan($precision) || is_nan($recall)) {
			return NAN;
		}
		if (($precision + $recall) === 0.0) {
			return 0.0;
		}
		return 2.0 * $precision * $recall / ($precision + $recall);
	}

	/**
	 * Returns the macro-averaged precision across labels (undefined classes count as 0).
	 * @return float The macro precision; 0.0 when no label has a defined precision.
	 */
	public function getMacroPrecision(): float
	{
		return $this->macroAverage(fn (string $label): float => $this->getPrecision($label));
	}

	/**
	 * Returns the macro-averaged recall across labels (undefined classes count as 0).
	 * @return float The macro recall; 0.0 when no label has a defined recall.
	 */
	public function getMacroRecall(): float
	{
		return $this->macroAverage(fn (string $label): float => $this->getRecall($label));
	}

	/**
	 * Returns the macro-averaged F1 across labels (undefined classes count as 0).
	 * @return float The macro F1; 0.0 when no label has a defined F1.
	 */
	public function getMacroF1(): float
	{
		return $this->macroAverage(fn (string $label): float => $this->getF1($label));
	}

	/**
	 * Returns the micro-averaged precision (equals accuracy for single-label classification).
	 * @return float The micro precision in [0, 1]; 0.0 when the matrix is empty.
	 */
	public function getMicroPrecision(): float
	{
		return $this->getAccuracy();
	}

	/**
	 * Returns the micro-averaged recall (equals accuracy for single-label classification).
	 * @return float The micro recall in [0, 1]; 0.0 when the matrix is empty.
	 */
	public function getMicroRecall(): float
	{
		return $this->getAccuracy();
	}

	/**
	 * Returns the micro-averaged F1 (equals accuracy for single-label classification).
	 * @return float The micro F1 in [0, 1]; 0.0 when the matrix is empty.
	 */
	public function getMicroF1(): float
	{
		return $this->getAccuracy();
	}

	/**
	 * Averages a per-label metric across the registered labels.
	 *
	 * A class whose per-label metric is undefined (NaN — e.g. a class that was never predicted,
	 * so its precision is 0/0) is counted as 0.0 and still divided into by the full label count.
	 * This is the standard macro-average convention (matching scikit-learn): dropping undefined
	 * classes and shrinking the denominator would over-report the average exactly on the
	 * class-imbalanced data macro averaging is meant for.
	 * @param callable(string): float $fn The per-label metric function.
	 * @return float The macro average; 0.0 when every label is undefined.
	 */
	private function macroAverage(callable $fn): float
	{
		// The matrix constructor guarantees at least one label.
		$labels = $this->_matrix->getLabels();
		$sum = 0.0;
		foreach ($labels as $label) {
			$value = $fn($label);
			if (!is_nan($value)) {
				$sum += $value;
			}
		}
		return $sum / count($labels);
	}
}
