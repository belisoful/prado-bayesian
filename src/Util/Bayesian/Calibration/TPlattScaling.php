<?php

/**
 * TPlattScaling class file.
 *
 * @author Brad Anderson <belisoful@icloud.com>
 * @link https://github.com/belisoful/prado-bayesian
 * @license https://github.com/pradosoft/prado/blob/master/LICENSE
 */

namespace Belisoful\Prado\Util\Bayesian\Calibration;

use Belisoful\Prado\Util\Bayesian\TBayesianPayload;
use Prado\Exceptions\TInvalidDataValueException;

/**
 * TPlattScaling class.
 *
 * Calibrates a binary decision value — a log-odds, a margin, any score that grows with the
 * probability of the positive class — into a probability with a fitted logistic function,
 * `P(positive) = 1 / (1 + exp(-(slope · value + intercept)))` (Platt 1999).  Two parameters
 * are fitted on held-out examples by Newton's method on the logistic loss, with Platt's target
 * smoothing so a set with few positives or few negatives still gives a sane curve (the
 * numerically stable formulation of Lin, Lin and Weng 2007).
 *
 * The multi-label {@see \Belisoful\Prado\Util\Bayesian\TBayesianTagger} fits one of these per
 * label, since each label is a separate yes/no decision; multi-class classifiers use
 * {@see TTemperatureScaling} instead.
 *
 * @author Brad Anderson <belisoful@icloud.com>
 * @since 0.2.0
 */
class TPlattScaling
{
	/** @var float The slope applied to the decision value. */
	private float $_slope = 1.0;

	/** @var float The intercept added after the slope. */
	private float $_intercept = 0.0;

	/** @var int How many examples the parameters were fitted on; 0 when set by hand. */
	private int $_sampleCount = 0;

	/**
	 * Creates a scaling; the defaults (slope 1, intercept 0) are the plain logistic function.
	 * @param float $slope The slope.
	 * @param float $intercept The intercept.
	 * @throws TInvalidDataValueException When either parameter is not finite.
	 */
	public function __construct(float $slope = 1.0, float $intercept = 0.0)
	{
		$this->setParameters($slope, $intercept);
	}

	/**
	 * Returns the slope.
	 * @return float The slope.
	 */
	public function getSlope(): float
	{
		return $this->_slope;
	}

	/**
	 * Returns the intercept.
	 * @return float The intercept.
	 */
	public function getIntercept(): float
	{
		return $this->_intercept;
	}

	/**
	 * Returns how many examples the parameters were fitted on.
	 * @return int The sample count; 0 when set by hand or never fitted.
	 */
	public function getSampleCount(): int
	{
		return $this->_sampleCount;
	}

	/**
	 * Sets the parameters by hand.
	 * @param float $slope The slope.
	 * @param float $intercept The intercept.
	 * @throws TInvalidDataValueException When either parameter is not finite.
	 */
	public function setParameters(float $slope, float $intercept): void
	{
		if (!is_finite($slope) || !is_finite($intercept)) {
			throw new TInvalidDataValueException('bayesian_calibration_parameters_invalid', TBayesianPayload::formatFloat($slope), TBayesianPayload::formatFloat($intercept));
		}
		$this->_slope = $slope;
		$this->_intercept = $intercept;
		$this->_sampleCount = 0;
	}

	/**
	 * Converts a decision value into a calibrated probability of the positive class.
	 * @param float $value The decision value (e.g. a log-odds).
	 * @return float The probability, in (0, 1).
	 */
	public function apply(float $value): float
	{
		return self::sigmoid($this->_slope * $value + $this->_intercept);
	}

	/**
	 * Fits the slope and intercept on held-out examples by minimizing the logistic loss with
	 * Newton's method and backtracking line search.  The targets are smoothed as Platt
	 * proposed — positives aim at `(n+ + 1) / (n+ + 2)`, negatives at `1 / (n- + 2)` — which
	 * keeps a set with very few of one class from producing an infinite slope.
	 * @param array<int, float> $values The decision values.
	 * @param array<int, bool> $positives Whether each example is positive, in the same order.
	 * @throws TInvalidDataValueException When the lists differ in length or are empty.
	 */
	public function fit(array $values, array $positives): void
	{
		if (count($values) !== count($positives)) {
			throw new TInvalidDataValueException('bayesian_calibration_samples_mismatch', (string) count($values), (string) count($positives));
		}
		$values = array_values(array_map('floatval', $values));
		$positives = array_values($positives);
		$count = count($values);
		if ($count === 0) {
			throw new TInvalidDataValueException('bayesian_calibration_set_empty');
		}
		$positiveCount = 0;
		foreach ($positives as $positive) {
			if ($positive) {
				$positiveCount++;
			}
		}
		$negativeCount = $count - $positiveCount;
		$highTarget = ($positiveCount + 1.0) / ($positiveCount + 2.0);
		$lowTarget = 1.0 / ($negativeCount + 2.0);
		$targets = [];
		foreach ($positives as $i => $positive) {
			$targets[$i] = $positive ? $highTarget : $lowTarget;
		}

		// Platt's convention inside the fit: p = 1 / (1 + exp(a·f + b)), so a comes out
		// negative for a value that grows with the positive class; the public parameters are
		// negated at the end so apply() reads naturally.
		$a = 0.0;
		$b = log(($negativeCount + 1.0) / ($positiveCount + 1.0));
		$loss = self::plattLoss($values, $targets, $a, $b);
		for ($iteration = 0; $iteration < 100; $iteration++) {
			$h11 = 1e-12;
			$h22 = 1e-12;
			$h21 = 0.0;
			$g1 = 0.0;
			$g2 = 0.0;
			foreach ($values as $i => $value) {
				$fApB = $value * $a + $b;
				if ($fApB >= 0.0) {
					$p = exp(-$fApB) / (1.0 + exp(-$fApB));
					$q = 1.0 / (1.0 + exp(-$fApB));
				} else {
					$p = 1.0 / (1.0 + exp($fApB));
					$q = exp($fApB) / (1.0 + exp($fApB));
				}
				$d2 = $p * $q;
				$h11 += $value * $value * $d2;
				$h22 += $d2;
				$h21 += $value * $d2;
				$d1 = $targets[$i] - $p;
				$g1 += $value * $d1;
				$g2 += $d1;
			}
			if (abs($g1) < 1e-5 && abs($g2) < 1e-5) {
				break;
			}
			$determinant = $h11 * $h22 - $h21 * $h21;
			$stepA = -($h22 * $g1 - $h21 * $g2) / $determinant;
			$stepB = -(-$h21 * $g1 + $h11 * $g2) / $determinant;
			$gd = $g1 * $stepA + $g2 * $stepB;
			$stepSize = 1.0;
			while ($stepSize >= 1e-10) {
				$newA = $a + $stepSize * $stepA;
				$newB = $b + $stepSize * $stepB;
				$newLoss = self::plattLoss($values, $targets, $newA, $newB);
				if ($newLoss < $loss + 0.0001 * $stepSize * $gd) {
					$a = $newA;
					$b = $newB;
					$loss = $newLoss;
					break;
				}
				$stepSize /= 2.0;
			}
			if ($stepSize < 1e-10) {
				break;
			}
		}
		$this->_slope = -$a;
		$this->_intercept = -$b;
		$this->_sampleCount = $count;
	}

	/**
	 * Returns the logistic loss of Platt's parameterization over the examples.
	 * @param array<int, float> $values The decision values.
	 * @param array<int, float> $targets The smoothed targets.
	 * @param float $a Platt's slope.
	 * @param float $b Platt's intercept.
	 * @return float The loss.
	 */
	private static function plattLoss(array $values, array $targets, float $a, float $b): float
	{
		$loss = 0.0;
		foreach ($values as $i => $value) {
			$fApB = $value * $a + $b;
			if ($fApB >= 0.0) {
				$loss += $targets[$i] * $fApB + log(1.0 + exp(-$fApB));
			} else {
				$loss += ($targets[$i] - 1.0) * $fApB + log(1.0 + exp($fApB));
			}
		}
		return $loss;
	}

	/**
	 * The logistic function, evaluated without overflow at either extreme.
	 * @param float $x The argument.
	 * @return float The value in (0, 1).
	 */
	private static function sigmoid(float $x): float
	{
		if ($x >= 0.0) {
			return 1.0 / (1.0 + exp(-$x));
		}
		$e = exp($x);
		return $e / (1.0 + $e);
	}

	/**
	 * Returns the JSON-safe state, as stored with a model.
	 * @return array{method: string, slope: float, intercept: float, samples: int} The state.
	 */
	public function export(): array
	{
		return ['method' => 'platt', 'slope' => $this->_slope, 'intercept' => $this->_intercept, 'samples' => $this->_sampleCount];
	}

	/**
	 * Restores a scaling from the state {@see export()} wrote.
	 * @param array<string, mixed> $state The state.
	 * @return ?self The scaling, or null when the state holds no usable parameters.
	 */
	public static function import(array $state): ?self
	{
		$slope = $state['slope'] ?? null;
		$intercept = $state['intercept'] ?? null;
		if ((!is_int($slope) && !is_float($slope)) || (!is_int($intercept) && !is_float($intercept))) {
			return null;
		}
		if (!is_finite((float) $slope) || !is_finite((float) $intercept)) {
			return null;
		}
		$scaling = new self((float) $slope, (float) $intercept);
		$samples = $state['samples'] ?? 0;
		$scaling->_sampleCount = is_int($samples) ? max(0, $samples) : 0;
		return $scaling;
	}
}
