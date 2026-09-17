<?php

/**
 * TCalibrationMetrics class file.
 *
 * @author Brad Anderson <belisoful@icloud.com>
 * @link https://github.com/belisoful/prado-bayesian
 * @license https://github.com/pradosoft/prado/blob/master/LICENSE
 */

namespace Belisoful\Prado\Util\Bayesian\Evaluation;

use Prado\Exceptions\TInvalidDataValueException;

/**
 * TCalibrationMetrics class.
 *
 * Measures how well probabilities match outcomes, which {@see TConfusionMatrix} cannot: a
 * classifier can be right 90% of the time and still claim 99.9% on every document.  Use these
 * on documents the model was not trained or calibrated on, before and after fitting a
 * {@see \Belisoful\Prado\Util\Bayesian\Calibration\TTemperatureScaling} or
 * {@see \Belisoful\Prado\Util\Bayesian\Calibration\TPlattScaling}, to see whether the
 * calibration helped.
 *
 * - {@see logLoss()} and {@see brierScore()} are proper scoring rules: lower is better, and
 *   the true probabilities minimize them.
 * - {@see expectedCalibrationError()} bins the predictions by confidence and averages the gap
 *   between the confidence and the observed frequency in each bin; 0 is perfect calibration.
 *
 * The binary methods take one probability per example (of the positive outcome); the
 * distribution methods take one probability map per example and its true category.
 *
 * @author Brad Anderson <belisoful@icloud.com>
 * @since 0.2.0
 */
final class TCalibrationMetrics
{
	/**
	 * Returns the mean negative log-likelihood of binary outcomes under their predicted
	 * probabilities.
	 * @param array<int, float> $probabilities The predicted probability of each positive outcome.
	 * @param array<int, bool> $outcomes Whether each outcome was positive, in the same order.
	 * @throws TInvalidDataValueException When the lists differ in length or are empty.
	 * @return float The log loss; 0 for perfect predictions.
	 */
	public static function logLoss(array $probabilities, array $outcomes): float
	{
		$pairs = self::pairs($probabilities, $outcomes);
		$total = 0.0;
		foreach ($pairs as [$probability, $positive]) {
			$total -= log(max($positive ? $probability : 1.0 - $probability, 1e-300));
		}
		return $total / count($pairs);
	}

	/**
	 * Returns the Brier score: the mean squared difference between each probability and its
	 * outcome (1 or 0).
	 * @param array<int, float> $probabilities The predicted probability of each positive outcome.
	 * @param array<int, bool> $outcomes Whether each outcome was positive, in the same order.
	 * @throws TInvalidDataValueException When the lists differ in length or are empty.
	 * @return float The Brier score, in [0, 1]; 0 for perfect predictions.
	 */
	public static function brierScore(array $probabilities, array $outcomes): float
	{
		$pairs = self::pairs($probabilities, $outcomes);
		$total = 0.0;
		foreach ($pairs as [$probability, $positive]) {
			$total += ($probability - ($positive ? 1.0 : 0.0)) ** 2;
		}
		return $total / count($pairs);
	}

	/**
	 * Returns the expected calibration error: predictions are grouped into equal-width
	 * confidence bins, and the absolute gap between the mean confidence and the observed
	 * positive rate of each bin is averaged, weighted by the bin's size.
	 * @param array<int, float> $probabilities The predicted probability of each positive outcome.
	 * @param array<int, bool> $outcomes Whether each outcome was positive, in the same order.
	 * @param int $bins The number of bins (>= 1).
	 * @throws TInvalidDataValueException When the lists differ in length or are empty.
	 * @return float The error, in [0, 1]; 0 when every bin's confidence matches its frequency.
	 */
	public static function expectedCalibrationError(array $probabilities, array $outcomes, int $bins = 10): float
	{
		$pairs = self::pairs($probabilities, $outcomes);
		$bins = max(1, $bins);
		$confidence = array_fill(0, $bins, 0.0);
		$positives = array_fill(0, $bins, 0);
		$sizes = array_fill(0, $bins, 0);
		foreach ($pairs as [$probability, $positive]) {
			$bin = min($bins - 1, max(0, (int) floor($probability * $bins)));
			$confidence[$bin] += $probability;
			$sizes[$bin]++;
			if ($positive) {
				$positives[$bin]++;
			}
		}
		$error = 0.0;
		$count = count($pairs);
		for ($bin = 0; $bin < $bins; $bin++) {
			if ($sizes[$bin] === 0) {
				continue;
			}
			$error += ($sizes[$bin] / $count) * abs($confidence[$bin] / $sizes[$bin] - $positives[$bin] / $sizes[$bin]);
		}
		return $error;
	}

	/**
	 * Returns the mean negative log-likelihood of true categories under predicted
	 * distributions, as {@see \Belisoful\Prado\Util\Bayesian\Classifier\IBayesianClassifier::score()}
	 * returns them.
	 * @param array<int, array<string, float>> $distributions One probability map per example.
	 * @param array<int, string> $labels The true category of each example, in the same order.
	 * @throws TInvalidDataValueException When the lists differ in length or are empty.
	 * @return float The log loss; 0 for perfect predictions.
	 */
	public static function distributionLogLoss(array $distributions, array $labels): float
	{
		if (count($distributions) !== count($labels) || $distributions === []) {
			throw new TInvalidDataValueException($distributions === [] ? 'bayesian_calibration_set_empty' : 'bayesian_calibration_samples_mismatch', (string) count($distributions), (string) count($labels));
		}
		$labels = array_values($labels);
		$total = 0.0;
		foreach (array_values($distributions) as $i => $distribution) {
			$total -= log(max((float) ($distribution[(string) $labels[$i]] ?? 0.0), 1e-300));
		}
		return $total / count($distributions);
	}

	/**
	 * Returns the expected calibration error of distributions, using each example's top
	 * probability as its confidence and whether the top category was the true one as its
	 * outcome — the multi-class form of {@see expectedCalibrationError()}.
	 * @param array<int, array<string, float>> $distributions One probability map per example.
	 * @param array<int, string> $labels The true category of each example, in the same order.
	 * @param int $bins The number of bins (>= 1).
	 * @throws TInvalidDataValueException When the lists differ in length or are empty.
	 * @return float The error, in [0, 1].
	 */
	public static function distributionCalibrationError(array $distributions, array $labels, int $bins = 10): float
	{
		if (count($distributions) !== count($labels) || $distributions === []) {
			throw new TInvalidDataValueException($distributions === [] ? 'bayesian_calibration_set_empty' : 'bayesian_calibration_samples_mismatch', (string) count($distributions), (string) count($labels));
		}
		$labels = array_values($labels);
		$confidences = [];
		$outcomes = [];
		foreach (array_values($distributions) as $i => $distribution) {
			if ($distribution === []) {
				$confidences[] = 0.0;
				$outcomes[] = false;
				continue;
			}
			$best = null;
			$bestProbability = -1.0;
			foreach ($distribution as $category => $probability) {
				if ($probability > $bestProbability) {
					$bestProbability = (float) $probability;
					$best = (string) $category;
				}
			}
			$confidences[] = $bestProbability;
			$outcomes[] = $best === (string) $labels[$i];
		}
		return self::expectedCalibrationError($confidences, $outcomes, $bins);
	}

	/**
	 * Validates and pairs the binary inputs.
	 * @param array<int, float> $probabilities The probabilities.
	 * @param array<int, bool> $outcomes The outcomes.
	 * @throws TInvalidDataValueException When the lists differ in length or are empty.
	 * @return array<int, array{0:float, 1:bool}> The pairs.
	 */
	private static function pairs(array $probabilities, array $outcomes): array
	{
		if (count($probabilities) !== count($outcomes)) {
			throw new TInvalidDataValueException('bayesian_calibration_samples_mismatch', (string) count($probabilities), (string) count($outcomes));
		}
		if ($probabilities === []) {
			throw new TInvalidDataValueException('bayesian_calibration_set_empty');
		}
		$outcomes = array_values($outcomes);
		$pairs = [];
		foreach (array_values($probabilities) as $i => $probability) {
			$pairs[] = [min(1.0, max(0.0, (float) $probability)), (bool) $outcomes[$i]];
		}
		return $pairs;
	}
}
