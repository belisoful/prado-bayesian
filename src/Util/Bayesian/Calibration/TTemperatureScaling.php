<?php

/**
 * TTemperatureScaling class file.
 *
 * @author Brad Anderson <belisoful@icloud.com>
 * @link https://github.com/belisoful/prado-bayesian
 * @license https://github.com/pradosoft/prado/blob/master/LICENSE
 */

namespace Belisoful\Prado\Util\Bayesian\Calibration;

use Belisoful\Prado\Util\Bayesian\Math\TBayesMath;
use Belisoful\Prado\Util\Bayesian\TBayesianPayload;
use Prado\Exceptions\TInvalidDataValueException;

/**
 * TTemperatureScaling class.
 *
 * Turns a classifier's log-posteriors into calibrated probabilities by dividing them by one
 * fitted constant, the temperature, before normalizing (Guo et al. 2017).  Naive Bayes is
 * overconfident by construction — its independence assumption multiplies correlated evidence,
 * so a document that is 70% likely to be spam routinely scores 0.99 — and a temperature above
 * one flattens exactly that: the ranking of the categories never changes, only how much of the
 * mass the winner gets.
 *
 * The temperature is fitted on held-out labeled documents by minimizing the negative
 * log-likelihood of the true labels, a one-dimensional convex search that needs only a few
 * dozen documents to be useful and cannot overfit the way a per-class model could.  Fit it
 * with {@see fit()} or through {@see \Belisoful\Prado\Util\Bayesian\Classifier\TNaiveBayesClassifier::calibrate()},
 * which stores the result with the model.
 *
 * ```php
 * $classifier->calibrate($heldOutSet);          // fits and installs a temperature
 * $classifier->score('free pills');             // now calibrated
 * $classifier->getCalibration()->getTemperature();
 * ```
 *
 * @author Brad Anderson <belisoful@icloud.com>
 * @since 0.2.0
 */
class TTemperatureScaling
{
	/** The lowest temperature the fit considers; below it the scores are sharpened absurdly. */
	public const MIN_TEMPERATURE = 0.01;

	/** The highest temperature the fit considers; above it every distribution is uniform. */
	public const MAX_TEMPERATURE = 100.0;

	/** @var float The temperature the log-scores are divided by. */
	private float $_temperature = 1.0;

	/** @var int How many held-out documents the temperature was fitted on; 0 when set by hand. */
	private int $_sampleCount = 0;

	/**
	 * Creates a scaling with the given temperature (1.0 leaves the scores as they are).
	 * @param float $temperature The temperature.
	 * @throws TInvalidDataValueException When the temperature is not positive and finite.
	 */
	public function __construct(float $temperature = 1.0)
	{
		$this->setTemperature($temperature);
	}

	/**
	 * Returns the temperature.
	 * @return float The temperature.
	 */
	public function getTemperature(): float
	{
		return $this->_temperature;
	}

	/**
	 * Sets the temperature by hand.  Above 1 flattens the distribution, below 1 sharpens it.
	 * @param float $value The temperature.
	 * @throws TInvalidDataValueException When the value is not positive and finite.
	 */
	public function setTemperature(float $value): void
	{
		if (!($value > 0.0) || !is_finite($value)) {
			throw new TInvalidDataValueException('bayesian_calibration_temperature_invalid', TBayesianPayload::formatFloat($value));
		}
		$this->_temperature = $value;
		$this->_sampleCount = 0;
	}

	/**
	 * Returns how many held-out documents the temperature was fitted on.
	 * @return int The sample count; 0 when the temperature was set by hand or never fitted.
	 */
	public function getSampleCount(): int
	{
		return $this->_sampleCount;
	}

	/**
	 * Converts log-scores into calibrated probabilities: the scores are divided by the
	 * temperature and normalized.
	 * @param array<string, float> $logScores The log-posteriors, keyed by category.
	 * @return array<string, float> The probabilities, keyed by category, summing to 1; empty
	 * when no category has a finite score.
	 */
	public function apply(array $logScores): array
	{
		$scaled = [];
		foreach ($logScores as $category => $score) {
			$scaled[$category] = $score / $this->_temperature;
		}
		return TBayesMath::normalize($scaled);
	}

	/**
	 * Fits the temperature on held-out documents.
	 *
	 * Each sample is the log-scores a classifier gave a document and the document's true
	 * category.  Samples whose true category has no finite score are skipped, since no
	 * temperature can help them.  The temperature that minimizes the mean negative
	 * log-likelihood of the true categories is found by golden-section search over its
	 * logarithm between {@see MIN_TEMPERATURE} and {@see MAX_TEMPERATURE}.
	 * @param array<int, array<string, float>> $logScores One log-score map per document.
	 * @param array<int, string> $labels The true category of each document, in the same order.
	 * @throws TInvalidDataValueException When the two lists differ in length, or no sample is usable.
	 * @return float The fitted temperature.
	 */
	public function fit(array $logScores, array $labels): float
	{
		if (count($logScores) !== count($labels)) {
			throw new TInvalidDataValueException('bayesian_calibration_samples_mismatch', (string) count($logScores), (string) count($labels));
		}
		$samples = [];
		foreach (array_values($logScores) as $i => $scores) {
			$label = (string) array_values($labels)[$i];
			if (!isset($scores[$label]) || !is_finite($scores[$label])) {
				continue;
			}
			$samples[] = [$scores, $label];
		}
		if ($samples === []) {
			throw new TInvalidDataValueException('bayesian_calibration_set_empty');
		}
		$low = log(self::MIN_TEMPERATURE);
		$high = log(self::MAX_TEMPERATURE);
		$ratio = (sqrt(5.0) - 1.0) / 2.0;
		$a = $high - $ratio * ($high - $low);
		$b = $low + $ratio * ($high - $low);
		$fa = self::loss($samples, exp($a));
		$fb = self::loss($samples, exp($b));
		for ($i = 0; $i < 100 && ($high - $low) > 1e-6; $i++) {
			if ($fa < $fb) {
				$high = $b;
				$b = $a;
				$fb = $fa;
				$a = $high - $ratio * ($high - $low);
				$fa = self::loss($samples, exp($a));
			} else {
				$low = $a;
				$a = $b;
				$fa = $fb;
				$b = $low + $ratio * ($high - $low);
				$fb = self::loss($samples, exp($b));
			}
		}
		$this->_temperature = exp(($low + $high) / 2.0);
		$this->_sampleCount = count($samples);
		return $this->_temperature;
	}

	/**
	 * Returns the mean negative log-likelihood of the true labels at a temperature.
	 * @param array<int, array{0:array<string, float>, 1:string}> $samples The usable samples.
	 * @param float $temperature The temperature.
	 * @return float The mean negative log-likelihood.
	 */
	private static function loss(array $samples, float $temperature): float
	{
		$total = 0.0;
		foreach ($samples as [$scores, $label]) {
			$scaled = [];
			foreach ($scores as $category => $score) {
				$scaled[$category] = $score / $temperature;
			}
			$probability = TBayesMath::normalize($scaled)[$label] ?? 0.0;
			$total -= log(max($probability, 1e-300));
		}
		return $total / count($samples);
	}

	/**
	 * Returns the mean negative log-likelihood the current temperature gives a labeled set, so
	 * a fit can be judged against the uncalibrated temperature of 1.
	 * @param array<int, array<string, float>> $logScores One log-score map per document.
	 * @param array<int, string> $labels The true category of each document.
	 * @throws TInvalidDataValueException When the lists differ in length or no sample is usable.
	 * @return float The mean negative log-likelihood.
	 */
	public function getLogLoss(array $logScores, array $labels): float
	{
		if (count($logScores) !== count($labels)) {
			throw new TInvalidDataValueException('bayesian_calibration_samples_mismatch', (string) count($logScores), (string) count($labels));
		}
		$samples = [];
		foreach (array_values($logScores) as $i => $scores) {
			$samples[] = [$scores, (string) array_values($labels)[$i]];
		}
		if ($samples === []) {
			throw new TInvalidDataValueException('bayesian_calibration_set_empty');
		}
		return self::loss($samples, $this->_temperature);
	}

	/**
	 * Returns the JSON-safe state, as stored with a model.
	 * @return array{method: string, temperature: float, samples: int} The state.
	 */
	public function export(): array
	{
		return ['method' => 'temperature', 'temperature' => $this->_temperature, 'samples' => $this->_sampleCount];
	}

	/**
	 * Restores a scaling from the state {@see export()} wrote.
	 * @param array<string, mixed> $state The state.
	 * @return ?self The scaling, or null when the state holds no usable temperature.
	 */
	public static function import(array $state): ?self
	{
		$temperature = $state['temperature'] ?? null;
		if (!is_int($temperature) && !is_float($temperature)) {
			return null;
		}
		$temperature = (float) $temperature;
		if (!($temperature > 0.0) || !is_finite($temperature)) {
			return null;
		}
		$scaling = new self($temperature);
		$samples = $state['samples'] ?? 0;
		$scaling->_sampleCount = is_int($samples) ? max(0, $samples) : 0;
		return $scaling;
	}
}
