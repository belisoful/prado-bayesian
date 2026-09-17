<?php

use Belisoful\Prado\Util\Bayesian\Evaluation\TCalibrationMetrics;
use Prado\Exceptions\TInvalidDataValueException;

class TCalibrationMetricsTest extends PHPUnit\Framework\TestCase
{
	public function testPerfectPredictionsScoreZero()
	{
		$probabilities = [1.0, 0.0, 1.0];
		$outcomes = [true, false, true];
		self::assertEqualsWithDelta(0.0, TCalibrationMetrics::logLoss($probabilities, $outcomes), 1e-9);
		self::assertSame(0.0, TCalibrationMetrics::brierScore($probabilities, $outcomes));
		self::assertSame(0.0, TCalibrationMetrics::expectedCalibrationError($probabilities, $outcomes));
	}

	public function testLogLossAndBrierOfKnownValues()
	{
		self::assertEqualsWithDelta(-log(0.5), TCalibrationMetrics::logLoss([0.5, 0.5], [true, false]), 1e-12);
		self::assertEqualsWithDelta(0.25, TCalibrationMetrics::brierScore([0.5, 0.5], [true, false]), 1e-12);
		// A confident wrong answer costs a lot, but never infinity.
		self::assertTrue(is_finite(TCalibrationMetrics::logLoss([1.0], [false])));
		self::assertGreaterThan(100.0, TCalibrationMetrics::logLoss([1.0], [false]));
		// Probabilities are clamped into [0, 1].
		self::assertSame(0.0, TCalibrationMetrics::brierScore([1.7, -0.2], [true, false]));
	}

	public function testExpectedCalibrationErrorMeasuresTheGapPerBin()
	{
		// Ten predictions at 0.9 of which nine are positive: perfectly calibrated.
		$probabilities = array_fill(0, 10, 0.9);
		$outcomes = array_fill(0, 9, true) + [9 => false];
		self::assertEqualsWithDelta(0.0, TCalibrationMetrics::expectedCalibrationError($probabilities, $outcomes), 1e-12);
		// Ten predictions at 0.9 of which only five are positive: a 0.4 gap.
		$outcomes = array_fill(0, 5, true) + array_fill(5, 5, false);
		self::assertEqualsWithDelta(0.4, TCalibrationMetrics::expectedCalibrationError($probabilities, $outcomes), 1e-12);
		// Bins weight by size: half the predictions in a perfect bin halves the error.
		$probabilities = array_merge($probabilities, array_fill(0, 10, 0.5));
		$outcomes = array_merge(array_values($outcomes), array_fill(0, 5, true), array_fill(0, 5, false));
		self::assertEqualsWithDelta(0.2, TCalibrationMetrics::expectedCalibrationError($probabilities, $outcomes), 1e-12);
		self::assertEqualsWithDelta(0.2, TCalibrationMetrics::expectedCalibrationError($probabilities, $outcomes, 1), 1e-12, 'one bin is the overall gap');
		self::assertSame(TCalibrationMetrics::expectedCalibrationError($probabilities, $outcomes, 0), TCalibrationMetrics::expectedCalibrationError($probabilities, $outcomes, 1), 'bins below one are treated as one');
	}

	public function testDistributionMetrics()
	{
		$distributions = [['spam' => 0.9, 'ham' => 0.1], ['spam' => 0.2, 'ham' => 0.8], ['spam' => 0.6, 'ham' => 0.4]];
		$labels = ['spam', 'ham', 'ham'];
		$expected = -(log(0.9) + log(0.8) + log(0.4)) / 3;
		self::assertEqualsWithDelta($expected, TCalibrationMetrics::distributionLogLoss($distributions, $labels), 1e-12);
		// Top probabilities 0.9, 0.8, 0.6 with outcomes right, right, wrong.
		$ece = TCalibrationMetrics::distributionCalibrationError($distributions, $labels, 10);
		self::assertEqualsWithDelta((abs(0.9 - 1.0) + abs(0.8 - 1.0) + abs(0.6 - 0.0)) / 3, $ece, 1e-12);
		// A missing label is a probability of zero; an empty distribution is a wrong answer at zero confidence.
		self::assertGreaterThan(100.0, TCalibrationMetrics::distributionLogLoss([['spam' => 1.0]], ['ham']));
		self::assertSame(0.0, TCalibrationMetrics::distributionCalibrationError([[]], ['ham']));
	}

	public function testMismatchedOrEmptyInputThrows()
	{
		foreach ([
			fn () => TCalibrationMetrics::logLoss([0.5], [true, false]),
			fn () => TCalibrationMetrics::brierScore([0.5, 0.5], [true]),
			fn () => TCalibrationMetrics::expectedCalibrationError([], [true]),
			fn () => TCalibrationMetrics::distributionLogLoss([['a' => 1.0]], []),
			fn () => TCalibrationMetrics::distributionCalibrationError([['a' => 1.0], ['a' => 1.0]], ['a']),
		] as $call) {
			try {
				$call();
				self::fail('expected exception');
			} catch (TInvalidDataValueException $e) {
				self::assertSame('bayesian_calibration_samples_mismatch', $e->getErrorCode());
			}
		}
		foreach ([
			fn () => TCalibrationMetrics::logLoss([], []),
			fn () => TCalibrationMetrics::distributionLogLoss([], []),
			fn () => TCalibrationMetrics::distributionCalibrationError([], []),
		] as $call) {
			try {
				$call();
				self::fail('expected exception');
			} catch (TInvalidDataValueException $e) {
				self::assertSame('bayesian_calibration_set_empty', $e->getErrorCode());
			}
		}
	}
}
