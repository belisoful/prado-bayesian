<?php

use Belisoful\Prado\Util\Bayesian\Calibration\TTemperatureScaling;
use Belisoful\Prado\Util\Bayesian\Math\TBayesMath;
use Prado\Exceptions\TInvalidDataValueException;

class TTemperatureScalingTest extends PHPUnit\Framework\TestCase
{
	/**
	 * An overconfident synthetic classifier: it gives the true label a large margin most of
	 * the time, but is wrong a fixed fraction of the time with the same margin.
	 * @param float $margin
	 * @param int $count
	 * @param float $accuracy
	 * @return array{0: array<int, array<string, float>>, 1: array<int, string>}
	 */
	private function overconfidentSamples(float $margin, int $count, float $accuracy): array
	{
		$scores = [];
		$labels = [];
		for ($i = 0; $i < $count; $i++) {
			$truth = $i % 2 === 0 ? 'spam' : 'ham';
			$other = $truth === 'spam' ? 'ham' : 'spam';
			$correct = ($i % 100) < (int) round($accuracy * 100);
			$winner = $correct ? $truth : $other;
			$scores[] = [$winner => 0.0, ($winner === 'spam' ? 'ham' : 'spam') => -$margin];
			$labels[] = $truth;
		}
		return [$scores, $labels];
	}

	public function testTemperatureOneIsPlainNormalization()
	{
		$scaling = new TTemperatureScaling();
		self::assertSame(1.0, $scaling->getTemperature());
		self::assertSame(0, $scaling->getSampleCount());
		$scores = ['spam' => -1.0, 'ham' => -3.0, 'news' => -8.0];
		self::assertSame(TBayesMath::normalize($scores), $scaling->apply($scores));
		self::assertSame([], $scaling->apply([]));
		self::assertSame([], $scaling->apply(['spam' => -INF]));
	}

	public function testHigherTemperatureFlattensAndLowerSharpens()
	{
		$scores = ['spam' => 0.0, 'ham' => -4.0];
		$raw = TBayesMath::normalize($scores)['spam'];
		$flat = (new TTemperatureScaling(4.0))->apply($scores);
		$sharp = (new TTemperatureScaling(0.25))->apply($scores);
		self::assertLessThan($raw, $flat['spam']);
		self::assertGreaterThan(0.5, $flat['spam'], 'the ranking never changes');
		self::assertGreaterThan($raw, $sharp['spam']);
		self::assertEqualsWithDelta(1.0, array_sum($flat), 1e-12);
		self::assertEqualsWithDelta(1.0, array_sum($sharp), 1e-12);
	}

	public function testFitRaisesTheTemperatureOfAnOverconfidentClassifier()
	{
		[$scores, $labels] = $this->overconfidentSamples(6.0, 200, 0.8);
		$scaling = new TTemperatureScaling();
		$before = $scaling->getLogLoss($scores, $labels);
		$temperature = $scaling->fit($scores, $labels);
		self::assertGreaterThan(1.0, $temperature, 'a classifier right 80% of the time at 99.7% confidence needs flattening');
		self::assertSame($temperature, $scaling->getTemperature());
		self::assertSame(200, $scaling->getSampleCount());
		$after = $scaling->getLogLoss($scores, $labels);
		self::assertLessThan($before, $after);
		// After calibration the winner's probability sits near the observed accuracy.
		$calibrated = $scaling->apply(['spam' => 0.0, 'ham' => -6.0]);
		self::assertEqualsWithDelta(0.8, $calibrated['spam'], 0.05);
	}

	public function testFitLowersTheTemperatureOfAnUnderconfidentClassifier()
	{
		[$scores, $labels] = $this->overconfidentSamples(0.5, 200, 1.0);
		$scaling = new TTemperatureScaling();
		$temperature = $scaling->fit($scores, $labels);
		self::assertLessThan(1.0, $temperature, 'always right at a tiny margin: sharpen');
		self::assertGreaterThanOrEqual(TTemperatureScaling::MIN_TEMPERATURE, $temperature);
	}

	public function testFitIsDeterministicAndBoundedAtBothEnds()
	{
		// Always wrong: nothing but the flattest possible distribution helps.
		[$scores, $labels] = $this->overconfidentSamples(3.0, 50, 0.0);
		$scaling = new TTemperatureScaling();
		$first = $scaling->fit($scores, $labels);
		$second = (new TTemperatureScaling())->fit($scores, $labels);
		self::assertSame($first, $second);
		self::assertEqualsWithDelta(TTemperatureScaling::MAX_TEMPERATURE, $first, 0.01);
	}

	public function testFitSkipsSamplesWhoseTrueLabelHasNoScore()
	{
		$scaling = new TTemperatureScaling();
		$scores = [['spam' => 0.0, 'ham' => -6.0], ['spam' => 0.0, 'ham' => -INF], ['spam' => 0.0, 'ham' => -6.0]];
		$labels = ['news', 'ham', 'spam'];
		$scaling->fit($scores, $labels);
		self::assertSame(1, $scaling->getSampleCount(), 'only the sample with a finite score for its label counts');
	}

	public function testFitRejectsUnusableInput()
	{
		$scaling = new TTemperatureScaling();
		try {
			$scaling->fit([['spam' => 0.0]], ['spam', 'ham']);
			self::fail('expected exception');
		} catch (TInvalidDataValueException $e) {
			self::assertSame('bayesian_calibration_samples_mismatch', $e->getErrorCode());
		}
		try {
			$scaling->fit([['spam' => 0.0]], ['ham']);
			self::fail('expected exception');
		} catch (TInvalidDataValueException $e) {
			self::assertSame('bayesian_calibration_set_empty', $e->getErrorCode());
		}
		try {
			$scaling->getLogLoss([], []);
			self::fail('expected exception');
		} catch (TInvalidDataValueException $e) {
			self::assertSame('bayesian_calibration_set_empty', $e->getErrorCode());
		}
		self::assertSame(1.0, $scaling->getTemperature(), 'a failed fit leaves the temperature alone');
	}

	public function testTemperatureMustBePositiveAndFinite()
	{
		foreach ([0.0, -1.0, NAN, INF] as $bad) {
			try {
				new TTemperatureScaling($bad);
				self::fail('expected exception');
			} catch (TInvalidDataValueException $e) {
				self::assertSame('bayesian_calibration_temperature_invalid', $e->getErrorCode());
			}
		}
		$scaling = new TTemperatureScaling(2.0);
		$scaling->setTemperature(3.0);
		self::assertSame(3.0, $scaling->getTemperature());
		self::assertSame(0, $scaling->getSampleCount(), 'a hand-set temperature has no samples');
	}

	public function testExportImportRoundTrip()
	{
		[$scores, $labels] = $this->overconfidentSamples(6.0, 40, 0.75);
		$scaling = new TTemperatureScaling();
		$scaling->fit($scores, $labels);
		$state = $scaling->export();
		self::assertSame('temperature', $state['method']);
		self::assertSame(40, $state['samples']);
		$restored = TTemperatureScaling::import($state);
		self::assertNotNull($restored);
		self::assertSame($scaling->getTemperature(), $restored->getTemperature());
		self::assertSame(40, $restored->getSampleCount());
		self::assertSame($scaling->apply(['spam' => 0.0, 'ham' => -6.0]), $restored->apply(['spam' => 0.0, 'ham' => -6.0]));
		self::assertNull(TTemperatureScaling::import([]));
		self::assertNull(TTemperatureScaling::import(['temperature' => 'hot']));
		self::assertNull(TTemperatureScaling::import(['temperature' => -1.0]));
		self::assertSame(1.5, TTemperatureScaling::import(['temperature' => 1.5, 'samples' => 'x'])->getTemperature());
	}
}
