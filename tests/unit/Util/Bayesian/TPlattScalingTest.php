<?php

use Belisoful\Prado\Util\Bayesian\Calibration\TPlattScaling;
use Prado\Exceptions\TInvalidDataValueException;

class TPlattScalingTest extends PHPUnit\Framework\TestCase
{
	public function testDefaultsAreThePlainLogisticFunction()
	{
		$scaling = new TPlattScaling();
		self::assertSame(1.0, $scaling->getSlope());
		self::assertSame(0.0, $scaling->getIntercept());
		self::assertSame(0, $scaling->getSampleCount());
		self::assertSame(0.5, $scaling->apply(0.0));
		self::assertEqualsWithDelta(1.0 / (1.0 + exp(-2.0)), $scaling->apply(2.0), 1e-12);
		self::assertSame(0.0, $scaling->apply(-1000.0), 'a huge negative argument underflows cleanly to zero, never to NaN');
		self::assertSame(1.0, $scaling->apply(1000.0), 'a huge positive argument saturates cleanly at one');
	}

	public function testFitOnSeparableDataGivesAMonotoneCurveCenteredOnTheBoundary()
	{
		$values = [];
		$positives = [];
		for ($i = 0; $i < 100; $i++) {
			$values[] = ($i - 50) / 10.0;
			$positives[] = $i >= 50;
		}
		$scaling = new TPlattScaling();
		$scaling->fit($values, $positives);
		self::assertSame(100, $scaling->getSampleCount());
		self::assertGreaterThan(0.0, $scaling->getSlope(), 'higher value, higher probability');
		self::assertGreaterThan($scaling->apply(1.0), $scaling->apply(2.0));
		self::assertGreaterThan(0.9, $scaling->apply(3.0));
		self::assertLessThan(0.1, $scaling->apply(-3.0));
		self::assertEqualsWithDelta(0.5, $scaling->apply(0.0), 0.1, 'the boundary sits at the crossover');
	}

	public function testFitShrinksOverconfidentValues()
	{
		// Values of ±8 (99.97% under the plain sigmoid) that are right only 80% of the time.
		$values = [];
		$positives = [];
		for ($i = 0; $i < 200; $i++) {
			$truth = $i % 2 === 0;
			$correct = ($i % 10) < 8;
			$values[] = ($truth === $correct) ? 8.0 : -8.0;
			$positives[] = $truth;
		}
		$scaling = new TPlattScaling();
		$scaling->fit($values, $positives);
		self::assertLessThan(1.0, $scaling->getSlope());
		self::assertEqualsWithDelta(0.8, $scaling->apply(8.0), 0.05);
		self::assertEqualsWithDelta(0.2, $scaling->apply(-8.0), 0.05);
	}

	public function testFitWithASingleClassStaysFinite()
	{
		$scaling = new TPlattScaling();
		$scaling->fit([1.0, 2.0, 3.0], [true, true, true]);
		self::assertTrue(is_finite($scaling->getSlope()));
		self::assertTrue(is_finite($scaling->getIntercept()));
		$probability = $scaling->apply(2.0);
		self::assertGreaterThan(0.5, $probability);
		self::assertLessThan(1.0, $probability, 'target smoothing keeps it off 1');
	}

	public function testFitIsDeterministic()
	{
		$values = [-2.0, -1.0, 0.5, 1.5, 3.0, -0.5];
		$positives = [false, false, true, true, true, false];
		$first = new TPlattScaling();
		$first->fit($values, $positives);
		$second = new TPlattScaling();
		$second->fit($values, $positives);
		self::assertSame($first->export(), $second->export());
	}

	public function testFitRejectsUnusableInput()
	{
		$scaling = new TPlattScaling();
		try {
			$scaling->fit([1.0, 2.0], [true]);
			self::fail('expected exception');
		} catch (TInvalidDataValueException $e) {
			self::assertSame('bayesian_calibration_samples_mismatch', $e->getErrorCode());
		}
		try {
			$scaling->fit([], []);
			self::fail('expected exception');
		} catch (TInvalidDataValueException $e) {
			self::assertSame('bayesian_calibration_set_empty', $e->getErrorCode());
		}
	}

	public function testParametersMustBeFinite()
	{
		try {
			new TPlattScaling(INF, 0.0);
			self::fail('expected exception');
		} catch (TInvalidDataValueException $e) {
			self::assertSame('bayesian_calibration_parameters_invalid', $e->getErrorCode());
		}
		$scaling = new TPlattScaling();
		try {
			$scaling->setParameters(1.0, NAN);
			self::fail('expected exception');
		} catch (TInvalidDataValueException $e) {
			self::assertSame('bayesian_calibration_parameters_invalid', $e->getErrorCode());
		}
		$scaling->setParameters(2.0, -1.0);
		self::assertSame(2.0, $scaling->getSlope());
		self::assertSame(-1.0, $scaling->getIntercept());
	}

	public function testExportImportRoundTrip()
	{
		$scaling = new TPlattScaling();
		$scaling->fit([-1.0, 1.0, -2.0, 2.0], [false, true, false, true]);
		$state = $scaling->export();
		self::assertSame('platt', $state['method']);
		self::assertSame(4, $state['samples']);
		$restored = TPlattScaling::import($state);
		self::assertNotNull($restored);
		self::assertSame($scaling->getSlope(), $restored->getSlope());
		self::assertSame($scaling->getIntercept(), $restored->getIntercept());
		self::assertSame(4, $restored->getSampleCount());
		self::assertSame($scaling->apply(0.7), $restored->apply(0.7));
		self::assertNull(TPlattScaling::import([]));
		self::assertNull(TPlattScaling::import(['slope' => 'steep', 'intercept' => 0]));
		self::assertNull(TPlattScaling::import(['slope' => 1.0, 'intercept' => INF]));
	}
}
