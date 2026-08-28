<?php

use Belisoful\Prado\Util\Bayesian\Evaluation\TBayesianMetrics;
use Belisoful\Prado\Util\Bayesian\Evaluation\TConfusionMatrix;
use Prado\Exceptions\TInvalidDataValueException;

class TBayesianMetricsTest extends PHPUnit\Framework\TestCase
{
	public function testAccuracyOnEmptyMatrix()
	{
		$matrix = new TConfusionMatrix(['a', 'b']);
		$metrics = new TBayesianMetrics($matrix);
		self::assertSame(0.0, $metrics->getAccuracy());
	}

	public function testAccuracyWithAllCorrect()
	{
		$matrix = new TConfusionMatrix(['a', 'b']);
		$matrix->record('a', 'a');
		$matrix->record('a', 'a');
		$matrix->record('b', 'b');
		$metrics = new TBayesianMetrics($matrix);
		self::assertEqualsWithDelta(1.0, $metrics->getAccuracy(), 1e-9);
	}

	public function testAccuracyWithHalfCorrect()
	{
		$matrix = new TConfusionMatrix(['a', 'b']);
		$matrix->record('a', 'a');
		$matrix->record('a', 'b');
		$matrix->record('b', 'a');
		$matrix->record('b', 'b');
		$metrics = new TBayesianMetrics($matrix);
		self::assertEqualsWithDelta(0.5, $metrics->getAccuracy(), 1e-9);
	}

	public function testPrecisionForLabel()
	{
		$matrix = new TConfusionMatrix(['spam', 'ham']);
		$matrix->record('spam', 'spam');
		$matrix->record('ham', 'spam');
		$matrix->record('ham', 'ham');
		$metrics = new TBayesianMetrics($matrix);
		self::assertEqualsWithDelta(0.5, $metrics->getPrecision('spam'), 1e-9);
		self::assertEqualsWithDelta(1.0, $metrics->getPrecision('ham'), 1e-9);
	}

	public function testPrecisionForLabelNeverPredictedIsNan()
	{
		$matrix = new TConfusionMatrix(['spam', 'ham']);
		$matrix->record('ham', 'ham');
		$metrics = new TBayesianMetrics($matrix);
		self::assertTrue(is_nan($metrics->getPrecision('spam')));
	}

	public function testRecallForLabel()
	{
		$matrix = new TConfusionMatrix(['spam', 'ham']);
		$matrix->record('spam', 'spam');
		$matrix->record('spam', 'ham');
		$matrix->record('ham', 'ham');
		$metrics = new TBayesianMetrics($matrix);
		self::assertEqualsWithDelta(0.5, $metrics->getRecall('spam'), 1e-9);
		self::assertEqualsWithDelta(1.0, $metrics->getRecall('ham'), 1e-9);
	}

	public function testRecallForLabelNeverExpectedIsNan()
	{
		$matrix = new TConfusionMatrix(['spam', 'ham']);
		$matrix->record('ham', 'ham');
		$metrics = new TBayesianMetrics($matrix);
		self::assertTrue(is_nan($metrics->getRecall('spam')));
	}

	public function testF1BalancesPrecisionAndRecall()
	{
		$matrix = new TConfusionMatrix(['spam', 'ham']);
		$matrix->record('spam', 'spam');
		$matrix->record('ham', 'spam');
		$matrix->record('ham', 'ham');
		$metrics = new TBayesianMetrics($matrix);
		// spam: precision = 1/(1+1) = 0.5, recall = 1/1 = 1.0; F1 = 2*0.5*1.0/(0.5+1.0) = 0.6667
		$expected = 2.0 * 0.5 * 1.0 / (0.5 + 1.0);
		self::assertEqualsWithDelta($expected, $metrics->getF1('spam'), 1e-9);
	}

	public function testF1WhenBothZeroIsNan()
	{
		$matrix = new TConfusionMatrix(['spam', 'ham']);
		$metrics = new TBayesianMetrics($matrix);
		self::assertTrue(is_nan($metrics->getF1('spam')));
	}

	public function testMacroPrecisionAveragesAcrossLabels()
	{
		$matrix = new TConfusionMatrix(['a', 'b']);
		$matrix->record('a', 'a');
		$matrix->record('b', 'a');
		$matrix->record('b', 'b');
		$metrics = new TBayesianMetrics($matrix);
		self::assertEqualsWithDelta((0.5 + 1.0) / 2, $metrics->getMacroPrecision(), 1e-9);
	}

	public function testMacroRecallAveragesAcrossLabels()
	{
		$matrix = new TConfusionMatrix(['a', 'b']);
		$matrix->record('a', 'a');
		$matrix->record('a', 'b');
		$matrix->record('b', 'b');
		$metrics = new TBayesianMetrics($matrix);
		self::assertEqualsWithDelta((0.5 + 1.0) / 2, $metrics->getMacroRecall(), 1e-9);
	}

	public function testMacroF1AveragesAcrossLabels()
	{
		$matrix = new TConfusionMatrix(['a', 'b']);
		$matrix->record('a', 'a');
		$matrix->record('a', 'b');
		$matrix->record('b', 'b');
		$metrics = new TBayesianMetrics($matrix);
		// a: precision = 1/1 = 1.0, recall = 1/2 = 0.5; F1 = 0.6667
		// b: precision = 1/2 = 0.5, recall = 1/1 = 1.0; F1 = 0.6667
		// macro F1 = 0.6667
		$expected = (2.0 * 1.0 * 0.5 / 1.5 + 2.0 * 0.5 * 1.0 / 1.5) / 2;
		self::assertEqualsWithDelta($expected, $metrics->getMacroF1(), 1e-9);
	}

	public function testMicroMetricsEqualAccuracy()
	{
		$matrix = new TConfusionMatrix(['a', 'b']);
		$matrix->record('a', 'a');
		$matrix->record('a', 'b');
		$matrix->record('b', 'b');
		$metrics = new TBayesianMetrics($matrix);
		self::assertEqualsWithDelta($metrics->getAccuracy(), $metrics->getMicroPrecision(), 1e-9);
		self::assertEqualsWithDelta($metrics->getAccuracy(), $metrics->getMicroRecall(), 1e-9);
		self::assertEqualsWithDelta($metrics->getAccuracy(), $metrics->getMicroF1(), 1e-9);
	}

	public function testMacroMetricsAreZeroWhenAllUndefined()
	{
		// An empty matrix has no predictions, so every per-class metric is undefined (0/0).
		// The macro average counts each undefined class as 0.0 rather than throwing.
		$matrix = new TConfusionMatrix(['a', 'b']);
		$metrics = new TBayesianMetrics($matrix);
		self::assertSame(0.0, $metrics->getMacroPrecision());
		self::assertSame(0.0, $metrics->getMacroRecall());
		self::assertSame(0.0, $metrics->getMacroF1());
	}

	public function testMacroAverageCountsUndefinedClassAsZero()
	{
		// Class 'c' is never predicted or expected, so its precision is undefined (NaN). It must
		// still be counted as 0 in the denominator, not silently dropped (which would report 1.0).
		$matrix = new TConfusionMatrix(['a', 'b', 'c']);
		$matrix->record('a', 'a');
		$matrix->record('b', 'b');
		$metrics = new TBayesianMetrics($matrix);
		self::assertEqualsWithDelta((1.0 + 1.0 + 0.0) / 3, $metrics->getMacroPrecision(), 1e-9);
		self::assertEqualsWithDelta((1.0 + 1.0 + 0.0) / 3, $metrics->getMacroRecall(), 1e-9);
		self::assertEqualsWithDelta((1.0 + 1.0 + 0.0) / 3, $metrics->getMacroF1(), 1e-9);
	}

	public function testPrecisionRejectsEmptyLabel()
	{
		$matrix = new TConfusionMatrix(['a', 'b']);
		$metrics = new TBayesianMetrics($matrix);
		$this->expectException(TInvalidDataValueException::class);
		$metrics->getPrecision('');
	}

	public function testRecallRejectsEmptyLabel()
	{
		$matrix = new TConfusionMatrix(['a', 'b']);
		$metrics = new TBayesianMetrics($matrix);
		$this->expectException(TInvalidDataValueException::class);
		$metrics->getRecall('');
	}

	public function testF1RejectsEmptyLabel()
	{
		$matrix = new TConfusionMatrix(['a', 'b']);
		$metrics = new TBayesianMetrics($matrix);
		$this->expectException(TInvalidDataValueException::class);
		$metrics->getF1('');
	}

	public function testF1IsZeroWhenPrecisionAndRecallAreDefinedButZero()
	{
		// 'a' is predicted once (wrongly) and expected once (missed): precision = recall = 0/1 = 0.
		$matrix = new TConfusionMatrix(['a', 'b']);
		$matrix->record('a', 'b');
		$matrix->record('b', 'a');
		$metrics = new TBayesianMetrics($matrix);
		self::assertSame(0.0, $metrics->getPrecision('a'));
		self::assertSame(0.0, $metrics->getRecall('a'));
		$f1 = $metrics->getF1('a');
		self::assertFalse(is_nan($f1));
		self::assertSame(0.0, $f1);
		self::assertSame(0.0, $metrics->getF1('b'));
		self::assertSame(0.0, $metrics->getMacroF1());
	}

	public function testPrecisionRejectsUnknownLabel()
	{
		$metrics = new TBayesianMetrics(new TConfusionMatrix(['a', 'b']));
		try {
			$metrics->getPrecision('zzz');
			self::fail('expected exception');
		} catch (TInvalidDataValueException $e) {
			self::assertSame('bayesian_confusion_label_unknown', $e->getErrorCode());
		}
	}

	public function testRecallRejectsUnknownLabel()
	{
		$metrics = new TBayesianMetrics(new TConfusionMatrix(['a', 'b']));
		try {
			$metrics->getRecall('zzz');
			self::fail('expected exception');
		} catch (TInvalidDataValueException $e) {
			self::assertSame('bayesian_confusion_label_unknown', $e->getErrorCode());
		}
	}

	public function testF1RejectsUnknownLabel()
	{
		$metrics = new TBayesianMetrics(new TConfusionMatrix(['a', 'b']));
		try {
			$metrics->getF1('zzz');
			self::fail('expected exception');
		} catch (TInvalidDataValueException $e) {
			self::assertSame('bayesian_confusion_label_unknown', $e->getErrorCode());
		}
	}

	public function testGetMatrixReturnsConstructorArgument()
	{
		$matrix = new TConfusionMatrix(['a']);
		$metrics = new TBayesianMetrics($matrix);
		self::assertSame($matrix, $metrics->getMatrix());
	}
}
