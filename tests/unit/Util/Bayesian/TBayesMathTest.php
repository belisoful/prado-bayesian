<?php

use Prado\Util\Bayesian\Math\TBayesMath;

class TBayesMathTest extends PHPUnit\Framework\TestCase
{
	public function testLogAddOfFiniteValues()
	{
		$sum = TBayesMath::logAdd(log(0.5), log(0.5));
		self::assertEqualsWithDelta(log(1.0), $sum, 1e-9);
	}

	public function testLogAddWithVeryDifferentMagnitudes()
	{
		$sum = TBayesMath::logAdd(log(1e-50), log(0.5));
		self::assertEqualsWithDelta(log(0.5), $sum, 1e-6);
	}

	public function testLogAddWithNegativeInfinityReturnsOtherOperand()
	{
		// log(exp(-INF) + exp(x)) = log(0 + exp(x)) = x, so the finite operand wins.
		self::assertSame(log(0.5), TBayesMath::logAdd(-INF, log(0.5)));
		self::assertSame(log(0.5), TBayesMath::logAdd(log(0.5), -INF));
	}

	public function testLogAddOfTwoNegativeInfinitiesIsNegativeInfinity()
	{
		self::assertSame(-INF, TBayesMath::logAdd(-INF, -INF));
	}

	public function testLogAddWithPositiveInfinity()
	{
		self::assertSame(INF, TBayesMath::logAdd(INF, log(0.5)));
		self::assertSame(INF, TBayesMath::logAdd(log(0.5), INF));
	}

	public function testLogSumOfEmptyArray()
	{
		self::assertSame(-INF, TBayesMath::logSum([]));
	}

	public function testLogSumOfValues()
	{
		$sum = TBayesMath::logSum([log(0.25), log(0.25), log(0.5)]);
		self::assertEqualsWithDelta(log(1.0), $sum, 1e-9);
	}

	public function testNormalizeEmptyInput()
	{
		self::assertSame([], TBayesMath::normalize([]));
	}

	public function testNormalizeAllNegativeInfinity()
	{
		self::assertSame([], TBayesMath::normalize(['a' => -INF, 'b' => -INF]));
	}

	public function testNormalizeProducesDistribution()
	{
		$normalized = TBayesMath::normalize(['a' => log(0.7), 'b' => log(0.2), 'c' => log(0.1)]);
		self::assertCount(3, $normalized);
		$sum = array_sum($normalized);
		self::assertEqualsWithDelta(1.0, $sum, 1e-9);
		self::assertEqualsWithDelta(0.7, $normalized['a'], 1e-6);
		self::assertEqualsWithDelta(0.2, $normalized['b'], 1e-6);
		self::assertEqualsWithDelta(0.1, $normalized['c'], 1e-6);
	}

	public function testLogComplementOfZero()
	{
		self::assertSame(-INF, TBayesMath::logComplement(0.0));
	}

	public function testLogComplementOfNegativeValue()
	{
		$p = 0.3;
		$complement = TBayesMath::logComplement(log($p));
		self::assertEqualsWithDelta(log(1.0 - $p), $complement, 1e-9);
	}

	public function testLogComplementOfVerySmall()
	{
		$p = 1e-300;
		$complement = TBayesMath::logComplement(log($p));
		self::assertEqualsWithDelta(log(1.0 - $p), $complement, 1e-9);
	}

	public function testLogAddOrdersOperandsEitherWay()
	{
		// The larger operand is normalized to the right; both argument orders agree.
		$expected = log(exp(1.0) + exp(0.5));
		self::assertEqualsWithDelta($expected, TBayesMath::logAdd(1.0, 0.5), 1e-12);
		self::assertEqualsWithDelta($expected, TBayesMath::logAdd(0.5, 1.0), 1e-12);
	}

	public function testLogComplementOfNonNegativeIsNegativeInfinity()
	{
		// log(1 - exp(v)) is undefined once exp(v) >= 1.
		self::assertSame(-INF, TBayesMath::logComplement(0.0));
		self::assertSame(-INF, TBayesMath::logComplement(1.5));
	}

	public function testLogComplementOfNegativeInfinityIsZero()
	{
		// p = exp(-INF) = 0, so log(1 - p) = log(1) = 0.
		self::assertSame(0.0, TBayesMath::logComplement(-INF));
	}

	public function testLogComplementOfSmallProbability()
	{
		$value = log(0.25);
		self::assertEqualsWithDelta(log(0.75), TBayesMath::logComplement($value), 1e-12);
	}

	public function testLogComplementOfNegativeValueTooSmallToRepresentIsNegativeInfinity()
	{
		// A negative log-probability can still round to exp(v) == 1.0 in double precision.
		// log(1 - 1) is undefined, so the guard on the *computed* probability - not just on
		// the sign of the input - is what keeps log1p(-1.0) from producing -INF by accident.
		self::assertSame(1.0, exp(-1e-300), 'precondition: exp() rounds this input to exactly 1.0');
		self::assertSame(-INF, TBayesMath::logComplement(-1e-300));
	}

	public function testLogComplementIsFiniteJustAboveTheRoundingThreshold()
	{
		// The neighbouring case: large enough that exp(v) < 1, so a real complement exists.
		// For tiny negative v, log(1 - exp(v)) tends to log(-v); the delta is loose because
		// double precision genuinely degrades as v approaches the threshold above.
		$result = TBayesMath::logComplement(-1e-10);
		self::assertTrue(is_finite($result), 'expected a finite complement, got ' . $result);
		self::assertEqualsWithDelta(log(1e-10), $result, 1e-6);
	}
}
