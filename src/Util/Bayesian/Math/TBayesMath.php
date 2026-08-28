<?php

/**
 * TBayesMath class file.
 *
 * @author Brad Anderson <belisoful@icloud.com>
 * @link https://github.com/belisoful/prado-bayesian
 * @license https://github.com/pradosoft/prado/blob/master/LICENSE
 */

namespace Prado\Util\Bayesian\Math;

/**
 * TBayesMath class.
 *
 * Log-space arithmetic helpers used by the Bayesian classifiers.  The product of thousands of
 * small probabilities (one per token) underflows float64 quickly; working in log space
 * ({@see logAdd()} for sums, addition for products) keeps every step in a numerically stable
 * range.  {@see normalize()} then exponentiates the relative log-scores back to probabilities
 * that sum to one.
 *
 * The class is intentionally a static utility — the math does not carry state, and the
 * classifiers call it as they accumulate log-scores.
 *
 * @author Brad Anderson <belisoful@icloud.com>
 * @since 0.1.0
 */
final class TBayesMath
{
	/**
	 * Adds two log-space values: log(exp(a) + exp(b)) without leaving log space.
	 *
	 * Uses the log-sum-exp trick — subtract the larger operand to keep the exponent bounded.
	 * @param float $a The first log-space value.
	 * @param float $b The second log-space value.
	 * @return float The log-space sum.
	 */
	public static function logAdd(float $a, float $b): float
	{
		if ($a === -INF || $a === INF) {
			return $a === -INF ? $b : $a;
		}
		if ($b === -INF || $b === INF) {
			return $b === -INF ? $a : $b;
		}
		if ($a > $b) {
			[$a, $b] = [$b, $a];
		}
		$diff = $b - $a;
		if ($diff > 50.0) {
			return $b;
		}
		return $a + log1p(exp($diff));
	}

	/**
	 * Adds a list of log-space values in a numerically stable way.
	 * @param float[] $values The log-space values.
	 * @return float The log-space sum; -INF when the list is empty.
	 */
	public static function logSum(array $values): float
	{
		$sum = -INF;
		foreach ($values as $value) {
			$sum = self::logAdd($sum, (float) $value);
		}
		return $sum;
	}

	/**
	 * Converts log-space scores to a normalized probability distribution.
	 *
	 * The largest score is treated as zero, the rest as offsets, so the result is dominated by
	 * the relative gaps and not by the absolute magnitudes.  An all -INF input yields an
	 * empty array so callers can detect "no information" without inspecting log-space.
	 * @param float[] $scores The log-space scores, keyed by category.
	 * @return float[] The normalized probabilities, summing to 1.0 (or empty when the input is empty).
	 */
	public static function normalize(array $scores): array
	{
		if ($scores === []) {
			return [];
		}
		$max = max($scores);
		if ($max === -INF) {
			return [];
		}
		$shifted = [];
		$sum = 0.0;
		foreach ($scores as $key => $value) {
			$shifted[$key] = exp($value - $max);
			$sum += $shifted[$key];
		}
		if ($sum <= 0.0) {
			return [];
		}
		$out = [];
		foreach ($shifted as $key => $value) {
			$out[$key] = $value / $sum;
		}
		return $out;
	}

	/**
	 * Computes log(1 - exp(value)) safely for negative inputs near zero.
	 *
	 * Used when a classifier needs P(absence | category) and the underlying probability is
	 * 1 - p.  Direct `log(1 - exp($v))` loses precision when $v is close to zero; this
	 * function uses `log1p(-exp($v))` and clamps so the argument is in (-1, 0).
	 * @param float $value The log-space probability in (-INF, 0].
	 * @return float The log-space complement.
	 */
	public static function logComplement(float $value): float
	{
		if ($value >= 0.0) {
			return -INF;
		}
		$prob = exp($value);
		if ($prob >= 1.0) {
			return -INF;
		}
		if ($prob <= 0.0) {
			return 0.0;
		}
		return log1p(-$prob);
	}
}
