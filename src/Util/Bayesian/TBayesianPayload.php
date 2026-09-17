<?php

/**
 * TBayesianPayload class file.
 *
 * @author Brad Anderson <belisoful@icloud.com>
 * @link https://github.com/belisoful/prado-bayesian
 * @license https://github.com/pradosoft/prado/blob/master/LICENSE
 */

namespace Belisoful\Prado\Util\Bayesian;

/**
 * TBayesianPayload class.
 *
 * Reads typed values out of decoded JSON and configuration arrays, whose contents are `mixed`
 * until checked.  A stored model, a per-token metadata row and a module configuration all
 * arrive this way, and every consumer wants an int, a float, a string or a map with a known
 * key type.  These helpers make that conversion explicit and total: a value of the wrong shape
 * becomes the default rather than a notice, a `TypeError` or a silently wrong cast.
 *
 * @author Brad Anderson <belisoful@icloud.com>
 * @since 0.2.0
 */
final class TBayesianPayload
{
	/**
	 * Returns an integer: ints as they are, floats and numeric strings truncated, booleans as
	 * 0/1; anything else is the default.
	 * @param mixed $value The raw value.
	 * @param int $default The value to use when the raw value is not numeric.
	 * @return int The integer.
	 */
	public static function int($value, int $default = 0): int
	{
		if (is_int($value)) {
			return $value;
		}
		if (is_float($value)) {
			return is_finite($value) ? (int) $value : $default;
		}
		if (is_bool($value)) {
			return $value ? 1 : 0;
		}
		if (is_string($value) && is_numeric($value)) {
			return (int) $value;
		}
		return $default;
	}

	/**
	 * Returns a float: ints and floats as they are, numeric strings converted; anything else
	 * is the default.
	 * @param mixed $value The raw value.
	 * @param float $default The value to use when the raw value is not numeric.
	 * @return float The float.
	 */
	public static function float($value, float $default = 0.0): float
	{
		if (is_int($value) || is_float($value)) {
			return (float) $value;
		}
		if (is_string($value) && is_numeric($value)) {
			return (float) $value;
		}
		return $default;
	}

	/**
	 * Returns a string: any scalar converted; anything else is the default.
	 * @param mixed $value The raw value.
	 * @param string $default The value to use when the raw value is not a scalar.
	 * @return string The string.
	 */
	public static function string($value, string $default = ''): string
	{
		if (is_string($value)) {
			return $value;
		}
		if (is_int($value) || is_float($value) || is_bool($value)) {
			return (string) $value;
		}
		return $default;
	}

	/**
	 * Returns a boolean: scalars by PHP's truthiness (so `"0"`, `0` and `""` are false);
	 * anything else is the default.
	 * @param mixed $value The raw value.
	 * @param bool $default The value to use when the raw value is not a scalar.
	 * @return bool The boolean.
	 */
	public static function bool($value, bool $default = false): bool
	{
		return is_scalar($value) ? (bool) $value : $default;
	}

	/**
	 * Formats a float for an error message, naming `NAN`, `INF` and `-INF` rather than
	 * coercing them (which PHP 8.5 warns about).
	 * @param float $value The value.
	 * @return string The text.
	 */
	public static function formatFloat(float $value): string
	{
		if (is_nan($value)) {
			return 'NAN';
		}
		if (is_infinite($value)) {
			return $value > 0 ? 'INF' : '-INF';
		}
		return (string) $value;
	}

	/**
	 * Returns the value as a map with string keys; a non-array is an empty map.
	 * @param mixed $value The raw value.
	 * @return array<string, mixed> The map.
	 */
	public static function map($value): array
	{
		if (!is_array($value)) {
			return [];
		}
		$out = [];
		foreach ($value as $key => $item) {
			$out[(string) $key] = $item;
		}
		return $out;
	}

	/**
	 * Returns the value as a list; a non-array is an empty list.
	 * @param mixed $value The raw value.
	 * @return array<int, mixed> The list.
	 */
	public static function list($value): array
	{
		return is_array($value) ? array_values($value) : [];
	}

	/**
	 * Returns the value as a map of string keys to integers, converting each entry with
	 * {@see int()}; a non-array is an empty map.
	 * @param mixed $value The raw value.
	 * @return array<string, int> The map.
	 */
	public static function intMap($value): array
	{
		$out = [];
		foreach (self::map($value) as $key => $item) {
			$out[$key] = self::int($item);
		}
		return $out;
	}

	/**
	 * Returns the value as a map of string keys to floats, converting each entry with
	 * {@see float()}; a non-array is an empty map.
	 * @param mixed $value The raw value.
	 * @return array<string, float> The map.
	 */
	public static function floatMap($value): array
	{
		$out = [];
		foreach (self::map($value) as $key => $item) {
			$out[$key] = self::float($item);
		}
		return $out;
	}

	/**
	 * Returns the value as a list of strings, converting each entry with {@see string()} and
	 * dropping entries that are not scalars; a non-array is an empty list.
	 * @param mixed $value The raw value.
	 * @return array<int, string> The list.
	 */
	public static function stringList($value): array
	{
		$out = [];
		foreach (self::list($value) as $item) {
			if (is_scalar($item)) {
				$out[] = self::string($item);
			}
		}
		return $out;
	}
}
