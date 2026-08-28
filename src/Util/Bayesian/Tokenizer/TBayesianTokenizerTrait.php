<?php

/**
 * TBayesianTokenizerTrait trait file.
 *
 * @author Brad Anderson <belisoful@icloud.com>
 * @link https://github.com/belisoful/prado-bayesian
 * @license https://github.com/pradosoft/prado/blob/master/LICENSE
 */

namespace Belisoful\Prado\Util\Bayesian\Tokenizer;

use Prado\Exceptions\TInvalidDataValueException;

/**
 * TBayesianTokenizerTrait trait.
 *
 * The features every {@see IBayesianTokenizer} shares, so each tokenizer only implements its
 * token strategy:
 *
 * - **Configuration round-trip.**  {@see exportConfig()} and {@see importConfig()} are driven
 *   by {@see getConfigProperties()}, a list of the tokenizer's PRADO property names (e.g.
 *   `['MinLength', 'StopWords', 'Pattern']`).  Export reads each property through its getter
 *   into a lower-camel-case key; import writes each present key back through its setter,
 *   coercing the value to the setter's declared scalar/array type so a JSON payload (which
 *   turns everything into ints/strings/bools/arrays) restores cleanly.  Tokenizers that hold
 *   nested tokenizers override the two public methods and call
 *   {@see exportConfigProperties()} / {@see importConfigProperties()} for the plain part.
 * - **Safe matching.**  {@see matchAll()} scrubs invalid UTF-8, runs `preg_match_all`, and
 *   throws {@see TInvalidDataValueException} `bayesian_tokenizer_pattern_failed` when PCRE
 *   fails (backtrack/recursion limit) instead of returning an empty match set;
 *   {@see normalizeText()} scrubs and lowercases input.  Both build on the static helpers in
 *   {@see TBayesianTokenizerFactory} (`scrubText()`, `checkPregError()`, `assertPattern()`).
 *
 * The trait expects to be used in a `TComponent` subclass (for `getSubProperty`-style getters
 * and setters named `getX()`/`setX()`).
 *
 * @author Brad Anderson <belisoful@icloud.com>
 * @since 0.1.0
 */
trait TBayesianTokenizerTrait
{
	/**
	 * Returns the PRADO property names that make up the tokenizer's configuration, in
	 * export order.  Each name `X` must have `getX()` and `setX()`.
	 * @return string[] The property names.
	 */
	protected function getConfigProperties(): array
	{
		return [];
	}

	/**
	 * {@see IBayesianTokenizer::exportConfig()}: the {@see getConfigProperties()} values.
	 * @return array<string, mixed> The configuration.
	 */
	public function exportConfig(): array
	{
		return $this->exportConfigProperties();
	}

	/**
	 * {@see IBayesianTokenizer::importConfig()}: applies the {@see getConfigProperties()} keys.
	 * @param array<string, mixed> $config The configuration.
	 * @throws TInvalidDataValueException When a setter rejects a value (e.g. a bad pattern).
	 */
	public function importConfig(array $config): void
	{
		$this->importConfigProperties($config);
	}

	/**
	 * Reads every {@see getConfigProperties()} property through its getter.
	 * @return array<string, mixed> Property values keyed by lower-camel-case name.
	 */
	protected function exportConfigProperties(): array
	{
		$config = [];
		foreach ($this->getConfigProperties() as $property) {
			$config[lcfirst($property)] = $this->{'get' . $property}();
		}
		return $config;
	}

	/**
	 * Writes every {@see getConfigProperties()} key present in `$config` through its setter,
	 * coercing to the setter's declared type.  Missing keys keep their current values; unknown
	 * keys are ignored.
	 * @param array<string, mixed> $config The configuration.
	 * @throws TInvalidDataValueException When a setter rejects a value.
	 */
	protected function importConfigProperties(array $config): void
	{
		foreach ($this->getConfigProperties() as $property) {
			$key = lcfirst($property);
			if (!array_key_exists($key, $config)) {
				continue;
			}
			$setter = 'set' . $property;
			$this->$setter($this->coerceConfigValue($setter, $config[$key]));
		}
	}

	/**
	 * Coerces a decoded configuration value to the scalar/array type declared by a setter's
	 * first parameter, so JSON round-trips (and hand-written arrays) restore without a
	 * `TypeError`.  A nullable parameter passes null through; other types are cast.
	 * @param string $setter The setter method name.
	 * @param mixed $value The raw value.
	 * @return mixed The coerced value.
	 */
	private function coerceConfigValue(string $setter, $value)
	{
		$parameter = (new \ReflectionMethod($this, $setter))->getParameters()[0] ?? null;
		$type = $parameter?->getType();
		if (!($type instanceof \ReflectionNamedType)) {
			return $value;
		}
		if ($value === null && $type->allowsNull()) {
			return null;
		}
		switch ($type->getName()) {
			case 'int':
				return is_scalar($value) ? (int) $value : 0;
			case 'float':
				return is_scalar($value) ? (float) $value : 0.0;
			case 'bool':
				return (bool) $value;
			case 'string':
				return is_scalar($value) ? (string) $value : '';
			case 'array':
				return is_array($value) ? $value : ($type->allowsNull() ? null : []);
			default:
				return $value;
		}
	}

	/**
	 * Replaces invalid UTF-8 with U+FFFD ({@see TBayesianTokenizerFactory::scrubText()}) and
	 * lowercases the text — the common first step of a case-insensitive tokenizer.
	 * @param string $text The text.
	 * @return string The scrubbed, lowercased text.
	 */
	protected function normalizeText(string $text): string
	{
		return (string) mb_strtolower(TBayesianTokenizerFactory::scrubText($text));
	}

	/**
	 * Runs `preg_match_all` over scrubbed text and returns the match set.  A PCRE failure
	 * (backtrack/recursion limit) throws instead of being mistaken for "no tokens".
	 * @param string $pattern The pattern.
	 * @param string $text The text (already normalized by the caller if desired).
	 * @param int $flags `preg_match_all` flags (default PREG_PATTERN_ORDER).
	 * @throws TInvalidDataValueException `bayesian_tokenizer_pattern_failed` when PCRE fails.
	 * @return array<int, array<int, string>> The `$matches` array; empty when nothing matched.
	 */
	protected function matchAll(string $pattern, string $text, int $flags = PREG_PATTERN_ORDER): array
	{
		$result = preg_match_all($pattern, TBayesianTokenizerFactory::scrubText($text), $matches, $flags);
		if ($result === false) {
			TBayesianTokenizerFactory::checkPregError($pattern);
			return [];
		}
		return $result === 0 ? [] : $matches;
	}
}
