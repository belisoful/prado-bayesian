<?php

/**
 * TBayesianTokenizerFactory class file.
 *
 * @author Brad Anderson <belisoful@icloud.com>
 * @link https://github.com/belisoful/prado-bayesian
 * @license https://github.com/pradosoft/prado/blob/master/LICENSE
 */

namespace Prado\Util\Bayesian\Tokenizer;

use Prado\Exceptions\TInvalidDataValueException;

/**
 * TBayesianTokenizerFactory class.
 *
 * Static helpers shared by the tokenizers (through {@see TBayesianTokenizerTrait}) and the
 * classifiers:
 *
 * - {@see export()} / {@see restore()} turn an {@see IBayesianTokenizer} into a JSON-safe
 *   `['class' => FQN, 'config' => [...]]` state and back.  Restoring only ever instantiates a
 *   class that exists and implements {@see IBayesianTokenizer} — a stored payload cannot make
 *   the loader construct an arbitrary class.  {@see restore()} prefers to re-configure an
 *   existing instance of the same class (so a tokenizer the application injected keeps its
 *   identity), and falls back to constructing a fresh one.
 * - {@see scrubText()} normalizes input to valid UTF-8 before it reaches a `/u` regex or the
 *   `mb_*` functions.  Invalid byte sequences are replaced by U+FFFD; without this, a single
 *   bad byte makes `preg_match_all` fail and silently turns a whole document into an empty
 *   feature vector.
 * - {@see assertPattern()} / {@see checkPregError()} turn regex configuration and matching
 *   errors into exceptions instead of empty results.
 *
 * @author Brad Anderson <belisoful@icloud.com>
 * @since 0.1.0
 */
final class TBayesianTokenizerFactory
{
	/**
	 * Serializes a tokenizer into a JSON-safe state array.
	 * @param IBayesianTokenizer $tokenizer The tokenizer.
	 * @return array{class: string, config: array<string, mixed>} The state.
	 */
	public static function export(IBayesianTokenizer $tokenizer): array
	{
		return ['class' => $tokenizer::class, 'config' => $tokenizer->exportConfig()];
	}

	/**
	 * Recreates a tokenizer from a state array produced by {@see export()}.
	 *
	 * When `$current` is an instance of the stored class, its configuration is updated in
	 * place and it is returned.  Otherwise a new instance is constructed.  Returns null when
	 * the state has no usable class name (the caller keeps its current tokenizer).
	 * @param array<string, mixed> $state The state.
	 * @param ?IBayesianTokenizer $current The tokenizer to reuse when its class matches.
	 * @throws TInvalidDataValueException When the stored class does not exist or is not a tokenizer.
	 * @return ?IBayesianTokenizer The restored tokenizer, or null when the state names no class.
	 */
	public static function restore(array $state, ?IBayesianTokenizer $current = null): ?IBayesianTokenizer
	{
		$class = $state['class'] ?? null;
		if (!is_string($class) || $class === '') {
			return null;
		}
		$class = ltrim($class, '\\');
		if (!class_exists($class) || !is_a($class, IBayesianTokenizer::class, true)) {
			throw new TInvalidDataValueException('bayesian_tokenizer_class_invalid', $class);
		}
		$config = $state['config'] ?? [];
		$config = is_array($config) ? $config : [];
		if ($current !== null && $current::class === $class) {
			$current->importConfig($config);
			return $current;
		}
		/** @var IBayesianTokenizer $tokenizer */
		$tokenizer = new $class();
		$tokenizer->importConfig($config);
		return $tokenizer;
	}

	/**
	 * Replaces invalid UTF-8 byte sequences with U+FFFD so the text is safe for `/u` regexes
	 * and `mb_*` functions.  Valid text is returned unchanged.
	 * @param string $text The text.
	 * @return string The scrubbed text.
	 */
	public static function scrubText(string $text): string
	{
		if ($text === '' || mb_check_encoding($text, 'UTF-8')) {
			return $text;
		}
		return (string) mb_convert_encoding($text, 'UTF-8', 'UTF-8');
	}

	/**
	 * Validates that a string is a compilable PCRE pattern.
	 * @param string $pattern The pattern.
	 * @throws TInvalidDataValueException When the pattern does not compile.
	 */
	public static function assertPattern(string $pattern): void
	{
		if ($pattern === '') {
			throw new TInvalidDataValueException('bayesian_tokenizer_pattern_invalid', $pattern, 'empty pattern');
		}
		$reason = null;
		set_error_handler(static function (int $severity, string $message) use (&$reason): bool {
			$reason = $message;
			return true;
		});
		try {
			$result = preg_match($pattern, '');
		} finally {
			restore_error_handler();
		}
		if ($result === false) {
			throw new TInvalidDataValueException('bayesian_tokenizer_pattern_invalid', $pattern, $reason ?? preg_last_error_msg());
		}
	}

	/**
	 * Throws when the last `preg_*` call failed (backtrack/recursion limit, bad UTF-8, ...).
	 * A failed match must not be mistaken for "no tokens".
	 * @param string $pattern The pattern that was run, for the error message.
	 * @throws TInvalidDataValueException When the last PCRE call reported an error.
	 */
	public static function checkPregError(string $pattern): void
	{
		if (preg_last_error() !== PREG_NO_ERROR) {
			throw new TInvalidDataValueException('bayesian_tokenizer_pattern_failed', $pattern, preg_last_error_msg());
		}
	}
}
