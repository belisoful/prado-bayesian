<?php

/**
 * TNGramTokenizer class file.
 *
 * @author Brad Anderson <belisoful@icloud.com>
 * @link https://github.com/belisoful/prado-bayesian
 * @license https://github.com/pradosoft/prado/blob/master/LICENSE
 */

namespace Belisoful\Prado\Util\Bayesian\Tokenizer;

use Prado\TComponent;

/**
 * TNGramTokenizer class.
 *
 * Splits input into character- or word-level n-grams.  N-grams are short sequences of
 * consecutive units (`n` is the width) and are useful when the discriminating feature is
 * morphology — common in language identification, spam filtering across scripts, and
 * misspellings — or when the meaningful signal is multi-word phrases.
 *
 * Word n-grams reuse the {@see TWordTokenizer} split; character n-grams slide across the raw
 * (lowercased) text.  Padded character n-grams emit the leading and trailing partials so the
 * model sees the boundaries; un-padded n-grams only emit complete-width windows.
 *
 * @author Brad Anderson <belisoful@icloud.com>
 * @since 0.1.0
 */
class TNGramTokenizer extends TComponent implements IBayesianTokenizer
{
	use TBayesianTokenizerTrait;

	/** @var int The n-gram width (>= 1). */
	private int $_n = 3;

	/** @var bool Whether to operate on character-level n-grams (true) or word-level (false). */
	private bool $_characters = true;

	/** @var bool Whether to emit the padded boundary n-grams (character mode only). */
	private bool $_pad = true;

	/** @var TWordTokenizer The word tokenizer reused in word mode. */
	private TWordTokenizer $_wordTokenizer;

	/**
	 * Initializes the tokenizer with a default {@see TWordTokenizer} for word mode.
	 */
	public function __construct()
	{
		$this->_wordTokenizer = new TWordTokenizer();
		parent::__construct();
	}

	/**
	 * Tokenizes text.  In character mode, runs of whitespace are collapsed to a single space
	 * and leading/trailing whitespace is trimmed first, so `" ab "` and `"ab"` produce the
	 * same n-grams and whitespace-only input yields no tokens.  Invalid UTF-8 is replaced by
	 * U+FFFD.  Character units are Unicode code points (a base letter and a following
	 * combining mark are two units).
	 * @param string $text The text to tokenize.
	 * @return string[] The n-gram tokens, in order.
	 */
	public function tokenize(string $text): array
	{
		$text = $this->normalizeText($text);
		if ($this->_characters) {
			$text = trim((string) preg_replace('/\s+/u', ' ', $text));
			return $this->characterNGrams($text);
		}
		return $this->wordNGrams($text);
	}

	/**
	 * Builds character n-grams from the (already-lowercased) text.
	 * @param string $text The text.
	 * @return string[] The n-grams.
	 */
	private function characterNGrams(string $text): array
	{
		if ($text === '') {
			return [];
		}
		$units = $this->splitCharacters($text);
		$n = $this->_n;
		if ($n <= 1) {
			return $units;
		}
		$count = count($units);
		if ($count < $n) {
			if ($this->_pad) {
				return [$this->mbPad($text, $n, STR_PAD_RIGHT)];
			}
			return [];
		}
		$tokens = [];
		if ($this->_pad) {
			for ($i = 1; $i < $n; $i++) {
				$tokens[] = $this->mbPad(implode('', array_slice($units, 0, $i)), $n, STR_PAD_RIGHT);
			}
		}
		for ($i = 0; $i <= $count - $n; $i++) {
			$tokens[] = implode('', array_slice($units, $i, $n));
		}
		if ($this->_pad) {
			for ($i = $count - $n + 1; $i < $count; $i++) {
				$tokens[] = $this->mbPad(implode('', array_slice($units, $i)), $n, STR_PAD_LEFT);
			}
		}
		return $tokens;
	}

	/**
	 * Builds word n-grams from the text by tokenizing with the inner {@see TWordTokenizer}.
	 * @param string $text The text.
	 * @return string[] The word n-grams, joined by a single space.
	 */
	private function wordNGrams(string $text): array
	{
		$words = $this->_wordTokenizer->tokenize($text);
		$n = $this->_n;
		if ($n <= 1) {
			return $words;
		}
		$count = count($words);
		if ($count < $n) {
			return [];
		}
		$tokens = [];
		for ($i = 0; $i <= $count - $n; $i++) {
			$tokens[] = implode(' ', array_slice($words, $i, $n));
		}
		return $tokens;
	}

	/**
	 * Splits a string into an array of single-character units, multibyte-safe.
	 * @param string $text The text.
	 * @return string[] The character units.
	 */
	private function splitCharacters(string $text): array
	{
		if ($text === '') {
			return [];
		}
		$out = [];
		$length = mb_strlen($text);
		for ($i = 0; $i < $length; $i++) {
			$out[] = mb_substr($text, $i, 1);
		}
		return $out;
	}

	/**
	 * Pads a string to a target character length with spaces, multibyte-safe.
	 *
	 * PHP's {@see str_pad()} counts bytes, which under-pads multibyte strings.  This helper
	 * counts characters via {@see mb_strlen()} so the result has the correct character width.
	 * @param string $text The text to pad.
	 * @param int $length The target character length.
	 * @param int $type STR_PAD_RIGHT or STR_PAD_LEFT.
	 * @return string The padded string.
	 */
	private function mbPad(string $text, int $length, int $type): string
	{
		$padding = $length - mb_strlen($text);
		if ($padding <= 0) {
			return $text;
		}
		$pad = str_repeat(' ', $padding);
		return $type === STR_PAD_RIGHT ? $text . $pad : $pad . $text;
	}

	/**
	 * Returns the n-gram width.
	 * @return int The width.
	 */
	public function getN(): int
	{
		return $this->_n;
	}

	/**
	 * Sets the n-gram width; values < 1 are clamped to 1.
	 * @param int $value The width.
	 */
	public function setN(int $value): void
	{
		$this->_n = $value < 1 ? 1 : $value;
	}

	/**
	 * Returns whether the tokenizer emits character (true) or word (false) n-grams.
	 * @return bool The mode.
	 */
	public function getCharacters(): bool
	{
		return $this->_characters;
	}

	/**
	 * Sets the n-gram mode.
	 * @param bool $value True for character n-grams, false for word n-grams.
	 */
	public function setCharacters(bool $value): void
	{
		$this->_characters = $value;
	}

	/**
	 * Returns whether boundary-padded n-grams are emitted (character mode).
	 * @return bool The padding flag.
	 */
	public function getPad(): bool
	{
		return $this->_pad;
	}

	/**
	 * Sets the boundary-padding flag (character mode only).
	 * @param bool $value The flag.
	 */
	public function setPad(bool $value): void
	{
		$this->_pad = $value;
	}

	/**
	 * {@inheritDoc}
	 * @return string[] `N`, `Characters`, `Pad` (the nested word tokenizer is handled separately).
	 */
	protected function getConfigProperties(): array
	{
		return ['N', 'Characters', 'Pad'];
	}

	/**
	 * {@inheritDoc}
	 * @return array<string, mixed> The `n`, `characters`, `pad`, and nested `wordTokenizer` settings.
	 */
	public function exportConfig(): array
	{
		return $this->exportConfigProperties() + ['wordTokenizer' => $this->_wordTokenizer->exportConfig()];
	}

	/**
	 * {@inheritDoc}
	 * @param array<string, mixed> $config The configuration.
	 */
	public function importConfig(array $config): void
	{
		$this->importConfigProperties($config);
		if (isset($config['wordTokenizer']) && is_array($config['wordTokenizer'])) {
			$this->_wordTokenizer->importConfig($config['wordTokenizer']);
		}
	}

	/**
	 * Returns the inner word tokenizer used in word n-gram mode.
	 * @return TWordTokenizer The word tokenizer.
	 */
	public function getWordTokenizer(): TWordTokenizer
	{
		return $this->_wordTokenizer;
	}

	/**
	 * Sets the inner word tokenizer used in word n-gram mode.
	 * @param TWordTokenizer $value The word tokenizer.
	 */
	public function setWordTokenizer(TWordTokenizer $value): void
	{
		$this->_wordTokenizer = $value;
	}
}
