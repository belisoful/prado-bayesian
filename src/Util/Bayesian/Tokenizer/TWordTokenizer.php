<?php

/**
 * TWordTokenizer class file.
 *
 * @author Brad Anderson <belisoful@icloud.com>
 * @link https://github.com/belisoful/prado-bayesian
 * @license https://github.com/pradosoft/prado/blob/master/LICENSE
 */

namespace Prado\Util\Bayesian\Tokenizer;

use Prado\Exceptions\TInvalidDataValueException;
use Prado\TComponent;

/**
 * TWordTokenizer class.
 *
 * The default word tokenizer for spam filtering and similar text classification.  It lowercases
 * the input (so "Free" and "free" match), splits on Unicode letter/digit runs, drops tokens
 * shorter than {@see getMinLength() min length} (so single letters and stray punctuation do
 * not bloat the vocabulary), and optionally drops a stop-word list.
 *
 * The stop-word list is consulted after lowercasing, so callers should provide lowercase
 * entries.  When {@see setStopWords() stop words} is null (the default), no filtering is
 * applied — set an empty array to filter all stop words (which is no-op until entries are
 * added).
 *
 * @author Brad Anderson <belisoful@icloud.com>
 * @since 0.1.0
 */
class TWordTokenizer extends TComponent implements IBayesianTokenizer
{
	use TBayesianTokenizerTrait;

	/** @var int The minimum token length kept (>= 1). */
	private int $_minLength = 2;

	/** @var ?array<string, true> Lowercased stop words; null disables the filter. */
	private ?array $_stopWords = null;

	/** @var string The Unicode-letters-and-digits pattern used to split the text. */
	private string $_pattern = '/[\p{L}\p{N}]+/u';

	/**
	 * Tokenizes text.  Invalid UTF-8 byte sequences are replaced by U+FFFD before matching
	 * (see {@see TBayesianTokenizerFactory::scrubText()}) so a stray byte cannot empty the
	 * whole feature vector.
	 * @param string $text The text to tokenize.
	 * @throws TInvalidDataValueException When the pattern fails to run (e.g. backtrack limit).
	 * @return string[] The tokens, in order, with stop words and short tokens removed.
	 */
	public function tokenize(string $text): array
	{
		$matches = $this->matchAll($this->_pattern, $this->normalizeText($text));
		if ($matches === []) {
			return [];
		}
		$tokens = [];
		$minLength = $this->_minLength;
		$stopWords = $this->_stopWords;
		foreach ($matches[0] as $token) {
			if (mb_strlen($token) < $minLength) {
				continue;
			}
			if ($stopWords !== null && isset($stopWords[$token])) {
				continue;
			}
			$tokens[] = $token;
		}
		return $tokens;
	}

	/**
	 * Returns the minimum token length kept.
	 * @return int The minimum length.
	 */
	public function getMinLength(): int
	{
		return $this->_minLength;
	}

	/**
	 * Sets the minimum token length kept; values < 1 are clamped to 1.
	 * @param int $value The minimum length.
	 */
	public function setMinLength(int $value): void
	{
		$this->_minLength = $value < 1 ? 1 : $value;
	}

	/**
	 * Returns the stop words as a list, or null when the filter is disabled.
	 * @return ?string[] The stop words, or null.
	 */
	public function getStopWords(): ?array
	{
		if ($this->_stopWords === null) {
			return null;
		}
		return array_keys($this->_stopWords);
	}

	/**
	 * Sets the stop words.  Pass null to disable the filter, an empty array to enable it with
	 * no entries (which is a no-op), or a list of lowercase words to filter.
	 * @param null|string[] $value The stop words.
	 */
	public function setStopWords(?array $value): void
	{
		if ($value === null) {
			$this->_stopWords = null;
			return;
		}
		$map = [];
		foreach ($value as $word) {
			$map[(string) mb_strtolower($word)] = true;
		}
		$this->_stopWords = $map;
	}

	/**
	 * Returns the regex pattern used to split the text.
	 * @return string The pattern.
	 */
	public function getPattern(): string
	{
		return $this->_pattern;
	}

	/**
	 * Sets the regex pattern used to split the text.  The pattern is run with
	 * `preg_match_all` and the full matches (group 0) become tokens; no capturing group
	 * is required.  Include the `u` modifier for non-ASCII text, otherwise multibyte
	 * characters split into byte fragments.
	 * @param string $value The pattern.
	 * @throws TInvalidDataValueException When the pattern does not compile.
	 */
	public function setPattern(string $value): void
	{
		TBayesianTokenizerFactory::assertPattern($value);
		$this->_pattern = $value;
	}

	/**
	 * {@inheritDoc}
	 * @return string[] `MinLength`, `StopWords`, `Pattern`.
	 */
	protected function getConfigProperties(): array
	{
		return ['MinLength', 'StopWords', 'Pattern'];
	}
}
