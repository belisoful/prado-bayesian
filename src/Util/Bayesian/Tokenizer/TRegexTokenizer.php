<?php

/**
 * TRegexTokenizer class file.
 *
 * @author Brad Anderson <belisoful@icloud.com>
 * @link https://github.com/belisoful/prado-bayesian
 * @license https://github.com/pradosoft/prado/blob/master/LICENSE
 */

namespace Belisoful\Prado\Util\Bayesian\Tokenizer;

use Prado\Exceptions\TInvalidDataValueException;
use Prado\TComponent;

/**
 * TRegexTokenizer class.
 *
 * A regex-driven tokenizer: every match of the configured pattern becomes a token, optionally
 * transformed (e.g. lowercased) before being emitted.  This is the escape hatch for
 * application-specific features — emails, phone numbers, hex hashes, structured identifiers
 * — that the default {@see TWordTokenizer} would miss or split.
 *
 * The pattern is run with `preg_match_all`.  When the pattern has a capturing group, the first
 * group is taken as the token — consistently for every match, so an optional group that did
 * not participate yields no token rather than the whole match.  A pattern without capturing
 * groups uses the whole match (so a plain pattern like `/\w+/u` works without parentheses).
 * When the pattern matches nothing the result is empty.  Include the `u` modifier for
 * non-ASCII text, otherwise multibyte characters split into byte fragments.
 *
 * Invalid UTF-8 in the input is replaced by U+FFFD before matching, and a pattern that fails
 * to run (backtrack limit, invalid pattern) throws instead of silently returning no tokens.
 *
 * @author Brad Anderson <belisoful@icloud.com>
 * @since 0.1.0
 */
class TRegexTokenizer extends TComponent implements IBayesianTokenizer
{
	use TBayesianTokenizerTrait;

	/** @var string The pattern; the first capturing group (or the whole match) is the token. */
	private string $_pattern = '/([\p{L}\p{N}]+)/u';

	/** @var bool Whether to lowercase each token after matching. */
	private bool $_lowercase = true;

	/**
	 * Tokenizes by running the configured pattern over the text.  When the pattern has a
	 * capturing group, group 1 supplies the tokens; otherwise the whole match does — so a
	 * pattern can either match tokens directly or match around them and capture the part it
	 * wants.
	 * @param string $text The text to tokenize.
	 * @throws TInvalidDataValueException When the pattern fails to run (e.g. backtrack limit).
	 * @return string[] The tokens, in match order.
	 */
	public function tokenize(string $text): array
	{
		$matches = $this->matchAll($this->_pattern, $text);
		if ($matches === []) {
			return [];
		}
		// PREG_PATTERN_ORDER always yields a group-1 column when the pattern has a capturing
		// group (unmatched groups are ''), so the group/whole-match choice is made once per
		// pattern instead of per match.
		$column = $matches[1] ?? $matches[0];
		$tokens = [];
		foreach ($column as $token) {
			$token = (string) $token;
			if ($token === '') {
				continue;
			}
			$tokens[] = $this->_lowercase ? (string) mb_strtolower($token) : $token;
		}
		return $tokens;
	}

	/**
	 * Returns the regex pattern.
	 * @return string The pattern.
	 */
	public function getPattern(): string
	{
		return $this->_pattern;
	}

	/**
	 * Sets the regex pattern.  If it has a capturing group, the first group is the token;
	 * otherwise the whole match is used.
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
	 * @return string[] `Pattern`, `Lowercase`.
	 */
	protected function getConfigProperties(): array
	{
		return ['Pattern', 'Lowercase'];
	}

	/**
	 * Returns whether matched tokens are lowercased.
	 * @return bool The flag.
	 */
	public function getLowercase(): bool
	{
		return $this->_lowercase;
	}

	/**
	 * Sets the lowercasing flag.
	 * @param bool $value The flag.
	 */
	public function setLowercase(bool $value): void
	{
		$this->_lowercase = $value;
	}
}
