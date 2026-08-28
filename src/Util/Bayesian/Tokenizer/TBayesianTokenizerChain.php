<?php

/**
 * TBayesianTokenizerChain class file.
 *
 * @author Brad Anderson <belisoful@icloud.com>
 * @link https://github.com/belisoful/prado-bayesian
 * @license https://github.com/pradosoft/prado/blob/master/LICENSE
 */

namespace Belisoful\Prado\Util\Bayesian\Tokenizer;

use Prado\Exceptions\TInvalidDataValueException;
use Prado\TComponent;

/**
 * TBayesianTokenizerChain class.
 *
 * Composes several {@see IBayesianTokenizer} instances, concatenating their token streams in
 * registration order.  Use it to combine orthogonal feature views — for example, a
 * {@see TWordTokenizer} for general words, a {@see TRegexTokenizer} for email-address
 * recognition, and a {@see TNGramTokenizer} for character-shape features — so the classifier
 * sees the union of their features.
 *
 * Tokenizers are kept in the order they were added; the chain re-runs each one over the same
 * input text and appends the results.
 *
 * @author Brad Anderson <belisoful@icloud.com>
 * @since 0.1.0
 */
class TBayesianTokenizerChain extends TComponent implements IBayesianTokenizer
{
	use TBayesianTokenizerTrait;

	/** @var IBayesianTokenizer[] The tokenizers, in chain order. */
	private array $_tokenizers = [];

	/**
	 * Tokenizes the text with every chain member and concatenates the results.  Each member
	 * sees the original text, not the previous member's output, so a chain composes
	 * complementary feature sets (words plus character n-grams, say) rather than piping one
	 * tokenizer into the next.  Duplicates are kept: a token produced by two members counts
	 * twice, which is what the multinomial event model expects.
	 * @param string $text The text to tokenize.
	 * @return string[] The concatenated tokens of every chain member, in order.
	 */
	public function tokenize(string $text): array
	{
		$tokens = [];
		foreach ($this->_tokenizers as $tokenizer) {
			foreach ($tokenizer->tokenize($text) as $token) {
				$tokens[] = $token;
			}
		}
		return $tokens;
	}

	/**
	 * Appends a tokenizer to the chain.
	 * @param IBayesianTokenizer $tokenizer The tokenizer.
	 */
	public function addTokenizer(IBayesianTokenizer $tokenizer): void
	{
		$this->_tokenizers[] = $tokenizer;
	}

	/**
	 * Removes a tokenizer from the chain by reference; subsequent calls to {@see tokenize()}
	 * no longer consult it.
	 * @param IBayesianTokenizer $tokenizer The tokenizer to remove.
	 * @return bool Whether the tokenizer was found and removed.
	 */
	public function removeTokenizer(IBayesianTokenizer $tokenizer): bool
	{
		foreach ($this->_tokenizers as $index => $existing) {
			if ($existing === $tokenizer) {
				array_splice($this->_tokenizers, $index, 1);
				return true;
			}
		}
		return false;
	}

	/**
	 * Returns the tokenizers in the chain, in execution order.
	 * @return IBayesianTokenizer[] The tokenizers.
	 */
	public function getTokenizers(): array
	{
		return $this->_tokenizers;
	}

	/**
	 * Clears the chain.
	 */
	public function clear(): void
	{
		$this->_tokenizers = [];
	}

	/**
	 * {@inheritDoc}
	 * @return array<string, mixed> A `tokenizers` list of exported member states, in chain order.
	 */
	public function exportConfig(): array
	{
		$members = [];
		foreach ($this->_tokenizers as $tokenizer) {
			$members[] = TBayesianTokenizerFactory::export($tokenizer);
		}
		return ['tokenizers' => $members];
	}

	/**
	 * {@inheritDoc}
	 *
	 * Replaces the chain members with those in the configuration.  Existing members of the
	 * same class at the same position are re-configured in place; others are constructed.
	 * @param array<string, mixed> $config The configuration.
	 * @throws TInvalidDataValueException When a stored member class is not a tokenizer.
	 */
	public function importConfig(array $config): void
	{
		if (!isset($config['tokenizers']) || !is_array($config['tokenizers'])) {
			return;
		}
		$members = [];
		foreach (array_values($config['tokenizers']) as $index => $state) {
			if (!is_array($state)) {
				continue;
			}
			$restored = TBayesianTokenizerFactory::restore($state, $this->_tokenizers[$index] ?? null);
			if ($restored !== null) {
				$members[] = $restored;
			}
		}
		$this->_tokenizers = $members;
	}
}
