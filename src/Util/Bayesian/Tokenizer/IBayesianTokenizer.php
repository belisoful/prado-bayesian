<?php

/**
 * IBayesianTokenizer interface file.
 *
 * @author Brad Anderson <belisoful@icloud.com>
 * @link https://github.com/belisoful/prado-bayesian
 * @license https://github.com/pradosoft/prado/blob/master/LICENSE
 */

namespace Belisoful\Prado\Util\Bayesian\Tokenizer;

/**
 * IBayesianTokenizer interface.
 *
 * The seam between raw text and the features a classifier trains on.  The classifier hands
 * the tokenizer a string and receives a list of tokens (with multiplicity — a token repeated
 * three times in the input should appear three times in the result) which the classifier
 * counts into the category statistics.
 *
 * Implementations decide the token strategy: {@see TWordTokenizer} for plain text,
 * {@see TNGramTokenizer} for character or word n-grams, {@see TRegexTokenizer} for custom
 * patterns, and {@see TBayesianTokenizerChain} to combine several tokenizers in series.
 *
 * A tokenizer is part of a trained model: the same text must tokenize the same way at
 * classification time as it did at training time, or the learned statistics are meaningless.
 * {@see exportConfig()} and {@see importConfig()} therefore serialize the tokenizer's settings
 * into the saved model state (see {@see TBayesianTokenizerFactory}) so a classifier restored
 * from storage tokenizes exactly as the one that was trained.
 *
 * @author Brad Anderson <belisoful@icloud.com>
 * @since 0.1.0
 */
interface IBayesianTokenizer
{
	/**
	 * Splits a string into tokens.  The order of tokens in the result is preserved.
	 * @param string $text The input text.
	 * @return string[] The tokens, in order.
	 */
	public function tokenize(string $text): array;

	/**
	 * Returns the tokenizer's settings as a JSON-serializable array (no objects), so the
	 * tokenizer can be recreated by {@see importConfig()} on a fresh instance of the same class.
	 * @return array<string, mixed> The configuration.
	 */
	public function exportConfig(): array;

	/**
	 * Applies settings previously produced by {@see exportConfig()}.  Missing keys keep the
	 * instance's current values; unknown keys are ignored.
	 * @param array<string, mixed> $config The configuration.
	 */
	public function importConfig(array $config): void;
}
