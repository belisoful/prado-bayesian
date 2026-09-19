<?php

/**
 * TBayesianTokenHistogram class file.
 *
 * @author Brad Anderson <belisoful@icloud.com>
 * @link https://github.com/belisoful/prado-bayesian
 * @license https://github.com/pradosoft/prado/blob/master/LICENSE
 */

namespace Belisoful\Prado\Util\Bayesian;

/**
 * TBayesianTokenHistogram class.
 *
 * The "count of counts" histograms that let a classifier sum a quantity over the whole
 * vocabulary without walking it.  {@see \Belisoful\Prado\Util\Bayesian\Classifier\TBernoulliNaiveBayes}
 * needs `sum over V of log(1 - P(token | category))` and
 * {@see \Belisoful\Prado\Util\Bayesian\Classifier\TComplementNaiveBayes} the L1 norm of a
 * category's complement weights; every term of both depends on a token only through a small
 * integer — how many of the category's documents contain it, how often it occurs outside the
 * category — so the sum over tokens is a sum over the distinct integers, weighted by how many
 * tokens share each one.  Those weights are integers, move by exact increments when a document
 * is trained or withdrawn, and do not depend on alpha, which is what lets a per-token storage
 * keep them current under concurrent writers.
 *
 * Four families exist.  Each counts the tokens of the vocabulary (the tokens some document
 * contains) that share a value:
 *
 * - {@see FAMILY_DOCUMENTS} `d`, per category: the number of the category's documents that
 *   contain the token, for tokens where that is at least one.
 * - {@see FAMILY_GLOBAL} `g`, model-wide (category `''`): the token's occurrences in all
 *   categories together.
 * - {@see FAMILY_CATEGORY_GLOBAL} `a`, per category: the same global count, for the tokens that
 *   occur in the category.
 * - {@see FAMILY_COMPLEMENT} `k`, per category: the token's occurrences outside the category,
 *   for the tokens that occur in it.
 *
 * A category's complement counts over the *whole* vocabulary are then `g - a + k`
 * ({@see complementCounts()}): every token counts by its global count, except the category's own
 * tokens, which count by their complement instead.  Keeping `a` and `k` only for the tokens a
 * category contains keeps a training write proportional to the document, not to the number of
 * categories.
 *
 * A histogram travels in two shapes: *flat*, `key => count` with the key built by {@see key()}
 * (what a storage moves by increments), and *expanded*, `family => category => value => count`
 * (what a classifier reads).
 *
 * @author Brad Anderson <belisoful@icloud.com>
 * @since 0.2.0
 */
final class TBayesianTokenHistogram
{
	/** Per category: documents of the category containing the token. */
	public const FAMILY_DOCUMENTS = 'd';

	/** Model-wide: the token's occurrences across all categories. */
	public const FAMILY_GLOBAL = 'g';

	/** Per category: the global occurrences of the tokens that occur in the category. */
	public const FAMILY_CATEGORY_GLOBAL = 'a';

	/** Per category: the occurrences outside the category of the tokens that occur in it. */
	public const FAMILY_COMPLEMENT = 'k';

	/** Every family, in the canonical order. */
	public const FAMILIES = [self::FAMILY_DOCUMENTS, self::FAMILY_GLOBAL, self::FAMILY_CATEGORY_GLOBAL, self::FAMILY_COMPLEMENT];

	/**
	 * Returns the valid family letters in a raw value (a stored `histograms` list), without
	 * duplicates and in the canonical order.
	 * @param mixed $value The raw value.
	 * @return string[] The families.
	 */
	public static function families($value): array
	{
		if (!is_array($value)) {
			return [];
		}
		return array_values(array_filter(self::FAMILIES, static fn (string $family): bool => in_array($family, $value, true)));
	}

	/**
	 * Returns the flat key of one histogram cell: the family letter, the value, a colon, and the
	 * category (empty for the model-wide family).
	 * @param string $family The family letter.
	 * @param string $category The category name.
	 * @param int $value The value the counted tokens share.
	 * @return string The key.
	 */
	public static function key(string $family, string $category, int $value): string
	{
		return $family . $value . ':' . $category;
	}

	/**
	 * Splits a flat key into its family, category and value.
	 * @param string $key The key.
	 * @return ?array{0:string, 1:string, 2:int} The parts, or null when the key is not one.
	 */
	public static function parseKey(string $key): ?array
	{
		$colon = strpos($key, ':');
		if ($colon === false || $colon < 2 || !in_array($key[0], self::FAMILIES, true)) {
			return null;
		}
		$value = substr($key, 1, $colon - 1);
		if (!ctype_digit($value)) {
			return null;
		}
		return [$key[0], substr($key, $colon + 1), (int) $value];
	}

	/**
	 * Returns the cells one token counts towards, given its statistics in every category.  A
	 * token no document contains is outside the vocabulary and counts towards nothing.
	 * @param array<string, array{count?:int, docCount?:int}> $rows The token's statistics, keyed by category.
	 * @param string[] $families The families wanted.
	 * @return array<string, int> The flat keys, each mapped to 1.
	 */
	public static function contributions(array $rows, array $families): array
	{
		$documents = 0;
		$global = 0;
		foreach ($rows as $stats) {
			$documents += max(0, (int) ($stats['docCount'] ?? 0));
			$global += max(0, (int) ($stats['count'] ?? 0));
		}
		if ($documents <= 0 || $families === []) {
			return [];
		}
		$wanted = array_fill_keys($families, true);
		$out = [];
		if (isset($wanted[self::FAMILY_DOCUMENTS])) {
			foreach ($rows as $category => $stats) {
				$docCount = (int) ($stats['docCount'] ?? 0);
				if ($docCount > 0) {
					$out[self::key(self::FAMILY_DOCUMENTS, (string) $category, $docCount)] = 1;
				}
			}
		}
		if (isset($wanted[self::FAMILY_GLOBAL])) {
			$out[self::key(self::FAMILY_GLOBAL, '', $global)] = 1;
		}
		foreach ($rows as $category => $stats) {
			$count = (int) ($stats['count'] ?? 0);
			if ($count <= 0) {
				continue;
			}
			if (isset($wanted[self::FAMILY_CATEGORY_GLOBAL])) {
				$out[self::key(self::FAMILY_CATEGORY_GLOBAL, (string) $category, $global)] = 1;
			}
			if (isset($wanted[self::FAMILY_COMPLEMENT])) {
				$out[self::key(self::FAMILY_COMPLEMENT, (string) $category, $global - $count)] = 1;
			}
		}
		return $out;
	}

	/**
	 * Returns what a write changes: the cells the tokens counted towards afterwards, minus the
	 * cells they counted towards before.  Cells that net to zero are left out.
	 * @param array<string, array<string, array{count?:int, docCount?:int}>> $before The touched tokens' statistics before the write.
	 * @param array<string, array<string, array{count?:int, docCount?:int}>> $after The same tokens' statistics after it.
	 * @param string[] $families The families wanted.
	 * @return array<string, int> The flat increments.
	 */
	public static function delta(array $before, array $after, array $families): array
	{
		$delta = [];
		foreach ($after as $rows) {
			foreach (self::contributions($rows, $families) as $key => $one) {
				$delta[$key] = ($delta[$key] ?? 0) + $one;
			}
		}
		foreach ($before as $rows) {
			foreach (self::contributions($rows, $families) as $key => $one) {
				$delta[$key] = ($delta[$key] ?? 0) - $one;
			}
		}
		return array_filter($delta);
	}

	/**
	 * Builds the flat histogram of a whole model from its per-token statistics.
	 * @param array<string, array<string, array{count?:int, docCount?:int}>> $tokens The statistics, keyed by token then category.
	 * @param string[] $families The families wanted.
	 * @return array<string, int> The flat histogram.
	 */
	public static function build(array $tokens, array $families): array
	{
		return self::delta([], $tokens, $families);
	}

	/**
	 * Builds the expanded histogram of a vocabulary that can be enumerated, without copying its
	 * per-token maps.
	 * @param IBayesianVocabulary $vocabulary The vocabulary; must support a full scan.
	 * @param string[] $families The families wanted.
	 * @return array<string, array<string, array<int, int>>> The expanded histogram.
	 */
	public static function fromVocabulary(IBayesianVocabulary $vocabulary, array $families): array
	{
		$wanted = array_fill_keys($families, true);
		if ($wanted === []) {
			return [];
		}
		$documentFrequency = $vocabulary->getDocumentFrequency();
		$out = [];
		if (isset($wanted[self::FAMILY_GLOBAL])) {
			foreach ($documentFrequency as $token => $_) {
				$global = $vocabulary->getTokenGlobalCount((string) $token);
				$out[self::FAMILY_GLOBAL][''][$global] = ($out[self::FAMILY_GLOBAL][''][$global] ?? 0) + 1;
			}
		}
		foreach ($vocabulary->getCategories() as $category) {
			$name = $category->getName();
			if (isset($wanted[self::FAMILY_DOCUMENTS])) {
				foreach ($category->getTokenDocumentCounts() as $token => $docCount) {
					if ($docCount > 0 && isset($documentFrequency[$token])) {
						$out[self::FAMILY_DOCUMENTS][$name][$docCount] = ($out[self::FAMILY_DOCUMENTS][$name][$docCount] ?? 0) + 1;
					}
				}
			}
			if (!isset($wanted[self::FAMILY_CATEGORY_GLOBAL]) && !isset($wanted[self::FAMILY_COMPLEMENT])) {
				continue;
			}
			foreach ($category->getTokenCounts() as $token => $count) {
				if ($count <= 0 || !isset($documentFrequency[$token])) {
					continue;
				}
				$global = $vocabulary->getTokenGlobalCount((string) $token);
				if (isset($wanted[self::FAMILY_CATEGORY_GLOBAL])) {
					$out[self::FAMILY_CATEGORY_GLOBAL][$name][$global] = ($out[self::FAMILY_CATEGORY_GLOBAL][$name][$global] ?? 0) + 1;
				}
				if (isset($wanted[self::FAMILY_COMPLEMENT])) {
					$complement = $global - $count;
					$out[self::FAMILY_COMPLEMENT][$name][$complement] = ($out[self::FAMILY_COMPLEMENT][$name][$complement] ?? 0) + 1;
				}
			}
		}
		return $out;
	}

	/**
	 * Converts a flat histogram into the expanded shape, dropping cells that are not positive
	 * and keys that do not parse.
	 * @param array<array-key, mixed> $flat The flat histogram.
	 * @return array<string, array<string, array<int, int>>> The expanded histogram.
	 */
	public static function expand(array $flat): array
	{
		$out = [];
		foreach ($flat as $key => $count) {
			$count = TBayesianPayload::int($count);
			$parts = self::parseKey((string) $key);
			if ($count <= 0 || $parts === null) {
				continue;
			}
			[$family, $category, $value] = $parts;
			$out[$family][$category][$value] = $count;
		}
		return $out;
	}

	/**
	 * Converts an expanded histogram into the flat shape, dropping cells that are not positive.
	 * @param array<string, array<string, array<int, int>>> $expanded The expanded histogram.
	 * @return array<string, int> The flat histogram.
	 */
	public static function flatten(array $expanded): array
	{
		$out = [];
		foreach ($expanded as $family => $categories) {
			foreach ($categories as $category => $values) {
				foreach ($values as $value => $count) {
					if ($count > 0) {
						$out[self::key((string) $family, (string) $category, (int) $value)] = (int) $count;
					}
				}
			}
		}
		return $out;
	}

	/**
	 * Returns how many vocabulary tokens have each complement count for a category — the
	 * token's occurrences outside it — merged from the three complement families as
	 * `g - a + k` and sorted by value, so a sum over it runs in the same order wherever the
	 * histogram came from.
	 * @param array<string, array<string, array<int, int>>> $expanded The expanded histogram.
	 * @param string $category The category name.
	 * @return array<int, int> The token count per complement count, without empty cells.
	 */
	public static function complementCounts(array $expanded, string $category): array
	{
		$out = $expanded[self::FAMILY_GLOBAL][''] ?? [];
		foreach ($expanded[self::FAMILY_CATEGORY_GLOBAL][$category] ?? [] as $value => $count) {
			$out[$value] = ($out[$value] ?? 0) - $count;
		}
		foreach ($expanded[self::FAMILY_COMPLEMENT][$category] ?? [] as $value => $count) {
			$out[$value] = ($out[$value] ?? 0) + $count;
		}
		$out = array_filter($out, static fn (int $count): bool => $count > 0);
		ksort($out, SORT_NUMERIC);
		return $out;
	}
}
