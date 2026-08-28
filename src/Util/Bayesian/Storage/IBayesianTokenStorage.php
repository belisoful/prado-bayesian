<?php

/**
 * IBayesianTokenStorage interface file.
 *
 * @author Brad Anderson <belisoful@icloud.com>
 * @link https://github.com/belisoful/prado-bayesian
 * @license https://github.com/pradosoft/prado/blob/master/LICENSE
 */

namespace Belisoful\Prado\Util\Bayesian\Storage;

/**
 * IBayesianTokenStorage interface.
 *
 * A storage backend that can serve a model's statistics **per token** rather than only as one
 * payload.  It is the seam behind {@see \Belisoful\Prado\Util\Bayesian\TLazyBayesianVocabulary}, which
 * lets a classifier score a document by reading the statistics of that document's tokens
 * instead of loading the whole model into the process.
 *
 * The interface extends {@see IBayesianStorage} rather than replacing it: a backend implementing
 * it still stores and returns whole payloads, so the same class can serve both modes and a
 * classifier can keep using the blob path when that is what fits.
 *
 * Two properties of the contract matter more than the method list:
 *
 * - **{@see loadTokens()} takes an array.**  It is the whole point.  A per-token method would
 *   be a lookup for every token of every document, where this is one round trip for the
 *   document.  Implementations must answer it in a bounded number of queries regardless of how
 *   many tokens are asked for, chunking internally if the driver limits bind parameters.
 * - **Derived quantities are not stored.**  A token's corpus-wide document frequency is the sum
 *   of its per-category document counts, and its corpus-wide occurrence count the sum of its
 *   per-category counts, because a training document belongs to exactly one category.  Both
 *   therefore come out of the same rows {@see loadTokens()} already returns; no separate
 *   vocabulary table is needed, and none can drift out of step.
 *
 * @author Brad Anderson <belisoful@icloud.com>
 * @since 0.1.0
 */
interface IBayesianTokenStorage extends IBayesianStorage
{
	/**
	 * Returns whether this backend is currently configured to store models per token.  A
	 * backend may implement the interface and still be configured for whole-payload storage.
	 * @return bool Whether per-token mode is active.
	 */
	public function getSupportsTokenLookup(): bool;

	/**
	 * Writes a model in per-token form, replacing any model already stored under the name.
	 *
	 * @param string $name The model name.
	 * @param array<string, mixed> $meta The model-level state: kind, tokenizer configuration,
	 * alpha, the scalar totals, and any classifier aggregates that cannot be recomputed without
	 * a full scan.
	 * @param array<string, array{documentCount:int, totalTokens:int}> $categories The
	 * per-category scalars, keyed by category name.
	 * @param array<string, array<string, array{count:int, docCount:int}>> $tokens The per-token
	 * statistics, keyed by token then by category name.
	 */
	public function saveTokenModel(string $name, array $meta, array $categories, array $tokens): void;

	/**
	 * Returns a model's model-level state, or null when the name is unknown.
	 * @param string $name The model name.
	 * @return ?array<string, mixed> The metadata, or null.
	 */
	public function loadTokenMeta(string $name): ?array;

	/**
	 * Returns a model's per-category scalars, keyed by category name.
	 * @param string $name The model name.
	 * @return array<string, array{documentCount:int, totalTokens:int}> The category scalars.
	 */
	public function loadTokenCategories(string $name): array;

	/**
	 * Returns the per-category statistics of the given tokens, in one batched read.
	 *
	 * Tokens with no stored row are simply absent from the result; a caller reads that as
	 * out-of-vocabulary. The returned shape is token => category => `['count' => int,
	 * 'docCount' => int]`.
	 * @param string $name The model name.
	 * @param string[] $tokens The tokens to fetch.
	 * @return array<string, array<string, array{count:int, docCount:int}>> The statistics.
	 */
	public function loadTokens(string $name, array $tokens): array;

	/**
	 * Applies one training document's deltas without rewriting the model.
	 *
	 * This is what makes incremental training proportional to the document rather than to the
	 * model: `$tokenDeltas` carries only the tokens the document contained. Implementations
	 * must apply the whole call atomically — a half-applied document leaves counts that no
	 * later training can reconcile.
	 * @param string $name The model name.
	 * @param string $category The category the document was filed under.
	 * @param array<string, array{count:int, docCount:int}> $tokenDeltas The per-token increments.
	 * @param array<string, mixed> $meta The updated model-level state.
	 * @param array{documentCount:int, totalTokens:int} $categoryStats The category's updated scalars.
	 */
	public function applyDeltas(string $name, string $category, array $tokenDeltas, array $meta, array $categoryStats): void;
}
