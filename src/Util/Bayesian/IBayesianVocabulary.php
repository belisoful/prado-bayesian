<?php

/**
 * IBayesianVocabulary interface file.
 *
 * @author Brad Anderson <belisoful@icloud.com>
 * @link https://github.com/belisoful/prado-bayesian
 * @license https://github.com/pradosoft/prado/blob/master/LICENSE
 */

namespace Belisoful\Prado\Util\Bayesian;

/**
 * IBayesianVocabulary interface.
 *
 * The training statistics a classifier scores against: one {@see TBayesianCategory} per label,
 * the cross-category document totals, and the per-token document frequencies.
 *
 * The interface exists so the statistics need not all be resident.  {@see TBayesianVocabulary}
 * holds the whole model in the process, which is the right thing for a model that fits;
 * a storage-backed implementation can instead fetch only the tokens a document contains.  The
 * split that makes that possible runs through this interface:
 *
 * - **Scalars and categories** ({@see getTotalDocuments()}, {@see getVocabularySize()},
 *   {@see getCategories()}) are always cheap.  There are as many categories as labels, not as
 *   many as tokens.
 * - **Per-token reads** ({@see hasToken()}, {@see getTokenDocumentFrequency()}, and
 *   {@see TBayesianCategory::getTokenCount()}) are cheap only for tokens that have been
 *   fetched.  Call {@see prefetch()} with the document's tokens first; an implementation that
 *   holds everything ignores it, and a storage-backed one turns it into a single batched read.
 * - **Whole-map reads** ({@see getDocumentFrequency()}) are not always available.  Check
 *   {@see getSupportsFullScan()} before iterating the vocabulary, and prefer a per-token read
 *   whenever the tokens of interest are known — which, on the classification path, they are.
 *
 * @author Brad Anderson <belisoful@icloud.com>
 * @since 0.1.0
 */
interface IBayesianVocabulary
{
	/**
	 * Returns the named category, or null if it does not exist.
	 * @param string $name The category name.
	 * @return ?TBayesianCategory The category, or null.
	 */
	public function getCategory(string $name): ?TBayesianCategory;

	/**
	 * Returns the categories in the order they were first added.
	 * @return TBayesianCategory[] The categories.
	 */
	public function getCategories(): array;

	/**
	 * Returns the category names in the order they were first added.
	 * @return string[] The category names.
	 */
	public function getCategoryNames(): array;

	/**
	 * Returns whether any category has been added.
	 * @return bool Whether the vocabulary is empty.
	 */
	public function getIsEmpty(): bool;

	/**
	 * Returns the total number of training documents across all categories.
	 * @return int The total.
	 */
	public function getTotalDocuments(): int;

	/**
	 * Returns the number of distinct tokens seen anywhere in the corpus — |V|, the size the
	 * classifiers smooth against.
	 *
	 * This is a scalar rather than `count()` of the document-frequency map on purpose: it is
	 * needed on every classification, and materializing the map to count it is exactly what a
	 * storage-backed vocabulary exists to avoid.
	 * @return int The vocabulary size.
	 */
	public function getVocabularySize(): int;

	/**
	 * Returns whether a token was seen anywhere in the corpus.  A token that was not carries no
	 * learned evidence and is skipped at classification time.
	 * @param string $token The token.
	 * @return bool Whether the token is in the vocabulary.
	 */
	public function hasToken(string $token): bool;

	/**
	 * Returns the number of training documents containing the token, across all categories.
	 * @param string $token The token.
	 * @return int The document frequency; 0 when the token is unknown.
	 */
	public function getTokenDocumentFrequency(string $token): int;

	/**
	 * Returns the total number of occurrences of the token across every category.
	 *
	 * Complement Naive Bayes needs the corpus-wide count so the complement of a category is a
	 * subtraction rather than a second pass: `complement(t, c) = global(t) - count(t, c)`.
	 * Reading it per token keeps that available without materializing a corpus-wide map.
	 * @param string $token The token.
	 * @return int The corpus-wide occurrence count; 0 when the token is unknown.
	 */
	public function getTokenGlobalCount(string $token): int;

	/**
	 * Returns the total number of token occurrences across every category — the denominator
	 * mass Complement Naive Bayes subtracts a category's own total from.
	 * @return int The corpus-wide token total.
	 */
	public function getGlobalTokenTotal(): int;

	/**
	 * Declares the tokens about to be read, so an implementation that fetches from storage can
	 * satisfy them in one batched round trip instead of one per token.
	 *
	 * Calling it is always safe and never required for correctness — a resident implementation
	 * does nothing — but omitting it against a storage-backed vocabulary turns one query into
	 * one query per token.
	 * @param string[] $tokens The tokens that are about to be read.
	 */
	public function prefetch(array $tokens): void;

	/**
	 * Returns whether the whole vocabulary can be enumerated, i.e. whether
	 * {@see getDocumentFrequency()} is available.
	 * @return bool Whether a full scan is supported.
	 */
	public function getSupportsFullScan(): bool;

	/**
	 * Returns the document-frequency map (token -> number of documents it appeared in).
	 * @throws \Prado\Exceptions\TInvalidOperationException When the implementation cannot
	 * enumerate the vocabulary; test {@see getSupportsFullScan()} first.
	 * @return array<string, int> The document frequencies.
	 */
	public function getDocumentFrequency(): array;

	/**
	 * Returns a token identifying the current state of the vocabulary and every category in it,
	 * for use as a cache key.  It changes on any mutation, including one made directly on a
	 * category this vocabulary handed out.
	 * @return string The signature.
	 */
	public function getStateSignature(): string;

	/**
	 * Records one training document in the named category.
	 * @param string $category The category name.
	 * @param string[] $tokens The document's tokens (with multiplicity).
	 */
	public function addDocument(string $category, array $tokens): void;

	/**
	 * Replaces the vocabulary wholesale, as {@see \Belisoful\Prado\Util\Bayesian\Classifier\TNaiveBayesClassifier::load()}
	 * does when restoring a saved model.
	 *
	 * A storage-backed implementation reads its per-token statistics from storage rather than
	 * from `$documentFrequency`, and receives categories carrying only their scalar totals; it
	 * takes |V| from the model's stored metadata instead of counting the map.
	 * @param TBayesianCategory[] $categories The categories.
	 * @param array<string, int> $documentFrequency The document frequencies; empty for an
	 * implementation that does not hold them.
	 * @param int $totalDocuments The total document count.
	 */
	public function setStats(array $categories, array $documentFrequency, int $totalDocuments): void;
}
