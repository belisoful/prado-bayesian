<?php

/**
 * TBernoulliNaiveBayes class file.
 *
 * @author Brad Anderson <belisoful@icloud.com>
 * @link https://github.com/belisoful/prado-bayesian
 * @license https://github.com/pradosoft/prado/blob/master/LICENSE
 */

namespace Belisoful\Prado\Util\Bayesian\Classifier;

use Belisoful\Prado\Util\Bayesian\TBayesianCategory;
use Belisoful\Prado\Util\Bayesian\TBayesianPayload;
use Belisoful\Prado\Util\Bayesian\TBayesianTokenHistogram;
use Prado\Exceptions\TInvalidOperationException;

/**
 * TBernoulliNaiveBayes class.
 *
 * The Bernoulli event model: each token is a binary feature (present in the document or not),
 * and the likelihood is
 *
 *   P(category | document) ∝ P(category) · ∏ P(token | category)^{present} · (1 - P(token | category))^{absent}
 *
 * where P(token | category) is the fraction of documents in the category that contain the
 * token (with Laplace smoothing).  Bernoulli NB is well-suited to short documents (titles,
 * queries) and to feature sets that are presence/absence by nature (e.g. "user has clicked
 * this ad").
 *
 * The per-token document counts come from {@see TBayesianCategory::getTokenDocumentCount()},
 * which the vocabulary records during training.
 *
 * Absence being evidence would make a literal implementation walk the whole vocabulary for
 * every category on every classification.  This one does not: the absent-token mass is a
 * constant per category, so it is summed once and cached, and scoring a document only corrects
 * that constant for the tokens the document actually contains.  See {@see absentLogMass()} for
 * the identity.  Classification is therefore O(document x categories), not O(vocabulary x
 * categories), and the result is arithmetically the same.
 *
 * @author Brad Anderson <belisoful@icloud.com>
 * @since 0.1.0
 */
class TBernoulliNaiveBayes extends TNaiveBayesClassifier
{
	/**
	 * @var array<string, float> The cached absent-token log mass per category name, valid only
	 * for the vocabulary state and alpha recorded in {@see $_massKey}.
	 */
	private array $_absentMass = [];

	/** @var ?string The vocabulary signature and alpha the cache in {@see $_absentMass} was built for. */
	private ?string $_massKey = null;

	/**
	 * @var null|array<string, array<string, array<int, int>>>|false The token histograms for
	 * the state in {@see $_massKey}: null before they are asked for, false when unavailable.
	 */
	private $_histograms;

	/**
	 * {@inheritDoc}
	 * @return string[] The per-category document-count family.
	 * @since 0.2.0
	 */
	protected function getHistogramFamilies(): array
	{
		return [TBayesianTokenHistogram::FAMILY_DOCUMENTS];
	}

	/**
	 * Returns the smoothed probability that a document of this category contains the token.
	 * @param TBayesianCategory $category The category.
	 * @param string $token The token.
	 * @param float $denominator The smoothed document mass of the category.
	 * @return float The probability, in (0, 1) whenever alpha is positive.
	 */
	private function presenceProbability(TBayesianCategory $category, string $token, float $denominator): float
	{
		return ($category->getTokenDocumentCount($token) + $this->_alpha) / $denominator;
	}

	/**
	 * Returns `sum over the whole vocabulary of log(1 - P(token | category))` — the score a
	 * document containing none of the vocabulary would get — computing it once per
	 * (vocabulary state, alpha) and caching it for every category.
	 *
	 * This is the constant term of the identity that makes scoring cheap.  Writing V for the
	 * vocabulary and d for the document's tokens:
	 *
	 *   sum over V of [ present ? log p : log(1-p) ]
	 *     = sum over V of log(1-p)  +  sum over (d intersect V) of [ log p - log(1-p) ]
	 *
	 * The left term depends only on the training statistics and alpha, so it is hoisted out of
	 * the per-document path; the right term ranges over the document's tokens alone.
	 *
	 * The constant itself is not summed token by token either.  `p` depends on a token only
	 * through how many of the category's documents contain it, so the sum runs over the
	 * distinct document counts, each weighted by the number of tokens sharing it — the
	 * {@see TBayesianTokenHistogram::FAMILY_DOCUMENTS} histogram — plus one term for the tokens
	 * none of the category's documents contain.  A resident vocabulary counts the histogram
	 * here; a storage-backed one reads it from the storage, which moves it with every training
	 * write.  Both then take the same sum in the same order, so the two layouts score
	 * identically, bit for bit, and a storage-backed model trains incrementally.
	 *
	 * The cache is keyed on {@see \Belisoful\Prado\Util\Bayesian\IBayesianVocabulary::getStateSignature()}
	 * rather than on a count of documents or categories.  A stale constant here does not throw
	 * — it silently shifts every score — so the key has to notice any mutation, including one
	 * made directly on a category obtained from {@see getVocabulary()}.
	 * @param TBayesianCategory $category The category.
	 * @param float $denominator The smoothed document mass of the category.
	 * @throws TInvalidOperationException `bayesian_classifier_aggregate_missing` when the model
	 * is storage-backed, its storage keeps no histograms, and the constant was not restored
	 * with the model either.
	 * @return float The summed log mass of the absent tokens.
	 */
	private function absentLogMass(TBayesianCategory $category, float $denominator): float
	{
		$key = $this->_vocabulary->getStateSignature() . '|' . $this->_alpha;
		if ($this->_massKey !== $key) {
			$this->_absentMass = [];
			$this->_histograms = null;
			$this->_massKey = $key;
		}
		$name = $category->getName();
		if (isset($this->_absentMass[$name])) {
			return $this->_absentMass[$name];
		}
		$this->_histograms ??= $this->getTokenHistograms() ?? false;
		if ($this->_histograms === false) {
			// A storage-backed vocabulary cannot be walked, its storage keeps no histograms,
			// and the constant was not restored with the model.  Silently scoring without it
			// would shift every result, so this says what is missing instead.
			throw new TInvalidOperationException('bayesian_classifier_aggregate_missing', (string) $this->getName(), 'absentMass:' . $name);
		}
		$histogram = $this->_histograms[TBayesianTokenHistogram::FAMILY_DOCUMENTS][$name] ?? [];
		ksort($histogram, SORT_NUMERIC);
		// The tokens none of the category's documents contain are the vocabulary less the
		// tokens the histogram counts; they all share the document count zero.
		$histogram = [0 => $this->_vocabulary->getVocabularySize() - array_sum($histogram)] + $histogram;
		$mass = 0.0;
		foreach ($histogram as $documentCount => $tokenCount) {
			$p = ($documentCount + $this->_alpha) / $denominator;
			// With alpha > 0 the smoothed p is strictly inside (0, 1); the guard only protects
			// against a hand-imported degenerate state.  A token skipped here must also be
			// skipped by the correction below, or the identity no longer holds.
			if ($tokenCount <= 0 || $p <= 0.0 || $p >= 1.0) {
				continue;
			}
			$mass += $tokenCount * log(1.0 - $p);
		}
		$this->_absentMass[$name] = $mass;
		return $mass;
	}

	/**
	 * {@inheritDoc}
	 *
	 * Bernoulli scores every category over the SAME fixed feature set — the global vocabulary —
	 * so the `(1 - p)` absent-term penalties are comparable across categories instead of
	 * depending on each category's own vocabulary size.  Each vocabulary token contributes
	 * `+ log P(token | category)` when the document contains it and `+ log(1 - P(token | category))`
	 * when it does not; tokens that appear only in the document (out-of-vocabulary) are skipped,
	 * as in the base classifier.
	 *
	 * The sum is evaluated as the cached absent-token constant from {@see absentLogMass()} plus
	 * a correction over the document's own tokens, so the vocabulary is walked once per
	 * (training state, alpha) rather than once per classification per category.
	 * @param string[] $tokens The document's tokens.
	 * @param TBayesianCategory $category The candidate category.
	 * @param int $totalDocs The total document count (unused for Bernoulli).
	 * @return float The log-likelihood.
	 */
	protected function logLikelihood(array $tokens, TBayesianCategory $category, int $totalDocs): float
	{
		$alpha = $this->_alpha;
		$denominator = $category->getDocumentCount() + 2.0 * $alpha;
		if ($denominator <= 0.0) {
			return -INF;
		}
		$logSum = $this->absentLogMass($category, $denominator);
		$present = [];
		foreach ($tokens as $token) {
			// Out-of-vocabulary tokens carry no learned evidence; a repeat adds nothing either,
			// because the feature is presence, not count.
			if (isset($present[$token]) || !$this->_vocabulary->hasToken((string) $token)) {
				continue;
			}
			$present[$token] = true;
			$p = $this->presenceProbability($category, (string) $token, $denominator);
			if ($p <= 0.0 || $p >= 1.0) {
				continue;
			}
			// Replace this token's absent contribution with its present one.
			$logSum += log($p) - log(1.0 - $p);
		}
		return $logSum;
	}

	/**
	 * {@inheritDoc}
	 *
	 * Also discards the cached absent-token constants, which depend on the training counts.
	 */
	protected function onTrainingChanged(): void
	{
		parent::onTrainingChanged();
		$this->_absentMass = [];
		$this->_histograms = null;
		$this->_massKey = null;
	}

	/**
	 * {@inheritDoc}
	 *
	 * Bernoulli's absent-token constant is a sum over the whole vocabulary.  A storage that
	 * keeps token histograms makes this copy redundant; it travels for a per-token storage
	 * that does not, where nothing on the read side could rebuild it.
	 * @return array<string, mixed> The per-category constants and the state they were built for.
	 */
	protected function exportAggregates(): array
	{
		if (!$this->_vocabulary->getSupportsFullScan()) {
			// Training incrementally against storage: the constants written at the last full
			// save are stale now, and writing stale ones would be worse than writing none.  A
			// reader computes them from the storage's histograms, or reports them missing.
			return [];
		}
		$mass = [];
		foreach ($this->_vocabulary->getCategories() as $category) {
			$denominator = $category->getDocumentCount() + 2.0 * $this->_alpha;
			if ($denominator > 0.0) {
				$mass[$category->getName()] = $this->absentLogMass($category, $denominator);
			}
		}
		return ['absentMass' => $mass, 'alpha' => $this->_alpha];
	}

	/**
	 * {@inheritDoc}
	 *
	 * Accepts the stored constants only when the histograms they would otherwise be computed
	 * from are unavailable, and only if they were built with the alpha now in force; a mismatch
	 * leaves them out, so the next score reports the aggregate as missing rather than using a
	 * number computed for different smoothing.
	 * @param array<string, mixed> $aggregates The stored aggregates.
	 */
	protected function importAggregates(array $aggregates): void
	{
		$mass = $aggregates['absentMass'] ?? null;
		if ($this->getHasTokenHistograms()) {
			// The histograms are current with every write; a stored constant is only as fresh
			// as the last full save.
			return;
		}
		if (!is_array($mass) || TBayesianPayload::float($aggregates['alpha'] ?? null) !== $this->_alpha) {
			return;
		}
		$this->_absentMass = TBayesianPayload::floatMap($mass);
		$this->_massKey = $this->_vocabulary->getStateSignature() . '|' . $this->_alpha;
	}

	/**
	 * {@inheritDoc}
	 *
	 * Bernoulli NB does not use TF-IDF re-weighting — each token's contribution is binary
	 * (present or absent).  The saved state still carries the flag for round-trip safety, but
	 * {@see logLikelihood()} above does not consult it.
	 * @return string The kind marker `bernoulli-naive-bayes`.
	 */
	protected function getKind(): string
	{
		return 'bernoulli-naive-bayes';
	}

	/**
	 * {@inheritDoc}
	 * @return string[] Only `bernoulli-naive-bayes`.
	 */
	protected function getCompatibleKinds(): array
	{
		return ['bernoulli-naive-bayes'];
	}
}
