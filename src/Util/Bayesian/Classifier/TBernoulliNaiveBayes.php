<?php

/**
 * TBernoulliNaiveBayes class file.
 *
 * @author Brad Anderson <belisoful@icloud.com>
 * @link https://github.com/belisoful/prado-bayesian
 * @license https://github.com/pradosoft/prado/blob/master/LICENSE
 */

namespace Prado\Util\Bayesian\Classifier;

use Prado\Exceptions\TInvalidOperationException;
use Prado\Util\Bayesian\TBayesianCategory;

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
	 * The cache is keyed on {@see \Prado\Util\Bayesian\IBayesianVocabulary::getStateSignature()}
	 * rather than on a count of documents or categories.  A stale constant here does not throw
	 * — it silently shifts every score — so the key has to notice any mutation, including one
	 * made directly on a category obtained from {@see getVocabulary()}.
	 * @param TBayesianCategory $category The category.
	 * @param float $denominator The smoothed document mass of the category.
	 * @return float The summed log mass of the absent tokens.
	 */
	private function absentLogMass(TBayesianCategory $category, float $denominator): float
	{
		$key = $this->_vocabulary->getStateSignature() . '|' . $this->_alpha;
		if ($this->_massKey !== $key) {
			$this->_absentMass = [];
			$this->_massKey = $key;
		}
		$name = $category->getName();
		if (isset($this->_absentMass[$name])) {
			return $this->_absentMass[$name];
		}
		if (!$this->_vocabulary->getSupportsFullScan()) {
			// A storage-backed vocabulary cannot be walked, and the constant was not restored
			// with the model.  Silently scoring without it would shift every result, so this
			// says what is missing instead.
			throw new TInvalidOperationException('bayesian_classifier_aggregate_missing', (string) $this->getName(), 'absentMass:' . $name);
		}
		$mass = 0.0;
		foreach ($this->_vocabulary->getDocumentFrequency() as $token => $_) {
			$p = $this->presenceProbability($category, (string) $token, $denominator);
			// With alpha > 0 the smoothed p is strictly inside (0, 1); the guard only protects
			// against a hand-imported degenerate state.  A token skipped here must also be
			// skipped by the correction below, or the identity no longer holds.
			if ($p <= 0.0 || $p >= 1.0) {
				continue;
			}
			$mass += log(1.0 - $p);
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
		$this->_massKey = null;
	}

	/**
	 * {@inheritDoc}
	 *
	 * Bernoulli's absent-token constant is a sum over the whole vocabulary, so a model stored
	 * per token has to carry it: nothing on the read side can rebuild it.
	 * @return array<string, mixed> The per-category constants and the state they were built for.
	 */
	protected function exportAggregates(): array
	{
		if (!$this->_vocabulary->getSupportsFullScan()) {
			// Training incrementally against storage: the aggregates cannot be recomputed
			// here, and writing stale ones would be worse than writing none.  Omitting them
			// makes the next score say so instead of quietly using the wrong constant.
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
	 * Accepts the stored constants only if they were built with the alpha now in force; a
	 * mismatch leaves them out, so the next score reports the aggregate as missing rather than
	 * using a number computed for different smoothing.
	 * @param array<string, mixed> $aggregates The stored aggregates.
	 */
	protected function importAggregates(array $aggregates): void
	{
		$mass = $aggregates['absentMass'] ?? null;
		if (!is_array($mass) || (float) ($aggregates['alpha'] ?? 0.0) !== $this->_alpha) {
			return;
		}
		$this->_absentMass = array_map('floatval', $mass);
		$this->_massKey = $this->_vocabulary->getStateSignature() . '|' . $this->_alpha;
	}

	/**
	 * {@inheritDoc}
	 *
	 * The Bernoulli model normalizes the log-posteriors in the standard way; the log prior
	 * is the only non-likelihood term.
	 * @param string[] $tokens The tokens.
	 * @return array<string, float> The log-posteriors.
	 */
	protected function scoreTokens(array $tokens): array
	{
		$categories = $this->_vocabulary->getCategories();
		if ($categories === []) {
			return [];
		}
		$this->_vocabulary->prefetch($tokens);
		$totalDocs = $this->_vocabulary->getTotalDocuments();
		$scores = [];
		foreach ($categories as $category) {
			$docCount = $category->getDocumentCount();
			if ($docCount === 0 || $totalDocs === 0) {
				$scores[$category->getName()] = -INF;
				continue;
			}
			$logPrior = log($docCount / $totalDocs);
			$logLikelihood = $this->logLikelihood($tokens, $category, $totalDocs);
			$scores[$category->getName()] = $logPrior + $logLikelihood;
		}
		return $scores;
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
