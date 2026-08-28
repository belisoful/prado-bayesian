<?php

/**
 * TComplementNaiveBayes class file.
 *
 * @author Brad Anderson <belisoful@icloud.com>
 * @link https://github.com/belisoful/prado-bayesian
 * @license https://github.com/pradosoft/prado/blob/master/LICENSE
 */

namespace Prado\Util\Bayesian\Classifier;

use Prado\Util\Bayesian\Math\TFIdf;
use Prado\Exceptions\TInvalidOperationException;
use Prado\Util\Bayesian\TBayesianCategory;

/**
 * TComplementNaiveBayes class.
 *
 * Complement Naive Bayes (CNB): for each candidate category, the score is the log-ratio of
 * the token's probability in the *complement* (all other categories) to its probability in
 * the candidate category.  This emphasizes tokens that distinguish the candidate from
 * everything else rather than tokens that are merely frequent in the candidate, and is
 * empirically the strongest of the three Naive Bayes variants on class-imbalanced text
 * classification.
 *
 *   θ_complement(token | category) = ( α + token_count_in_complement )
 *                                     / ( α·|V| + total_token_count_in_complement )
 *   score(category) = ∑_{token in document} tf(token) · ( log θ_complement(token) / ‖w_category‖₁ )
 *
 * where the complement is every other category combined, |V| is the global vocabulary size, and
 * the per-category weight vector is L1-normalized (the WCNB refinement of Rennie et al. 2003) so
 * categories of differing token mass stay comparable.  The category with the *lowest* complement
 * score wins (the document looks least like everything else); {@see logLikelihood()} negates the
 * score so the "highest value wins" convention used by {@see classify()} and {@see score()} holds.
 *
 * TF-IDF re-weighting is applied to the document's term frequencies when enabled, the same way as
 * the base classifier, and out-of-vocabulary tokens are skipped.
 *
 * The corpus-wide token counts and each category's L1 weight norm depend only on the training
 * statistics, so they are computed once and cached until the next {@see trainOne()} or
 * {@see load()} rather than rebuilt for every category on every call.
 *
 * @author Brad Anderson <belisoful@icloud.com>
 * @since 0.1.0
 */
class TComplementNaiveBayes extends TNaiveBayesClassifier
{
	/** @var array<string, float> Cached per-category L1 norms of the complement weight vector. */
	private array $_norms = [];

	/**
	 * @var ?string The vocabulary signature and alpha the caches were built for, held in its own
	 * property rather than inside {@see $_norms} — a category is free to be named anything, so
	 * a reserved key in that map would be a name a caller could collide with.
	 */
	private ?string $_cacheKey = null;

	/**
	 * {@inheritDoc}
	 */
	protected function onTrainingChanged(): void
	{
		$this->_norms = [];
		$this->_cacheKey = null;
	}

	/**
	 * {@inheritDoc}
	 *
	 * Complement's per-category weight norms are L1 sums over the whole vocabulary, so a model
	 * stored per token has to carry them.  The complement denominator each norm was computed
	 * against travels too: the norm is only meaningful paired with it.
	 * @return array<string, mixed> The per-category norms and the state they were built for.
	 */
	protected function exportAggregates(): array
	{
		if (!$this->_vocabulary->getSupportsFullScan()) {
			// Training incrementally against storage: the aggregates cannot be recomputed
			// here, and writing stale ones would be worse than writing none.  Omitting them
			// makes the next score say so instead of quietly using the wrong constant.
			return [];
		}
		$this->ensureFresh();
		$norms = [];
		$globalTotal = $this->_vocabulary->getGlobalTokenTotal();
		$vocabularySize = $this->_vocabulary->getVocabularySize();
		foreach ($this->_vocabulary->getCategories() as $category) {
			$denominator = ($globalTotal - $category->getTotalTokens()) + $this->_alpha * $vocabularySize;
			if ($denominator > 0.0) {
				$norms[$category->getName()] = $this->categoryNorm($category, $denominator);
			}
		}
		return ['norms' => $norms, 'alpha' => $this->_alpha];
	}

	/**
	 * {@inheritDoc}
	 *
	 * Accepts the stored norms only if they were built with the alpha now in force.
	 * @param array<string, mixed> $aggregates The stored aggregates.
	 */
	protected function importAggregates(array $aggregates): void
	{
		$norms = $aggregates['norms'] ?? null;
		if (!is_array($norms) || (float) ($aggregates['alpha'] ?? 0.0) !== $this->_alpha) {
			return;
		}
		$this->_norms = array_map('floatval', $norms);
		$this->_cacheKey = $this->cacheKey();
	}

	/**
	 * Returns the key the derived caches are valid for: the vocabulary's state signature and
	 * the current alpha.
	 *
	 * Both cached quantities — the corpus-wide token counts and the per-category weight norms —
	 * are O(|V|) to rebuild and are read on every classification.  A stale one does not throw;
	 * it silently shifts every score.  So the key is taken from
	 * {@see \Prado\Util\Bayesian\IBayesianVocabulary::getStateSignature()}, which changes on
	 * any mutation including one made directly on a category handed out by
	 * {@see getVocabulary()}, rather than from totals that a rearrangement could leave
	 * unchanged.
	 * @return string The cache key.
	 */
	private function cacheKey(): string
	{
		return $this->_vocabulary->getStateSignature() . '|' . $this->_alpha;
	}

	/**
	 * {@inheritDoc}
	 * @param float $value The Laplace smoothing constant (> 0).
	 */
	public function setAlpha(float $value): void
	{
		parent::setAlpha($value);
		$this->onTrainingChanged();
	}

	/**
	 * Discards the cached per-category norms when the statistics or alpha have moved.
	 */
	private function ensureFresh(): void
	{
		$key = $this->cacheKey();
		if ($this->_cacheKey !== $key) {
			$this->_norms = [];
			$this->_cacheKey = $key;
		}
	}

	/**
	 * Returns the L1 norm of a category's complement weight vector over the full vocabulary,
	 * computing and caching it on first use.
	 * @param TBayesianCategory $category The category.
	 * @param float $denominator The smoothed complement token mass.
	 * @return float The norm.
	 */
	private function categoryNorm(TBayesianCategory $category, float $denominator): float
	{
		$name = $category->getName();
		if (isset($this->_norms[$name])) {
			return (float) $this->_norms[$name];
		}
		if (!$this->_vocabulary->getSupportsFullScan()) {
			// The norm is an L1 sum over the whole vocabulary; a storage-backed vocabulary
			// cannot supply that, and it was not restored with the model.
			throw new TInvalidOperationException('bayesian_classifier_aggregate_missing', (string) $this->getName(), 'norm:' . $name);
		}
		$alpha = $this->_alpha;
		$categoryCounts = $category->getTokenCounts();
		$norm = 0.0;
		foreach ($this->_vocabulary->getDocumentFrequency() as $token => $_) {
			$token = (string) $token;
			$complementForToken = $this->_vocabulary->getTokenGlobalCount($token) - ($categoryCounts[$token] ?? 0);
			$norm += abs(log(($complementForToken + $alpha) / $denominator));
		}
		$this->_norms[$name] = $norm;
		return $norm;
	}

	/**
	 * {@inheritDoc}
	 *
	 * Returns the negative of the complement score so the "largest value wins" convention
	 * holds.  The implementation sums `log((alpha + complement_count) / (alpha + category_count))`
	 * weighted by the document's token count (or TF-IDF weight when enabled).
	 * @param string[] $tokens The document's tokens.
	 * @param TBayesianCategory $category The candidate category.
	 * @param int $totalDocs The total document count.
	 * @return float The negative complement log-score.
	 */
	protected function logLikelihood(array $tokens, TBayesianCategory $category, int $totalDocs): float
	{
		$alpha = $this->_alpha;
		$counts = $this->aggregate($tokens);
		$this->ensureFresh();
		// The complement (everything but the candidate category) is a subtraction rather than a
		// second pass: complement_count(t) = global_count(t) - count(t, category).  Both come
		// from the vocabulary per token, so no corpus-wide map is built.
		// CNB smoothes the complement count into a proper probability by dividing by the TOTAL
		// token mass of the complement plus alpha*|V|.
		$complementTotal = $this->_vocabulary->getGlobalTokenTotal() - $category->getTotalTokens();
		$denominator = $complementTotal + $alpha * $this->_vocabulary->getVocabularySize();
		if ($denominator <= 0.0) {
			return -INF;
		}
		// WCNB normalization: the per-category weight vector is L1-normalized over the FULL
		// vocabulary (cached per category).  Normalizing over only the document's terms would
		// discard the weight magnitudes that discriminate between categories.
		$norm = $this->categoryNorm($category, $denominator);
		if ($norm <= 0.0) {
			return -INF;
		}
		$logSum = 0.0;
		foreach ($counts as $token => $count) {
			$token = (string) $token;
			$documentFrequency = $this->_vocabulary->getTokenDocumentFrequency($token);
			if ($documentFrequency === 0) {
				continue;
			}
			$tf = $this->_useTfidf
				? TFIdf::weight($count, $documentFrequency, $totalDocs)
				: (float) $count;
			$complementForToken = $this->_vocabulary->getTokenGlobalCount($token) - $category->getTokenCount($token);
			$logSum += $tf * (log(($complementForToken + $alpha) / $denominator) / $norm);
		}
		// CNB picks the category whose complement the document looks LEAST like (lowest score);
		// negate so the "largest value wins" convention used by classify()/score() holds.
		return -$logSum;
	}

	/**
	 * {@inheritDoc}
	 *
	 * CNB does not use a per-category prior in its score; the prior information is folded
	 * into the per-token log-ratio, so the log-posteriors are the negated complement scores.
	 * @param string[] $tokens The tokens.
	 * @return array<string, float> The negated complement log-scores, keyed by category.
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
			if ($category->getDocumentCount() === 0) {
				$scores[$category->getName()] = -INF;
				continue;
			}
			$scores[$category->getName()] = $this->logLikelihood($tokens, $category, $totalDocs);
		}
		return $scores;
	}

	/**
	 * {@inheritDoc}
	 * @return string The kind marker `complement-naive-bayes`.
	 */
	protected function getKind(): string
	{
		return 'complement-naive-bayes';
	}

	/**
	 * {@inheritDoc}
	 * @return string[] Only `complement-naive-bayes`.
	 */
	protected function getCompatibleKinds(): array
	{
		return ['complement-naive-bayes'];
	}
}
