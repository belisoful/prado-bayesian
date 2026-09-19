<?php

/**
 * TComplementNaiveBayes class file.
 *
 * @author Brad Anderson <belisoful@icloud.com>
 * @link https://github.com/belisoful/prado-bayesian
 * @license https://github.com/pradosoft/prado/blob/master/LICENSE
 */

namespace Belisoful\Prado\Util\Bayesian\Classifier;

use Belisoful\Prado\Util\Bayesian\Math\TFIdf;
use Belisoful\Prado\Util\Bayesian\TBayesianCategory;
use Belisoful\Prado\Util\Bayesian\TBayesianPayload;
use Belisoful\Prado\Util\Bayesian\TBayesianTokenHistogram;
use Prado\Exceptions\TInvalidOperationException;

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
	 * @var null|array<string, array<string, array<int, int>>>|false The token histograms for
	 * the state in {@see $_cacheKey}: null before they are asked for, false when unavailable.
	 */
	private $_histograms;

	/**
	 * {@inheritDoc}
	 */
	protected function onTrainingChanged(): void
	{
		$this->_norms = [];
		$this->_histograms = null;
		$this->_cacheKey = null;
	}

	/**
	 * {@inheritDoc}
	 * @return string[] The three families a category's complement counts are merged from.
	 * @since 0.2.0
	 */
	protected function getHistogramFamilies(): array
	{
		return [
			TBayesianTokenHistogram::FAMILY_GLOBAL,
			TBayesianTokenHistogram::FAMILY_CATEGORY_GLOBAL,
			TBayesianTokenHistogram::FAMILY_COMPLEMENT,
		];
	}

	/**
	 * {@inheritDoc}
	 *
	 * Complement's per-category weight norms are L1 sums over the whole vocabulary.  A storage
	 * that keeps token histograms makes this copy redundant; it travels for a per-token storage
	 * that does not, where nothing on the read side could rebuild it.
	 * @return array<string, mixed> The per-category norms and the state they were built for.
	 */
	protected function exportAggregates(): array
	{
		if (!$this->_vocabulary->getSupportsFullScan()) {
			// Training incrementally against storage: the norms written at the last full save
			// are stale now, and writing stale ones would be worse than writing none.  A reader
			// computes them from the storage's histograms, or reports them missing.
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
	 * Accepts the stored norms only when the histograms they would otherwise be computed from
	 * are unavailable, and only if they were built with the alpha now in force.
	 * @param array<string, mixed> $aggregates The stored aggregates.
	 */
	protected function importAggregates(array $aggregates): void
	{
		$norms = $aggregates['norms'] ?? null;
		if ($this->getHasTokenHistograms()) {
			// The histograms are current with every write; a stored norm is only as fresh as
			// the last full save.
			return;
		}
		if (!is_array($norms) || TBayesianPayload::float($aggregates['alpha'] ?? null) !== $this->_alpha) {
			return;
		}
		$this->_norms = TBayesianPayload::floatMap($norms);
		$this->_cacheKey = $this->cacheKey();
	}

	/**
	 * Returns the key the derived caches are valid for: the vocabulary's state signature and
	 * the current alpha.
	 *
	 * Both cached quantities — the corpus-wide token counts and the per-category weight norms —
	 * are O(|V|) to rebuild and are read on every classification.  A stale one does not throw;
	 * it silently shifts every score.  So the key is taken from
	 * {@see \Belisoful\Prado\Util\Bayesian\IBayesianVocabulary::getStateSignature()}, which changes on
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
			$this->_histograms = null;
			$this->_cacheKey = $key;
		}
	}

	/**
	 * Returns the L1 norm of a category's complement weight vector over the full vocabulary,
	 * computing and caching it on first use.
	 * @param TBayesianCategory $category The category.
	 * @param float $denominator The smoothed complement token mass.
	 * @throws TInvalidOperationException `bayesian_classifier_aggregate_missing` when the model
	 * is storage-backed and the norm was neither restored with it nor recomputable here.
	 * @return float The norm.
	 */
	private function categoryNorm(TBayesianCategory $category, float $denominator): float
	{
		$name = $category->getName();
		if (isset($this->_norms[$name])) {
			return (float) $this->_norms[$name];
		}
		$this->_histograms ??= $this->getTokenHistograms() ?? false;
		if ($this->_histograms === false) {
			// The norm is an L1 sum over the whole vocabulary; a storage-backed vocabulary
			// cannot supply that, its storage keeps no histograms, and the norm was not
			// restored with the model.
			throw new TInvalidOperationException('bayesian_classifier_aggregate_missing', (string) $this->getName(), 'norm:' . $name);
		}
		// A token's weight depends on it only through its complement count, so the sum over the
		// vocabulary is a sum over the distinct complement counts, each weighted by how many
		// tokens share it.  Resident and storage-backed vocabularies both arrive here with the
		// same integers and sum them in the same order, so they score identically.
		$alpha = $this->_alpha;
		$norm = 0.0;
		foreach (TBayesianTokenHistogram::complementCounts($this->_histograms, $name) as $complementCount => $tokenCount) {
			$norm += $tokenCount * abs(log(($complementCount + $alpha) / $denominator));
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
			// Every weight is zero: no token tells this category apart from its complement
			// (two categories trained on the same text, say).  That is an absence of evidence,
			// not an impossibility, so the category scores neutrally rather than -INF — which
			// would make a trained model unable to classify at all.
			return 0.0;
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
