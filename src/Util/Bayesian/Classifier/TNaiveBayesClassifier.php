<?php

/**
 * TNaiveBayesClassifier class file.
 *
 * @author Brad Anderson <belisoful@icloud.com>
 * @link https://github.com/belisoful/prado-bayesian
 * @license https://github.com/pradosoft/prado/blob/master/LICENSE
 */

namespace Belisoful\Prado\Util\Bayesian\Classifier;

use Belisoful\Prado\Util\Bayesian\Calibration\TTemperatureScaling;
use Belisoful\Prado\Util\Bayesian\IBayesianVocabulary;
use Belisoful\Prado\Util\Bayesian\Math\TBayesMath;
use Belisoful\Prado\Util\Bayesian\Math\TFIdf;
use Belisoful\Prado\Util\Bayesian\Storage\IBayesianStorage;
use Belisoful\Prado\Util\Bayesian\Storage\IBayesianTokenStorage;
use Belisoful\Prado\Util\Bayesian\TBayesianCategory;
use Belisoful\Prado\Util\Bayesian\TBayesianPayload;
use Belisoful\Prado\Util\Bayesian\TBayesianTrainingSet;
use Belisoful\Prado\Util\Bayesian\TBayesianVocabulary;
use Belisoful\Prado\Util\Bayesian\TBayesianTokenHistogram;
use Belisoful\Prado\Util\Bayesian\TLazyBayesianVocabulary;
use Belisoful\Prado\Util\Bayesian\Tokenizer\IBayesianTokenizer;
use Belisoful\Prado\Util\Bayesian\Tokenizer\TBayesianTokenizerFactory;
use Belisoful\Prado\Util\Bayesian\Tokenizer\TWordTokenizer;
use Prado\Exceptions\TConfigurationException;
use Prado\Exceptions\TInvalidDataValueException;
use Prado\Exceptions\TInvalidOperationException;
use Prado\TComponent;

/**
 * TNaiveBayesClassifier class.
 *
 * The canonical Naive Bayes text classifier, used here as the default spam filter and the
 * base for the recommender.  The implementation is the multinomial event model with
 * Laplace (add-one) smoothing and a {@see TFIdf} re-weighting knob:
 *
 *   P(category | document) ∝ P(category) · ∏ P(token | category)^{tfidf_weight(token, count)}
 *
 * Computations happen in log space ({@see TBayesMath}) to avoid the underflow that bites
 * straight multiplication of thousands of small probabilities, then the relative log-scores
 * are normalized back to a distribution that sums to one.  Those are *not* calibrated
 * probabilities — Naive Bayes is overconfident by construction — until a
 * {@see TTemperatureScaling} is fitted on held-out documents with {@see calibrate()} and
 * installed as the {@see setCalibration() Calibration}; {@see score()} then applies it, and
 * the fitted temperature is saved with the model.  {@see logScores()} always returns the raw
 * log-posteriors.
 *
 * The feature vocabulary is fixed by training: a token never seen in any category is
 * out-of-vocabulary and is skipped at classification time (the convention of scikit-learn
 * and Manning et al.).  Penalizing unseen tokens instead would push every novel document
 * toward the smallest category, because the smoothed penalty shrinks as the category grows.
 *
 * Category names are used as PHP array keys in {@see score()}, so a purely numeric name
 * such as `"2024"` comes back as the integer key `2024`; {@see classify()} always returns a
 * string.
 *
 * The classifier owns its {@see IBayesianVocabulary} and a configured {@see IBayesianTokenizer};
 * an optional {@see IBayesianStorage} persists the trained state under a name.  The same
 * component is reusable for any number of categories — train it once, classify forever.
 *
 * The {@see isSpam()} helper is a two-category shortcut assuming one category is "spam"
 * (or whatever the user calls the negative class via {@see setSpamCategory()}); in a
 * multi-class setup, prefer {@see score()} and inspect the probabilities directly.
 *
 * @author Brad Anderson <belisoful@icloud.com>
 * @since 0.1.0
 */
class TNaiveBayesClassifier extends TComponent implements IBayesianClassifier
{
	/**
	 * The version of the saved-state shape this class writes, stored as `formatVersion` in
	 * every payload and per-token metadata.  A payload without it was written by 0.1.0 and is
	 * read as version 1; a payload from a later version than this is refused with
	 * `bayesian_model_format_unsupported`, so a newer shape is never silently misread.
	 * @since 0.2.0
	 */
	public const FORMAT_VERSION = 1;

	/** @var ?string The model identifier (used as the storage key when persisted). */
	private ?string $_name = null;

	/** @var IBayesianTokenizer The tokenizer that turns text into features. */
	private IBayesianTokenizer $_tokenizer;

	/** @var ?IBayesianStorage The storage backend. */
	private ?IBayesianStorage $_storage = null;

	/** @var IBayesianVocabulary The vocabulary owned by the classifier. */
	protected IBayesianVocabulary $_vocabulary;

	/** @var bool Whether per-token TF-IDF re-weighting is applied to the log-scores. */
	protected bool $_useTfidf = true;

	/** @var float The Laplace smoothing constant (the "alpha"). */
	protected float $_alpha = 1.0;

	/** @var string The label treated as "spam" by {@see isSpam()}. */
	private string $_spamCategory = 'spam';

	/** @var ?TTemperatureScaling The calibration {@see score()} applies, when one is installed. */
	private ?TTemperatureScaling $_calibration = null;

	/**
	 * @var array<string, array<string, mixed>> State other components keep with the model
	 * (a tagger's per-label calibrations, say), saved and loaded with it, keyed by owner.
	 */
	private array $_extra = [];

	/**
	 * Initializes the classifier with a default {@see TWordTokenizer}.
	 */
	public function __construct()
	{
		$this->_tokenizer = new TWordTokenizer();
		$this->_vocabulary = new TBayesianVocabulary();
		parent::__construct();
	}

	/**
	 * Trains the classifier on a labeled training set.
	 * @param TBayesianTrainingSet $set The training set.
	 * @throws TInvalidDataValueException When the training set is empty.
	 */
	public function train(TBayesianTrainingSet $set): void
	{
		if ($set->getIsEmpty()) {
			throw new TInvalidDataValueException('bayesian_training_set_empty');
		}
		foreach ($set->each() as $category => $document) {
			$this->trainOne($category, $document);
		}
	}

	/**
	 * Adds one document to a category.
	 * @param string $category The category name.
	 * @param string|string[] $document The document.
	 * @throws TInvalidDataValueException When the category name is empty.
	 */
	public function trainOne(string $category, $document): void
	{
		if ($category === '') {
			throw new TInvalidDataValueException('bayesian_category_required');
		}
		$tokens = is_array($document) ? array_values($document) : $this->_tokenizer->tokenize($document);
		$vocabulary = $this->_vocabulary;
		if ($vocabulary instanceof TLazyBayesianVocabulary) {
			// Storage-backed: write this document's deltas rather than accumulating a whole
			// model in the process.  The cost is the document, not the model.
			$vocabulary->applyDocument($category, $tokens, $this->exportTokenMeta());
			$this->onTrainingChanged();
			return;
		}
		$vocabulary->addDocument($category, $tokens);
		$this->onTrainingChanged();
	}

	/**
	 * Withdraws every document of a training set, reversing {@see train()}.
	 * @param TBayesianTrainingSet $set The training set.
	 * @throws TInvalidDataValueException When the training set is empty.
	 * @since 0.2.0
	 */
	public function untrain(TBayesianTrainingSet $set): void
	{
		if ($set->getIsEmpty()) {
			throw new TInvalidDataValueException('bayesian_training_set_empty');
		}
		foreach ($set->each() as $category => $document) {
			$this->untrainOne($category, $document);
		}
	}

	/**
	 * Withdraws one document from a category, reversing {@see trainOne()} for the same text
	 * (or the same pre-tokenized list).  Every count stops at zero, a token no document
	 * contains any more leaves the vocabulary, and a category left without documents is
	 * removed, so the model ends as it would have without the document.  Withdrawing a document
	 * that was never trained, or from a category that does not exist, changes nothing beyond
	 * the clamped counts.
	 * @param string $category The category name.
	 * @param string|string[] $document The document.
	 * @throws TInvalidDataValueException When the category name is empty.
	 * @since 0.2.0
	 */
	public function untrainOne(string $category, $document): void
	{
		if ($category === '') {
			throw new TInvalidDataValueException('bayesian_category_required');
		}
		$tokens = is_array($document) ? array_values($document) : $this->_tokenizer->tokenize($document);
		$vocabulary = $this->_vocabulary;
		if ($vocabulary instanceof TLazyBayesianVocabulary) {
			$vocabulary->withdrawDocument($category, $tokens, $this->exportTokenMeta());
			$this->onTrainingChanged();
			return;
		}
		$vocabulary->removeDocument($category, $tokens);
		$this->onTrainingChanged();
	}

	/**
	 * Hook called whenever the training statistics change (after {@see trainOne()},
	 * {@see untrainOne()} and {@see importState()}), so subclasses can drop derived caches.
	 */
	protected function onTrainingChanged(): void
	{
	}

	/**
	 * Classifies a document.
	 * @param string|string[] $document The document.
	 * @throws TInvalidOperationException `bayesian_classifier_not_trained` when no training has
	 * been recorded; `bayesian_classifier_score_undefined` when the classifier is trained but
	 * every category scores -INF or NaN (e.g. training produced no tokens at all).
	 * @return string The predicted category.
	 */
	public function classify($document): string
	{
		if (!$this->getIsTrained()) {
			throw new TInvalidOperationException('bayesian_classifier_not_trained', $this->_name ?? '');
		}
		$scores = $this->score($document);
		$best = null;
		$bestScore = -INF;
		foreach ($scores as $category => $score) {
			if (!is_nan($score) && $score > $bestScore) {
				$bestScore = $score;
				$best = $category;
			}
		}
		if ($best === null) {
			throw new TInvalidOperationException('bayesian_classifier_score_undefined', $this->_name ?? '');
		}
		return (string) $best;
	}

	/**
	 * Returns the normalized score of every category for a document.
	 *
	 * Without a {@see getCalibration() Calibration} the log-posteriors are normalized as they
	 * are, which is a ranking that sums to one rather than a probability; with one they are
	 * scaled by the fitted temperature first, so the value is a probability estimate.
	 * @param string|string[] $document The document.
	 * @return array<string, float> The scores, keyed by category, summing to 1.0; empty when
	 * the classifier is untrained or no category has a finite score.
	 */
	public function score($document): array
	{
		$logScores = $this->logScores($document);
		return $this->_calibration !== null ? $this->_calibration->apply($logScores) : TBayesMath::normalize($logScores);
	}

	/**
	 * Returns the raw log-posterior of every category for a document, before any
	 * normalization or calibration: what {@see calibrate()} fits on and what a custom
	 * calibration would consume.
	 * @param string|string[] $document The document.
	 * @return array<string, float> The log-posteriors, keyed by category; `-INF` for a category
	 * that cannot score the document; empty when the classifier is untrained.
	 * @since 0.2.0
	 */
	public function logScores($document): array
	{
		$tokens = is_array($document) ? array_values($document) : $this->_tokenizer->tokenize($document);
		return $this->scoreTokens($tokens);
	}

	/**
	 * Fits a {@see TTemperatureScaling} on held-out labeled documents and installs it, so
	 * {@see score()} returns calibrated probabilities from then on.  The held-out documents
	 * must not be ones the model was trained on: a model is far more confident about its
	 * training data than about anything else.  The fitted calibration is saved with the model.
	 * Refit after substantial further training.
	 * @param TBayesianTrainingSet $heldOut Labeled documents the model was not trained on.
	 * @throws TInvalidOperationException When the classifier has not been trained.
	 * @throws TInvalidDataValueException When the set is empty, or no document of it can be
	 * scored for its label.
	 * @return TTemperatureScaling The installed calibration.
	 * @since 0.2.0
	 */
	public function calibrate(TBayesianTrainingSet $heldOut): TTemperatureScaling
	{
		if (!$this->getIsTrained()) {
			throw new TInvalidOperationException('bayesian_classifier_not_trained', $this->_name ?? '');
		}
		if ($heldOut->getIsEmpty()) {
			throw new TInvalidDataValueException('bayesian_calibration_set_empty');
		}
		$logScores = [];
		$labels = [];
		foreach ($heldOut->each() as $category => $document) {
			$logScores[] = $this->logScores($document);
			$labels[] = (string) $category;
		}
		$calibration = new TTemperatureScaling();
		$calibration->fit($logScores, $labels);
		$this->_calibration = $calibration;
		return $calibration;
	}

	/**
	 * Returns the calibration {@see score()} applies, or null when scores are uncalibrated.
	 * @return ?TTemperatureScaling The calibration, or null.
	 * @since 0.2.0
	 */
	public function getCalibration(): ?TTemperatureScaling
	{
		return $this->_calibration;
	}

	/**
	 * Installs a calibration (fitted elsewhere, or with a hand-set temperature), or removes it
	 * with null so {@see score()} returns the plain normalized scores again.
	 * @param ?TTemperatureScaling $value The calibration, or null.
	 * @since 0.2.0
	 */
	public function setCalibration(?TTemperatureScaling $value): void
	{
		$this->_calibration = $value;
	}

	/**
	 * Returns whether {@see score()} returns calibrated probabilities.
	 * @return bool Whether a calibration is installed.
	 * @since 0.2.0
	 */
	public function getIsCalibrated(): bool
	{
		return $this->_calibration !== null;
	}

	/**
	 * Returns state another component keeps with this model, saved and loaded alongside it.
	 * @param string $owner The owner key, e.g. `tagger`.
	 * @return ?array<string, mixed> The state, or null when the owner has none.
	 * @since 0.2.0
	 */
	public function getExtraState(string $owner): ?array
	{
		return $this->_extra[$owner] ?? null;
	}

	/**
	 * Stores state another component keeps with this model, or removes it with null.  The
	 * state must be JSON-serializable; it travels in the saved payload and per-token metadata.
	 * @param string $owner The owner key.
	 * @param ?array<string, mixed> $state The state, or null to remove it.
	 * @since 0.2.0
	 */
	public function setExtraState(string $owner, ?array $state): void
	{
		if ($state === null) {
			unset($this->_extra[$owner]);
			return;
		}
		$this->_extra[$owner] = $state;
	}

	/**
	 * Computes the log-posterior over categories for a token sequence.
	 *
	 * The smoothing constant `alpha` keeps tokens unseen in a category from zeroing the score:
	 * every (token, category) cell is treated as having seen `alpha` extra observations.
	 * Tokens absent from the whole vocabulary are skipped.  TF-IDF weighting (when enabled)
	 * substitutes the document's raw token count with `1 + log(count)` times the term's IDF —
	 * a long document's repetitions stop dominating the score.
	 * @param string[] $tokens The token sequence.
	 * @return array<string, float> The log-posteriors, keyed by category.
	 */
	protected function scoreTokens(array $tokens): array
	{
		$categories = $this->_vocabulary->getCategories();
		if ($categories === []) {
			return [];
		}
		// One batched read for the whole document before any category is scored.  A resident
		// vocabulary ignores this; a storage-backed one turns what would be a lookup per token
		// per category into a single round trip.
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
	 * Computes the log-likelihood contribution of a token sequence against a category.
	 *
	 * Out-of-vocabulary tokens (never seen in training) are skipped.  Returns -INF when the
	 * smoothed denominator is zero (an empty vocabulary), so the log-posterior is undefined.
	 * @param string[] $tokens The tokens of the document.
	 * @param TBayesianCategory $category The candidate category.
	 * @param int $totalDocs The total document count.
	 * @return float The log-likelihood (sum of per-token log-probabilities).
	 */
	protected function logLikelihood(array $tokens, TBayesianCategory $category, int $totalDocs): float
	{
		$alpha = $this->_alpha;
		// Laplace smoothing uses the GLOBAL vocabulary size |V| — the number of distinct terms
		// across the whole corpus, counted once.  (Summing each category's own distinct-token
		// count would double-count every shared term and over-smooth.)
		$denominator = $category->getTotalTokens() + $alpha * $this->_vocabulary->getVocabularySize();
		if ($denominator <= 0.0) {
			return -INF;
		}
		$counts = $this->aggregate($tokens);
		$logSum = 0.0;
		foreach ($counts as $token => $count) {
			$token = (string) $token;
			$documentFrequency = $this->_vocabulary->getTokenDocumentFrequency($token);
			if ($documentFrequency === 0) {
				continue;
			}
			$weight = $this->_useTfidf
				? TFIdf::weight($count, $documentFrequency, $totalDocs)
				: (float) $count;
			if ($weight === 0.0) {
				continue;
			}
			$numerator = $category->getTokenCount($token) + $alpha;
			$logP = log($numerator / $denominator);
			$logSum += $weight * $logP;
		}
		return $logSum;
	}

	/**
	 * Aggregates a token sequence into a (token => count) map.
	 * @param string[] $tokens The tokens.
	 * @return array<string, int> The counts.
	 */
	protected function aggregate(array $tokens): array
	{
		$counts = [];
		foreach ($tokens as $token) {
			$counts[$token] = ($counts[$token] ?? 0) + 1;
		}
		return $counts;
	}

	/**
	 * Convenience helper for the spam-filter use case: returns true when the document is
	 * more likely to be in the {@see getSpamCategory() spam category} than any other.
	 * @param string|string[] $document The document.
	 * @return bool Whether the document is classified as spam.
	 */
	public function isSpam($document): bool
	{
		return $this->classify($document) === $this->_spamCategory;
	}

	/**
	 * Persists the trained state under the configured name.  Requires a storage backend.
	 * @throws TConfigurationException When name or storage is unset.
	 */
	public function save(): void
	{
		if ($this->_name === null || $this->_name === '') {
			throw new TConfigurationException('bayesian_classifier_name_required');
		}
		if ($this->_storage === null) {
			throw new TConfigurationException('bayesian_storage_required');
		}
		if ($this->getIsTokenStorage()) {
			$this->saveTokenModel();
			return;
		}
		$this->_storage->save($this->_name, $this->exportState());
	}

	/**
	 * Returns whether the configured storage is set up for per-token lookup.
	 * @return bool Whether per-token storage is in use.
	 */
	protected function getIsTokenStorage(): bool
	{
		return $this->tokenStorage() !== null;
	}

	/**
	 * Returns the storage narrowed to {@see IBayesianTokenStorage} when it is one and is
	 * configured for per-token lookup, or null otherwise.
	 * @return ?IBayesianTokenStorage The token storage, or null.
	 */
	private function tokenStorage(): ?IBayesianTokenStorage
	{
		$storage = $this->_storage;
		if ($storage instanceof IBayesianTokenStorage && $storage->getSupportsTokenLookup()) {
			return $storage;
		}
		return null;
	}

	/**
	 * Writes the model to a per-token backend: the scalars and aggregates as metadata, the
	 * per-category totals, and one entry per (token, category).
	 *
	 * The vocabulary must be resident for this — the statistics are being written out, which
	 * means reading all of them.  A classifier that loaded a model lazily has no whole model to
	 * write back; train against a resident vocabulary and save from there.
	 * @throws TConfigurationException When the classifier has no name.
	 * @throws TInvalidOperationException When the vocabulary cannot be enumerated.
	 */
	protected function saveTokenModel(): void
	{
		$modelName = $this->_name;
		if ($modelName === null || $modelName === '') {
			throw new TConfigurationException('bayesian_classifier_name_required');
		}
		$storage = $this->tokenStorage();
		if ($storage === null) {
			throw new TInvalidOperationException('bayesian_storage_token_mode_required');
		}
		if (!$this->_vocabulary->getSupportsFullScan()) {
			throw new TInvalidOperationException('bayesian_vocabulary_full_scan_unavailable', (string) $this->_name);
		}
		$categories = [];
		$tokens = [];
		foreach ($this->_vocabulary->getCategories() as $category) {
			$name = $category->getName();
			$categories[$name] = [
				'documentCount' => $category->getDocumentCount(),
				'totalTokens' => $category->getTotalTokens(),
			];
			$documentCounts = $category->getTokenDocumentCounts();
			foreach ($category->getTokenCounts() as $token => $count) {
				$tokens[(string) $token][$name] = [
					'count' => (int) $count,
					'docCount' => (int) ($documentCounts[$token] ?? 0),
				];
			}
		}
		$storage->saveTokenModel($modelName, $this->exportTokenMeta(), $categories, $tokens);
	}

	/**
	 * Builds the model-level state for a per-token save: everything from {@see exportState()}
	 * except the per-token maps, plus the scalars a lazy vocabulary cannot derive.
	 * @return array<string, mixed> The metadata.
	 */
	protected function exportTokenMeta(): array
	{
		// Built directly rather than by stripping exportState(): that walks every category's
		// token maps, which is exactly what a storage-backed vocabulary cannot produce — and
		// this method has to work while training one incrementally.
		return [
			'formatVersion' => self::FORMAT_VERSION,
			'kind' => $this->getKind(),
			'name' => $this->_name,
			'alpha' => $this->_alpha,
			'useTfidf' => $this->_useTfidf,
			'spamCategory' => $this->_spamCategory,
			'tokenizer' => $this->exportTokenizer(),
			'totalDocuments' => $this->_vocabulary->getTotalDocuments(),
			'vocabularySize' => $this->_vocabulary->getVocabularySize(),
			// The categories come back from a table, which has no inherent order; carrying the
			// order here keeps score() returning its keys in the order the categories were
			// first trained, as a resident model does.
			'categoryOrder' => array_map('strval', $this->_vocabulary->getCategoryNames()),
			'tokenMode' => true,
			// The histogram families the storage keeps current for this model, so the
			// aggregates that span the vocabulary can be computed without walking it.
			'histograms' => $this->getHistogramFamilies(),
			'aggregates' => $this->exportAggregates(),
			'calibration' => $this->_calibration?->export(),
			'extra' => $this->_extra,
		];
	}

	/**
	 * Loads a previously-saved trained state by name.  Requires a storage backend.
	 * @param string $name The model name.
	 * @throws TConfigurationException When storage is unset or the model is unknown.
	 */
	public function load(string $name): void
	{
		if ($this->_storage === null) {
			throw new TConfigurationException('bayesian_storage_required');
		}
		if ($this->getIsTokenStorage()) {
			$this->loadTokenModel($name);
			return;
		}
		$payload = $this->_storage->load($name);
		if ($payload === null) {
			throw new TConfigurationException('bayesian_classifier_model_missing', $name);
		}
		$this->importState($payload, $name);
		$this->_name = $name;
	}

	/**
	 * Loads a model from a per-token backend, binding the classifier to a
	 * {@see TLazyBayesianVocabulary} that reads statistics a document at a time.
	 *
	 * After this the classifier scores normally but holds only the scalars and categories; the
	 * per-token statistics stay in storage and arrive one batched query per classification.
	 * @param string $name The model name.
	 * @throws TConfigurationException When the model is unknown.
	 * @throws TInvalidDataValueException When the payload belongs to another classifier variant
	 * or a newer saved-state format.
	 * @throws TInvalidOperationException When the storage is not configured for per-token lookup.
	 */
	protected function loadTokenModel(string $name): void
	{
		$storage = $this->tokenStorage();
		if ($storage === null) {
			throw new TInvalidOperationException('bayesian_storage_token_mode_required');
		}
		$meta = $storage->loadTokenMeta($name);
		if ($meta === null) {
			throw new TConfigurationException('bayesian_classifier_model_missing', $name);
		}
		$this->assertFormatVersion($meta, $name);
		$kind = $meta['kind'] ?? null;
		if (is_string($kind) && $kind !== '' && !in_array($kind, $this->getCompatibleKinds(), true)) {
			throw new TInvalidDataValueException('bayesian_classifier_kind_mismatch', $kind, $this->getKind());
		}
		$vocabulary = new TLazyBayesianVocabulary($storage, $name);
		$vocabulary->initialize($meta, $storage->loadTokenCategories($name));
		$this->_vocabulary = $vocabulary;

		$alpha = TBayesianPayload::float($meta['alpha'] ?? null, 1.0);
		$this->_alpha = $alpha > 0.0 ? $alpha : 1.0;
		$this->_useTfidf = TBayesianPayload::bool($meta['useTfidf'] ?? null, true);
		$this->_spamCategory = TBayesianPayload::string($meta['spamCategory'] ?? null, 'spam');
		$tokenizer = $meta['tokenizer'] ?? null;
		if (is_array($tokenizer)) {
			$this->importTokenizer(TBayesianPayload::map($tokenizer));
		}
		$this->onTrainingChanged();
		$this->importAggregates(TBayesianPayload::map($meta['aggregates'] ?? null));
		$this->importCalibrationAndExtra($meta);
		$this->_name = $name;
	}

	/**
	 * Returns the {@see TBayesianTokenHistogram} families this classifier computes its
	 * likelihood from.  A per-token storage keeps exactly these current for the model.
	 *
	 * The base classifier needs none: every term of the multinomial likelihood comes from the
	 * document's own tokens.
	 * @return string[] The family letters.
	 * @since 0.2.0
	 */
	protected function getHistogramFamilies(): array
	{
		return [];
	}

	/**
	 * Returns whether {@see getTokenHistograms()} can answer for the current vocabulary, without
	 * counting or reading anything.
	 * @return bool Whether token histograms are available.
	 * @since 0.2.0
	 */
	protected function getHasTokenHistograms(): bool
	{
		$vocabulary = $this->_vocabulary;
		return $vocabulary->getSupportsFullScan()
			|| ($vocabulary instanceof TLazyBayesianVocabulary && $vocabulary->getSupportsTokenHistograms());
	}

	/**
	 * Returns the token histograms of {@see getHistogramFamilies()} for the current vocabulary:
	 * counted here when the vocabulary is resident, read from the storage when it is not.
	 * @return ?array<string, array<string, array<int, int>>> The histograms, as family =>
	 * category => value => token count; null when the vocabulary is storage-backed and its
	 * storage keeps no histograms.
	 * @since 0.2.0
	 */
	protected function getTokenHistograms(): ?array
	{
		$families = $this->getHistogramFamilies();
		$vocabulary = $this->_vocabulary;
		if ($vocabulary->getSupportsFullScan()) {
			return TBayesianTokenHistogram::fromVocabulary($vocabulary, $families);
		}
		return $vocabulary instanceof TLazyBayesianVocabulary ? $vocabulary->getTokenHistograms($families) : null;
	}

	/**
	 * Returns the derived quantities that cost a full pass over the vocabulary to compute, so
	 * they can travel with a per-token model.
	 *
	 * The base classifier has none: every term of the multinomial likelihood comes from the
	 * document's own tokens.  {@see TBernoulliNaiveBayes} and {@see TComplementNaiveBayes}
	 * override this, because theirs cannot be recomputed against a vocabulary that cannot be
	 * enumerated.
	 * @return array<string, mixed> The aggregates.
	 */
	protected function exportAggregates(): array
	{
		return [];
	}

	/**
	 * Restores the aggregates written by {@see exportAggregates()}.
	 * @param array<string, mixed> $aggregates The stored aggregates.
	 */
	protected function importAggregates(array $aggregates): void
	{
	}

	/**
	 * Builds the JSON-serializable state representation of the trained classifier.
	 *
	 * The shape is shared by all storage backends: a "kind" marker ({@see getKind()}) so a
	 * loader can refuse a payload from a different classifier variant, the per-category
	 * statistics, the document-frequency map, and the configuration that affects the math
	 * (alpha, TF-IDF flag, spam category, and the tokenizer class + settings).
	 * @return array<string, mixed> The state payload.
	 */
	protected function exportState(): array
	{
		$categories = [];
		foreach ($this->_vocabulary->getCategories() as $category) {
			$categories[] = [
				'name' => $category->getName(),
				'documentCount' => $category->getDocumentCount(),
				'tokenCounts' => $category->getTokenCounts(),
				'tokenDocumentCounts' => $category->getTokenDocumentCounts(),
				'totalTokens' => $category->getTotalTokens(),
			];
		}
		return [
			'formatVersion' => self::FORMAT_VERSION,
			'kind' => $this->getKind(),
			'name' => $this->_name,
			'alpha' => $this->_alpha,
			'useTfidf' => $this->_useTfidf,
			'spamCategory' => $this->_spamCategory,
			'tokenizer' => $this->exportTokenizer(),
			'categories' => $categories,
			'documentFrequency' => $this->_vocabulary->getDocumentFrequency(),
			'totalDocuments' => $this->_vocabulary->getTotalDocuments(),
			'calibration' => $this->_calibration?->export(),
			'extra' => $this->_extra,
		];
	}

	/**
	 * Returns the "kind" marker written into the saved state, identifying the classifier variant.
	 * @return string The kind.
	 */
	protected function getKind(): string
	{
		return 'naive-bayes';
	}

	/**
	 * Returns the kind markers this classifier can load.  The base and multinomial classifiers
	 * are mathematically identical, so each accepts the other's payload.
	 * @return string[] The accepted kinds.
	 */
	protected function getCompatibleKinds(): array
	{
		return ['naive-bayes', 'multinomial-naive-bayes'];
	}

	/**
	 * Refuses a payload written in a saved-state format newer than this class reads.
	 * @param array<string, mixed> $payload The payload or per-token metadata.
	 * @param string $name The model name, for the message.
	 * @throws TInvalidDataValueException When the payload's `formatVersion` exceeds {@see FORMAT_VERSION}.
	 * @since 0.2.0
	 */
	protected function assertFormatVersion(array $payload, string $name): void
	{
		$version = TBayesianPayload::int($payload['formatVersion'] ?? null, 1);
		if ($version > self::FORMAT_VERSION) {
			throw new TInvalidDataValueException('bayesian_model_format_unsupported', $name, (string) $version, (string) self::FORMAT_VERSION);
		}
	}

	/**
	 * Restores a state payload produced by {@see exportState()}.
	 * @param array<string, mixed> $payload The payload.
	 * @param ?string $name The model name being loaded, for error messages; the payload's own
	 * name when null.
	 * @throws TInvalidDataValueException When the payload's format is newer than this class
	 * reads, its kind marker belongs to a different classifier variant, or its tokenizer state
	 * names a class that is not a tokenizer.
	 */
	protected function importState(array $payload, ?string $name = null): void
	{
		$this->assertFormatVersion($payload, $name ?? TBayesianPayload::string($payload['name'] ?? null, $this->_name ?? ''));
		$kind = $payload['kind'] ?? null;
		if (is_string($kind) && $kind !== '' && !in_array($kind, $this->getCompatibleKinds(), true)) {
			throw new TInvalidDataValueException('bayesian_classifier_kind_mismatch', $kind, $this->getKind());
		}
		if (!empty($payload['tokenMode'])) {
			// The two layouts share a metadata row, so a per-token model read through the blob
			// path arrives as a payload with no categories and no token maps.  That would
			// import cleanly as an untrained classifier and classify everything as whatever the
			// empty model says, which is the worst possible failure: silent.
			throw new TInvalidDataValueException('bayesian_classifier_token_mode_payload', TBayesianPayload::string($payload['name'] ?? null));
		}
		$categories = [];
		foreach (TBayesianPayload::list($payload['categories'] ?? null) as $row) {
			$row = TBayesianPayload::map($row);
			$cat = new TBayesianCategory(TBayesianPayload::string($row['name'] ?? null));
			$cat->setStats(
				TBayesianPayload::int($row['documentCount'] ?? null),
				TBayesianPayload::intMap($row['tokenCounts'] ?? null),
				TBayesianPayload::intMap($row['tokenDocumentCounts'] ?? null),
				TBayesianPayload::int($row['totalTokens'] ?? null)
			);
			$categories[] = $cat;
		}
		$this->_vocabulary->setStats(
			$categories,
			TBayesianPayload::intMap($payload['documentFrequency'] ?? null),
			TBayesianPayload::int($payload['totalDocuments'] ?? null)
		);
		$alpha = TBayesianPayload::float($payload['alpha'] ?? null, 1.0);
		$this->_alpha = $alpha > 0.0 ? $alpha : 1.0;
		$this->_useTfidf = TBayesianPayload::bool($payload['useTfidf'] ?? null, true);
		$this->_spamCategory = TBayesianPayload::string($payload['spamCategory'] ?? null, 'spam');
		$tokenizer = $payload['tokenizer'] ?? null;
		if (is_array($tokenizer)) {
			$this->importTokenizer(TBayesianPayload::map($tokenizer));
		}
		$this->importCalibrationAndExtra($payload);
		$this->onTrainingChanged();
	}

	/**
	 * Restores the calibration and the extra state a payload or per-token metadata carries;
	 * a payload without them (0.1.0, or never calibrated) leaves both empty.
	 * @param array<string, mixed> $payload The payload or metadata.
	 */
	private function importCalibrationAndExtra(array $payload): void
	{
		$calibration = $payload['calibration'] ?? null;
		$this->_calibration = is_array($calibration) ? TTemperatureScaling::import(TBayesianPayload::map($calibration)) : null;
		$this->_extra = [];
		foreach (TBayesianPayload::map($payload['extra'] ?? null) as $owner => $state) {
			if (is_array($state)) {
				$this->_extra[$owner] = TBayesianPayload::map($state);
			}
		}
	}

	/**
	 * Serializes the tokenizer (class and settings) for the saved state, via
	 * {@see TBayesianTokenizerFactory::export()}.
	 * @return array<string, mixed> The tokenizer state.
	 */
	protected function exportTokenizer(): array
	{
		return TBayesianTokenizerFactory::export($this->_tokenizer);
	}

	/**
	 * Restores the tokenizer from a saved-state entry via {@see TBayesianTokenizerFactory::restore()}.
	 * When the current tokenizer is of the stored class it is re-configured in place; otherwise
	 * a new instance of the stored class replaces it.  A state without a class name keeps the
	 * current tokenizer (legacy payloads).
	 * @param array<string, mixed> $state The tokenizer state.
	 * @throws TInvalidDataValueException When the stored class is not a tokenizer.
	 */
	protected function importTokenizer(array $state): void
	{
		$restored = TBayesianTokenizerFactory::restore($state, $this->_tokenizer);
		if ($restored !== null) {
			$this->_tokenizer = $restored;
		}
	}

	/**
	 * Returns the model name used as the storage key by {@see save()} and {@see load()}.
	 * @return ?string The name, or null when the classifier is unnamed.
	 */
	public function getName(): ?string
	{
		return $this->_name;
	}

	/**
	 * Sets the model name used as the storage key.  An empty string is normalized to null, so
	 * `Name=""` in a configuration leaves the classifier unnamed rather than keying on `''`.
	 * @param ?string $value The name.
	 */
	public function setName(?string $value): void
	{
		$this->_name = $value === '' ? null : $value;
	}

	/**
	 * Returns the tokenizer that turns input text into feature tokens.  Defaults to a
	 * {@see TWordTokenizer} when none has been set.
	 * @return IBayesianTokenizer The tokenizer.
	 */
	public function getTokenizer(): IBayesianTokenizer
	{
		return $this->_tokenizer;
	}

	/**
	 * Sets the tokenizer.  Changing it after training invalidates the learned vocabulary,
	 * because the stored tokens were produced by the previous tokenizer; retrain (or
	 * {@see load()} a model saved with the new tokenizer) before classifying again.
	 * @param IBayesianTokenizer $value The tokenizer.
	 */
	public function setTokenizer(IBayesianTokenizer $value): void
	{
		$this->_tokenizer = $value;
	}

	/**
	 * Returns the storage backend used by {@see save()} and {@see load()}.
	 * @return ?IBayesianStorage The storage, or null when none is configured.
	 */
	public function getStorage(): ?IBayesianStorage
	{
		return $this->_storage;
	}

	/**
	 * Sets the storage backend.  With no storage configured, {@see save()} and {@see load()}
	 * raise a configuration exception rather than silently doing nothing.
	 * @param ?IBayesianStorage $value The storage.
	 */
	public function setStorage(?IBayesianStorage $value): void
	{
		$this->_storage = $value;
	}

	/**
	 * Returns the learned vocabulary: the per-category token counts and document frequencies
	 * accumulated by training.  The instance is live, not a copy — training mutates it.
	 * @return IBayesianVocabulary The vocabulary.
	 */
	public function getVocabulary(): IBayesianVocabulary
	{
		return $this->_vocabulary;
	}

	/**
	 * Returns whether the classifier has anything to classify with, i.e. whether its
	 * vocabulary is non-empty after training or a {@see load()}.
	 * @return bool Whether the classifier has been trained.
	 */
	public function getIsTrained(): bool
	{
		return !$this->_vocabulary->getIsEmpty();
	}

	/**
	 * Returns whether token contributions are re-weighted by TF-IDF at classification time.
	 * @return bool Whether TF-IDF re-weighting is applied.
	 */
	public function getUseTfidf(): bool
	{
		return $this->_useTfidf;
	}

	/**
	 * Sets whether token contributions are re-weighted by TF-IDF, damping terms that occur in
	 * most documents.  The flag is part of the saved payload, so a model reloads with the
	 * same weighting it was trained under.
	 * @param bool $value Whether to apply TF-IDF re-weighting.
	 */
	public function setUseTfidf(bool $value): void
	{
		$this->_useTfidf = $value;
	}

	/**
	 * Returns the Laplace (additive) smoothing constant added to every token count, which
	 * keeps an unseen token from zeroing out a category's probability.
	 * @return float The Laplace smoothing constant.
	 */
	public function getAlpha(): float
	{
		return $this->_alpha;
	}

	/**
	 * Sets the Laplace smoothing constant.  It must be positive: with zero or negative alpha
	 * the smoothed probabilities are 0, negative, or undefined and every score degenerates.
	 * @param float $value The Laplace smoothing constant (> 0).
	 * @throws TInvalidDataValueException When the value is not positive or not finite.
	 */
	public function setAlpha(float $value): void
	{
		if (!($value > 0.0) || !is_finite($value)) {
			throw new TInvalidDataValueException('bayesian_alpha_invalid', TBayesianPayload::formatFloat($value));
		}
		$this->_alpha = $value;
	}

	/**
	 * Returns the category name treated as "spam" by {@see isSpam()}, and the key looked up in
	 * the {@see score()} map by the spam-filter convenience path.
	 * @return string The spam category.
	 */
	public function getSpamCategory(): string
	{
		return $this->_spamCategory;
	}

	/**
	 * Sets the category name treated as "spam" by the spam-filter convenience methods.  It is
	 * an ordinary category label — the classifier itself is not spam-specific.
	 * @param string $value The spam category.
	 */
	public function setSpamCategory(string $value): void
	{
		$this->_spamCategory = $value;
	}
}
