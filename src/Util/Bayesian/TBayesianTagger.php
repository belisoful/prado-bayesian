<?php

/**
 * TBayesianTagger class file.
 *
 * @author Brad Anderson <belisoful@icloud.com>
 * @link https://github.com/belisoful/prado-bayesian
 * @license https://github.com/pradosoft/prado/blob/master/LICENSE
 */

namespace Belisoful\Prado\Util\Bayesian;

use Belisoful\Prado\Util\Bayesian\Calibration\TPlattScaling;
use Belisoful\Prado\Util\Bayesian\Classifier\TNaiveBayesClassifier;
use Belisoful\Prado\Util\Bayesian\Math\TFIdf;
use Prado\Exceptions\TInvalidDataValueException;
use Prado\Exceptions\TInvalidOperationException;
use Prado\TComponent;

/**
 * TBayesianTagger class.
 *
 * Multi-label classification — tagging — as one-versus-rest Naive Bayes over one shared model.
 * A document is trained under each of its labels and, once, under a *background* category
 * that every document goes into.  For any label the "rest" is then the background minus the
 * label: `count(t, ¬L) = count(t, *) − count(t, L)`, `docs(¬L) = docs(*) − docs(L)`, and so
 * on, so each label gets an exact binary Naive Bayes decision from statistics the model
 * already holds.  Training a document costs one write per label plus one; nothing is stored
 * per label pair, and every storage backend — whole-payload or per-token, resident or lazy —
 * works unchanged, including concurrent training.
 *
 * For each label the tagger computes the log-odds
 *
 *   log P(L | d) − log P(¬L | d) = log (n_L + 1)/(n_¬L + 1) + Σ_t w_t · [ log θ_L(t) − log θ_¬L(t) ]
 *
 * with Laplace-smoothed multinomial likelihoods `θ_c(t) = (count(t, c) + α) / (total(c) + α|V|)`,
 * the classifier's `Alpha` and (optionally) TF-IDF weights, and turns it into a probability
 * with the logistic function.  The probabilities are independent: they need not sum to one,
 * and a document may earn every tag or none.  Like every Naive Bayes output they are
 * overconfident until calibrated; {@see calibrate()} fits a {@see TPlattScaling} per label on
 * held-out documents and stores it with the model.
 *
 * ```php
 * $tagger = new TBayesianTagger();
 * $tagger->getClassifier()->setStorage($storage);
 * $tagger->getClassifier()->setName('post-tags');
 * $tagger->train(['php', 'security'], 'Validate every request parameter before it reaches SQL');
 * $tagger->train(['cooking'], 'Simmer the sauce for twenty minutes');
 * $tagger->train([], 'Nothing in particular');            // a negative example for every label
 * $tagger->tag('Escape the parameters of the query');     // ['php' => 0.9, 'security' => 0.8]
 * $tagger->getClassifier()->save();
 * ```
 *
 * The underlying classifier's own `classify()` and `score()` see the background as a category
 * and are not meaningful for a tagged model; score through the tagger.  The background
 * category's name is {@see setBackgroundCategory BackgroundCategory} (default `*`) and may
 * not be used as a label.
 *
 * @author Brad Anderson <belisoful@icloud.com>
 * @since 0.2.0
 */
class TBayesianTagger extends TComponent implements IBayesianTagger
{
	/** The key the tagger's state is stored under in the classifier's extra state. */
	public const EXTRA_STATE_KEY = 'tagger';

	/** @var TNaiveBayesClassifier The classifier holding the label statistics. */
	private TNaiveBayesClassifier $_classifier;

	/** @var string The category every document is counted in. */
	private string $_backgroundCategory = '*';

	/** @var float The probability a label needs to be returned by {@see tag()}. */
	private float $_threshold = 0.5;

	/** @var int The most tags {@see tag()} returns; 0 for no limit. */
	private int $_maxTags = 0;

	/**
	 * Initializes the tagger with a default {@see TNaiveBayesClassifier}.
	 */
	public function __construct()
	{
		$this->_classifier = new TNaiveBayesClassifier();
		parent::__construct();
	}

	/**
	 * Returns the classifier holding the label statistics.
	 * @return TNaiveBayesClassifier The classifier.
	 */
	public function getClassifier(): TNaiveBayesClassifier
	{
		return $this->_classifier;
	}

	/**
	 * Sets the classifier holding the label statistics.  Its tokenizer, storage, name, `Alpha`
	 * and `UseTfidf` are what the tagger trains and scores with; its own scoring is not used.
	 * @param TNaiveBayesClassifier $value The classifier.
	 */
	public function setClassifier(TNaiveBayesClassifier $value): void
	{
		$this->_classifier = $value;
	}

	/**
	 * Returns the category every document is counted in.
	 * @return string The background category name.
	 */
	public function getBackgroundCategory(): string
	{
		return $this->_backgroundCategory;
	}

	/**
	 * Sets the category every document is counted in.  Change it before training; a model
	 * trained under one name cannot be read under another.
	 * @param string $value The background category name.
	 * @throws TInvalidDataValueException When the name is empty.
	 */
	public function setBackgroundCategory(string $value): void
	{
		if ($value === '') {
			throw new TInvalidDataValueException('bayesian_category_required');
		}
		$this->_backgroundCategory = $value;
	}

	/**
	 * Returns the probability a label needs to be returned by {@see tag()}.
	 * @return float The threshold.
	 */
	public function getThreshold(): float
	{
		return $this->_threshold;
	}

	/**
	 * Sets the probability a label needs to be returned by {@see tag()}; clamped into [0, 1].
	 * @param float $value The threshold.
	 */
	public function setThreshold(float $value): void
	{
		$this->_threshold = is_nan($value) ? 0.5 : min(1.0, max(0.0, $value));
	}

	/**
	 * Returns the most tags {@see tag()} returns; 0 for no limit.
	 * @return int The limit.
	 */
	public function getMaxTags(): int
	{
		return $this->_maxTags;
	}

	/**
	 * Sets the most tags {@see tag()} returns; values below 1 mean no limit.
	 * @param int $value The limit.
	 */
	public function setMaxTags(int $value): void
	{
		$this->_maxTags = max(0, $value);
	}

	/**
	 * Returns whether any document has been trained.
	 * @return bool Whether the background category exists.
	 */
	public function getIsTrained(): bool
	{
		return $this->_classifier->getVocabulary()->getCategory($this->_backgroundCategory) !== null;
	}

	/**
	 * Returns every label the tagger has seen a document for, in the order they were first
	 * trained.
	 * @return string[] The labels.
	 */
	public function getLabels(): array
	{
		$labels = [];
		foreach ($this->_classifier->getVocabulary()->getCategoryNames() as $name) {
			$name = (string) $name;
			if ($name !== $this->_backgroundCategory) {
				$labels[] = $name;
			}
		}
		return $labels;
	}

	/**
	 * Tokenizes a document once, so every category it is trained under sees the same tokens.
	 * @param string|string[] $document The document.
	 * @return string[] The tokens.
	 */
	private function tokens($document): array
	{
		if (is_array($document)) {
			$tokens = [];
			foreach ($document as $token) {
				$tokens[] = (string) $token;
			}
			return $tokens;
		}
		return $this->_classifier->getTokenizer()->tokenize((string) $document);
	}

	/**
	 * Validates and de-duplicates a label list.
	 * @param string[] $labels The labels.
	 * @throws TInvalidDataValueException When a label is empty or is the background category.
	 * @return string[] The distinct labels, in order.
	 */
	private function labels(array $labels): array
	{
		$out = [];
		foreach ($labels as $label) {
			$label = (string) $label;
			if ($label === '') {
				throw new TInvalidDataValueException('bayesian_category_required');
			}
			if ($label === $this->_backgroundCategory) {
				throw new TInvalidDataValueException('bayesian_tagger_label_reserved', $label);
			}
			$out[$label] = true;
		}
		return array_keys($out);
	}

	/**
	 * Records a training document under each of its labels and under the background
	 * category.  An empty label list is a valid document: it is a negative example for every
	 * label, and teaches the model the words that mean nothing.
	 * @param string[] $labels The document's labels.
	 * @param string|string[] $document The document (text or pre-tokenized).
	 * @throws TInvalidDataValueException When a label is empty or is the background category.
	 */
	public function train(array $labels, $document): void
	{
		$labels = $this->labels($labels);
		$tokens = $this->tokens($document);
		foreach ($labels as $label) {
			$this->_classifier->trainOne($label, $tokens);
		}
		$this->_classifier->trainOne($this->_backgroundCategory, $tokens);
	}

	/**
	 * Withdraws a training document from each of its labels and from the background category;
	 * the inverse of {@see train()} for the same document and labels.
	 * @param string[] $labels The labels the document was trained under.
	 * @param string|string[] $document The document, as it was trained.
	 * @throws TInvalidDataValueException When a label is empty or is the background category.
	 */
	public function untrain(array $labels, $document): void
	{
		$labels = $this->labels($labels);
		$tokens = $this->tokens($document);
		foreach ($labels as $label) {
			$this->_classifier->untrainOne($label, $tokens);
		}
		$this->_classifier->untrainOne($this->_backgroundCategory, $tokens);
	}

	/**
	 * Returns the raw log-odds of every label for a document — `log P(L|d) − log P(¬L|d)`
	 * before calibration — in label order.  What {@see calibrate()} fits on.
	 * @param string|string[] $document The document.
	 * @throws TInvalidOperationException When nothing has been trained.
	 * @return array<string, float> The log-odds, keyed by label.
	 */
	public function logOdds($document): array
	{
		$vocabulary = $this->_classifier->getVocabulary();
		$background = $vocabulary->getCategory($this->_backgroundCategory);
		if ($background === null) {
			throw new TInvalidOperationException('bayesian_classifier_not_trained', $this->_classifier->getName() ?? '');
		}
		$tokens = $this->tokens($document);
		$vocabulary->prefetch($tokens);
		$counts = [];
		foreach ($tokens as $token) {
			$counts[$token] = ($counts[$token] ?? 0) + 1;
		}
		$alpha = $this->_classifier->getAlpha();
		$useTfidf = $this->_classifier->getUseTfidf();
		$allDocuments = $background->getDocumentCount();
		$allTokens = $background->getTotalTokens();
		$smoothing = $alpha * $vocabulary->getVocabularySize();
		$out = [];
		foreach ($this->getLabels() as $label) {
			$category = $vocabulary->getCategory($label);
			if ($category === null) {
				continue;
			}
			$labelDocuments = $category->getDocumentCount();
			$restDocuments = max(0, $allDocuments - $labelDocuments);
			$logOdds = log(($labelDocuments + 1.0) / ($restDocuments + 1.0));
			$labelDenominator = $category->getTotalTokens() + $smoothing;
			$restDenominator = max(0, $allTokens - $category->getTotalTokens()) + $smoothing;
			if ($labelDenominator <= 0.0 || $restDenominator <= 0.0) {
				$out[$label] = $logOdds;
				continue;
			}
			foreach ($counts as $token => $count) {
				$token = (string) $token;
				// A token no document contains carries no evidence; the background's document
				// count for it is the corpus document frequency, each document counted once.
				$frequency = $background->getTokenDocumentCount($token);
				if ($frequency === 0) {
					continue;
				}
				$weight = $useTfidf ? TFIdf::weight($count, $frequency, $allDocuments) : (float) $count;
				$labelCount = $category->getTokenCount($token);
				$restCount = max(0, $background->getTokenCount($token) - $labelCount);
				$logOdds += $weight * (log(($labelCount + $alpha) / $labelDenominator) - log(($restCount + $alpha) / $restDenominator));
			}
			$out[$label] = $logOdds;
		}
		return $out;
	}

	/**
	 * Returns the probability of every label for a document, independently of one another:
	 * the logistic function of each label's log-odds, through the label's calibration when
	 * one has been fitted.
	 * @param string|string[] $document The document.
	 * @throws TInvalidOperationException When nothing has been trained.
	 * @return array<string, float> The probability of each label, in [0, 1], in label order.
	 */
	public function probabilities($document): array
	{
		$calibrations = $this->getCalibrations();
		$out = [];
		foreach ($this->logOdds($document) as $label => $logOdds) {
			$calibration = $calibrations[$label] ?? null;
			$out[$label] = $calibration !== null ? $calibration->apply($logOdds) : self::sigmoid($logOdds);
		}
		return $out;
	}

	/**
	 * Returns the labels whose probability reaches {@see getThreshold() Threshold}, highest
	 * first (ties keep label order), at most {@see getMaxTags() MaxTags} of them.
	 * @param string|string[] $document The document.
	 * @throws TInvalidOperationException When nothing has been trained.
	 * @return array<string, float> The tags and their probabilities.
	 */
	public function tag($document): array
	{
		$tags = [];
		foreach ($this->probabilities($document) as $label => $probability) {
			if ($probability >= $this->_threshold) {
				$tags[$label] = $probability;
			}
		}
		arsort($tags);
		if ($this->_maxTags > 0 && count($tags) > $this->_maxTags) {
			$tags = array_slice($tags, 0, $this->_maxTags, true);
		}
		return $tags;
	}

	/**
	 * Fits a {@see TPlattScaling} per label on held-out tagged documents and stores them with
	 * the model, so {@see probabilities()} and {@see tag()} return calibrated probabilities
	 * from then on.  The documents must not be ones the tagger was trained on.  A label with no
	 * held-out document keeps its previous calibration, or none.
	 * @param array<int, array{0: string[], 1: string|string[]}> $examples Pairs of the
	 * document's true labels and the document.
	 * @throws TInvalidOperationException When nothing has been trained.
	 * @throws TInvalidDataValueException When the list is empty.
	 * @return array<string, TPlattScaling> The calibrations fitted, keyed by label.
	 */
	public function calibrate(array $examples): array
	{
		if ($examples === []) {
			throw new TInvalidDataValueException('bayesian_calibration_set_empty');
		}
		$values = [];
		$positives = [];
		foreach ($examples as [$labels, $document]) {
			$truth = array_fill_keys($this->labels($labels), true);
			foreach ($this->logOdds($document) as $label => $logOdds) {
				$values[$label][] = $logOdds;
				$positives[$label][] = isset($truth[$label]);
			}
		}
		$calibrations = $this->getCalibrations();
		$fitted = [];
		foreach ($values as $label => $labelValues) {
			$scaling = new TPlattScaling();
			$scaling->fit($labelValues, $positives[$label]);
			$calibrations[$label] = $scaling;
			$fitted[$label] = $scaling;
		}
		$this->setCalibrations($calibrations);
		return $fitted;
	}

	/**
	 * Returns the per-label calibrations stored with the model.
	 * @return array<string, TPlattScaling> The calibrations, keyed by label; empty when uncalibrated.
	 */
	public function getCalibrations(): array
	{
		$state = $this->_classifier->getExtraState(self::EXTRA_STATE_KEY);
		$out = [];
		foreach (TBayesianPayload::map($state['calibration'] ?? null) as $label => $scaling) {
			if (is_array($scaling)) {
				$restored = TPlattScaling::import(TBayesianPayload::map($scaling));
				if ($restored !== null) {
					$out[$label] = $restored;
				}
			}
		}
		return $out;
	}

	/**
	 * Stores per-label calibrations with the model, replacing any there; null or an empty
	 * array removes them, so the plain logistic function applies again.
	 * @param ?array<string, TPlattScaling> $calibrations The calibrations, keyed by label.
	 */
	public function setCalibrations(?array $calibrations): void
	{
		if ($calibrations === null || $calibrations === []) {
			$this->_classifier->setExtraState(self::EXTRA_STATE_KEY, null);
			return;
		}
		$state = [];
		foreach ($calibrations as $label => $scaling) {
			$state[(string) $label] = $scaling->export();
		}
		$this->_classifier->setExtraState(self::EXTRA_STATE_KEY, ['calibration' => $state]);
	}

	/**
	 * Returns whether at least one label has a fitted calibration.
	 * @return bool Whether the probabilities are calibrated.
	 */
	public function getIsCalibrated(): bool
	{
		return $this->getCalibrations() !== [];
	}

	/**
	 * The logistic function, evaluated without overflow at either extreme.
	 * @param float $x The log-odds.
	 * @return float The probability.
	 */
	private static function sigmoid(float $x): float
	{
		if ($x >= 0.0) {
			return 1.0 / (1.0 + exp(-$x));
		}
		$e = exp($x);
		return $e / (1.0 + $e);
	}
}
