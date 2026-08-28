<?php

/**
 * TBayesianRecommender class file.
 *
 * @author Brad Anderson <belisoful@icloud.com>
 * @link https://github.com/belisoful/prado-bayesian
 * @license https://github.com/pradosoft/prado/blob/master/LICENSE
 */

namespace Prado\Util\Bayesian;

use Prado\Exceptions\TInvalidDataValueException;
use Prado\Exceptions\TInvalidOperationException;
use Prado\TComponent;
use Prado\Util\Bayesian\Classifier\IBayesianClassifier;
use Prado\Util\Bayesian\Classifier\TNaiveBayesClassifier;

/**
 * TBayesianRecommender class.
 *
 * A probabilistic recommender built on top of any {@see IBayesianClassifier}.  Train the
 * classifier on positive/negative user interactions (e.g. "liked"/"ignored"), then ask it to
 * score candidate items for a new user context.
 *
 * The default scoring strategy is the posterior probability of the positive class given the
 * candidate's features — so a candidate whose token profile is typical of the positive class
 * ranks highest.  When the classifier is set up with the {@see TBernoulliNaiveBayes} model,
 * the scoring naturally handles presence/absence features (e.g. "user has watched this
 * director's other films").
 *
 * The {@see IBayesianClassifier} interface is reused for the train/persist surface, so a
 * recommender can be saved and reloaded through the same storage backends as a classifier.
 *
 * @author Brad Anderson <belisoful@icloud.com>
 * @since 0.1.0
 */
class TBayesianRecommender extends TComponent implements IBayesianRecommender
{
	/** @var IBayesianClassifier The classifier that scores candidates. */
	private IBayesianClassifier $_classifier;

	/** @var string The category treated as a positive interaction. */
	private string $_positiveCategory = 'liked';

	/**
	 * Initializes the recommender with a default {@see TNaiveBayesClassifier}.
	 */
	public function __construct()
	{
		$this->_classifier = new TNaiveBayesClassifier();
		parent::__construct();
	}

	/**
	 * Scores a list of candidate items against a user context and returns them ranked
	 * from best to worst.
	 *
	 * The probe for each candidate is the concatenation of the context and the candidate,
	 * so the score is P(positive | context + candidate).  The result is a map of candidate
	 * identifier to score; the keys appear in descending score order (ties keep candidate
	 * order).  Candidates are treated as a unique set — a repeated identifier is scored once —
	 * and blank (empty or whitespace-only) identifiers are ignored.  An empty candidate list,
	 * or one with only blank identifiers, throws.
	 *
	 * Identifiers are array keys, so a purely numeric identifier such as `"123"` comes back as
	 * the integer key `123`; encode the result with `JSON_FORCE_OBJECT` if it must stay a map.
	 * @param string[] $context The items the user has interacted with (positive or negative).
	 * @param string[] $candidates The items to rank.
	 * @throws TInvalidDataValueException When the candidate list is empty.
	 * @throws TInvalidOperationException When the classifier has not been trained, or the
	 * {@see getPositiveCategory() positive category} is not one of its trained categories.
	 * @return array<string, float> The ranked scores, highest first.
	 */
	public function recommend(array $context, array $candidates): array
	{
		if ($candidates === []) {
			throw new TInvalidDataValueException('bayesian_recommendation_candidates_empty');
		}
		if (!$this->_classifier->getIsTrained()) {
			throw new TInvalidOperationException('bayesian_classifier_not_trained', $this->_classifier->getName() ?? '');
		}
		// Without the positive category in the trained vocabulary every candidate scores 0.0 and
		// the ranking is meaningless; that is a misconfiguration, not a result.
		$categories = array_map('strval', $this->_classifier->getVocabulary()->getCategoryNames());
		if (!in_array($this->_positiveCategory, $categories, true)) {
			throw new TInvalidOperationException(
				'bayesian_recommender_category_unknown',
				$this->_positiveCategory,
				implode(', ', $categories)
			);
		}
		$contextText = trim(implode(' ', $context));
		$scores = [];
		foreach ($candidates as $candidate) {
			$candidate = trim((string) $candidate);
			// A candidate identifier is a unique result key; do not rescore a repeat.
			if ($candidate === '' || array_key_exists($candidate, $scores)) {
				continue;
			}
			$probe = trim($contextText . ' ' . $candidate);
			$distribution = $this->_classifier->score($probe);
			$scores[$candidate] = $distribution[$this->_positiveCategory] ?? 0.0;
		}
		if ($scores === []) {
			throw new TInvalidDataValueException('bayesian_recommendation_candidates_empty');
		}
		arsort($scores);
		return $scores;
	}

	/**
	 * Returns the classifier whose per-category scores the ranking is derived from.
	 * @return IBayesianClassifier The classifier.
	 */
	public function getClassifier(): IBayesianClassifier
	{
		return $this->_classifier;
	}

	/**
	 * Sets the classifier backing the recommendations.  Any {@see IBayesianClassifier} works —
	 * the recommender adds ranking on top and does no training of its own.
	 * @param IBayesianClassifier $value The classifier.
	 */
	public function setClassifier(IBayesianClassifier $value): void
	{
		$this->_classifier = $value;
	}

	/**
	 * Returns the category whose probability candidates are ranked by.
	 * @return string The positive category.
	 */
	public function getPositiveCategory(): string
	{
		return $this->_positiveCategory;
	}

	/**
	 * Sets the category whose probability candidates are ranked by — the "liked" or "relevant"
	 * label the classifier was trained with.  A candidate scoring high on this category ranks
	 * first.
	 * @param string $value The positive category.
	 */
	public function setPositiveCategory(string $value): void
	{
		$this->_positiveCategory = $value;
	}
}
