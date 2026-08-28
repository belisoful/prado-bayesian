<?php

use Belisoful\Prado\Util\Bayesian\Classifier\TNaiveBayesClassifier;
use Belisoful\Prado\Util\Bayesian\Evaluation\TBayesianMetrics;
use Belisoful\Prado\Util\Bayesian\Evaluation\TConfusionMatrix;
use Belisoful\Prado\Util\Bayesian\Storage\TFileBayesianStorage;
use Belisoful\Prado\Util\Bayesian\Storage\TMemoryBayesianStorage;
use Belisoful\Prado\Util\Bayesian\TBayesianRecommender;
use Belisoful\Prado\Util\Bayesian\TBayesianTrainingSet;

/**
 * Integration tests that exercise end-to-end workflows across multiple classes.
 *
 * These verify the full lifecycle: train a classifier, persist it, reload it in a
 * fresh instance, and classify; plus the recommender and evaluation workflows.
 */
class TBayesianIntegrationTest extends PHPUnit\Framework\TestCase
{
	private string $_dir;

	protected function setUp(): void
	{
		$this->_dir = sys_get_temp_dir() . '/bayesian-int-' . uniqid('', true);
	}

	protected function tearDown(): void
	{
		if (is_dir($this->_dir)) {
			foreach (glob($this->_dir . '/*') ?: [] as $file) {
				@unlink($file);
			}
			@rmdir($this->_dir);
		}
	}

	/**
	 * Train → save to file → load in a fresh classifier → classify.
	 * Simulates a model trained in one process and used in another.
	 */
	public function testTrainSaveLoadClassifyRoundTrip()
	{
		$storage = new TFileBayesianStorage();
		$storage->setDirectory($this->_dir);

		$classifier = new TNaiveBayesClassifier();
		$classifier->setStorage($storage);
		$classifier->setName('spam-filter');
		$classifier->trainOne('spam', 'buy cheap watches now click here');
		$classifier->trainOne('spam', 'limited time offer low price winner');
		$classifier->trainOne('spam', 'congratulations you have won a prize');
		$classifier->trainOne('ham', 'are we meeting for lunch tomorrow');
		$classifier->trainOne('ham', 'I attached the report you asked for');
		$classifier->trainOne('ham', 'thanks for the help with the bug fix');
		$classifier->save();

		// Load in a fresh instance with a fresh storage (simulates a new request).
		$restoredStorage = new TFileBayesianStorage();
		$restoredStorage->setDirectory($this->_dir);
		$restored = new TNaiveBayesClassifier();
		$restored->setStorage($restoredStorage);
		$restored->load('spam-filter');

		self::assertTrue($restored->getIsTrained());
		self::assertSame('spam', $restored->classify('FREE VIAGRA cheapest price online'));
		self::assertSame('ham', $restored->classify('are we meeting for lunch tomorrow'));
		self::assertTrue($restored->isSpam('buy cheap click here'));
		self::assertFalse($restored->isSpam('meeting report tomorrow'));
	}

	/**
	 * Train → save → load → continue training → save → load → classify.
	 * Verifies incremental training after a load.
	 */
	public function testIncrementalTrainingAfterLoad()
	{
		$storage = new TFileBayesianStorage();
		$storage->setDirectory($this->_dir);

		$classifier = new TNaiveBayesClassifier();
		$classifier->setStorage($storage);
		$classifier->setName('inc-model');
		$classifier->trainOne('a', 'foo bar');
		$classifier->trainOne('b', 'baz qux');
		$classifier->save();

		$restored = new TNaiveBayesClassifier();
		$restored->setStorage($storage);
		$restored->load('inc-model');
		// Continue training after load.
		$restored->trainOne('a', 'foo baz');
		$restored->save();

		$final = new TNaiveBayesClassifier();
		$final->setStorage($storage);
		$final->load('inc-model');
		self::assertSame('a', $final->classify('foo'));
	}

	/**
	 * Full recommender workflow: train on positive/negative interactions,
	 * rank candidates, evaluate with a confusion matrix.
	 */
	public function testRecommenderAndEvaluationWorkflow()
	{
		$rec = new TBayesianRecommender();
		$rec->setPositiveCategory('liked');
		$rec->getClassifier()->trainOne('liked', 'red shoes blue sneakers leather boots');
		$rec->getClassifier()->trainOne('liked', 'running sneakers gym fitness');
		$rec->getClassifier()->trainOne('ignored', 'red hat blue scarf leather belt');
		$rec->getClassifier()->trainOne('ignored', 'formal tie suit office');

		$top = $rec->recommend(['red shoes'], ['red sneakers', 'blue sneakers', 'red hat', 'leather wallet']);
		self::assertNotEmpty($top);
		// 'blue sneakers' is verbatim "liked" training; 'red hat' is verbatim "ignored" training,
		// so the recommender must rank the liked-like candidate strictly above the ignored one.
		self::assertGreaterThan(
			$top['red hat'],
			$top['blue sneakers'],
			'A liked-trained candidate must outrank an ignored-trained candidate.'
		);
		// The clearly ignored-like candidate must not be ranked first.
		self::assertNotSame('red hat', array_key_first($top));
		// Scores are in descending order.
		$previous = INF;
		foreach ($top as $candidate => $score) {
			self::assertLessThanOrEqual($previous, $score);
			$previous = $score;
		}

		// Evaluate the classifier with a confusion matrix.
		$classifier = $rec->getClassifier();
		$matrix = new TConfusionMatrix(['liked', 'ignored']);
		$matrix->record('liked', $classifier->classify('red shoes'));
		$matrix->record('liked', $classifier->classify('blue sneakers'));
		$matrix->record('ignored', $classifier->classify('red hat'));
		$matrix->record('ignored', $classifier->classify('leather belt'));
		$metrics = new TBayesianMetrics($matrix);
		self::assertGreaterThan(0.0, $metrics->getAccuracy());
		self::assertGreaterThan(0.0, $metrics->getF1('liked'));
	}

	/**
	 * Train via TBayesianTrainingSet → score → verify probabilities sum to 1.
	 */
	public function testTrainingSetToScoreWorkflow()
	{
		$set = new TBayesianTrainingSet();
		$set->add('spam', 'buy cheap watches');
		$set->add('spam', 'limited offer click');
		$set->add('ham', 'lunch meeting tomorrow');
		$set->add('ham', 'report attached thanks');

		$classifier = new TNaiveBayesClassifier();
		$classifier->train($set);
		$scores = $classifier->score('cheap offer click');
		self::assertEqualsWithDelta(1.0, array_sum($scores), 1e-9);
		self::assertSame('spam', $classifier->classify('cheap offer click'));
	}

	/**
	 * Memory storage round-trip with all classifier variants.
	 */
	public function testAllClassifierVariantsRoundTripViaMemoryStorage()
	{
		$variants = [
			\Belisoful\Prado\Util\Bayesian\Classifier\TNaiveBayesClassifier::class,
			\Belisoful\Prado\Util\Bayesian\Classifier\TMultinomialNaiveBayes::class,
			\Belisoful\Prado\Util\Bayesian\Classifier\TBernoulliNaiveBayes::class,
			\Belisoful\Prado\Util\Bayesian\Classifier\TComplementNaiveBayes::class,
		];
		foreach ($variants as $class) {
			$storage = new TMemoryBayesianStorage();
			$classifier = new $class();
			$classifier->setStorage($storage);
			$classifier->setName('variant');
			$classifier->trainOne('spam', 'cheap offer click free');
			$classifier->trainOne('ham', 'meeting report tomorrow');
			$classifier->save();

			$restored = new $class();
			$restored->setStorage($storage);
			$restored->load('variant');
			self::assertSame('spam', $restored->classify('cheap click'), $class . ' round-trip');
			self::assertTrue($restored->getIsTrained(), $class . ' is trained after load');
		}
	}
}
