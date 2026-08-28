<?php

use Belisoful\Prado\Util\Bayesian\Classifier\IBayesianClassifier;
use Belisoful\Prado\Util\Bayesian\Classifier\TNaiveBayesClassifier;
use Belisoful\Prado\Util\Bayesian\IBayesianRecommender;
use Belisoful\Prado\Util\Bayesian\TBayesianRecommender;
use Prado\Exceptions\TInvalidDataValueException;

class TBayesianRecommenderTest extends PHPUnit\Framework\TestCase
{
	public function testRecommendsByLikelihood()
	{
		$rec = new TBayesianRecommender();
		$rec->setPositiveCategory('liked');
		$rec->getClassifier()->trainOne('liked', 'red shoes blue sneakers leather boots');
		$rec->getClassifier()->trainOne('ignored', 'red hat blue scarf leather belt');

		$top = $rec->recommend(['red shoes'], ['red sneakers', 'blue sneakers', 'red hat', 'leather wallet']);
		self::assertNotEmpty($top);
		$best = array_key_first($top);
		// The best candidate is whichever looks most like "liked" (red shoes, blue sneakers)
		// and least like "ignored" (red hat).
		self::assertContains($best, ['red sneakers', 'blue sneakers']);
		// The scores are sorted descending.
		$previous = INF;
		foreach ($top as $candidate => $score) {
			self::assertLessThanOrEqual($previous, $score);
			$previous = $score;
		}
	}

	public function testEmptyCandidatesThrows()
	{
		$rec = new TBayesianRecommender();
		$this->expectException(TInvalidDataValueException::class);
		$rec->recommend(['item'], []);
	}

	public function testReturnsEmptyForUnknownPositiveCategory()
	{
		$rec = new TBayesianRecommender();
		$rec->setPositiveCategory('nonexistent');
		$rec->getClassifier()->trainOne('liked', 'shoes');
		$rec->getClassifier()->trainOne('ignored', 'hat');
		// A positive category the classifier was never trained on would score every candidate
		// 0.0; that is a misconfiguration and is reported rather than ranked.
		try {
			$rec->recommend(['shoes'], ['shoes', 'hat']);
			self::fail('expected exception');
		} catch (\Prado\Exceptions\TInvalidOperationException $e) {
			self::assertSame('bayesian_recommender_category_unknown', $e->getErrorCode());
			self::assertStringContainsString('nonexistent', $e->getErrorMessage());
		}
	}

	public function testSetClassifierReplacesDefault()
	{
		$rec = new TBayesianRecommender();
		$custom = new TNaiveBayesClassifier();
		$custom->trainOne('liked', 'foo bar');
		$custom->trainOne('ignored', 'baz qux');
		$rec->setClassifier($custom);
		self::assertSame($custom, $rec->getClassifier());
	}

	public function testImplementsRecommenderInterface()
	{
		$rec = new TBayesianRecommender();
		self::assertInstanceOf(IBayesianRecommender::class, $rec);
	}

	public function testDefaultClassifierIsNaiveBayes()
	{
		$rec = new TBayesianRecommender();
		self::assertInstanceOf(IBayesianClassifier::class, $rec->getClassifier());
		self::assertInstanceOf(TNaiveBayesClassifier::class, $rec->getClassifier());
	}

	public function testSetPositiveCategory()
	{
		$rec = new TBayesianRecommender();
		$rec->setPositiveCategory('clicked');
		self::assertSame('clicked', $rec->getPositiveCategory());
	}

	public function testNumericStringCandidatesAreScored()
	{
		$rec = new TBayesianRecommender();
		$rec->getClassifier()->trainOne('liked', ['2024', 'shoes', 'red']);
		$rec->getClassifier()->trainOne('ignored', ['1999', 'hat', 'blue']);
		$scores = $rec->recommend(['red'], ['2024', '1999', '42']);
		self::assertCount(3, $scores);
		// PHP turns numeric-string keys into ints; compare as strings.
		self::assertEqualsCanonicalizing(['2024', '1999', '42'], array_map('strval', array_keys($scores)));
		self::assertGreaterThan($scores[1999], $scores[2024]);
		foreach ($scores as $score) {
			self::assertIsFloat($score);
		}
	}

	public function testBlankAndWhitespaceCandidatesAreIgnored()
	{
		$rec = new TBayesianRecommender();
		$rec->getClassifier()->trainOne('liked', 'red shoes');
		$rec->getClassifier()->trainOne('ignored', 'blue hat');
		$scores = $rec->recommend(['red'], ['', '   ', "\t", 'shoes', ' hat ']);
		self::assertEqualsCanonicalizing(['shoes', 'hat'], array_keys($scores));
		self::assertArrayNotHasKey('', $scores);
	}

	public function testAllBlankCandidatesThrows()
	{
		$rec = new TBayesianRecommender();
		$rec->getClassifier()->trainOne('liked', 'red shoes');
		$rec->getClassifier()->trainOne('ignored', 'blue hat');
		try {
			$rec->recommend(['red'], ['', '  ', "\n"]);
			self::fail('expected exception');
		} catch (TInvalidDataValueException $e) {
			self::assertSame('bayesian_recommendation_candidates_empty', $e->getErrorCode());
		}
	}

	public function testDuplicateCandidatesAreScoredOnce()
	{
		$rec = new TBayesianRecommender();
		$rec->getClassifier()->trainOne('liked', 'red shoes');
		$rec->getClassifier()->trainOne('ignored', 'blue hat');
		$scores = $rec->recommend(['red'], ['shoes', 'shoes', ' shoes ', 'hat']);
		self::assertCount(2, $scores);
	}

	public function testUntrainedClassifierThrowsNotTrained()
	{
		$rec = new TBayesianRecommender();
		try {
			$rec->recommend(['red'], ['shoes', 'hat']);
			self::fail('expected exception');
		} catch (\Prado\Exceptions\TInvalidOperationException $e) {
			self::assertSame('bayesian_classifier_not_trained', $e->getErrorCode());
		}
	}

	public function testEmptyCandidatesThrowsBeforeTrainingCheck()
	{
		$rec = new TBayesianRecommender();
		self::assertFalse($rec->getClassifier()->getIsTrained());
		try {
			$rec->recommend(['red'], []);
			self::fail('expected exception');
		} catch (TInvalidDataValueException $e) {
			self::assertSame('bayesian_recommendation_candidates_empty', $e->getErrorCode());
		}
	}

	public function testEmptyContextScoresCandidatesAlone()
	{
		$rec = new TBayesianRecommender();
		$rec->getClassifier()->trainOne('liked', 'red shoes');
		$rec->getClassifier()->trainOne('ignored', 'blue hat');
		$scores = $rec->recommend([], ['shoes', 'hat']);
		self::assertCount(2, $scores);
		self::assertGreaterThan($scores['hat'], $scores['shoes']);
	}
}
