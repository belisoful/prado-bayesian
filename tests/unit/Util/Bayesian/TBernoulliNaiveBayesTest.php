<?php

use Prado\Util\Bayesian\Classifier\TBernoulliNaiveBayes;
use Prado\Util\Bayesian\Classifier\TNaiveBayesClassifier;

class TBernoulliNaiveBayesTest extends PHPUnit\Framework\TestCase
{
	public function testInheritsFromNaiveBayes()
	{
		$classifier = new TBernoulliNaiveBayes();
		self::assertInstanceOf(TNaiveBayesClassifier::class, $classifier);
	}

	public function testClassifiesShortDocuments()
	{
		$classifier = new TBernoulliNaiveBayes();
		$classifier->trainOne('spam', 'free viagra');
		$classifier->trainOne('spam', 'cheap offer');
		$classifier->trainOne('spam', 'limited time');
		$classifier->trainOne('ham', 'meeting tomorrow');
		$classifier->trainOne('ham', 'lunch soon');
		$classifier->trainOne('ham', 'report attached');
		self::assertSame('spam', $classifier->classify('free cheap'));
		self::assertSame('ham', $classifier->classify('meeting soon'));
	}

	public function testExportsBernoulliKind()
	{
		$classifier = new TBernoulliNaiveBayes();
		$classifier->setStorage(new \Prado\Util\Bayesian\Storage\TMemoryBayesianStorage());
		$classifier->setName('bern');
		$classifier->trainOne('a', 'foo bar');
		$classifier->trainOne('b', 'baz qux');
		$classifier->save();
		$payload = $classifier->getStorage()->load('bern');
		self::assertSame('bernoulli-naive-bayes', $payload['kind']);
	}

	public function testCategoryWithZeroDocumentsScoresZeroAndIsNeverChosen()
	{
		// A hand-built payload with a category that has no documents: its log-posterior is -INF,
		// which normalizes to probability 0.0, so classify() never picks it.
		$storage = new \Prado\Util\Bayesian\Storage\TMemoryBayesianStorage();
		$storage->save('bern', [
			'kind' => 'bernoulli-naive-bayes',
			'categories' => [
				[
					'name' => 'spam',
					'documentCount' => 1,
					'tokenCounts' => ['free' => 1],
					'tokenDocumentCounts' => ['free' => 1],
					'totalTokens' => 1,
				],
				[
					'name' => 'ghost',
					'documentCount' => 0,
					'tokenCounts' => [],
					'tokenDocumentCounts' => [],
					'totalTokens' => 0,
				],
			],
			'documentFrequency' => ['free' => 1],
			'totalDocuments' => 1,
		]);
		$classifier = new TBernoulliNaiveBayes();
		$classifier->setStorage($storage);
		$classifier->load('bern');
		$scores = $classifier->score('free');
		self::assertArrayHasKey('ghost', $scores);
		self::assertSame(0.0, $scores['ghost']);
		self::assertEqualsWithDelta(1.0, $scores['spam'], 1e-12);
		self::assertSame('spam', $classifier->classify('free'));
		self::assertSame('spam', $classifier->classify('unseen'));
	}

	public function testOutOfVocabularyTokenDoesNotChangeScore()
	{
		// Bernoulli scores over the fixed global vocabulary; a token absent from it is not a
		// feature at all, so a document of only unseen tokens scores like an empty document.
		$classifier = new TBernoulliNaiveBayes();
		for ($i = 0; $i < 20; $i++) {
			$classifier->trainOne('spam', ['cheap', 'offer', 'token' . $i]);
		}
		$classifier->trainOne('ham', ['meeting', 'report']);
		$classifier->trainOne('ham', ['lunch', 'tomorrow']);
		$empty = $classifier->score([]);
		$oov = $classifier->score(['zzz', 'never', 'seen']);
		self::assertEqualsWithDelta($empty['spam'], $oov['spam'], 1e-12);
		self::assertEqualsWithDelta($empty['ham'], $oov['ham'], 1e-12);
		// A known token does move the score.
		$known = $classifier->score(['meeting']);
		self::assertGreaterThan($empty['ham'], $known['ham']);
	}

	public function testSetAlphaRejectsNonPositive()
	{
		$classifier = new TBernoulliNaiveBayes();
		foreach ([0.0, -1.0, INF] as $value) {
			try {
				$classifier->setAlpha($value);
				self::fail('expected exception for ' . $value);
			} catch (\Prado\Exceptions\TInvalidDataValueException $e) {
				self::assertSame('bayesian_alpha_invalid', $e->getErrorCode());
			}
		}
	}

	public function testUntrainedClassifyThrows()
	{
		$classifier = new TBernoulliNaiveBayes();
		try {
			$classifier->classify('x');
			self::fail('expected exception');
		} catch (\Prado\Exceptions\TInvalidOperationException $e) {
			self::assertSame('bayesian_classifier_not_trained', $e->getErrorCode());
		}
	}

	public function testDegenerateTrainingFallsBackToPrior()
	{
		// Unlike the multinomial variants, Bernoulli's smoothed denominator (docCount + 2*alpha)
		// stays positive with an empty vocabulary: no features means no likelihood term, so the
		// document is scored by the prior alone and classify() still succeeds.
		$classifier = new TBernoulliNaiveBayes();
		$classifier->trainOne('a', 'a b c');
		$classifier->trainOne('a', 'd e');
		$classifier->trainOne('b', 'f g');
		self::assertSame([], $classifier->getVocabulary()->getDocumentFrequency());
		self::assertSame('a', $classifier->classify('a b c'));
		$scores = $classifier->score('a b c');
		self::assertEqualsWithDelta(2.0 / 3.0, $scores['a'], 1e-12);
		self::assertEqualsWithDelta(1.0 / 3.0, $scores['b'], 1e-12);
	}

	public function testLoadRejectsNaiveBayesPayload()
	{
		$storage = new \Prado\Util\Bayesian\Storage\TMemoryBayesianStorage();
		$naive = new TNaiveBayesClassifier();
		$naive->setStorage($storage);
		$naive->setName('nb');
		$naive->trainOne('a', 'foo bar');
		$naive->trainOne('b', 'baz qux');
		$naive->save();
		$classifier = new TBernoulliNaiveBayes();
		$classifier->setStorage($storage);
		try {
			$classifier->load('nb');
			self::fail('expected exception');
		} catch (\Prado\Exceptions\TInvalidDataValueException $e) {
			self::assertSame('bayesian_classifier_kind_mismatch', $e->getErrorCode());
		}
	}

	public function testSaveLoadRoundTripsTokenizerAndConfiguration()
	{
		$storage = new \Prado\Util\Bayesian\Storage\TMemoryBayesianStorage();
		$tokenizer = new \Prado\Util\Bayesian\Tokenizer\TNGramTokenizer();
		$tokenizer->setN(4);
		$classifier = new TBernoulliNaiveBayes();
		$classifier->setTokenizer($tokenizer);
		$classifier->setAlpha(0.5);
		$classifier->setSpamCategory('junk');
		$classifier->setStorage($storage);
		$classifier->setName('bern');
		$classifier->trainOne('junk', 'free offer');
		$classifier->trainOne('ham', 'meeting report');
		$classifier->save();

		$restored = new TBernoulliNaiveBayes();
		$restored->setStorage($storage);
		$restored->load('bern');
		self::assertInstanceOf(\Prado\Util\Bayesian\Tokenizer\TNGramTokenizer::class, $restored->getTokenizer());
		self::assertSame(4, $restored->getTokenizer()->getN());
		self::assertEqualsWithDelta(0.5, $restored->getAlpha(), 1e-12);
		self::assertSame('junk', $restored->getSpamCategory());
		$expected = $classifier->score('free meeting');
		$actual = $restored->score('free meeting');
		foreach ($expected as $category => $probability) {
			self::assertEqualsWithDelta($probability, $actual[$category], 1e-12);
		}
	}

	public function testSaveAndLoadRoundTripPreservesBernoulliBehavior()
	{
		$storage = new \Prado\Util\Bayesian\Storage\TMemoryBayesianStorage();
		$classifier = new TBernoulliNaiveBayes();
		$classifier->setStorage($storage);
		$classifier->setName('berni');
		$classifier->trainOne('spam', 'free offer');
		$classifier->trainOne('ham', 'meeting report');
		$classifier->save();

		$restored = new TBernoulliNaiveBayes();
		$restored->setStorage($storage);
		$restored->load('berni');
		self::assertSame('spam', $restored->classify('free'));
		self::assertSame('ham', $restored->classify('meeting'));
	}

	public function testProbabilitiesMatchHandComputation()
	{
		// alpha = 1. Train spam={a,b}, ham={b,c}; one doc each, denominator = 1 + 2 = 3.
		// The GLOBAL vocabulary {a,b,c} is scored for EVERY category (the global-vocab fix), so
		// the absent-term penalties are comparable. For the document {a}:
		//   spam: a present 2/3, b absent (1-2/3), c absent (1-1/3) -> (2/3)(1/3)(2/3) = 4/27
		//   ham:  a present 1/3, b absent 1/3,     c absent 1/3     -> (1/3)^3        = 1/27
		//   equal priors -> posterior ∝ 4/27 : 1/27 = 4 : 1
		$classifier = new TBernoulliNaiveBayes();
		$classifier->trainOne('spam', ['a', 'b']);
		$classifier->trainOne('ham', ['b', 'c']);
		$scores = $classifier->score(['a']);
		self::assertEqualsWithDelta(0.8, $scores['spam'], 1e-9);
		self::assertEqualsWithDelta(0.2, $scores['ham'], 1e-9);
	}

	public function testUntrainedScoreIsEmpty()
	{
		$classifier = new TBernoulliNaiveBayes();
		self::assertSame([], $classifier->score('anything'));
	}

	public function testInconsistentTokenDocumentCountsAreSkipped()
	{
		// A hand-built payload whose per-token document count exceeds the category's document
		// count would make the smoothed presence probability >= 1 (log(1 - p) undefined), so the
		// feature is skipped instead of poisoning the score with NaN.
		$storage = new \Prado\Util\Bayesian\Storage\TMemoryBayesianStorage();
		$storage->save('bad', [
			'kind' => 'bernoulli-naive-bayes',
			'categories' => [
				[
					'name' => 'spam',
					'documentCount' => 1,
					'tokenCounts' => ['free' => 1],
					'tokenDocumentCounts' => ['free' => 100],
					'totalTokens' => 1,
				],
				[
					'name' => 'ham',
					'documentCount' => 1,
					'tokenCounts' => ['free' => 1],
					'tokenDocumentCounts' => ['free' => 1],
					'totalTokens' => 1,
				],
			],
			'documentFrequency' => ['free' => 2],
			'totalDocuments' => 2,
		]);
		$classifier = new TBernoulliNaiveBayes();
		$classifier->setStorage($storage);
		$classifier->load('bad');
		$scores = $classifier->score('free');
		self::assertCount(2, $scores);
		foreach ($scores as $score) {
			self::assertFalse(is_nan($score));
		}
		self::assertEqualsWithDelta(1.0, array_sum($scores), 1e-9);
	}
}
