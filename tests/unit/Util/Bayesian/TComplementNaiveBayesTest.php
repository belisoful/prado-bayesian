<?php

use Prado\Util\Bayesian\Classifier\TComplementNaiveBayes;
use Prado\Util\Bayesian\Classifier\TNaiveBayesClassifier;

class TComplementNaiveBayesTest extends PHPUnit\Framework\TestCase
{
	public function testInheritsFromNaiveBayes()
	{
		$classifier = new TComplementNaiveBayes();
		self::assertInstanceOf(TNaiveBayesClassifier::class, $classifier);
	}

	public function testClassifiesDocuments()
	{
		$classifier = new TComplementNaiveBayes();
		$classifier->trainOne('spam', 'cheap offer buy click');
		$classifier->trainOne('spam', 'free prize winner');
		$classifier->trainOne('ham', 'meeting lunch tomorrow');
		$classifier->trainOne('ham', 'report attached thanks');
		self::assertSame('spam', $classifier->classify('cheap click buy'));
		self::assertSame('ham', $classifier->classify('meeting lunch'));
	}

	public function testExportsComplementKind()
	{
		$classifier = new TComplementNaiveBayes();
		$classifier->setStorage(new \Prado\Util\Bayesian\Storage\TMemoryBayesianStorage());
		$classifier->setName('cnb');
		$classifier->trainOne('a', 'foo bar');
		$classifier->trainOne('b', 'baz qux');
		$classifier->save();
		$payload = $classifier->getStorage()->load('cnb');
		self::assertSame('complement-naive-bayes', $payload['kind']);
	}

	public function testSaveAndLoadRoundTrip()
	{
		$storage = new \Prado\Util\Bayesian\Storage\TMemoryBayesianStorage();
		$classifier = new TComplementNaiveBayes();
		$classifier->setStorage($storage);
		$classifier->setName('cnb');
		$classifier->trainOne('spam', 'cheap offer click free');
		$classifier->trainOne('ham', 'meeting report tomorrow');
		$classifier->save();

		$restored = new TComplementNaiveBayes();
		$restored->setStorage($storage);
		$restored->load('cnb');
		self::assertSame('spam', $restored->classify('cheap click free'));
		self::assertSame('ham', $restored->classify('meeting report'));
	}

	public function testScoresAreRealNumbers()
	{
		$classifier = new TComplementNaiveBayes();
		$classifier->trainOne('spam', 'foo bar');
		$classifier->trainOne('ham', 'baz qux');
		$scores = $classifier->score('foo');
		self::assertArrayHasKey('spam', $scores);
		self::assertArrayHasKey('ham', $scores);
		self::assertIsFloat($scores['spam']);
		self::assertIsFloat($scores['ham']);
	}

	public function testComplementScoreIsSymmetric()
	{
		// With a perfectly symmetric corpus (spam={a}, ham={b}), scoring {a} for spam must equal
		// scoring {b} for ham. Correct WCNB normalization (over the full vocabulary) preserves
		// this symmetry; the earlier per-document normalization collapsed both scores to a tie.
		$classifier = new TComplementNaiveBayes();
		$classifier->setUseTfidf(false);
		$classifier->trainOne('spam', ['a']);
		$classifier->trainOne('ham', ['b']);
		$scoresA = $classifier->score(['a']);
		$scoresB = $classifier->score(['b']);
		self::assertEqualsWithDelta($scoresA['spam'], $scoresB['ham'], 1e-12);
		self::assertEqualsWithDelta($scoresA['ham'], $scoresB['spam'], 1e-12);
		self::assertGreaterThan($scoresA['ham'], $scoresA['spam']);
	}

	public function testAllOutOfVocabularyDocumentScoresEqually()
	{
		// CNB has no prior term and OOV tokens are skipped, so a document of only unseen tokens
		// contributes nothing: every category gets the same (zero) log-score, hence equal
		// probabilities even with heavily imbalanced training.
		$classifier = new TComplementNaiveBayes();
		for ($i = 0; $i < 20; $i++) {
			$classifier->trainOne('spam', ['cheap', 'offer', 'token' . $i]);
		}
		$classifier->trainOne('ham', ['meeting', 'report']);
		$classifier->trainOne('ham', ['lunch', 'tomorrow']);
		$scores = $classifier->score(['zzz', 'never', 'seen']);
		self::assertEqualsWithDelta(0.5, $scores['spam'], 1e-12);
		self::assertEqualsWithDelta(0.5, $scores['ham'], 1e-12);
	}

	public function testSetAlphaRejectsNonPositive()
	{
		$classifier = new TComplementNaiveBayes();
		foreach ([0.0, -1.0, INF] as $value) {
			try {
				$classifier->setAlpha($value);
				self::fail('expected exception for ' . $value);
			} catch (\Prado\Exceptions\TInvalidDataValueException $e) {
				self::assertSame('bayesian_alpha_invalid', $e->getErrorCode());
			}
		}
		self::assertEqualsWithDelta(1.0, $classifier->getAlpha(), 1e-12);
	}

	public function testUntrainedClassifyThrows()
	{
		$classifier = new TComplementNaiveBayes();
		try {
			$classifier->classify('x');
			self::fail('expected exception');
		} catch (\Prado\Exceptions\TInvalidOperationException $e) {
			self::assertSame('bayesian_classifier_not_trained', $e->getErrorCode());
		}
	}

	public function testDegenerateTrainingThrowsScoreUndefined()
	{
		$classifier = new TComplementNaiveBayes();
		$classifier->trainOne('a', 'a b c');
		try {
			$classifier->classify('a b c');
			self::fail('expected exception');
		} catch (\Prado\Exceptions\TInvalidOperationException $e) {
			self::assertSame('bayesian_classifier_score_undefined', $e->getErrorCode());
		}
	}

	public function testAdditionalTrainingInvalidatesCachedScores()
	{
		$classifier = new TComplementNaiveBayes();
		$classifier->setUseTfidf(false);
		$classifier->trainOne('spam', ['cheap', 'offer']);
		$classifier->trainOne('ham', ['meeting', 'report']);
		$before = $classifier->score(['cheap']);
		self::assertSame('spam', $classifier->classify(['cheap']));
		// Now flood 'cheap' into ham: the cached global counts / norms must be dropped so the
		// new statistics are reflected.
		for ($i = 0; $i < 10; $i++) {
			$classifier->trainOne('ham', ['cheap', 'cheap', 'meeting' . $i]);
		}
		$after = $classifier->score(['cheap']);
		self::assertNotEqualsWithDelta($before['spam'], $after['spam'], 1e-9);
		self::assertSame('ham', $classifier->classify(['cheap']));
	}

	public function testSetAlphaInvalidatesCachedScores()
	{
		$classifier = new TComplementNaiveBayes();
		$classifier->setUseTfidf(false);
		$classifier->trainOne('spam', ['cheap', 'offer', 'click']);
		$classifier->trainOne('ham', ['meeting', 'report']);
		$before = $classifier->score(['cheap', 'meeting', 'meeting']);
		$classifier->setAlpha(0.01);
		$after = $classifier->score(['cheap', 'meeting', 'meeting']);
		self::assertNotEqualsWithDelta($before['spam'], $after['spam'], 1e-9);
	}

	public function testLoadInvalidatesCachedScores()
	{
		$storage = new \Prado\Util\Bayesian\Storage\TMemoryBayesianStorage();
		$source = new TComplementNaiveBayes();
		$source->setStorage($storage);
		$source->setName('other');
		$source->trainOne('spam', ['meeting', 'meeting', 'report']);
		$source->trainOne('ham', ['cheap', 'offer']);
		$source->save();

		$classifier = new TComplementNaiveBayes();
		$classifier->setStorage($storage);
		$classifier->trainOne('spam', ['cheap', 'offer']);
		$classifier->trainOne('ham', ['meeting', 'report']);
		self::assertSame('spam', $classifier->classify(['cheap']));
		// Loading a model with the opposite labeling must replace the cached statistics.
		$classifier->load('other');
		self::assertSame('ham', $classifier->classify(['cheap']));
		self::assertSame('spam', $classifier->classify(['meeting']));
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
		$classifier = new TComplementNaiveBayes();
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
		$tokenizer = new \Prado\Util\Bayesian\Tokenizer\TRegexTokenizer();
		$tokenizer->setPattern('/([a-z]{3,})/');
		$classifier = new TComplementNaiveBayes();
		$classifier->setTokenizer($tokenizer);
		$classifier->setAlpha(0.5);
		$classifier->setUseTfidf(false);
		$classifier->setSpamCategory('junk');
		$classifier->setStorage($storage);
		$classifier->setName('cnb');
		$classifier->trainOne('junk', 'cheap offer click free');
		$classifier->trainOne('ham', 'meeting report tomorrow');
		$classifier->save();

		$restored = new TComplementNaiveBayes();
		$restored->setStorage($storage);
		$restored->load('cnb');
		self::assertInstanceOf(\Prado\Util\Bayesian\Tokenizer\TRegexTokenizer::class, $restored->getTokenizer());
		self::assertSame('/([a-z]{3,})/', $restored->getTokenizer()->getPattern());
		self::assertEqualsWithDelta(0.5, $restored->getAlpha(), 1e-12);
		self::assertFalse($restored->getUseTfidf());
		self::assertSame('junk', $restored->getSpamCategory());
		$expected = $classifier->score('cheap meeting');
		$actual = $restored->score('cheap meeting');
		foreach ($expected as $category => $probability) {
			self::assertEqualsWithDelta($probability, $actual[$category], 1e-12);
		}
	}

	public function testUntrainedScoreIsEmpty()
	{
		$classifier = new TComplementNaiveBayes();
		self::assertSame([], $classifier->score('anything'));
	}

	public function testCategoryWithoutDocumentsScoresZero()
	{
		$storage = new \Prado\Util\Bayesian\Storage\TMemoryBayesianStorage();
		$storage->save('cnb', [
			'kind' => 'complement-naive-bayes',
			'categories' => [
				[
					'name' => 'spam',
					'documentCount' => 1,
					'tokenCounts' => ['free' => 2, 'offer' => 1],
					'tokenDocumentCounts' => ['free' => 1, 'offer' => 1],
					'totalTokens' => 3,
				],
				[
					'name' => 'ham',
					'documentCount' => 1,
					'tokenCounts' => ['meeting' => 2],
					'tokenDocumentCounts' => ['meeting' => 1],
					'totalTokens' => 2,
				],
				[
					'name' => 'ghost',
					'documentCount' => 0,
					'tokenCounts' => [],
					'tokenDocumentCounts' => [],
					'totalTokens' => 0,
				],
			],
			'documentFrequency' => ['free' => 1, 'offer' => 1, 'meeting' => 1],
			'totalDocuments' => 2,
		]);
		$classifier = new TComplementNaiveBayes();
		$classifier->setStorage($storage);
		$classifier->load('cnb');
		$scores = $classifier->score('free');
		// The complement of a category with no documents is the whole corpus, so it carries no
		// evidence: its score is -INF, which normalizes to 0.0 and never wins.
		self::assertSame(0.0, $scores['ghost']);
		self::assertSame('spam', $classifier->classify('free'));
	}

	public function testEmptyVocabularyMakesEveryScoreUndefined()
	{
		// A payload with token counts but no document frequencies leaves the smoothed
		// complement mass at zero, so no category has a defined score.
		$storage = new \Prado\Util\Bayesian\Storage\TMemoryBayesianStorage();
		$storage->save('degenerate', [
			'kind' => 'complement-naive-bayes',
			'categories' => [
				[
					'name' => 'spam',
					'documentCount' => 1,
					'tokenCounts' => ['free' => 1],
					'tokenDocumentCounts' => ['free' => 1],
					'totalTokens' => 1,
				],
			],
			'documentFrequency' => [],
			'totalDocuments' => 1,
		]);
		$classifier = new TComplementNaiveBayes();
		$classifier->setStorage($storage);
		$classifier->load('degenerate');
		self::assertSame([], $classifier->score('free'));
		try {
			$classifier->classify('free');
			self::fail('expected exception');
		} catch (\Prado\Exceptions\TInvalidOperationException $e) {
			self::assertSame('bayesian_classifier_score_undefined', $e->getErrorCode());
		}
	}

	public function testSingleCategoryHasNoDefinedComplementScore()
	{
		// Complement NB scores a category against everything else; with only one category the
		// complement is empty, its weight vector normalizes to zero, and no score is defined.
		$classifier = new TComplementNaiveBayes();
		$classifier->trainOne('spam', 'free');
		self::assertTrue($classifier->getIsTrained());
		self::assertSame([], $classifier->score('free'));
		try {
			$classifier->classify('free');
			self::fail('expected exception');
		} catch (\Prado\Exceptions\TInvalidOperationException $e) {
			self::assertSame('bayesian_classifier_score_undefined', $e->getErrorCode());
		}
	}
}
