<?php

use Prado\Exceptions\TConfigurationException;
use Prado\Exceptions\TInvalidDataValueException;
use Prado\Exceptions\TInvalidOperationException;
use Prado\Util\Bayesian\Classifier\TNaiveBayesClassifier;
use Prado\Util\Bayesian\Storage\TMemoryBayesianStorage;
use Prado\Util\Bayesian\TBayesianTrainingSet;
use Prado\Util\Bayesian\Tokenizer\TWordTokenizer;

class TNaiveBayesClassifierTest extends PHPUnit\Framework\TestCase
{
	public function testInitialState()
	{
		$classifier = new TNaiveBayesClassifier();
		self::assertNull($classifier->getName());
		self::assertFalse($classifier->getIsTrained());
		self::assertNull($classifier->getStorage());
		self::assertInstanceOf(TWordTokenizer::class, $classifier->getTokenizer());
	}

	public function testTrainOneAddsDocument()
	{
		$classifier = new TNaiveBayesClassifier();
		$classifier->trainOne('spam', 'buy cheap watches now');
		self::assertTrue($classifier->getIsTrained());
		$spam = $classifier->getVocabulary()->getCategory('spam');
		self::assertSame(1, $spam->getDocumentCount());
		self::assertSame(4, $spam->getTotalTokens());
	}

	public function testTrainAcceptsTrainingSet()
	{
		$classifier = new TNaiveBayesClassifier();
		$set = new TBayesianTrainingSet();
		$set->add('spam', 'buy cheap watches now');
		$set->add('ham', 'lunch tomorrow?');
		$classifier->train($set);
		self::assertSame(2, count($classifier->getVocabulary()->getCategories()));
	}

	public function testTrainOneRejectsEmptyCategory()
	{
		$classifier = new TNaiveBayesClassifier();
		$this->expectException(TInvalidDataValueException::class);
		$classifier->trainOne('', 'text');
	}

	public function testTrainRejectsEmptyTrainingSet()
	{
		$classifier = new TNaiveBayesClassifier();
		$set = new TBayesianTrainingSet();
		$this->expectException(TInvalidDataValueException::class);
		$classifier->train($set);
	}

	public function testClassifySpam()
	{
		$classifier = new TNaiveBayesClassifier();
		$classifier->trainOne('spam', 'buy cheap watches now click here');
		$classifier->trainOne('spam', 'limited time offer low price');
		$classifier->trainOne('spam', 'congratulations you have won a prize');
		$classifier->trainOne('ham', 'are we meeting for lunch tomorrow');
		$classifier->trainOne('ham', 'I attached the report you asked for');
		$classifier->trainOne('ham', 'thanks for the help with the bug fix');
		self::assertSame('spam', $classifier->classify('FREE VIAGRA cheapest price online'));
		self::assertSame('ham', $classifier->classify('Hey, are we still meeting for lunch tomorrow?'));
	}

	public function testClassifyWithoutTrainingThrows()
	{
		$classifier = new TNaiveBayesClassifier();
		$this->expectException(TInvalidOperationException::class);
		$classifier->classify('text');
	}

	public function testScoreReturnsNormalizedProbabilities()
	{
		$classifier = new TNaiveBayesClassifier();
		$classifier->trainOne('spam', 'cheap deal offer');
		$classifier->trainOne('ham', 'meeting lunch tomorrow');
		$scores = $classifier->score('cheap deal');
		self::assertArrayHasKey('spam', $scores);
		self::assertArrayHasKey('ham', $scores);
		// score() returns normalized probabilities that sum to 1.0.
		$sum = $scores['spam'] + $scores['ham'];
		self::assertEqualsWithDelta(1.0, $sum, 1e-9);
	}

	public function testScorePreservesHighestProbability()
	{
		$classifier = new TNaiveBayesClassifier();
		$classifier->trainOne('spam', 'cheap deal offer');
		$classifier->trainOne('ham', 'meeting lunch tomorrow');
		$scores = $classifier->score('cheap deal');
		$winner = array_search(max($scores), $scores, true);
		self::assertSame($winner, $classifier->classify('cheap deal'));
		// The highest probability belongs to the winning category.
		self::assertEqualsWithDelta(max($scores), $scores[$winner], 1e-9);
		$sum = array_sum($scores);
		self::assertEqualsWithDelta(1.0, $sum, 1e-9);
	}

	public function testScoreBeforeTrainingReturnsEmpty()
	{
		$classifier = new TNaiveBayesClassifier();
		self::assertSame([], $classifier->score('text'));
	}

	public function testIsSpam()
	{
		$classifier = new TNaiveBayesClassifier();
		$classifier->trainOne('spam', 'free viagra cheap click');
		$classifier->trainOne('ham', 'meeting lunch tomorrow report');
		self::assertTrue($classifier->isSpam('cheap click here'));
		self::assertFalse($classifier->isSpam('lunch meeting tomorrow'));
	}

	public function testCustomSpamCategory()
	{
		$classifier = new TNaiveBayesClassifier();
		$classifier->setSpamCategory('junk');
		$classifier->trainOne('junk', 'cheap offer buy now');
		$classifier->trainOne('good', 'meeting report tomorrow');
		self::assertTrue($classifier->isSpam('cheap offer now'));
	}

	public function testClassifyAcceptsPretokenizedInput()
	{
		$classifier = new TNaiveBayesClassifier();
		$classifier->trainOne('spam', 'cheap offer buy now');
		$classifier->trainOne('ham', 'meeting report tomorrow');
		self::assertSame('spam', $classifier->classify(['cheap', 'buy']));
	}

	public function testAlphaSmoothingAffectsScore()
	{
		$classifierLow = new TNaiveBayesClassifier();
		$classifierLow->setAlpha(0.001);
		$classifierHigh = new TNaiveBayesClassifier();
		$classifierHigh->setAlpha(10.0);
		// Use imbalanced training data so the priors differ; this makes the per-token
		// smoothing effect observable in the relative log-scores.
		$classifierLow->trainOne('spam', 'free offer click buy cheap price');
		$classifierLow->trainOne('spam', 'limited time winner prize');
		$classifierLow->trainOne('ham', 'meeting tomorrow');
		$classifierHigh->trainOne('spam', 'free offer click buy cheap price');
		$classifierHigh->trainOne('spam', 'limited time winner prize');
		$classifierHigh->trainOne('ham', 'meeting tomorrow');
		// 'meeting' is a ham-only token: with low alpha the ham/spam likelihood ratio is
		// large; with high alpha smoothing flattens the per-token probabilities and the
		// gap between the categories shrinks.  (Out-of-vocabulary tokens are skipped, so
		// they cannot be used to observe alpha.)
		$lowScores = $classifierLow->score('meeting');
		$highScores = $classifierHigh->score('meeting');
		$lowDiff = abs($lowScores['spam'] - $lowScores['ham']);
		$highDiff = abs($highScores['spam'] - $highScores['ham']);
		self::assertLessThan($lowDiff, $highDiff);
	}

	public function testTfidfToggle()
	{
		$withTfidf = new TNaiveBayesClassifier();
		$withTfidf->setUseTfidf(true);
		$withoutTfidf = new TNaiveBayesClassifier();
		$withoutTfidf->setUseTfidf(false);
		$doc1 = 'spam spam spam offer cheap';
		$doc2 = 'ham ham ham meeting lunch';
		$withTfidf->trainOne('spam', $doc1);
		$withTfidf->trainOne('ham', $doc2);
		$withoutTfidf->trainOne('spam', $doc1);
		$withoutTfidf->trainOne('ham', $doc2);
		$withScores = $withTfidf->score('spam offer');
		$withoutScores = $withoutTfidf->score('spam offer');
		self::assertNotEquals($withScores, $withoutScores);
	}

	public function testSaveRequiresName()
	{
		$classifier = new TNaiveBayesClassifier();
		$classifier->setStorage(new TMemoryBayesianStorage());
		$this->expectException(TConfigurationException::class);
		$classifier->save();
	}

	public function testSaveRequiresStorage()
	{
		$classifier = new TNaiveBayesClassifier();
		$classifier->setName('model');
		$this->expectException(TConfigurationException::class);
		$classifier->save();
	}

	public function testSaveAndLoadRoundTrip()
	{
		$storage = new TMemoryBayesianStorage();
		$classifier = new TNaiveBayesClassifier();
		$classifier->setStorage($storage);
		$classifier->setName('spam-filter');
		$classifier->trainOne('spam', 'free offer click cheap');
		$classifier->trainOne('ham', 'meeting report tomorrow lunch');
		$classifier->save();

		$restored = new TNaiveBayesClassifier();
		$restored->setStorage($storage);
		$restored->load('spam-filter');
		self::assertSame('spam', $restored->classify('cheap click free'));
		self::assertSame('ham', $restored->classify('meeting report lunch'));
		self::assertTrue($restored->getIsTrained());
	}

	public function testLoadMissingModelThrows()
	{
		$classifier = new TNaiveBayesClassifier();
		$classifier->setStorage(new TMemoryBayesianStorage());
		$this->expectException(TConfigurationException::class);
		$classifier->load('nope');
	}

	public function testLoadWithoutStorageThrows()
	{
		$classifier = new TNaiveBayesClassifier();
		$this->expectException(TConfigurationException::class);
		$classifier->load('any');
	}

	public function testSetNameRejectsEmptyString()
	{
		$classifier = new TNaiveBayesClassifier();
		$classifier->setName('');
		self::assertNull($classifier->getName());
	}

	public function testGetVocabularyReturnsSameInstance()
	{
		$classifier = new TNaiveBayesClassifier();
		self::assertSame($classifier->getVocabulary(), $classifier->getVocabulary());
	}

	public function testMultinomialNaiveBayesExportsCorrectKind()
	{
		$classifier = new \Prado\Util\Bayesian\Classifier\TMultinomialNaiveBayes();
		$classifier->setStorage(new TMemoryBayesianStorage());
		$classifier->setName('model');
		$classifier->trainOne('a', 'foo bar');
		$classifier->trainOne('b', 'baz qux');
		$classifier->save();
		$loaded = $classifier->getStorage()->load('model');
		self::assertSame('multinomial-naive-bayes', $loaded['kind']);
	}

	public function testSingleCategoryModelAlwaysClassifiesToThatCategory()
	{
		$classifier = new TNaiveBayesClassifier();
		$classifier->trainOne('only', 'alpha beta gamma');
		self::assertSame('only', $classifier->classify('something entirely different'));
		// With one category the normalized posterior is degenerate: all mass on that category.
		self::assertSame(['only' => 1.0], $classifier->score('alpha beta'));
	}

	public function testEmptyDocumentFallsBackToPriorDistribution()
	{
		// Pre-tokenized so the tokens survive (the word tokenizer would drop single chars).
		$classifier = new TNaiveBayesClassifier();
		$classifier->setUseTfidf(false);
		$classifier->trainOne('a', ['xx', 'yy']);
		$classifier->trainOne('a', ['xx', 'zz']);
		$classifier->trainOne('b', ['pp', 'qq']);
		// An empty document has no likelihood term, so the posterior is the prior: 2/3 : 1/3.
		$scores = $classifier->score([]);
		self::assertEqualsWithDelta(2.0 / 3.0, $scores['a'], 1e-9);
		self::assertEqualsWithDelta(1.0 / 3.0, $scores['b'], 1e-9);
	}

	public function testOutOfVocabularyTokensAreSkippedAndPriorDecides()
	{
		// Imbalanced training: 20 spam docs, 2 ham docs.  A document made only of unseen tokens
		// contributes no likelihood term, so the posterior equals the prior and spam wins.  (If
		// OOV tokens were penalized instead, the smaller category would win.)
		$classifier = new TNaiveBayesClassifier();
		for ($i = 0; $i < 20; $i++) {
			$classifier->trainOne('spam', ['cheap', 'offer', 'token' . $i]);
		}
		$classifier->trainOne('ham', ['meeting', 'report']);
		$classifier->trainOne('ham', ['lunch', 'tomorrow']);
		$scores = $classifier->score(['zzz', 'qqq', 'never', 'seen']);
		self::assertEqualsWithDelta(20.0 / 22.0, $scores['spam'], 1e-9);
		self::assertEqualsWithDelta(2.0 / 22.0, $scores['ham'], 1e-9);
		self::assertSame('spam', $classifier->classify(['zzz', 'qqq', 'never', 'seen']));
		// The unseen tokens are identical in effect to an empty document.
		self::assertEqualsWithDelta($classifier->score([])['spam'], $scores['spam'], 1e-12);
	}

	public function testSetAlphaRejectsZero()
	{
		$classifier = new TNaiveBayesClassifier();
		try {
			$classifier->setAlpha(0);
			self::fail('expected exception');
		} catch (TInvalidDataValueException $e) {
			self::assertSame('bayesian_alpha_invalid', $e->getErrorCode());
		}
		self::assertEqualsWithDelta(1.0, $classifier->getAlpha(), 1e-12, 'alpha is unchanged after a rejected value');
	}

	public function testSetAlphaRejectsNegative()
	{
		$classifier = new TNaiveBayesClassifier();
		try {
			$classifier->setAlpha(-1);
			self::fail('expected exception');
		} catch (TInvalidDataValueException $e) {
			self::assertSame('bayesian_alpha_invalid', $e->getErrorCode());
		}
	}

	public function testSetAlphaRejectsInfinity()
	{
		$classifier = new TNaiveBayesClassifier();
		try {
			$classifier->setAlpha(INF);
			self::fail('expected exception');
		} catch (TInvalidDataValueException $e) {
			self::assertSame('bayesian_alpha_invalid', $e->getErrorCode());
		}
	}

	public function testSetAlphaRejectsNan()
	{
		$classifier = new TNaiveBayesClassifier();
		try {
			$classifier->setAlpha(NAN);
			self::fail('expected exception');
		} catch (TInvalidDataValueException $e) {
			self::assertSame('bayesian_alpha_invalid', $e->getErrorCode());
		}
	}

	public function testSetAlphaAcceptsPositiveValue()
	{
		$classifier = new TNaiveBayesClassifier();
		$classifier->setAlpha(0.25);
		self::assertEqualsWithDelta(0.25, $classifier->getAlpha(), 1e-12);
	}

	public function testUntrainedClassifyThrowsNotTrainedCode()
	{
		$classifier = new TNaiveBayesClassifier();
		$classifier->setName('empty');
		try {
			$classifier->classify('anything');
			self::fail('expected exception');
		} catch (TInvalidOperationException $e) {
			self::assertSame('bayesian_classifier_not_trained', $e->getErrorCode());
		}
	}

	public function testDegenerateTrainingThrowsScoreUndefined()
	{
		// TWordTokenizer (minLength 2) drops every 1-character token, so a category is recorded
		// (the classifier reports trained) but the vocabulary is empty: every score is -INF.
		$classifier = new TNaiveBayesClassifier();
		$classifier->trainOne('a', 'a b c');
		self::assertTrue($classifier->getIsTrained());
		self::assertSame([], $classifier->getVocabulary()->getDocumentFrequency());
		try {
			$classifier->classify('a b c');
			self::fail('expected exception');
		} catch (TInvalidOperationException $e) {
			self::assertSame('bayesian_classifier_score_undefined', $e->getErrorCode());
		}
	}

	public function testSaveLoadRoundTripsNGramTokenizer()
	{
		$storage = new TMemoryBayesianStorage();
		$tokenizer = new \Prado\Util\Bayesian\Tokenizer\TNGramTokenizer();
		$tokenizer->setN(4);
		$tokenizer->setCharacters(true);
		$classifier = new TNaiveBayesClassifier();
		$classifier->setTokenizer($tokenizer);
		$classifier->setStorage($storage);
		$classifier->setName('ngram');
		$classifier->trainOne('spam', 'cheap offer click free');
		$classifier->trainOne('ham', 'meeting report tomorrow lunch');
		$classifier->save();

		$restored = new TNaiveBayesClassifier();
		$restored->setStorage($storage);
		$restored->load('ngram');
		$restoredTokenizer = $restored->getTokenizer();
		self::assertInstanceOf(\Prado\Util\Bayesian\Tokenizer\TNGramTokenizer::class, $restoredTokenizer);
		self::assertNotSame($tokenizer, $restoredTokenizer);
		self::assertSame(4, $restoredTokenizer->getN());
		self::assertTrue($restoredTokenizer->getCharacters());
		$sample = 'cheap click meeting';
		$expected = $classifier->score($sample);
		$actual = $restored->score($sample);
		self::assertSame(array_keys($expected), array_keys($actual));
		foreach ($expected as $category => $probability) {
			self::assertEqualsWithDelta($probability, $actual[$category], 1e-12);
		}
	}

	public function testSaveLoadRoundTripsRegexTokenizer()
	{
		$storage = new TMemoryBayesianStorage();
		$tokenizer = new \Prado\Util\Bayesian\Tokenizer\TRegexTokenizer();
		$tokenizer->setPattern('/([a-z]{3,})/');
		$tokenizer->setLowercase(false);
		$classifier = new TNaiveBayesClassifier();
		$classifier->setTokenizer($tokenizer);
		$classifier->setStorage($storage);
		$classifier->setName('regex');
		$classifier->trainOne('spam', 'cheap offer click free');
		$classifier->trainOne('ham', 'meeting report tomorrow lunch');
		$classifier->save();

		$restored = new TNaiveBayesClassifier();
		$restored->setStorage($storage);
		$restored->load('regex');
		$restoredTokenizer = $restored->getTokenizer();
		self::assertInstanceOf(\Prado\Util\Bayesian\Tokenizer\TRegexTokenizer::class, $restoredTokenizer);
		self::assertSame('/([a-z]{3,})/', $restoredTokenizer->getPattern());
		self::assertFalse($restoredTokenizer->getLowercase());
		$sample = 'cheap click meeting';
		$expected = $classifier->score($sample);
		$actual = $restored->score($sample);
		foreach ($expected as $category => $probability) {
			self::assertEqualsWithDelta($probability, $actual[$category], 1e-12);
		}
	}

	public function testSaveLoadRoundTripsTokenizerChain()
	{
		$storage = new TMemoryBayesianStorage();
		$chain = new \Prado\Util\Bayesian\Tokenizer\TBayesianTokenizerChain();
		$word = new TWordTokenizer();
		$word->setMinLength(3);
		$ngram = new \Prado\Util\Bayesian\Tokenizer\TNGramTokenizer();
		$ngram->setN(2);
		$chain->addTokenizer($word);
		$chain->addTokenizer($ngram);
		$classifier = new TNaiveBayesClassifier();
		$classifier->setTokenizer($chain);
		$classifier->setStorage($storage);
		$classifier->setName('chain');
		$classifier->trainOne('spam', 'cheap offer click free');
		$classifier->trainOne('ham', 'meeting report tomorrow lunch');
		$classifier->save();

		$restored = new TNaiveBayesClassifier();
		$restored->setStorage($storage);
		$restored->load('chain');
		$restoredChain = $restored->getTokenizer();
		self::assertInstanceOf(\Prado\Util\Bayesian\Tokenizer\TBayesianTokenizerChain::class, $restoredChain);
		$members = $restoredChain->getTokenizers();
		self::assertCount(2, $members);
		self::assertInstanceOf(TWordTokenizer::class, $members[0]);
		self::assertSame(3, $members[0]->getMinLength());
		self::assertInstanceOf(\Prado\Util\Bayesian\Tokenizer\TNGramTokenizer::class, $members[1]);
		self::assertSame(2, $members[1]->getN());
		$sample = 'cheap click meeting';
		self::assertSame($chain->tokenize($sample), $restoredChain->tokenize($sample));
		$expected = $classifier->score($sample);
		$actual = $restored->score($sample);
		foreach ($expected as $category => $probability) {
			self::assertEqualsWithDelta($probability, $actual[$category], 1e-12);
		}
	}

	public function testLoadReconfiguresExistingTokenizerOfSameClassInPlace()
	{
		$storage = new TMemoryBayesianStorage();
		$classifier = new TNaiveBayesClassifier();
		$classifier->getTokenizer()->setMinLength(4);
		$classifier->getTokenizer()->setStopWords(['the']);
		$classifier->setStorage($storage);
		$classifier->setName('word');
		$classifier->trainOne('spam', 'cheap offer click free');
		$classifier->trainOne('ham', 'meeting report tomorrow lunch');
		$classifier->save();

		$restored = new TNaiveBayesClassifier();
		$original = $restored->getTokenizer();
		self::assertInstanceOf(TWordTokenizer::class, $original);
		self::assertSame(2, $original->getMinLength());
		$restored->setStorage($storage);
		$restored->load('word');
		self::assertSame($original, $restored->getTokenizer(), 'the existing tokenizer object is kept');
		self::assertSame(4, $original->getMinLength());
		self::assertSame(['the'], $original->getStopWords());
	}

	public function testLoadRejectsPayloadOfDifferentKind()
	{
		$storage = new TMemoryBayesianStorage();
		$bernoulli = new \Prado\Util\Bayesian\Classifier\TBernoulliNaiveBayes();
		$bernoulli->setStorage($storage);
		$bernoulli->setName('bern');
		$bernoulli->trainOne('a', 'foo bar');
		$bernoulli->trainOne('b', 'baz qux');
		$bernoulli->save();

		$classifier = new TNaiveBayesClassifier();
		$classifier->setStorage($storage);
		try {
			$classifier->load('bern');
			self::fail('expected exception');
		} catch (TInvalidDataValueException $e) {
			self::assertSame('bayesian_classifier_kind_mismatch', $e->getErrorCode());
		}
		self::assertFalse($classifier->getIsTrained(), 'a rejected payload leaves the classifier untouched');
	}

	public function testNaiveBayesAndMultinomialPayloadsAreInterchangeable()
	{
		$storage = new TMemoryBayesianStorage();
		$naive = new TNaiveBayesClassifier();
		$naive->setStorage($storage);
		$naive->setName('nb');
		$naive->trainOne('spam', 'cheap offer click free');
		$naive->trainOne('ham', 'meeting report tomorrow lunch');
		$naive->save();

		$multinomial = new \Prado\Util\Bayesian\Classifier\TMultinomialNaiveBayes();
		$multinomial->setStorage($storage);
		$multinomial->load('nb');
		self::assertTrue($multinomial->getIsTrained());
		self::assertSame('spam', $multinomial->classify('cheap click'));
		$multinomial->setName('mnb');
		$multinomial->save();
		self::assertSame('multinomial-naive-bayes', $storage->load('mnb')['kind']);

		$back = new TNaiveBayesClassifier();
		$back->setStorage($storage);
		$back->load('mnb');
		self::assertTrue($back->getIsTrained());
		self::assertSame('ham', $back->classify('meeting report'));
	}

	public function testLegacyPayloadWithoutKindOrTokenizerLoads()
	{
		$storage = new TMemoryBayesianStorage();
		$storage->save('legacy', [
			'name' => 'legacy',
			'alpha' => 1.0,
			'useTfidf' => false,
			'spamCategory' => 'spam',
			'categories' => [
				[
					'name' => 'spam',
					'documentCount' => 1,
					'tokenCounts' => ['cheap' => 1, 'offer' => 1],
					'tokenDocumentCounts' => ['cheap' => 1, 'offer' => 1],
					'totalTokens' => 2,
				],
				[
					'name' => 'ham',
					'documentCount' => 1,
					'tokenCounts' => ['meeting' => 1, 'report' => 1],
					'tokenDocumentCounts' => ['meeting' => 1, 'report' => 1],
					'totalTokens' => 2,
				],
			],
			'documentFrequency' => ['cheap' => 1, 'offer' => 1, 'meeting' => 1, 'report' => 1],
			'totalDocuments' => 2,
		]);
		$classifier = new TNaiveBayesClassifier();
		$classifier->setStorage($storage);
		$original = $classifier->getTokenizer();
		$classifier->load('legacy');
		self::assertTrue($classifier->getIsTrained());
		self::assertSame($original, $classifier->getTokenizer(), 'a payload without a tokenizer keeps the current one');
		self::assertSame('spam', $classifier->classify('cheap offer'));
		self::assertSame('ham', $classifier->classify('meeting'));
		self::assertFalse($classifier->getUseTfidf());
	}

	public function testLoadRejectsNonTokenizerClassInPayload()
	{
		$storage = new TMemoryBayesianStorage();
		$storage->save('bad', [
			'kind' => 'naive-bayes',
			'tokenizer' => ['class' => 'stdClass', 'config' => []],
			'categories' => [
				[
					'name' => 'a',
					'documentCount' => 1,
					'tokenCounts' => ['foo' => 1],
					'tokenDocumentCounts' => ['foo' => 1],
					'totalTokens' => 1,
				],
			],
			'documentFrequency' => ['foo' => 1],
			'totalDocuments' => 1,
		]);
		$classifier = new TNaiveBayesClassifier();
		$classifier->setStorage($storage);
		try {
			$classifier->load('bad');
			self::fail('expected exception');
		} catch (TInvalidDataValueException $e) {
			self::assertSame('bayesian_tokenizer_class_invalid', $e->getErrorCode());
		}
	}

	public function testAlphaTfidfAndSpamCategoryRoundTripThroughSaveLoad()
	{
		$storage = new TMemoryBayesianStorage();
		$classifier = new TNaiveBayesClassifier();
		$classifier->setAlpha(0.3);
		$classifier->setUseTfidf(false);
		$classifier->setSpamCategory('junk');
		$classifier->setStorage($storage);
		$classifier->setName('cfg');
		$classifier->trainOne('junk', 'cheap offer');
		$classifier->trainOne('ham', 'meeting report');
		$classifier->save();

		$restored = new TNaiveBayesClassifier();
		$restored->setStorage($storage);
		$restored->load('cfg');
		self::assertEqualsWithDelta(0.3, $restored->getAlpha(), 1e-12);
		self::assertFalse($restored->getUseTfidf());
		self::assertSame('junk', $restored->getSpamCategory());
		self::assertSame('cfg', $restored->getName());
		self::assertTrue($restored->isSpam('cheap offer'));
	}

	public function testCategoryWithoutDocumentsIsNeverChosen()
	{
		// A hand-built payload with an empty category: its prior is zero, so its log-posterior
		// is -INF and it normalizes to probability 0.0.
		$storage = new \Prado\Util\Bayesian\Storage\TMemoryBayesianStorage();
		$storage->save('nb', [
			'kind' => 'naive-bayes',
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
		$classifier = new TNaiveBayesClassifier();
		$classifier->setStorage($storage);
		$classifier->load('nb');
		$scores = $classifier->score('free');
		self::assertSame(0.0, $scores['ghost']);
		self::assertEqualsWithDelta(1.0, $scores['spam'], 1e-12);
		self::assertSame('spam', $classifier->classify('free'));
	}
}
