<?php

use Belisoful\Prado\Util\Bayesian\Classifier\TMultinomialNaiveBayes;
use Belisoful\Prado\Util\Bayesian\Classifier\TNaiveBayesClassifier;

class TMultinomialNaiveBayesTest extends PHPUnit\Framework\TestCase
{
	public function testInheritsFromNaiveBayes()
	{
		$classifier = new TMultinomialNaiveBayes();
		self::assertInstanceOf(TNaiveBayesClassifier::class, $classifier);
	}

	public function testClassifiesSpam()
	{
		$classifier = new TMultinomialNaiveBayes();
		$classifier->trainOne('spam', 'free offer click cheap');
		$classifier->trainOne('spam', 'limited time winner prize');
		$classifier->trainOne('ham', 'meeting lunch tomorrow report');
		$classifier->trainOne('ham', 'attached fix thanks help');
		self::assertSame('spam', $classifier->classify('free cheap click'));
		self::assertSame('ham', $classifier->classify('meeting report tomorrow'));
	}

	public function testSaveAndLoadRoundTrip()
	{
		$storage = new \Belisoful\Prado\Util\Bayesian\Storage\TMemoryBayesianStorage();
		$classifier = new TMultinomialNaiveBayes();
		$classifier->setStorage($storage);
		$classifier->setName('mnb');
		$classifier->trainOne('spam', 'free offer click cheap');
		$classifier->trainOne('ham', 'meeting report tomorrow');
		$classifier->save();
		$restored = new TMultinomialNaiveBayes();
		$restored->setStorage($storage);
		$restored->load('mnb');
		self::assertSame('spam', $restored->classify('free cheap'));
	}

	public function testProbabilitiesMatchHandComputation()
	{
		// Pre-tokenized input so the tokenizer is not involved; tf-idf off; alpha = 1.
		// Train spam={a,a,b} (totalTokens=3), ham={b,c} (totalTokens=2). The GLOBAL vocabulary
		// is {a,b,c}, so |V| = 3 (the smoothing denominator that the |V| fix corrects). For the
		// document {a}, with equal priors (one doc each):
		//   P(a|spam) = (2+1)/(3 + 1*3) = 3/6 ;  P(a|ham) = (0+1)/(2 + 1*3) = 1/5
		//   posterior ∝ 0.5*0.5 (spam) : 0.5*0.2 (ham) = 0.25 : 0.10
		$classifier = new TMultinomialNaiveBayes();
		$classifier->setUseTfidf(false);
		$classifier->trainOne('spam', ['a', 'a', 'b']);
		$classifier->trainOne('ham', ['b', 'c']);
		$scores = $classifier->score(['a']);
		self::assertEqualsWithDelta(0.25 / 0.35, $scores['spam'], 1e-9);
		self::assertEqualsWithDelta(0.10 / 0.35, $scores['ham'], 1e-9);
	}
}
