<?php

use Prado\Util\Bayesian\TBayesianVocabulary;

class TBayesianVocabularyTest extends PHPUnit\Framework\TestCase
{
	public function testNewVocabularyIsEmpty()
	{
		$vocab = new TBayesianVocabulary();
		self::assertTrue($vocab->getIsEmpty());
		self::assertSame([], $vocab->getCategories());
		self::assertSame([], $vocab->getCategoryNames());
		self::assertSame(0, $vocab->getTotalDocuments());
		self::assertSame([], $vocab->getDocumentFrequency());
	}

	public function testGetOrCreateCategoryIsIdempotent()
	{
		$vocab = new TBayesianVocabulary();
		$a = $vocab->getOrCreateCategory('spam');
		$b = $vocab->getOrCreateCategory('spam');
		self::assertSame($a, $b);
		self::assertCount(1, $vocab->getCategories());
	}

	public function testGetCategoryReturnsNullWhenAbsent()
	{
		$vocab = new TBayesianVocabulary();
		self::assertNull($vocab->getCategory('nope'));
	}

	public function testGetCategoryReturnsCategoryWhenPresent()
	{
		$vocab = new TBayesianVocabulary();
		$cat = $vocab->getOrCreateCategory('spam');
		self::assertSame($cat, $vocab->getCategory('spam'));
	}

	public function testAddDocumentRecordsCounts()
	{
		$vocab = new TBayesianVocabulary();
		$vocab->addDocument('spam', ['free', 'money', 'free']);
		$cat = $vocab->getCategory('spam');
		self::assertSame(1, $cat->getDocumentCount());
		self::assertSame(2, $cat->getTokenCount('free'));
		self::assertSame(1, $cat->getTokenCount('money'));
		self::assertSame(3, $cat->getTotalTokens());
		self::assertSame(1, $vocab->getTotalDocuments());
		// One document containing "free" -> document frequency is 1, not 2.
		self::assertSame(1, $vocab->getDocumentFrequency()['free']);
		self::assertSame(1, $vocab->getDocumentFrequency()['money']);
	}

	public function testAddDocumentCountsDocumentFrequencyOncePerUniqueToken()
	{
		$vocab = new TBayesianVocabulary();
		$vocab->addDocument('spam', ['free', 'free', 'free', 'money']);
		$vocab->addDocument('ham', ['meeting', 'free']);
		self::assertSame(2, $vocab->getDocumentFrequency()['free']);
		self::assertSame(1, $vocab->getDocumentFrequency()['money']);
		self::assertSame(1, $vocab->getDocumentFrequency()['meeting']);
	}

	public function testCategoryNamesPreserveInsertionOrder()
	{
		$vocab = new TBayesianVocabulary();
		$vocab->getOrCreateCategory('c');
		$vocab->getOrCreateCategory('a');
		$vocab->getOrCreateCategory('b');
		self::assertSame(['c', 'a', 'b'], $vocab->getCategoryNames());
	}

	public function testSetStatsReplacesVocabulary()
	{
		$vocab = new TBayesianVocabulary();
		$cat1 = new \Prado\Util\Bayesian\TBayesianCategory('spam');
		$cat1->setStats(2, ['a' => 4], ['a' => 1], 4);
		$cat2 = new \Prado\Util\Bayesian\TBayesianCategory('ham');
		$cat2->setStats(1, ['b' => 2], ['b' => 1], 2);
		$vocab->setStats([$cat1, $cat2], ['a' => 1, 'b' => 1], 3);
		self::assertSame(2, count($vocab->getCategories()));
		self::assertSame(3, $vocab->getTotalDocuments());
		self::assertSame(4, $vocab->getCategory('spam')->getTokenCount('a'));
	}

	public function testSetStatsClampsNegativeTotal()
	{
		$vocab = new TBayesianVocabulary();
		$vocab->setStats([], [], -1);
		self::assertSame(0, $vocab->getTotalDocuments());
	}
}
