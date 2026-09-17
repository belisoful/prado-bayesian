<?php

use Belisoful\Prado\Util\Bayesian\TBayesianVocabulary;

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
		$cat1 = new \Belisoful\Prado\Util\Bayesian\TBayesianCategory('spam');
		$cat1->setStats(2, ['a' => 4], ['a' => 1], 4);
		$cat2 = new \Belisoful\Prado\Util\Bayesian\TBayesianCategory('ham');
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
	public function testRemoveDocumentReversesAddDocumentExactly()
	{
		$reference = new TBayesianVocabulary();
		$reference->addDocument('spam', ['cheap', 'pills']);
		$reference->addDocument('ham', ['team', 'meeting']);

		$vocabulary = new TBayesianVocabulary();
		$vocabulary->addDocument('spam', ['cheap', 'pills']);
		$vocabulary->addDocument('ham', ['team', 'meeting']);
		$vocabulary->addDocument('spam', ['cheap', 'cheap', 'watches']);
		$generation = $vocabulary->getGeneration();
		$vocabulary->removeDocument('spam', ['cheap', 'cheap', 'watches']);

		self::assertSame($reference->getTotalDocuments(), $vocabulary->getTotalDocuments());
		self::assertSame($reference->getDocumentFrequency(), $vocabulary->getDocumentFrequency());
		self::assertSame($reference->getVocabularySize(), $vocabulary->getVocabularySize());
		self::assertFalse($vocabulary->hasToken('watches'), 'a token no document contains any more has left the vocabulary');
		self::assertTrue($vocabulary->hasToken('cheap'));
		self::assertSame($reference->getCategory('spam')->getTokenCounts(), $vocabulary->getCategory('spam')->getTokenCounts());
		self::assertSame($reference->getCategory('spam')->getTokenDocumentCounts(), $vocabulary->getCategory('spam')->getTokenDocumentCounts());
		self::assertSame($reference->getCategory('spam')->getTotalTokens(), $vocabulary->getCategory('spam')->getTotalTokens());
		self::assertGreaterThan($generation, $vocabulary->getGeneration());
	}

	public function testRemovingTheLastDocumentOfACategoryRemovesTheCategory()
	{
		$vocabulary = new TBayesianVocabulary();
		$vocabulary->addDocument('spam', ['cheap']);
		$vocabulary->addDocument('ham', ['team']);
		$vocabulary->removeDocument('ham', ['team']);
		self::assertSame(['spam'], $vocabulary->getCategoryNames());
		self::assertNull($vocabulary->getCategory('ham'));
		self::assertFalse($vocabulary->hasToken('team'));
		self::assertSame(1, $vocabulary->getTotalDocuments());
		$vocabulary->removeDocument('spam', ['cheap']);
		self::assertTrue($vocabulary->getIsEmpty());
		self::assertSame(0, $vocabulary->getVocabularySize());
	}

	public function testRemoveDocumentClampsAndIgnoresTheUnknown()
	{
		$vocabulary = new TBayesianVocabulary();
		$vocabulary->addDocument('spam', ['cheap']);
		// Removing from a category that does not exist changes nothing.
		$vocabulary->removeDocument('ghost', ['cheap']);
		self::assertSame(['spam'], $vocabulary->getCategoryNames());
		self::assertSame(1, $vocabulary->getTokenDocumentFrequency('cheap'));
		// Removing tokens the category never saw leaves the counts alone; the document count
		// still drops, since a document was withdrawn.
		$vocabulary->addDocument('spam', ['pills']);
		$vocabulary->removeDocument('spam', ['never', 'seen']);
		self::assertSame(1, $vocabulary->getCategory('spam')->getDocumentCount());
		self::assertSame(1, $vocabulary->getTokenDocumentFrequency('cheap'));
		self::assertSame(1, $vocabulary->getTokenDocumentFrequency('pills'));
		self::assertSame(0, $vocabulary->getTokenDocumentFrequency('never'));
		self::assertSame(2, $vocabulary->getVocabularySize());
	}
}
