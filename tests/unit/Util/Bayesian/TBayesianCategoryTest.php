<?php

use Belisoful\Prado\Util\Bayesian\TBayesianCategory;

class TBayesianCategoryTest extends PHPUnit\Framework\TestCase
{
	public function testInitialStateIsEmpty()
	{
		$cat = new TBayesianCategory('spam');
		self::assertSame('spam', $cat->getName());
		self::assertSame(0, $cat->getDocumentCount());
		self::assertSame(0, $cat->getTotalTokens());
		self::assertSame(0, $cat->getVocabularySize());
		self::assertSame(0, $cat->getTokenCount('foo'));
	}

	public function testAddDocument()
	{
		$cat = new TBayesianCategory('spam');
		$cat->addDocument();
		$cat->addDocument();
		self::assertSame(2, $cat->getDocumentCount());
	}

	public function testAddToken()
	{
		$cat = new TBayesianCategory('spam');
		$cat->addToken('foo');
		$cat->addToken('foo');
		$cat->addToken('bar');
		self::assertSame(2, $cat->getTokenCount('foo'));
		self::assertSame(1, $cat->getTokenCount('bar'));
		self::assertSame(3, $cat->getTotalTokens());
		self::assertSame(2, $cat->getVocabularySize());
	}

	public function testAddTokenWithCount()
	{
		$cat = new TBayesianCategory('spam');
		$cat->addToken('foo', 5);
		self::assertSame(5, $cat->getTokenCount('foo'));
		self::assertSame(5, $cat->getTotalTokens());
	}

	public function testAddTokenClampsNegativeCountToOne()
	{
		$cat = new TBayesianCategory('spam');
		$cat->addToken('foo', 0);
		$cat->addToken('foo', -3);
		self::assertSame(2, $cat->getTokenCount('foo'));
	}

	public function testAddTokenDocument()
	{
		$cat = new TBayesianCategory('spam');
		$cat->addTokenDocument('foo');
		$cat->addTokenDocument('foo');
		$cat->addTokenDocument('bar');
		self::assertSame(2, $cat->getTokenDocumentCount('foo'));
		self::assertSame(1, $cat->getTokenDocumentCount('bar'));
		self::assertSame(0, $cat->getTokenDocumentCount('baz'));
	}

	public function testGetTokenDocumentCountsReturnsAllEntries()
	{
		$cat = new TBayesianCategory('spam');
		$cat->addTokenDocument('foo');
		$cat->addTokenDocument('bar');
		$counts = $cat->getTokenDocumentCounts();
		self::assertArrayHasKey('foo', $counts);
		self::assertArrayHasKey('bar', $counts);
	}

	public function testSetStatsReplacesAllState()
	{
		$cat = new TBayesianCategory('spam');
		$cat->addDocument();
		$cat->addToken('old');
		$cat->setStats(5, ['a' => 3, 'b' => 7], ['a' => 2, 'b' => 4], 10);
		self::assertSame(5, $cat->getDocumentCount());
		self::assertSame(3, $cat->getTokenCount('a'));
		self::assertSame(7, $cat->getTokenCount('b'));
		self::assertSame(10, $cat->getTotalTokens());
		self::assertSame(2, $cat->getTokenDocumentCount('a'));
	}

	public function testSetStatsClampsNegativesToZero()
	{
		$cat = new TBayesianCategory('spam');
		$cat->setStats(-1, ['a' => -1], ['a' => -1], -5);
		self::assertSame(0, $cat->getDocumentCount());
		self::assertSame(0, $cat->getTotalTokens());
	}
}
