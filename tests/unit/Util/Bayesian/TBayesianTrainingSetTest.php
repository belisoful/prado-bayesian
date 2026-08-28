<?php

use Prado\Util\Bayesian\TBayesianTrainingSet;

class TBayesianTrainingSetTest extends PHPUnit\Framework\TestCase
{
	public function testNewSetIsEmpty()
	{
		$set = new TBayesianTrainingSet();
		self::assertTrue($set->getIsEmpty());
		self::assertSame(0, $set->getTotalDocuments());
		self::assertSame([], $set->getCategories());
	}

	public function testAddAndRetrieveStringDocument()
	{
		$set = new TBayesianTrainingSet();
		$set->add('spam', 'buy cheap watches now');
		$set->add('ham', 'lunch tomorrow?');
		self::assertSame(['spam', 'ham'], $set->getCategories());
		self::assertSame('buy cheap watches now', $set->getCategoryDocuments('spam')[0]);
		self::assertSame('lunch tomorrow?', $set->getCategoryDocuments('ham')[0]);
		self::assertSame(2, $set->getTotalDocuments());
		self::assertFalse($set->getIsEmpty());
	}

	public function testAddPretokenizedDocument()
	{
		$set = new TBayesianTrainingSet();
		$set->add('spam', ['buy', 'cheap', 'watches']);
		self::assertSame(['buy', 'cheap', 'watches'], $set->getCategoryDocuments('spam')[0]);
	}

	public function testGetCategoryDocumentsForUnknownCategory()
	{
		$set = new TBayesianTrainingSet();
		self::assertSame([], $set->getCategoryDocuments('nope'));
	}

	public function testEachIteratesAll()
	{
		$set = new TBayesianTrainingSet();
		$set->add('spam', 'one');
		$set->add('spam', 'two');
		$set->add('ham', 'three');
		$collected = [];
		foreach ($set->each() as $category => $document) {
			$collected[] = [$category, $document];
		}
		self::assertCount(3, $collected);
		self::assertSame(['spam', 'one'], $collected[0]);
		self::assertSame(['spam', 'two'], $collected[1]);
		self::assertSame(['ham', 'three'], $collected[2]);
	}

	public function testGetIsEmptyAfterAddingDocument()
	{
		$set = new TBayesianTrainingSet();
		self::assertTrue($set->getIsEmpty());
		$set->add('spam', 'doc');
		self::assertFalse($set->getIsEmpty());
	}
}
