<?php

use Belisoful\Prado\Util\Bayesian\Evaluation\TConfusionMatrix;
use Prado\Exceptions\TInvalidDataValueException;

class TConfusionMatrixTest extends PHPUnit\Framework\TestCase
{
	public function testConstructorRejectsEmptyLabels()
	{
		$this->expectException(TInvalidDataValueException::class);
		new TConfusionMatrix([]);
	}

	public function testConstructorRecordsLabelsInOrder()
	{
		$matrix = new TConfusionMatrix(['spam', 'ham', 'borderline']);
		self::assertSame(['spam', 'ham', 'borderline'], $matrix->getLabels());
	}

	public function testConstructorDeduplicatesLabels()
	{
		$matrix = new TConfusionMatrix(['a', 'b', 'a']);
		self::assertSame(['a', 'b'], $matrix->getLabels());
	}

	public function testCellsInitializeToZero()
	{
		$matrix = new TConfusionMatrix(['a', 'b']);
		self::assertSame(0, $matrix->getCell('a', 'a'));
		self::assertSame(0, $matrix->getCell('a', 'b'));
		self::assertSame(0, $matrix->getCell('b', 'a'));
		self::assertSame(0, $matrix->getCell('b', 'b'));
		self::assertSame(0, $matrix->getTotal());
	}

	public function testRecordIncrementsCount()
	{
		$matrix = new TConfusionMatrix(['spam', 'ham']);
		$matrix->record('spam', 'spam');
		$matrix->record('spam', 'ham');
		$matrix->record('ham', 'ham');
		self::assertSame(1, $matrix->getCell('spam', 'spam'));
		self::assertSame(1, $matrix->getCell('spam', 'ham'));
		self::assertSame(1, $matrix->getCell('ham', 'ham'));
		self::assertSame(0, $matrix->getCell('ham', 'spam'));
		self::assertSame(3, $matrix->getTotal());
	}

	public function testRecordRejectsUnknownExpected()
	{
		$matrix = new TConfusionMatrix(['a', 'b']);
		$this->expectException(TInvalidDataValueException::class);
		$matrix->record('c', 'a');
	}

	public function testRecordRejectsUnknownPredicted()
	{
		$matrix = new TConfusionMatrix(['a', 'b']);
		$this->expectException(TInvalidDataValueException::class);
		$matrix->record('a', 'c');
	}

	public function testGetCountsReturnsFullMatrix()
	{
		$matrix = new TConfusionMatrix(['a', 'b']);
		$matrix->record('a', 'a');
		$counts = $matrix->getCounts();
		self::assertArrayHasKey('a', $counts);
		self::assertArrayHasKey('b', $counts);
		self::assertArrayHasKey('a', $counts['a']);
		self::assertArrayHasKey('b', $counts['a']);
		self::assertArrayHasKey('a', $counts['b']);
		self::assertArrayHasKey('b', $counts['b']);
	}
}
