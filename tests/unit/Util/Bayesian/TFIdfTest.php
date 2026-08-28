<?php

use Belisoful\Prado\Util\Bayesian\Math\TFIdf;

class TFIdfTest extends PHPUnit\Framework\TestCase
{
	public function testIdfOfCommonTermIsLow()
	{
		$common = TFIdf::idf(100, 100);   // appears in every document
		$rare = TFIdf::idf(5, 100);
		self::assertGreaterThan($common, $rare);
	}

	public function testIdfIsAlwaysAtLeastOne()
	{
		$idf = TFIdf::idf(1000, 1000);
		self::assertGreaterThanOrEqual(1.0, $idf);
	}

	public function testIdfOfUnknownTermMatchesSingleOccurrenceIdf()
	{
		$idf = TFIdf::idf(0, 10);
		self::assertEqualsWithDelta(log(11.0 / 1.0) + 1.0, $idf, 1e-9);
	}

	public function testIdfTreatsANegativeDocumentFrequencyAsZero()
	{
		// A storage-backed lookup returns a count, and a missing row must not produce a
		// negative one that would invert the logarithm.
		self::assertSame(TFIdf::idf(0, 10), TFIdf::idf(-5, 10));
	}

	public function testIdfWithEmptyCorpus()
	{
		self::assertSame(1.0, TFIdf::idf(0, 0));
	}

	public function testTermFrequencyOfOne()
	{
		self::assertEqualsWithDelta(1.0 + log(1), TFIdf::termFrequency(1), 1e-9);
	}

	public function testTermFrequencyOfZero()
	{
		self::assertSame(0.0, TFIdf::termFrequency(0));
	}

	public function testTermFrequencyCompressesRepetition()
	{
		$tf1 = TFIdf::termFrequency(1);
		$tf2 = TFIdf::termFrequency(2);
		$tf100 = TFIdf::termFrequency(100);
		self::assertLessThan($tf1 * 100, $tf100);
		self::assertGreaterThan($tf1, $tf2);
	}

	public function testWeightOfAbsentTermIsZero()
	{
		self::assertSame(0.0, TFIdf::weight(0, 1, 10));
	}

	public function testWeightMultipliesTfAndIdf()
	{
		$weight = TFIdf::weight(3, 2, 10);
		$expected = TFIdf::termFrequency(3) * TFIdf::idf(2, 10);
		self::assertEqualsWithDelta($expected, $weight, 1e-9);
	}
}
