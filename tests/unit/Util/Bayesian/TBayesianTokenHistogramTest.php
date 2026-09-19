<?php

use Belisoful\Prado\Util\Bayesian\TBayesianTokenHistogram;
use Belisoful\Prado\Util\Bayesian\TBayesianVocabulary;

class TBayesianTokenHistogramTest extends PHPUnit\Framework\TestCase
{
	/** @return array<string, array<string, array{count:int, docCount:int}>> Three tokens over two categories. */
	private function tokens(): array
	{
		return [
			'cheap' => ['spam' => ['count' => 3, 'docCount' => 2], 'ham' => ['count' => 1, 'docCount' => 1]],
			'pills' => ['spam' => ['count' => 1, 'docCount' => 1]],
			'team' => ['ham' => ['count' => 2, 'docCount' => 2]],
		];
	}

	public function testFamiliesKeepsOnlyKnownLettersInAStableOrder()
	{
		self::assertSame(['d', 'g'], TBayesianTokenHistogram::families(['g', 'x', 'd', 'g', 7]));
		self::assertSame([], TBayesianTokenHistogram::families(null));
		self::assertSame([], TBayesianTokenHistogram::families('d'));
		self::assertSame(TBayesianTokenHistogram::FAMILIES, TBayesianTokenHistogram::families(array_reverse(TBayesianTokenHistogram::FAMILIES)));
	}

	public function testKeysRoundTripIncludingACategoryWithAColon()
	{
		$key = TBayesianTokenHistogram::key('k', 'a:b:c', 42);
		self::assertSame('k42:a:b:c', $key);
		self::assertSame(['k', 'a:b:c', 42], TBayesianTokenHistogram::parseKey($key));
		self::assertSame(['g', '', 7], TBayesianTokenHistogram::parseKey(TBayesianTokenHistogram::key('g', '', 7)));
		self::assertNull(TBayesianTokenHistogram::parseKey('x1:spam'));
		self::assertNull(TBayesianTokenHistogram::parseKey('d:spam'));
		self::assertNull(TBayesianTokenHistogram::parseKey('d12'));
		self::assertNull(TBayesianTokenHistogram::parseKey(''));
	}

	public function testContributionsOfOneToken()
	{
		$rows = $this->tokens()['cheap'];
		self::assertSame(
			['d2:spam' => 1, 'd1:ham' => 1],
			TBayesianTokenHistogram::contributions($rows, ['d'])
		);
		// Global count 4; in spam the complement holds 1 occurrence, in ham 3.
		self::assertSame(
			['g4:' => 1, 'a4:spam' => 1, 'k1:spam' => 1, 'a4:ham' => 1, 'k3:ham' => 1],
			TBayesianTokenHistogram::contributions($rows, ['g', 'a', 'k'])
		);
	}

	public function testATokenNoDocumentContainsContributesNothing()
	{
		// Out of the vocabulary, whatever its occurrence counts say.
		$rows = ['spam' => ['count' => 2, 'docCount' => 0], 'ham' => ['count' => 0, 'docCount' => 0]];
		self::assertSame([], TBayesianTokenHistogram::contributions($rows, TBayesianTokenHistogram::FAMILIES));
		self::assertSame([], TBayesianTokenHistogram::contributions([], TBayesianTokenHistogram::FAMILIES));
	}

	public function testACategoryWithoutTheTokenIsLeftOutOfThePerCategoryFamilies()
	{
		$rows = ['spam' => ['count' => 2, 'docCount' => 1], 'ham' => ['count' => 0, 'docCount' => 0]];
		self::assertSame(
			['d1:spam' => 1, 'g2:' => 1, 'a2:spam' => 1, 'k0:spam' => 1],
			TBayesianTokenHistogram::contributions($rows, TBayesianTokenHistogram::FAMILIES)
		);
	}

	public function testBuildCountsEveryToken()
	{
		$flat = TBayesianTokenHistogram::build($this->tokens(), TBayesianTokenHistogram::FAMILIES);
		ksort($flat);
		$expected = [
			'd2:spam' => 1, 'd1:spam' => 1, 'd1:ham' => 1, 'd2:ham' => 1,
			'g4:' => 1, 'g1:' => 1, 'g2:' => 1,
			'a4:spam' => 1, 'a1:spam' => 1, 'a4:ham' => 1, 'a2:ham' => 1,
			'k1:spam' => 1, 'k0:spam' => 1, 'k3:ham' => 1, 'k0:ham' => 1,
		];
		ksort($expected);
		self::assertSame($expected, $flat);
	}

	public function testDeltaIsWhatTrainingOneDocumentChanges()
	{
		$before = ['cheap' => $this->tokens()['cheap'], 'new' => []];
		$after = [
			'cheap' => ['spam' => ['count' => 4, 'docCount' => 3], 'ham' => ['count' => 1, 'docCount' => 1]],
			'new' => ['spam' => ['count' => 1, 'docCount' => 1]],
		];
		$delta = TBayesianTokenHistogram::delta($before, $after, TBayesianTokenHistogram::FAMILIES);
		ksort($delta);
		$expected = [
			'd2:spam' => -1, 'd3:spam' => 1, 'd1:spam' => 1,
			'g4:' => -1, 'g5:' => 1, 'g1:' => 1,
			'a4:spam' => -1, 'a5:spam' => 1, 'a1:spam' => 1,
			// cheap's complement in spam is unchanged (ham's 1), so k1:spam nets to zero and is absent.
			'k0:spam' => 1,
			'a4:ham' => -1, 'a5:ham' => 1,
			'k3:ham' => -1, 'k4:ham' => 1,
		];
		ksort($expected);
		self::assertSame($expected, $delta);
	}

	public function testApplyingTheDeltaToTheOldHistogramGivesTheNewOne()
	{
		$tokens = $this->tokens();
		$old = TBayesianTokenHistogram::build($tokens, TBayesianTokenHistogram::FAMILIES);
		$before = ['cheap' => $tokens['cheap'], 'team' => $tokens['team']];
		$tokens['cheap']['ham'] = ['count' => 0, 'docCount' => 0];
		$tokens['team']['ham'] = ['count' => 0, 'docCount' => 0];
		$after = ['cheap' => $tokens['cheap'], 'team' => $tokens['team']];
		foreach (TBayesianTokenHistogram::delta($before, $after, TBayesianTokenHistogram::FAMILIES) as $key => $change) {
			$old[$key] = ($old[$key] ?? 0) + $change;
		}
		$old = array_filter($old);
		$new = TBayesianTokenHistogram::build($tokens, TBayesianTokenHistogram::FAMILIES);
		ksort($old);
		ksort($new);
		self::assertSame($new, $old);
	}

	public function testExpandAndFlattenAreInverses()
	{
		$flat = TBayesianTokenHistogram::build($this->tokens(), TBayesianTokenHistogram::FAMILIES);
		$expanded = TBayesianTokenHistogram::expand($flat);
		self::assertSame(1, $expanded['d']['spam'][2]);
		self::assertSame(1, $expanded['g'][''][4]);
		$back = TBayesianTokenHistogram::flatten($expanded);
		ksort($flat);
		ksort($back);
		self::assertSame($flat, $back);
		// Zero and negative counts, and keys that do not parse, are dropped.
		self::assertSame([], TBayesianTokenHistogram::expand(['d1:spam' => 0, 'g2:' => -1, 'junk' => 3]));
	}

	public function testFromVocabularyMatchesBuild()
	{
		$vocabulary = new TBayesianVocabulary();
		$vocabulary->addDocument('spam', ['cheap', 'cheap', 'pills']);
		$vocabulary->addDocument('spam', ['cheap']);
		$vocabulary->addDocument('ham', ['cheap', 'team']);
		$vocabulary->addDocument('ham', ['team']);
		$fromVocabulary = TBayesianTokenHistogram::flatten(TBayesianTokenHistogram::fromVocabulary($vocabulary, TBayesianTokenHistogram::FAMILIES));
		$built = TBayesianTokenHistogram::build($this->tokens(), TBayesianTokenHistogram::FAMILIES);
		ksort($fromVocabulary);
		ksort($built);
		self::assertSame($built, $fromVocabulary);
	}

	public function testFromVocabularyBuildsOnlyTheFamiliesAskedFor()
	{
		$vocabulary = new TBayesianVocabulary();
		$vocabulary->addDocument('spam', ['cheap']);
		self::assertSame(['d' => ['spam' => [1 => 1]]], TBayesianTokenHistogram::fromVocabulary($vocabulary, ['d']));
		self::assertSame([], TBayesianTokenHistogram::fromVocabulary($vocabulary, []));
	}

	public function testComplementCountsMergesTheThreeFamilies()
	{
		$expanded = TBayesianTokenHistogram::expand(TBayesianTokenHistogram::build($this->tokens(), ['g', 'a', 'k']));
		// spam's complement counts over the vocabulary: cheap 1, pills 0, team 2.
		self::assertSame([0 => 1, 1 => 1, 2 => 1], TBayesianTokenHistogram::complementCounts($expanded, 'spam'));
		// ham's: cheap 3, pills 1, team 0.
		self::assertSame([0 => 1, 1 => 1, 3 => 1], TBayesianTokenHistogram::complementCounts($expanded, 'ham'));
		// A category with no tokens of its own sees the global counts.
		self::assertSame([1 => 1, 2 => 1, 4 => 1], TBayesianTokenHistogram::complementCounts($expanded, 'news'));
	}
}
