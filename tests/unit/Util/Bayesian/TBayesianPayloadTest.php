<?php

use Belisoful\Prado\Util\Bayesian\TBayesianPayload;
use Prado\Exceptions\TInvalidDataValueException;

class TBayesianPayloadTest extends PHPUnit\Framework\TestCase
{
	public function testIntCoercesScalarsAndFallsBackOtherwise()
	{
		self::assertSame(3, TBayesianPayload::int(3));
		self::assertSame(3, TBayesianPayload::int(3.9));
		self::assertSame(1, TBayesianPayload::int(true));
		self::assertSame(12, TBayesianPayload::int('12'));
		self::assertSame(7, TBayesianPayload::int(INF, 7));
		self::assertSame(7, TBayesianPayload::int('twelve', 7));
		self::assertSame(7, TBayesianPayload::int(null, 7));
		self::assertSame(7, TBayesianPayload::int([], 7));
	}

	public function testFloatStringAndBoolCoerceScalarsAndFallBackOtherwise()
	{
		self::assertSame(2.5, TBayesianPayload::float('2.5'));
		self::assertSame(2.0, TBayesianPayload::float(2));
		self::assertSame(0.5, TBayesianPayload::float(null, 0.5));
		self::assertSame('1.5', TBayesianPayload::string(1.5));
		self::assertSame('1', TBayesianPayload::string(true));
		self::assertSame('x', TBayesianPayload::string([], 'x'));
		self::assertFalse(TBayesianPayload::bool('0'));
		self::assertTrue(TBayesianPayload::bool('yes'));
		self::assertTrue(TBayesianPayload::bool([], true));
	}

	public function testFormatFloatNamesTheNonFiniteValues()
	{
		self::assertSame('NAN', TBayesianPayload::formatFloat(NAN));
		self::assertSame('INF', TBayesianPayload::formatFloat(INF));
		self::assertSame('-INF', TBayesianPayload::formatFloat(-INF));
		self::assertSame('0.25', TBayesianPayload::formatFloat(0.25));
	}

	public function testMapListAndTypedCollectionsConvertTheirEntries()
	{
		self::assertSame(['1' => 'a', 'b' => 'c'], TBayesianPayload::map([1 => 'a', 'b' => 'c']));
		self::assertSame([], TBayesianPayload::map('no'));
		self::assertSame(['a', 'b'], TBayesianPayload::list([5 => 'a', 9 => 'b']));
		self::assertSame([], TBayesianPayload::list(null));
		self::assertSame(['a' => 1, 'b' => 0], TBayesianPayload::intMap(['a' => '1', 'b' => 'x']));
		self::assertSame(['a' => 1.5, 'b' => 0.0], TBayesianPayload::floatMap(['a' => '1.5', 'b' => []]));
		self::assertSame(['1', 'x'], TBayesianPayload::stringList([1, ['nested'], 'x']));
	}

	public function testEncodeTokenMetaDropsTheDerivedTotalsAndStampsTheLayout()
	{
		$json = TBayesianPayload::encodeTokenMeta([
			'kind' => 'bernoulli',
			'totalDocuments' => 12,
			'vocabularySize' => 40,
			'tokenizer' => ['class' => 'TWordTokenizer', 'Pattern' => '/\w+/u'],
			'label' => 'café',
		], 2);
		$decoded = json_decode($json, true);
		self::assertSame(
			['kind' => 'bernoulli', 'tokenizer' => ['class' => 'TWordTokenizer', 'Pattern' => '/\w+/u'], 'label' => 'café', 'layoutVersion' => 2],
			$decoded
		);
		self::assertStringContainsString('"/\\\\w+/u"', $json, 'slashes are not escaped, backslashes are');
		self::assertStringContainsString('"café"', $json, 'unicode is written as is');
		self::assertSame('{"layoutVersion":3}', TBayesianPayload::encodeTokenMeta(['totalDocuments' => 1], 3));
	}

	public function testEncodeTokenMetaRefusesWhatJsonCannotCarry()
	{
		try {
			TBayesianPayload::encodeTokenMeta(['token' => "caf\xE9"], 2);
			self::fail('expected exception');
		} catch (TInvalidDataValueException $e) {
			self::assertSame('bayesian_storage_encode_failed', $e->getErrorCode());
		}
	}
}
