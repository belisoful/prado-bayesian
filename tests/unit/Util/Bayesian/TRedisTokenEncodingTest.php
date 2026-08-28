<?php

use Belisoful\Prado\Util\Bayesian\Storage\TRedisBayesianStorage;

require_once(__DIR__ . '/../../../test_tools/BayesianBackends.php');

/**
 * Covers the pure per-token encode/decode helpers of the Redis backend, without a live Redis.
 *
 * The Redis integration tests need both `ext-redis` and a server and therefore skip on most
 * developer machines.  The bugs those tests would catch, though, mostly live in one place: the
 * mapping between the nested `{token: {category: {count, docCount}}}` shape and Redis's flat
 * hash fields.  That mapping is pure and reachable by reflection, so it is pinned down here
 * where it always runs — and in particular the case a category named `cat` or `dog` (starting
 * with a type byte) does not corrupt the parse.
 *
 * These call private static methods through reflection: the methods never touch Redis, and
 * reflection does not invoke the class constructor (which would require `ext-redis`).
 */
class TRedisTokenEncodingTest extends PHPUnit\Framework\TestCase
{
	private function invoke(string $method, array $args)
	{
		$m = new \ReflectionMethod(TRedisBayesianStorage::class, $method);
		$m->setAccessible(true);
		return $m->invoke(null, ...$args);
	}

	public function testTokenFieldEncodesTypeAndCategoryWithoutCollision()
	{
		self::assertSame('cspam', $this->invoke('tokenField', ['c', 'spam']));
		self::assertSame('dspam', $this->invoke('tokenField', ['d', 'spam']));
		// 'c' + 'at' must not equal 'd' + something; the fixed one-byte prefix guarantees it.
		self::assertNotSame(
			$this->invoke('tokenField', ['c', 'at']),
			$this->invoke('tokenField', ['d', 'cat'])
		);
		self::assertSame('c', $this->invoke('tokenField', ['c', '']));
	}

	public function testParseTokenHashRoundTripsTwoCategories()
	{
		$parsed = $this->invoke('parseTokenHash', [['cspam' => '5', 'dspam' => '2', 'cham' => '1', 'dham' => '1']]);
		self::assertSame([
			'spam' => ['count' => 5, 'docCount' => 2],
			'ham' => ['count' => 1, 'docCount' => 1],
		], $parsed);
	}

	public function testParseTokenHashHandlesCategoriesStartingWithTheTypeBytes()
	{
		$parsed = $this->invoke('parseTokenHash', [['ccat' => '3', 'dcat' => '1', 'cdog' => '7', 'ddog' => '2']]);
		self::assertSame([
			'cat' => ['count' => 3, 'docCount' => 1],
			'dog' => ['count' => 7, 'docCount' => 2],
		], $parsed);
	}

	public function testParseTokenHashDefaultsAMissingHalfToZero()
	{
		self::assertSame(
			['lonely' => ['count' => 4, 'docCount' => 0]],
			$this->invoke('parseTokenHash', [['clonely' => '4']])
		);
	}

	public function testParseTokenHashSkipsUnrecognizedFields()
	{
		self::assertSame(
			['spam' => ['count' => 1, 'docCount' => 1]],
			$this->invoke('parseTokenHash', [['xbogus' => '9', 'cspam' => '1', 'dspam' => '1', '' => '0']])
		);
	}

	public function testParseTokenHashOfEmptyHashIsEmpty()
	{
		self::assertSame([], $this->invoke('parseTokenHash', [[]]));
	}

	public function testCategoryScalarsRoundTrip()
	{
		self::assertSame('12:340', $this->invoke('packCategoryScalars', [12, 340]));
		self::assertSame(
			['documentCount' => 12, 'totalTokens' => 340],
			$this->invoke('unpackCategoryScalars', ['12:340'])
		);
		self::assertSame(
			['documentCount' => 0, 'totalTokens' => 0],
			$this->invoke('unpackCategoryScalars', ['0:0'])
		);
	}

	public function testUnpackToleratesAMalformedScalar()
	{
		self::assertSame(
			['documentCount' => 5, 'totalTokens' => 0],
			$this->invoke('unpackCategoryScalars', ['5'])
		);
	}
}
