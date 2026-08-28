<?php

use Prado\Util\Bayesian\Storage\TRedisBayesianStorage;

require_once(__DIR__ . '/../../../test_tools/BayesianBackends.php');

class TRedisBayesianStorageTest extends PHPUnit\Framework\TestCase
{
	public function testConstructorThrowsWhenRedisExtensionMissing()
	{
		// The inverse of every other test here: this one is the only coverage the class gets
		// on a machine without ext-redis, and it is exactly the branch that makes that machine
		// unable to run the rest.  Plain markTestSkipped, not requireBackend - a present
		// extension is not a missing backend, so BAYESIAN_REQUIRE_BACKENDS must not fail it.
		if (extension_loaded('redis')) {
			self::markTestSkipped('redis extension is loaded; the missing-extension guard cannot fire');
		}
		try {
			new TRedisBayesianStorage();
			self::fail('expected a TConfigurationException when ext-redis is missing');
		} catch (\Prado\Exceptions\TConfigurationException $e) {
			self::assertSame('bayesian_storage_redis_missing', $e->getErrorCode());
		}
	}

	public function testSaveAndLoadRoundTrip()
	{
		$storage = $this->newStorage();
		$storage->save('model', ['spam' => 42, 'ham' => 10]);
		self::assertTrue($storage->exists('model'));
		self::assertSame(['spam' => 42, 'ham' => 10], $storage->load('model'));
	}

	public function testLoadReturnsNullForMissing()
	{
		$storage = $this->newStorage();
		self::assertNull($storage->load('missing'));
	}

	public function testDeleteRemovesModel()
	{
		$storage = $this->newStorage();
		$storage->save('model', ['a' => 1]);
		$storage->delete('model');
		self::assertFalse($storage->exists('model'));
	}

	public function testListReturnsAllModelNames()
	{
		$storage = $this->newStorage();
		$storage->save('a', []);
		$storage->save('b', []);
		self::assertSame(['a', 'b'], $storage->list());
	}

	public function testCustomKeyPrefix()
	{
		$storage = $this->newStorage();
		$storage->setKeyPrefix('test:');
		$storage->save('model', ['x' => 1]);
		self::assertTrue($storage->exists('model'));
	}

	public function testListReturnsEmptyForFreshStorage()
	{
		$storage = $this->newStorage();
		self::assertSame([], $storage->list());
	}

	public function testLoadCorruptJsonReturnsNull()
	{
		$storage = $this->newStorage();
		// The storage under test uses a unique prefix; write under that prefix so load('bad')
		// really reads the corrupt value.
		$storage->getRedis()->set($storage->getKeyPrefix() . 'bad', 'not-json');
		self::assertNull($storage->load('bad'));
	}

	private function newStorage(): TRedisBayesianStorage
	{
		// Gate on the extension here rather than in setUp(): the constructor throws without
		// it, so a blanket skip would also hide testConstructorThrowsWhenRedisExtensionMissing.
		BayesianBackends::requireBackend($this, extension_loaded('redis'), 'redis extension not available');
		$storage = new TRedisBayesianStorage();
		$storage->setKeyPrefix('test-' . uniqid('', true) . ':');
		$storage->setIndexKey('test-' . uniqid('', true) . ':index');
		// Use an in-process Redis via the redis extension.  Many test environments
		// don't have Redis running, so this test only runs when ext-redis is loaded
		// AND a real Redis is available.
		try {
			$storage->setHost('127.0.0.1');
			$storage->setPort(6379);
			$storage->setTimeout(0.1);
			$storage->save('init', []);   // forces connection attempt
			$storage->delete('init');     // clean up
		} catch (\Exception $e) {
			BayesianBackends::requireBackend($this, false, 'No reachable Redis at 127.0.0.1:6379: ' . $e->getMessage());
		}
		return $storage;
	}
}
