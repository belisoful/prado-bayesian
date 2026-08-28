<?php

use Prado\Util\Bayesian\Storage\TMemoryBayesianStorage;

class TMemoryBayesianStorageTest extends PHPUnit\Framework\TestCase
{
	public function testNewStorageIsEmpty()
	{
		$storage = new TMemoryBayesianStorage();
		self::assertSame([], $storage->list());
		self::assertFalse($storage->exists('foo'));
		self::assertNull($storage->load('foo'));
	}

	public function testSaveAndLoadRoundTrip()
	{
		$storage = new TMemoryBayesianStorage();
		$storage->save('spam-filter', ['spam' => 42, 'ham' => 10]);
		self::assertTrue($storage->exists('spam-filter'));
		self::assertSame(['spam' => 42, 'ham' => 10], $storage->load('spam-filter'));
	}

	public function testSaveOverwritesPrevious()
	{
		$storage = new TMemoryBayesianStorage();
		$storage->save('model', ['a' => 1]);
		$storage->save('model', ['a' => 2]);
		self::assertSame(['a' => 2], $storage->load('model'));
	}

	public function testDeleteRemovesModel()
	{
		$storage = new TMemoryBayesianStorage();
		$storage->save('model', ['a' => 1]);
		$storage->delete('model');
		self::assertFalse($storage->exists('model'));
		self::assertNull($storage->load('model'));
	}

	public function testDeleteNonexistentIsNoOp()
	{
		$storage = new TMemoryBayesianStorage();
		$storage->delete('missing');
		self::assertFalse($storage->exists('missing'));
	}

	public function testListReturnsAllNames()
	{
		$storage = new TMemoryBayesianStorage();
		$storage->save('a', []);
		$storage->save('b', []);
		$storage->save('c', []);
		self::assertSame(['a', 'b', 'c'], $storage->list());
	}

	public function testListSortsRatherThanReturningInsertionOrder()
	{
		// IBayesianStorage::list() promises ascending order from every backend.  Saving in
		// already-sorted order cannot tell a sort apart from insertion order, so save out of
		// order: this is the case that separates the in-process backend from the others.
		$storage = new TMemoryBayesianStorage();
		$storage->save('zeta', []);
		$storage->save('alpha', []);
		$storage->save('mid', []);
		self::assertSame(['alpha', 'mid', 'zeta'], $storage->list());
	}

	public function testListStaysSortedAfterDeleteAndReinsert()
	{
		$storage = new TMemoryBayesianStorage();
		$storage->save('alpha', []);
		$storage->save('mid', []);
		$storage->save('zeta', []);
		// Deleting and re-saving moves the key to the end of the backing array; the sort has
		// to survive that, not just the initial insertion order.
		$storage->delete('alpha');
		$storage->save('alpha', []);
		self::assertSame(['alpha', 'mid', 'zeta'], $storage->list());
	}
}
