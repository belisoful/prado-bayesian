<?php

use Prado\Exceptions\TConfigurationException;
use Prado\Util\Bayesian\Storage\TFileBayesianStorage;

class TFileBayesianStorageTest extends PHPUnit\Framework\TestCase
{
	private string $_dir;

	protected function setUp(): void
	{
		$this->_dir = sys_get_temp_dir() . '/bayesian-test-' . uniqid('', true);
	}

	protected function tearDown(): void
	{
		if (is_dir($this->_dir)) {
			foreach (glob($this->_dir . '/*') ?: [] as $file) {
				@unlink($file);
			}
			@rmdir($this->_dir);
		}
	}

	public function testSaveAndLoadRoundTrip()
	{
		$storage = new TFileBayesianStorage();
		$storage->setDirectory($this->_dir);
		$storage->save('model', ['spam' => 1, 'ham' => 2]);
		self::assertTrue($storage->exists('model'));
		self::assertSame(['spam' => 1, 'ham' => 2], $storage->load('model'));
	}

	public function testLoadReturnsNullForMissingModel()
	{
		$storage = new TFileBayesianStorage();
		$storage->setDirectory($this->_dir);
		self::assertNull($storage->load('missing'));
	}

	public function testExistsReturnsFalseForMissingModel()
	{
		$storage = new TFileBayesianStorage();
		$storage->setDirectory($this->_dir);
		self::assertFalse($storage->exists('missing'));
	}

	public function testDeleteRemovesFile()
	{
		$storage = new TFileBayesianStorage();
		$storage->setDirectory($this->_dir);
		$storage->save('model', ['x' => 1]);
		$storage->delete('model');
		self::assertFalse($storage->exists('model'));
	}

	public function testListReturnsAllModelNames()
	{
		$storage = new TFileBayesianStorage();
		$storage->setDirectory($this->_dir);
		$storage->save('aaa', []);
		$storage->save('bbb', []);
		$storage->save('ccc', []);
		self::assertSame(['aaa', 'bbb', 'ccc'], $storage->list());
	}

	public function testListSortsRatherThanReturningDirectoryOrder()
	{
		// Directory order is filesystem-dependent; list() must sort regardless of save order.
		$storage = new TFileBayesianStorage();
		$storage->setDirectory($this->_dir);
		$storage->save('zeta', []);
		$storage->save('alpha', []);
		$storage->save('mid', []);
		self::assertSame(['alpha', 'mid', 'zeta'], $storage->list());
	}

	public function testListSkipsTempFilesAndKeepsDotTmpModels()
	{
		$storage = new TFileBayesianStorage();
		$storage->setDirectory($this->_dir);
		$storage->save('good', []);
		// A stray in-progress temp file (dotfile with a random suffix) is never listed ...
		file_put_contents($this->_dir . '/.good.json.Ab12Cd', '{}');
		// ... while a model whose name happens to end in '.tmp' is a real model.
		$storage->save('bad.tmp', ['a' => 1]);
		self::assertSame(['bad.tmp', 'good'], $storage->list());
		self::assertTrue($storage->exists('bad.tmp'));
	}

	public function testSaveLeavesNoTempFilesBehind()
	{
		$storage = new TFileBayesianStorage();
		$storage->setDirectory($this->_dir);
		$storage->save('m', ['a' => 1]);
		$storage->save('m', ['a' => 2]);
		$entries = array_values(array_diff(scandir($this->_dir) ?: [], ['.', '..']));
		self::assertSame(['m.json'], $entries);
		self::assertSame(['a' => 2], $storage->load('m'));
	}

	public function testEmptyDirectoryIsTreatedAsUnset()
	{
		$storage = new TFileBayesianStorage();
		$storage->setDirectory('');
		self::assertFalse($storage->exists('x'));
		self::assertNull($storage->load('x'));
		self::assertSame([], $storage->list());
		$storage->delete('x');
		$this->expectException(\Prado\Exceptions\TConfigurationException::class);
		$storage->save('x', []);
	}

	public function testSaveThrowsWhenDirectoryUnset()
	{
		$storage = new TFileBayesianStorage();
		$this->expectException(TConfigurationException::class);
		$storage->save('model', []);
	}

	public function testSaveCreatesMissingDirectory()
	{
		$storage = new TFileBayesianStorage();
		$storage->setDirectory($this->_dir);
		$storage->save('model', ['k' => 'v']);
		self::assertDirectoryExists($this->_dir);
	}

	public function testGetDirectoryReturnsSetValue()
	{
		$storage = new TFileBayesianStorage();
		$storage->setDirectory($this->_dir);
		self::assertSame($this->_dir, $storage->getDirectory());
	}

	public function testListReturnsEmptyForNullDirectory()
	{
		$storage = new TFileBayesianStorage();
		self::assertSame([], $storage->list());
	}

	public function testSaveRejectsPathTraversalName()
	{
		$storage = new TFileBayesianStorage();
		$storage->setDirectory($this->_dir);
		$this->expectException(\Prado\Exceptions\TInvalidDataValueException::class);
		$storage->save('../escape', ['x' => 1]);
	}

	public function testTraversalNameDoesNotEscapeDirectory()
	{
		$storage = new TFileBayesianStorage();
		$storage->setDirectory($this->_dir);
		$sentinel = sys_get_temp_dir() . '/bayesian-escape-' . uniqid('', true) . '.json';
		try {
			$storage->save('../' . basename($sentinel, '.json'), ['x' => 1]);
		} catch (\Prado\Exceptions\TInvalidDataValueException $e) {
			// expected
		}
		self::assertFileDoesNotExist($sentinel, 'A traversal name must not write outside the storage directory.');
	}

	public function testLoadRejectsPathTraversalName()
	{
		$storage = new TFileBayesianStorage();
		$storage->setDirectory($this->_dir);
		$this->expectException(\Prado\Exceptions\TInvalidDataValueException::class);
		$storage->load('../../etc/passwd');
	}

	public function testSaveWithInvalidUtf8PayloadThrowsEncodeFailed()
	{
		$storage = new TFileBayesianStorage();
		$storage->setDirectory($this->_dir);
		try {
			$storage->save('bad', ['token' => "caf\xE9"]);
			self::fail('expected exception');
		} catch (\Prado\Exceptions\TIOException $e) {
			self::assertSame('bayesian_storage_encode_failed', $e->getErrorCode());
		}
		self::assertFalse($storage->exists('bad'), 'nothing is written when encoding fails');
		self::assertSame([], glob($this->_dir . '/.*.json.*') ?: [], 'no temp file is left behind');
	}

	public function testLoadReturnsNullForMissingAndCorruptFiles()
	{
		$storage = new TFileBayesianStorage();
		$storage->setDirectory($this->_dir);
		$storage->save('good', ['a' => 1]);
		self::assertNull($storage->load('absent'));
		file_put_contents($this->_dir . '/broken.json', '{not json');
		self::assertNull($storage->load('broken'));
		file_put_contents($this->_dir . '/scalar.json', '"a string"');
		self::assertNull($storage->load('scalar'));
		self::assertSame(['a' => 1], $storage->load('good'));
	}

	public function testSaveThrowsWhenDirectoryIsNotWritable()
	{
		if (function_exists('posix_geteuid') && posix_geteuid() === 0) {
			self::markTestSkipped('root bypasses directory permissions');
		}
		$dir = $this->_dir . '/readonly';
		mkdir($dir, 0o755, true);
		chmod($dir, 0o500);
		if (is_writable($dir)) {
			chmod($dir, 0o755);
			self::markTestSkipped('the filesystem does not enforce the directory mode');
		}
		$storage = new TFileBayesianStorage();
		$storage->setDirectory($dir);
		try {
			$storage->save('m', ['a' => 1]);
			self::fail('expected exception');
		} catch (\Prado\Exceptions\TConfigurationException $e) {
			self::assertSame('bayesian_storage_directory_unwritable', $e->getErrorCode());
		} finally {
			chmod($dir, 0o755);
		}
	}

	public function testSaveThrowsWhenDirectoryCannotBeCreated()
	{
		// A child path of a regular file can never be a directory.
		mkdir($this->_dir, 0o755, true);
		$file = $this->_dir . '/plain-file';
		file_put_contents($file, 'x');
		$storage = new TFileBayesianStorage();
		$storage->setDirectory($file . '/sub');
		try {
			$storage->save('m', ['a' => 1]);
			self::fail('expected exception');
		} catch (\Prado\Exceptions\TConfigurationException $e) {
			self::assertSame('bayesian_storage_directory_unwritable', $e->getErrorCode());
		}
	}

	public function testSaveThrowsWhenTheModelPathIsADirectory()
	{
		// rename() into an existing directory fails; the temp file must not be left behind.
		$storage = new TFileBayesianStorage();
		$storage->setDirectory($this->_dir);
		mkdir($this->_dir . '/blocked.json', 0o755, true);
		try {
			$storage->save('blocked', ['a' => 1]);
			self::fail('expected exception');
		} catch (\Prado\Exceptions\TIOException $e) {
			self::assertSame('bayesian_storage_save_failed', $e->getErrorCode());
		}
		$entries = array_values(array_diff(scandir($this->_dir) ?: [], ['.', '..']));
		self::assertSame(['blocked.json'], $entries);
		rmdir($this->_dir . '/blocked.json');
	}

	public function testLoadReturnsNullWhenTheFileCannotBeRead()
	{
		if (function_exists('posix_geteuid') && posix_geteuid() === 0) {
			self::markTestSkipped('root bypasses file permissions');
		}
		$storage = new TFileBayesianStorage();
		$storage->setDirectory($this->_dir);
		$storage->save('locked', ['a' => 1]);
		$path = $this->_dir . '/locked.json';
		chmod($path, 0o000);
		if (is_readable($path)) {
			chmod($path, 0o644);
			self::markTestSkipped('the filesystem does not enforce the file mode');
		}
		try {
			self::assertTrue($storage->exists('locked'));
			self::assertNull($storage->load('locked'));
		} finally {
			chmod($path, 0o644);
		}
	}
}
