<?php

use Belisoful\Prado\Util\Bayesian\Storage\TSqlBayesianStorage;
use Prado\Data\TDbConnection;
use Prado\Exceptions\TConfigurationException;

require_once(__DIR__ . '/../../../test_tools/BayesianTestApplication.php');
require_once(__DIR__ . '/../../../test_tools/BayesianBackends.php');

class TSqlBayesianStorageTest extends PHPUnit\Framework\TestCase
{
	protected function setUp(): void
	{
		BayesianBackends::requireBackend($this, extension_loaded('pdo'), 'pdo extension not available');
		BayesianBackends::requireBackend(
			$this,
			in_array('sqlite', PDO::getAvailableDrivers(), true),
			'pdo_sqlite driver not available'
		);
	}

	private function newStorage(): TSqlBayesianStorage
	{
		$storage = new TSqlBayesianStorage();
		$storage->setConnectionString('sqlite::memory:');
		return $storage;
	}

	public function testSaveAndLoadRoundTrip()
	{
		$storage = $this->newStorage();
		$storage->save('model', ['spam' => 42, 'ham' => 10]);
		self::assertTrue($storage->exists('model'));
		self::assertSame(['spam' => 42, 'ham' => 10], $storage->load('model'));
	}

	public function testSaveOverwritesPrevious()
	{
		$storage = $this->newStorage();
		$storage->save('model', ['a' => 1]);
		$storage->save('model', ['a' => 2]);
		self::assertSame(['a' => 2], $storage->load('model'));
	}

	public function testLoadReturnsNullForMissing()
	{
		$storage = $this->newStorage();
		self::assertNull($storage->load('nope'));
	}

	public function testExistsReturnsFalseForMissing()
	{
		$storage = $this->newStorage();
		self::assertFalse($storage->exists('nope'));
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
		$storage->save('c', []);
		self::assertSame(['a', 'b', 'c'], $storage->list());
	}

	public function testListSortsRatherThanReturningInsertionOrder()
	{
		// ORDER BY name ASC, not row order: saving out of order must still come back sorted.
		$storage = $this->newStorage();
		$storage->save('zeta', []);
		$storage->save('alpha', []);
		$storage->save('mid', []);
		self::assertSame(['alpha', 'mid', 'zeta'], $storage->list());
	}

	public function testCustomTableName()
	{
		$storage = $this->newStorage();
		$storage->setTable('custom_models');
		$storage->save('model', ['x' => 1]);
		self::assertTrue($storage->exists('model'));
	}

	public function testSetConnectionStringResetsConnection()
	{
		$storage = $this->newStorage();
		$storage->save('model', ['a' => 1]);
		$storage->setConnectionString('sqlite::memory:');
		// After connection string change, the model is gone (fresh in-memory database).
		self::assertFalse($storage->exists('model'));
	}

	public function testLoadCorruptJsonReturnsNull()
	{
		$storage = $this->newStorage();
		$storage->save('bad', ['x' => 1]);
		// Corrupt the stored payload directly in the storage's own table.
		$connection = $storage->getDbConnection();
		$command = $connection->createCommand('UPDATE ' . $storage->getTable() . ' SET payload = :payload WHERE name = :name');
		$command->bindValue(':payload', 'not-json');
		$command->bindValue(':name', 'bad');
		$command->execute();
		self::assertNull($storage->load('bad'));
	}

	public function testSetDbConnectionInjectsConnection()
	{
		$connection = new TDbConnection('sqlite::memory:');
		$storage = new TSqlBayesianStorage();
		$storage->setDbConnection($connection);
		$storage->save('model', ['a' => 1]);
		self::assertTrue($storage->exists('model'));
		self::assertSame(['a' => 1], $storage->load('model'));
	}

	public function testGetDbConnectionThrowsWithoutDsnOrConnection()
	{
		$storage = new TSqlBayesianStorage();
		$this->expectException(TConfigurationException::class);
		$storage->getDbConnection();
	}

	public function testConnectionIDWithoutModuleThrows()
	{
		$storage = new TSqlBayesianStorage();
		$storage->setConnectionID('nonexistent');
		$this->expectException(TConfigurationException::class);
		$storage->getDbConnection();
	}

	public function testSetTableRejectsInjectionAttempt()
	{
		$storage = $this->newStorage();
		$this->expectException(\Prado\Exceptions\TInvalidDataValueException::class);
		$storage->setTable('models; DROP TABLE users');
	}

	public function testSetTableAcceptsPlainIdentifier()
	{
		$storage = $this->newStorage();
		$storage->setTable('custom_models_2');
		self::assertSame('custom_models_2', $storage->getTable());
	}

	public function testSaveWithInvalidUtf8PayloadThrowsAndLeavesNoRow()
	{
		$storage = $this->newStorage();
		try {
			$storage->save('bad', ['token' => "caf\xE9"]);
			self::fail('expected exception');
		} catch (\Prado\Exceptions\TInvalidDataValueException $e) {
			self::assertSame('bayesian_storage_encode_failed', $e->getErrorCode());
		}
		self::assertFalse($storage->exists('bad'));
		self::assertSame([], $storage->list());
	}

	public function testCreateTableSqlForMysql()
	{
		$storage = $this->newStorage();
		$sql = $storage->getCreateTableSql('mysql');
		self::assertStringContainsString('CREATE TABLE IF NOT EXISTS bayesian_models', $sql);
		self::assertStringContainsString('VARCHAR(191)', $sql);
		self::assertStringContainsString('LONGTEXT', $sql);
		self::assertStringContainsString('BIGINT', $sql);
	}

	public function testCreateTableSqlForSqlite()
	{
		$storage = $this->newStorage();
		$sql = $storage->getCreateTableSql('sqlite');
		self::assertStringContainsString('CREATE TABLE IF NOT EXISTS bayesian_models', $sql);
		self::assertStringContainsString('TEXT', $sql);
		self::assertStringNotContainsString('LONGTEXT', $sql);
	}

	public function testCreateTableSqlForPgsql()
	{
		$storage = $this->newStorage();
		$sql = $storage->getCreateTableSql('pgsql');
		self::assertStringContainsString('BIGINT', $sql);
		self::assertStringContainsString('VARCHAR(191)', $sql);
		self::assertStringNotContainsString('LONGTEXT', $sql);
	}

	public function testCreateTableSqlUsesConfiguredTable()
	{
		$storage = $this->newStorage();
		$storage->setTable('my_models');
		self::assertStringStartsWith('CREATE TABLE IF NOT EXISTS my_models (', $storage->getCreateTableSql('sqlite'));
	}

	public function testAutoCreateTableDefaultsToTrue()
	{
		$storage = $this->newStorage();
		self::assertTrue($storage->getAutoCreateTable());
		$storage->setAutoCreateTable(false);
		self::assertFalse($storage->getAutoCreateTable());
	}

	public function testLoadWithoutAutoCreateTableThrowsOnFreshDatabase()
	{
		$storage = $this->newStorage();
		$storage->setAutoCreateTable(false);
		$this->expectException(\Prado\Exceptions\TDbException::class);
		$storage->load('anything');
	}

	public function testAutoCreateTableIssuesCreateOnlyOnce()
	{
		$connection = new class ('sqlite::memory:') extends TDbConnection {
			/** @var string[] */
			public array $sqls = [];
			public function createCommand($sql)
			{
				$this->sqls[] = (string) $sql;
				return parent::createCommand($sql);
			}
		};
		$storage = new TSqlBayesianStorage();
		$storage->setDbConnection($connection);
		$storage->save('a', ['x' => 1]);
		$storage->save('b', ['y' => 2]);
		self::assertSame(['x' => 1], $storage->load('a'));
		self::assertTrue($storage->exists('b'));
		self::assertSame(['a', 'b'], $storage->list());
		$storage->delete('a');
		$creates = array_values(array_filter($connection->sqls, static fn (string $sql): bool => str_starts_with($sql, 'CREATE TABLE')));
		self::assertCount(1, $creates, 'CREATE TABLE is issued once per connection');
		self::assertGreaterThan(5, count($connection->sqls));
	}

	public function testTableIsEnsuredAgainOnNewConnection()
	{
		// The table has been ensured on the first connection; a fresh connection (via
		// setConnectionString) needs it created again, so with autoCreate off it is missing.
		$storage = $this->newStorage();
		$storage->save('a', ['x' => 1]);
		$storage->setAutoCreateTable(false);
		self::assertSame(['x' => 1], $storage->load('a'), 'the ensured connection keeps working');
		$storage->setConnectionString('sqlite::memory:');
		$this->expectException(\Prado\Exceptions\TDbException::class);
		$storage->list();
	}

	public function testConnectionPropertiesRoundTripAndResetTheConnection()
	{
		$storage = $this->newStorage();
		self::assertSame('sqlite::memory:', $storage->getConnectionString());
		$first = $storage->getDbConnection();

		self::assertNull($storage->getUsername());
		$storage->setUsername('bayes');
		self::assertSame('bayes', $storage->getUsername());
		// Every connection property invalidates the lazily-built connection.
		self::assertNotSame($first, $storage->getDbConnection());

		self::assertNull($storage->getPassword());
		$storage->setPassword('secret');
		self::assertSame('secret', $storage->getPassword());

		self::assertSame('', $storage->getCharset());
		$storage->setCharset('UTF8');
		self::assertSame('UTF8', $storage->getCharset());

		self::assertSame([], $storage->getAttributes());
		$storage->setAttributes(['Active' => false]);
		self::assertSame(['Active' => false], $storage->getAttributes());

		// The framework's database components report "not set" as an empty string, not null
		// (see Prado\Util\IDbModule); this storage follows that convention.
		self::assertSame('', $storage->getConnectionID());
		$storage->setConnectionID('db');
		self::assertSame('db', $storage->getConnectionID());
	}

	public function testConnectionFollowsTheFrameworkDatabaseComponentContract()
	{
		// TDbPropertiesTrait is the framework's standard way to reach a TDataSourceConfig, and
		// brings these with it; a caller that knows any other Prado database component knows
		// this one.
		$storage = $this->newStorage();
		self::assertFalse($storage->getHasDbConnection(), 'no connection before first use');
		$storage->setConnectionString('sqlite::memory:');
		$connection = $storage->getDbConnection();
		self::assertInstanceOf(\Prado\Data\IDataConnection::class, $connection);
		self::assertTrue($storage->getHasDbConnection());
		self::assertTrue($connection->getActive(), 'the connection is activated on retrieval');

		$storage->deactivateDbConnection();
		self::assertTrue($storage->getHasDbConnection(), 'deactivating keeps the instance');
		$storage->deactivateDbConnection(true);
		self::assertFalse($storage->getHasDbConnection(), 'clearing drops it');
	}

	public function testChangingTheDsnDiscardsTheOpenConnection()
	{
		$storage = $this->newStorage();
		$storage->setConnectionString('sqlite::memory:');
		$storage->save('m', ['a' => 1]);
		self::assertTrue($storage->getHasDbConnection());

		// A second in-memory database is a different database; the old connection must not be
		// reused, or the save above would still be visible.
		$storage->setConnectionString('sqlite::memory:');
		self::assertFalse($storage->getHasDbConnection());
		self::assertNull($storage->load('m'));
	}

	public function testAnInvalidConnectionIdIsReportedWithTheExtensionErrorCode()
	{
		$app = BayesianTestApplication::get();
		$id = 'notADataSource' . uniqid();
		$app->setModule($id, new \Prado\Util\TParameterModule());

		$storage = new TSqlBayesianStorage();
		$storage->setConnectionID($id);
		try {
			$storage->getDbConnection();
			self::fail('expected a ConnectionID naming a non-datasource module to raise');
		} catch (\Prado\Exceptions\TConfigurationException $e) {
			self::assertSame('bayesian_storage_pdo_connect_failed', $e->getErrorCode());
		}
	}

	public function testAttributesAreAppliedToTheConnection()
	{
		$storage = $this->newStorage();
		$storage->setAttributes(['ConnectionString' => 'sqlite::memory:']);
		$storage->save('m', ['a' => 1]);
		self::assertSame(['a' => 1], $storage->load('m'));
	}

	public function testConnectionIDResolvesADataSourceModule()
	{
		$app = BayesianTestApplication::get();
		$source = new \Prado\Data\TDataSourceConfig();
		// TDataSourceConfig builds its own TDbConnection; configure that instance.
		$source->getDbConnection()->setConnectionString('sqlite::memory:');
		$id = 'bayesianDb' . uniqid();
		$app->setModule($id, $source);

		$storage = new TSqlBayesianStorage();
		$storage->setConnectionID($id);
		$storage->save('m', ['a' => 1]);
		self::assertSame(['a' => 1], $storage->load('m'));
		self::assertSame($source->getDbConnection(), $storage->getDbConnection());
	}

	public function testUnknownConnectionIDThrows()
	{
		BayesianTestApplication::get();
		$storage = new TSqlBayesianStorage();
		$storage->setConnectionID('missingDataSource' . uniqid());
		try {
			$storage->load('m');
			self::fail('expected exception');
		} catch (TConfigurationException $e) {
			self::assertSame('bayesian_storage_pdo_connect_failed', $e->getErrorCode());
		}
	}

	public function testDeleteRemovesARowAndIsANoOpOtherwise()
	{
		$storage = $this->newStorage();
		$storage->save('m', ['a' => 1]);
		self::assertTrue($storage->exists('m'));
		$storage->delete('m');
		self::assertFalse($storage->exists('m'));
		self::assertNull($storage->load('m'));
		$storage->delete('m');
		self::assertSame([], $storage->list());
	}

	/**
	 * Builds a storage against a server-backed driver configured through the environment
	 * (`BAYESIAN_MYSQL_DSN` / `BAYESIAN_PGSQL_DSN`), on a table of its own.
	 * @param string $dsnEnv
	 * @param string $userEnv
	 * @param string $passEnv
	 * @param string $driver
	 */
	private function serverStorage(string $dsnEnv, string $userEnv, string $passEnv, string $driver): TSqlBayesianStorage
	{
		BayesianBackends::requireBackend(
			$this,
			in_array($driver, PDO::getAvailableDrivers(), true),
			'pdo_' . $driver . ' driver not available'
		);
		$dsn = BayesianBackends::dsn($dsnEnv);
		BayesianBackends::requireBackend($this, $dsn !== null, $dsnEnv . ' is not configured');

		$storage = new TSqlBayesianStorage();
		$storage->setConnectionString((string) $dsn);
		$storage->setUsername(BayesianBackends::credential($userEnv));
		$storage->setPassword(BayesianBackends::credential($passEnv));
		$storage->setTable('bayes_t' . str_replace('.', '', uniqid('', true)));
		$this->_serverTables[] = $storage;
		return $storage;
	}

	/** @var TSqlBayesianStorage[] Storages whose tables must be dropped after the test. */
	private array $_serverTables = [];

	protected function tearDown(): void
	{
		foreach ($this->_serverTables as $storage) {
			try {
				$storage->getDbConnection()->createCommand('DROP TABLE IF EXISTS ' . $storage->getTable())->execute();
			} catch (\Exception $e) {
				// The table may never have been created; nothing to clean up.
			}
		}
		$this->_serverTables = [];
	}

	/**
	 * Exercises a real server round-trip: the driver-aware DDL must create a usable table, the
	 * upsert must replace an existing row, and the payload column must hold a model far larger
	 * than the 64 KB a plain `TEXT` column allows on MySQL.
	 * @param TSqlBayesianStorage $storage
	 */
	private function assertServerRoundTrip(TSqlBayesianStorage $storage): void
	{
		$storage->save('model', ['a' => 1]);
		self::assertTrue($storage->exists('model'));
		self::assertSame(['a' => 1], $storage->load('model'));

		// Upsert: the same name is replaced, not duplicated.
		$storage->save('model', ['a' => 2]);
		self::assertSame(['a' => 2], $storage->load('model'));
		self::assertSame(['model'], $storage->list());

		// A realistic vocabulary is well over 64 KB of JSON.
		$big = [];
		for ($i = 0; $i < 20000; $i++) {
			$big['token' . $i] = $i;
		}
		$storage->save('big', $big);
		$loaded = $storage->load('big');
		self::assertIsArray($loaded);
		self::assertCount(20000, $loaded);
		self::assertSame(19999, $loaded['token19999']);

		$storage->delete('model');
		self::assertFalse($storage->exists('model'));
		self::assertSame(['big'], $storage->list());
	}

	public function testMySqlRoundTrip()
	{
		$this->assertServerRoundTrip(
			$this->serverStorage('BAYESIAN_MYSQL_DSN', 'BAYESIAN_MYSQL_USER', 'BAYESIAN_MYSQL_PASSWORD', 'mysql')
		);
	}

	public function testPostgresRoundTrip()
	{
		$this->assertServerRoundTrip(
			$this->serverStorage('BAYESIAN_PGSQL_DSN', 'BAYESIAN_PGSQL_USER', 'BAYESIAN_PGSQL_PASSWORD', 'pgsql')
		);
	}
}
