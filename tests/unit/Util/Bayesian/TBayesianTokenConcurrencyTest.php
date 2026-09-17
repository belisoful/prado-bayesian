<?php

use Belisoful\Prado\Util\Bayesian\Classifier\TNaiveBayesClassifier;
use Belisoful\Prado\Util\Bayesian\Storage\IBayesianTokenStorage;
use Belisoful\Prado\Util\Bayesian\Storage\TRedisBayesianStorage;
use Belisoful\Prado\Util\Bayesian\Storage\TSqlBayesianStorage;

require_once(__DIR__ . '/../../../test_tools/BayesianBackends.php');

/**
 * Trains one per-token model from several processes at once and asserts that no count was
 * lost.  This is the deployment shape the package is meant for — web requests and background
 * workers training the same model — so the counters have to be exact, not approximately right.
 *
 * SQLite runs everywhere; MySQL, PostgreSQL and Redis run when their environment is configured
 * (see tests/test_tools/BayesianBackends.php).
 */
class TBayesianTokenConcurrencyTest extends PHPUnit\Framework\TestCase
{
	private const WORKERS = 4;
	private const DOCUMENTS_PER_WORKER = 25;

	/** @var string[] Files to remove after the test. */
	private array $_files = [];

	/** @var IBayesianTokenStorage[] Storages whose models must be dropped after the test. */
	private array $_storages = [];

	protected function tearDown(): void
	{
		foreach ($this->_storages as $storage) {
			try {
				$storage->delete('shared');
				if ($storage instanceof TSqlBayesianStorage) {
					$connection = $storage->getDbConnection();
					foreach (['', '_tokens', '_categories', '_vocab', '_counters'] as $suffix) {
						$connection->createCommand('DROP TABLE IF EXISTS ' . $storage->getTable() . $suffix)->execute();
					}
				}
			} catch (\Throwable $e) {
				// Best-effort cleanup.
			}
		}
		foreach ($this->_files as $file) {
			@unlink($file);
		}
		$this->_storages = [];
		$this->_files = [];
	}

	/**
	 * @return array<string, array{0:string}> The backends to run against.
	 */
	public static function backends(): array
	{
		return ['sqlite' => ['sqlite'], 'mysql' => ['mysql'], 'pgsql' => ['pgsql'], 'redis' => ['redis']];
	}

	/**
	 * @return array{0:IBayesianTokenStorage, 1:array<string, mixed>} The storage and the worker's description of it.
	 */
	private function storage(string $backend): array
	{
		if ($backend === 'redis') {
			BayesianBackends::requireBackend($this, extension_loaded('redis'), 'redis extension not available');
			$storage = new TRedisBayesianStorage();
			$prefix = 'test-' . uniqid('', true) . ':model:';
			$index = 'test-' . uniqid('', true) . ':index';
			$storage->setKeyPrefix($prefix);
			$storage->setIndexKey($index);
			$storage->setMode(TRedisBayesianStorage::MODE_TOKEN);
			try {
				$storage->setTimeout(0.5);
				$storage->getRedis();
			} catch (\Throwable $e) {
				BayesianBackends::requireBackend($this, false, 'No reachable Redis at 127.0.0.1:6379: ' . $e->getMessage());
			}
			$this->_storages[] = $storage;
			return [$storage, ['backend' => 'redis', 'prefix' => $prefix, 'index' => $index]];
		}
		BayesianBackends::requireBackend($this, extension_loaded('pdo'), 'pdo extension not available');
		BayesianBackends::requireBackend($this, in_array($backend, \PDO::getAvailableDrivers(), true), 'pdo_' . $backend . ' driver not available');
		$user = '';
		$password = '';
		if ($backend === 'sqlite') {
			$file = sys_get_temp_dir() . '/bayesian-concurrent-' . uniqid('', true) . '.sqlite';
			$this->_files[] = $file;
			$dsn = 'sqlite:' . $file;
		} else {
			$env = strtoupper($backend);
			$dsn = BayesianBackends::dsn('BAYESIAN_' . $env . '_DSN');
			BayesianBackends::requireBackend($this, $dsn !== null, 'BAYESIAN_' . $env . '_DSN is not configured');
			$user = BayesianBackends::credential('BAYESIAN_' . $env . '_USER');
			$password = BayesianBackends::credential('BAYESIAN_' . $env . '_PASSWORD');
		}
		$table = 'bayes_c' . str_replace('.', '', uniqid('', true));
		$storage = new TSqlBayesianStorage();
		$storage->setConnectionString((string) $dsn);
		$storage->setUsername($user);
		$storage->setPassword($password);
		$storage->setTable($table);
		$storage->setMode(TSqlBayesianStorage::MODE_TOKEN);
		$this->_storages[] = $storage;
		return [$storage, ['backend' => 'sql', 'dsn' => $dsn, 'user' => $user, 'password' => $password, 'table' => $table]];
	}

	/**
	 * @dataProvider backends
	 */
	public function testParallelWorkersLoseNoCounts(string $backend): void
	{
		[$storage, $description] = $this->storage($backend);
		$seed = new TNaiveBayesClassifier();
		$seed->setStorage($storage);
		$seed->setName('shared');
		$seed->trainOne('spam', 'cheap pills');
		$seed->trainOne('ham', 'team meeting');
		$seed->save();

		// Every worker trains the shared token "cheap" plus tokens of its own, so both the
		// contended row and the vocabulary growth are exercised.
		$processes = [];
		$pipes = [];
		for ($w = 0; $w < self::WORKERS; $w++) {
			$documents = [];
			for ($d = 0; $d < self::DOCUMENTS_PER_WORKER; $d++) {
				$documents[] = "cheap shared w{$w}tok{$d} common{$d}";
			}
			$command = [
				PHP_BINARY,
				__DIR__ . '/../../../test_tools/bayesian-train-worker.php',
				json_encode($description),
				'shared',
				'spam',
				json_encode($documents),
			];
			$process = proc_open($command, [1 => ['pipe', 'w'], 2 => ['pipe', 'w']], $procPipes);
			self::assertIsResource($process);
			$processes[$w] = $process;
			$pipes[$w] = $procPipes;
		}
		foreach ($processes as $w => $process) {
			$stderr = stream_get_contents($pipes[$w][2]);
			fclose($pipes[$w][1]);
			fclose($pipes[$w][2]);
			self::assertSame(0, proc_close($process), "worker {$w} failed: {$stderr}");
		}

		$trained = self::WORKERS * self::DOCUMENTS_PER_WORKER;
		$reader = new TNaiveBayesClassifier();
		$reader->setStorage($storage);
		$reader->load('shared');
		$vocabulary = $reader->getVocabulary();
		self::assertSame(2 + $trained, $vocabulary->getTotalDocuments(), $backend . ': every document is counted');
		// Seed: cheap, pills, team, meeting. Workers add "shared", 25 "commonN" and 25 tokens each.
		$expectedVocabulary = 4 + 1 + self::DOCUMENTS_PER_WORKER + self::WORKERS * self::DOCUMENTS_PER_WORKER;
		self::assertSame($expectedVocabulary, $vocabulary->getVocabularySize(), $backend . ': vocabulary growth is exact');
		$spam = $vocabulary->getCategory('spam');
		self::assertSame(1 + $trained, $spam->getDocumentCount(), $backend . ': category document count');
		self::assertSame(2 + 4 * $trained, $spam->getTotalTokens(), $backend . ': category token total');
		$rows = $storage->loadTokens('shared', ['cheap', 'shared', 'common0']);
		self::assertSame(1 + $trained, $rows['cheap']['spam']['count'], $backend . ': the contended token count');
		self::assertSame(1 + $trained, $rows['cheap']['spam']['docCount'], $backend . ': the contended token document count');
		self::assertSame($trained, $rows['shared']['spam']['count']);
		self::assertSame(self::WORKERS, $rows['common0']['spam']['count']);
	}
}
