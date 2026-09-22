<?php

use Belisoful\Prado\Util\Bayesian\Classifier\TBernoulliNaiveBayes;
use Belisoful\Prado\Util\Bayesian\Classifier\TComplementNaiveBayes;
use Belisoful\Prado\Util\Bayesian\Classifier\TNaiveBayesClassifier;
use Belisoful\Prado\Util\Bayesian\Storage\IBayesianHistogramStorage;
use Belisoful\Prado\Util\Bayesian\Storage\IBayesianTokenStorage;
use Belisoful\Prado\Util\Bayesian\Storage\TRedisBayesianStorage;
use Belisoful\Prado\Util\Bayesian\Storage\TSqlBayesianStorage;
use PHPUnit\Framework\Attributes\DataProvider;

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
					foreach (['', '_tokens', '_categories', '_vocab', '_counters', '_hist', '_journal', '_folded'] as $suffix) {
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
	 * Starts a worker process with its stdout and stderr merged on one pipe, so a chatty child
	 * can never fill a pipe nobody reads and block against the reader of the other one.
	 * @param string[] $command The command line, PHP binary first.
	 * @return array{0:resource, 1:resource} The process and its output pipe.
	 */
	private function startWorker(array $command): array
	{
		$process = proc_open($command, [1 => ['pipe', 'w'], 2 => ['redirect', 1]], $pipes);
		self::assertIsResource($process);
		return [$process, $pipes[1]];
	}

	/**
	 * Waits for a worker and asserts that it exited cleanly, quoting its output when it did not.
	 * @param resource $process The process.
	 * @param resource $pipe Its output pipe.
	 * @param string $what The worker, for the failure message.
	 */
	private function finishWorker($process, $pipe, string $what): void
	{
		$output = stream_get_contents($pipe);
		fclose($pipe);
		self::assertSame(0, proc_close($process), "{$what} failed: {$output}");
	}

	/**
	 * @return array<string, array{0:string}> The backends to run against.
	 */
	public static function backends(): array
	{
		return ['sqlite' => ['sqlite'], 'mysql' => ['mysql'], 'pgsql' => ['pgsql'], 'redis' => ['redis']];
	}

	/**
	 * @param string $backend
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

	#[DataProvider('backends')]
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
			[$processes[$w], $pipes[$w]] = $this->startWorker($command);
		}
		foreach ($processes as $w => $process) {
			$this->finishWorker($process, $pipes[$w], "worker {$w}");
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

	/**
	 * @return array<string, array{0:string, 1:class-string<TNaiveBayesClassifier>}> Each backend with each variant.
	 */
	public static function variantBackends(): array
	{
		$cases = [];
		foreach (array_keys(self::backends()) as $backend) {
			foreach (['bernoulli' => TBernoulliNaiveBayes::class, 'complement' => TComplementNaiveBayes::class] as $label => $class) {
				$cases[$backend . ' ' . $label] = [$backend, $class];
			}
		}
		return $cases;
	}

	/**
	 * @return array<string, array{0:string, 1:class-string<TNaiveBayesClassifier>}> Each backend with every classifier.
	 */
	public static function classifierBackends(): array
	{
		$cases = self::variantBackends();
		foreach (array_keys(self::backends()) as $backend) {
			$cases[$backend . ' multinomial'] = [$backend, TNaiveBayesClassifier::class];
		}
		return $cases;
	}

	/**
	 * Bernoulli and Complement keep histograms over the whole vocabulary in the store.  Workers
	 * training different categories at once all move them, so after the dust settles they must
	 * equal a rebuild from the token rows, and the model must score exactly as a resident model
	 * trained on the same documents.
	 * @param class-string<TNaiveBayesClassifier> $class
	 */
	#[DataProvider('variantBackends')]
	public function testParallelWorkersKeepTheVariantAggregatesExact(string $backend, string $class): void
	{
		[$storage, $description] = $this->storage($backend);
		self::assertInstanceOf(IBayesianHistogramStorage::class, $storage);
		$reference = new $class();
		$seed = new $class();
		$seed->setStorage($storage);
		$seed->setName('shared');
		foreach ([['spam', 'cheap pills'], ['ham', 'team meeting cheap'], ['news', 'election results']] as [$category, $document]) {
			$seed->trainOne($category, $document);
			$reference->trainOne($category, $document);
		}
		$seed->save();

		$categories = ['spam', 'ham', 'news', 'spam'];
		$processes = [];
		$pipes = [];
		for ($w = 0; $w < self::WORKERS; $w++) {
			$documents = [];
			for ($d = 0; $d < self::DOCUMENTS_PER_WORKER; $d++) {
				// "cheap" and "commonN" are contended across categories; the rest is the worker's own.
				$documents[] = "cheap cheap shared w{$w}tok{$d} common{$d}";
				$reference->trainOne($categories[$w], end($documents));
			}
			$command = [
				PHP_BINARY,
				__DIR__ . '/../../../test_tools/bayesian-train-worker.php',
				json_encode($description),
				'shared',
				$categories[$w],
				json_encode($documents),
				$class,
			];
			[$processes[$w], $pipes[$w]] = $this->startWorker($command);
		}
		foreach ($processes as $w => $process) {
			$this->finishWorker($process, $pipes[$w], "worker {$w}");
		}

		$reader = new $class();
		$reader->setStorage($storage);
		$reader->load('shared');
		foreach (['cheap pills common3', 'team meeting election', 'shared w1tok2 results', 'nothing known'] as $probe) {
			self::assertSame($reference->logScores($probe), $reader->logScores($probe), $backend . ' ' . $class . ': ' . $probe);
		}
		$families = \Belisoful\Prado\Util\Bayesian\TBayesianTokenHistogram::families($storage->loadTokenMeta('shared')['histograms'] ?? null);
		self::assertNotSame([], $families);
		$maintained = $storage->loadTokenHistograms('shared', $families);
		$storage->rebuildTokenHistograms('shared', $families);
		self::assertEquals($maintained, $storage->loadTokenHistograms('shared', $families), $backend . ': the maintained histograms equal a rebuild');
	}

	/**
	 * Withdrawing documents moves the histograms the other way, through the clamped-decrement
	 * path.  Workers withdraw what was trained, all at once, two of them in the same category
	 * on the same rows, and the model must end exactly where a resident one does.  The
	 * multinomial classifier runs too: it keeps no histograms, but the vocabulary must still
	 * shrink by exactly the tokens no document contains any more.
	 * @param class-string<TNaiveBayesClassifier> $class
	 */
	#[DataProvider('classifierBackends')]
	public function testParallelWorkersUntrainingKeepTheVariantAggregatesExact(string $backend, string $class): void
	{
		[$storage, $description] = $this->storage($backend);
		$reference = new $class();
		$seed = new $class();
		$seed->setStorage($storage);
		$seed->setName('shared');
		$categories = ['spam', 'ham', 'news', 'spam'];
		$batches = [];
		foreach ([['spam', 'cheap pills'], ['ham', 'team meeting cheap'], ['news', 'election results']] as [$category, $document]) {
			$seed->trainOne($category, $document);
			$reference->trainOne($category, $document);
		}
		for ($w = 0; $w < self::WORKERS; $w++) {
			for ($d = 0; $d < self::DOCUMENTS_PER_WORKER; $d++) {
				$document = "cheap cheap shared w{$w}tok{$d} common{$d}";
				$batches[$w][] = $document;
				$seed->trainOne($categories[$w], $document);
			}
		}
		$seed->save();

		$processes = [];
		$pipes = [];
		foreach ($batches as $w => $documents) {
			$command = [
				PHP_BINARY,
				__DIR__ . '/../../../test_tools/bayesian-train-worker.php',
				json_encode($description),
				'shared',
				$categories[$w],
				json_encode($documents),
				$class,
				'untrain',
			];
			[$processes[$w], $pipes[$w]] = $this->startWorker($command);
		}
		foreach ($processes as $w => $process) {
			$this->finishWorker($process, $pipes[$w], "worker {$w}");
		}

		$reader = new $class();
		$reader->setStorage($storage);
		$reader->load('shared');
		self::assertSame(3, $reader->getVocabulary()->getTotalDocuments(), $backend . ': only the seed documents remain');
		foreach (['cheap pills common3', 'team meeting election', 'shared w1tok2 results', 'nothing known'] as $probe) {
			self::assertSame($reference->logScores($probe), $reader->logScores($probe), $backend . ' ' . $class . ': ' . $probe);
		}
		// Seed: cheap, pills, team, meeting, election, results.
		self::assertSame(6, $reader->getVocabulary()->getVocabularySize(), $backend . ': the vocabulary shrank by exactly the withdrawn tokens');
		$families = \Belisoful\Prado\Util\Bayesian\TBayesianTokenHistogram::families($storage->loadTokenMeta('shared')['histograms'] ?? null);
		$maintained = $storage->loadTokenHistograms('shared', $families);
		$storage->rebuildTokenHistograms('shared', $families);
		self::assertEquals($maintained, $storage->loadTokenHistograms('shared', $families), $backend . ': the maintained histograms equal a rebuild');
	}

	/**
	 * @return array<string, array{0:string, 1:string}> Each SQL backend with each relaxed histogram mode.
	 */
	public static function relaxedBackends(): array
	{
		$cases = [];
		foreach (['sqlite', 'mysql', 'pgsql'] as $backend) {
			foreach ([TSqlBayesianStorage::HISTOGRAM_DEFERRED, TSqlBayesianStorage::HISTOGRAM_PERIODIC] as $mode) {
				$cases[$backend . ' ' . $mode] = [$backend, $mode];
			}
		}
		return $cases;
	}

	/**
	 * A Complement model that leaves its histograms to maintenance: four workers train it side
	 * by side while a fifth process folds (or recounts) continuously.  Whatever the interleaving,
	 * once training has stopped and maintenance has run, the model scores exactly as a resident
	 * one, and the histograms equal a recount.
	 */
	#[DataProvider('relaxedBackends')]
	public function testMaintenanceRunningBesideTrainersConvergesExactly(string $backend, string $mode): void
	{
		[$storage, $description] = $this->storage($backend);
		self::assertInstanceOf(TSqlBayesianStorage::class, $storage);
		$storage->setHistogramMode($mode);
		$reference = new TComplementNaiveBayes();
		$seed = new TComplementNaiveBayes();
		$seed->setStorage($storage);
		$seed->setName('shared');
		foreach ([['spam', 'cheap pills'], ['ham', 'team meeting cheap'], ['news', 'election results']] as [$category, $document]) {
			$seed->trainOne($category, $document);
			$reference->trainOne($category, $document);
		}
		$seed->save();

		$stopFile = sys_get_temp_dir() . '/bayesian-maintain-stop-' . uniqid('', true);
		$this->_files[] = $stopFile;
		[$maintainer, $maintainerPipe] = $this->startWorker(
			[PHP_BINARY, __DIR__ . '/../../../test_tools/bayesian-maintain-worker.php', json_encode($description), 'shared', $stopFile]
		);

		$categories = ['spam', 'ham', 'news', 'spam'];
		$processes = [];
		$pipes = [];
		try {
			for ($w = 0; $w < self::WORKERS; $w++) {
				$documents = [];
				for ($d = 0; $d < self::DOCUMENTS_PER_WORKER; $d++) {
					$documents[] = "cheap cheap shared w{$w}tok{$d} common{$d}";
					$reference->trainOne($categories[$w], end($documents));
				}
				$command = [
					PHP_BINARY,
					__DIR__ . '/../../../test_tools/bayesian-train-worker.php',
					json_encode($description),
					'shared',
					$categories[$w],
					json_encode($documents),
					TComplementNaiveBayes::class,
				];
				[$processes[$w], $pipes[$w]] = $this->startWorker($command);
			}
			foreach ($processes as $w => $process) {
				$this->finishWorker($process, $pipes[$w], "worker {$w}");
			}
		} finally {
			// A failed trainer must not leave the maintainer running to its deadline: the process
			// resource is released when this frame unwinds, and releasing it waits for the child.
			touch($stopFile);
		}
		$this->finishWorker($maintainer, $maintainerPipe, 'maintenance worker');

		self::assertSame($mode, $storage->loadTokenMeta('shared')['histogramMode'], 'the trainers kept the model in its mode');
		$storage->maintainTokenHistograms('shared');
		self::assertSame(0, $storage->getPendingTokenCount('shared'));
		$reader = new TComplementNaiveBayes();
		$reader->setStorage($storage);
		$reader->load('shared');
		foreach (['cheap pills common3', 'team meeting election', 'shared w1tok2 results', 'nothing known'] as $probe) {
			self::assertSame($reference->logScores($probe), $reader->logScores($probe), $backend . ' ' . $mode . ': ' . $probe);
		}
		$families = \Belisoful\Prado\Util\Bayesian\TBayesianTokenHistogram::families($storage->loadTokenMeta('shared')['histograms'] ?? null);
		$maintained = $storage->loadTokenHistograms('shared', $families);
		$storage->rebuildTokenHistograms('shared', $families);
		self::assertEquals($maintained, $storage->loadTokenHistograms('shared', $families), $backend . ' ' . $mode . ': the maintained histograms equal a recount');
	}
}
