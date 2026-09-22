<?php

/**
 * Measures what a model costs in each storage layout, so the figures quoted in the
 * documentation can be reproduced rather than trusted.
 *
 *     php tests/benchmark/benchmark-storage.php [vocabulary-size ...]
 *     composer benchmark
 *
 * For each vocabulary size (default 5000, 20000 and 100000 tokens) the script trains a
 * two-category multinomial model whose categories have both seen every token, saves it as a
 * whole-payload JSON file and as a per-token SQLite model, then measures:
 *
 * - the JSON payload size and the PHP memory the decoded model occupies;
 * - the time and memory to load the model through each layout;
 * - the time to train one more document through each layout (whole-payload mode re-saves the
 *   model; per-token mode writes the document's rows).
 *
 * Numbers depend on the machine, the PHP build and the SQLite version; treat them as ratios,
 * not absolutes.  The script needs `ext-pdo` with the SQLite driver and no other backend.
 */
ini_set('memory_limit', '2G');

require_once(__DIR__ . '/../../vendor/autoload.php');

\Prado\Exceptions\TException::addMessageFile(__DIR__ . '/../../config/errorMessages.txt');

use Belisoful\Prado\Util\Bayesian\Classifier\TNaiveBayesClassifier;
use Belisoful\Prado\Util\Bayesian\Storage\TFileBayesianStorage;
use Belisoful\Prado\Util\Bayesian\Storage\TSqlBayesianStorage;

$sizes = array_map('intval', array_slice($argv, 1));
if ($sizes === []) {
	$sizes = [5000, 20000, 100000];
}

$workDir = sys_get_temp_dir() . '/bayesian-benchmark-' . uniqid('', true);
mkdir($workDir, 0o755, true);

/**
 * @param int $vocabulary The number of distinct tokens.
 * @return array<int, array{0:string, 1:string[]}> Training documents: [category, tokens].
 */
function corpus(int $vocabulary): array
{
	// Every category sees the whole vocabulary, which is the worst case for payload size:
	// each token has a row (or hash field) in every category.
	$documents = [];
	$perDocument = 50;
	foreach (['spam', 'ham'] as $category) {
		for ($start = 0; $start < $vocabulary; $start += $perDocument) {
			$tokens = [];
			for ($i = $start; $i < min($vocabulary, $start + $perDocument); $i++) {
				$tokens[] = 'tok' . $i;
			}
			$documents[] = [$category, $tokens];
		}
	}
	return $documents;
}

/**
 * @param callable(): mixed $callback The work to time.
 * @return array{0:float, 1:int, 2:mixed} Milliseconds, peak bytes above the start, and the result.
 */
function measure(callable $callback): array
{
	gc_collect_cycles();
	$memoryBefore = memory_get_usage();
	$peakBefore = memory_get_peak_usage();
	$start = hrtime(true);
	$result = $callback();
	$elapsed = (hrtime(true) - $start) / 1e6;
	$held = memory_get_usage() - $memoryBefore;
	$peak = memory_get_peak_usage() - $peakBefore;
	return [$elapsed, max($held, $peak), $result];
}

function mb(int $bytes): string
{
	return sprintf('%.1f MB', $bytes / 1048576);
}

printf("PHP %s, %s\n", PHP_VERSION, php_uname('m'));
printf("%-10s %-11s %-13s %-14s %-14s %-14s %-14s\n", 'vocabulary', 'json', 'decoded', 'payload load', 'token load', 'payload train', 'token train');

foreach ($sizes as $vocabulary) {
	$documents = corpus($vocabulary);
	$name = 'bench' . $vocabulary;

	$fileStorage = new TFileBayesianStorage();
	$fileStorage->setDirectory($workDir);
	$payload = new TNaiveBayesClassifier();
	$payload->setStorage($fileStorage);
	$payload->setName($name);
	foreach ($documents as [$category, $tokens]) {
		$payload->trainOne($category, $tokens);
	}
	$payload->save();
	$jsonBytes = filesize($workDir . '/' . $name . '.json');
	unset($payload);

	$sqlStorage = new TSqlBayesianStorage();
	$sqlStorage->setConnectionString('sqlite:' . $workDir . '/' . $name . '.sqlite');
	$sqlStorage->setMode(TSqlBayesianStorage::MODE_TOKEN);
	$token = new TNaiveBayesClassifier();
	$token->setStorage($sqlStorage);
	$token->setName($name);
	foreach ($documents as [$category, $tokens]) {
		$token->trainOne($category, $tokens);
	}
	$token->save();
	unset($token, $documents);

	[$payloadLoadMs, $payloadLoadBytes, $payloadLoaded] = measure(static function () use ($fileStorage, $name) {
		$classifier = new TNaiveBayesClassifier();
		$classifier->setStorage($fileStorage);
		$classifier->load($name);
		return $classifier;
	});
	[$decodedMs, $decodedBytes] = measure(static fn () => $fileStorage->load($name));
	[$tokenLoadMs, $tokenLoadBytes, $tokenLoaded] = measure(static function () use ($sqlStorage, $name) {
		$classifier = new TNaiveBayesClassifier();
		$classifier->setStorage($sqlStorage);
		$classifier->load($name);
		return $classifier;
	});
	$probe = 'tok1 tok2 tok3 brand new words';
	if ($payloadLoaded->score($probe) !== $tokenLoaded->score($probe)) {
		fwrite(STDERR, "the two layouts disagree on a score; the benchmark is invalid\n");
		exit(1);
	}
	[$payloadTrainMs] = measure(static function () use ($payloadLoaded) {
		$payloadLoaded->trainOne('spam', 'tok1 tok2 another document');
		$payloadLoaded->save();
	});
	[$tokenTrainMs] = measure(static function () use ($tokenLoaded) {
		$tokenLoaded->trainOne('spam', 'tok1 tok2 another document');
	});

	printf(
		"%-10s %-11s %-13s %-14s %-14s %-14s %-14s\n",
		number_format($vocabulary),
		mb((int) $jsonBytes),
		mb($decodedBytes),
		sprintf('%.1f ms', $payloadLoadMs),
		sprintf('%.1f ms', $tokenLoadMs),
		sprintf('%.1f ms', $payloadTrainMs),
		sprintf('%.1f ms', $tokenTrainMs)
	);
	printf("%-10s %-11s %-13s %-14s %-14s\n", '', '', '', mb($payloadLoadBytes), mb($tokenLoadBytes));
	unset($payloadLoaded, $tokenLoaded, $fileStorage, $sqlStorage);
	gc_collect_cycles();
}

foreach (glob($workDir . '/*') ?: [] as $file) {
	@unlink($file);
}
@rmdir($workDir);
