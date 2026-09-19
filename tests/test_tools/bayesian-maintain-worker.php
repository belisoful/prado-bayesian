<?php

/**
 * Keeps a per-token model's histograms maintained from a separate process, for the concurrency
 * tests: the singleton worker a deployment would run for a model in the deferred or periodic
 * histogram mode.
 *
 *     php bayesian-maintain-worker.php <storage-json> <model> <stop-file>
 *
 * Calls maintainTokenHistograms() in a loop until the stop file appears.  Exits non-zero on any
 * failure so the test can tell a crashed worker from a wrong histogram.
 */

require_once(__DIR__ . '/../../vendor/autoload.php');
\Prado\Exceptions\TException::addMessageFile(__DIR__ . '/../../config/errorMessages.txt');

use Belisoful\Prado\Util\Bayesian\Storage\TSqlBayesianStorage;

[, $storageJson, $model, $stopFile] = $argv + [null, '', '', ''];
$config = json_decode((string) $storageJson, true);
if (!is_array($config) || $model === '' || $stopFile === '') {
	fwrite(STDERR, "usage: bayesian-maintain-worker.php <storage-json> <model> <stop-file>\n");
	exit(2);
}
set_exception_handler(static function (\Throwable $e): void {
	fwrite(STDERR, get_class($e) . ': ' . $e->getMessage() . "\n");
	exit(1);
});

$storage = new TSqlBayesianStorage();
$storage->setConnectionString((string) $config['dsn']);
$storage->setUsername((string) ($config['user'] ?? ''));
$storage->setPassword((string) ($config['password'] ?? ''));
$storage->setTable((string) ($config['table'] ?? 'bayesian_models'));
$storage->setMode('token');

$deadline = microtime(true) + 120;
while (!file_exists($stopFile) && microtime(true) < $deadline) {
	$storage->maintainTokenHistograms($model, 50);
	usleep(2000);
}
