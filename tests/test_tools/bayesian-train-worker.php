<?php

/**
 * Trains documents into an existing per-token model from a separate process, for the
 * concurrency tests.  Several of these run at once against one model; the tests then assert
 * that every count is exact.
 *
 *     php bayesian-train-worker.php <storage-json> <model> <category> <documents-json> [classifier-class] [train|untrain]
 *
 * `storage-json` is `{"backend":"sql","dsn":...,"user":...,"password":...,"table":...}` or
 * `{"backend":"redis","prefix":...,"index":...}`.  Exits non-zero on any failure so the test
 * can tell a crashed worker from a lost update.
 */

require_once(__DIR__ . '/../../vendor/autoload.php');

\Prado\Exceptions\TException::addMessageFile(__DIR__ . '/../../config/errorMessages.txt');

use Belisoful\Prado\Util\Bayesian\Classifier\TNaiveBayesClassifier;
use Belisoful\Prado\Util\Bayesian\Storage\TRedisBayesianStorage;
use Belisoful\Prado\Util\Bayesian\Storage\TSqlBayesianStorage;

[, $storageJson, $model, $category, $documentsJson, $classifierClass, $operation] = $argv + [null, '', '', '', '[]', TNaiveBayesClassifier::class, 'train'];
$config = json_decode((string) $storageJson, true);
$documents = json_decode((string) $documentsJson, true);
if (!is_array($config) || !is_array($documents) || $model === '' || $category === '') {
	fwrite(STDERR, "usage: bayesian-train-worker.php <storage-json> <model> <category> <documents-json>\n");
	exit(2);
}

if (($config['backend'] ?? '') === 'redis') {
	$storage = new TRedisBayesianStorage();
	$storage->setKeyPrefix((string) $config['prefix']);
	$storage->setIndexKey((string) $config['index']);
} else {
	$storage = new TSqlBayesianStorage();
	$storage->setConnectionString((string) $config['dsn']);
	$storage->setUsername((string) ($config['user'] ?? ''));
	$storage->setPassword((string) ($config['password'] ?? ''));
	$storage->setTable((string) ($config['table'] ?? 'bayesian_models'));
}
set_exception_handler(static function (\Throwable $e): void {
	fwrite(STDERR, get_class($e) . ': ' . $e->getMessage() . "\n");
	exit(1);
});
$storage->setMode('token');

if (!is_a($classifierClass, TNaiveBayesClassifier::class, true)) {
	fwrite(STDERR, "not a classifier class: {$classifierClass}\n");
	exit(2);
}
$classifier = new $classifierClass();
$classifier->setStorage($storage);
$classifier->load($model);
foreach ($documents as $document) {
	if ($operation === 'untrain') {
		$classifier->untrainOne($category, (string) $document);
	} else {
		$classifier->trainOne($category, (string) $document);
	}
}
