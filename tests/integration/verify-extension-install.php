<?php

/**
 * Composer-extension integration checks.
 *
 * The unit suite registers the extension's error messages and class map by hand, so it cannot
 * prove that a real `composer require` wires them up.  This script runs inside a throwaway
 * consumer project that installed the extension through Composer and asserts the three things
 * `composer.json`'s `extra.prado` section promises:
 *
 * - `error-messages` registers `config/errorMessages.txt`, so `bayesian_*` codes resolve to text.
 * - `class-map` registers the Prado3 short names, so `TNaiveBayesClassifier` resolves to its FQN.
 * - `bootstrap` names the module, so `<module id="belisoful/prado-bayesian"/>` boots it.
 *
 * Each mode runs in its own process because a Prado application is a per-process singleton.
 *
 *     php verify-extension-install.php <mode> <consumer-dir>
 *
 * Modes: `capture` (Composer metadata), `boot` (configure, train, save), `serve` (eager load in
 * a fresh process, then serve a real request).  Diagnostics go to stderr so that in `serve`
 * mode stdout carries only the HTTP response body, which the caller inspects.  Exits non-zero
 * with a message on the first failed check.
 */

use Belisoful\Prado\Util\Bayesian\TBayesianModule;
use Prado\Exceptions\TException;
use Prado\Exceptions\TInvalidDataValueException;
use Prado\Prado;
use Prado\TApplication;
use Prado\TApplicationConfiguration;

$mode = $argv[1] ?? '';
$dir = $argv[2] ?? '';
if ($mode === '' || $dir === '' || !is_file($dir . '/vendor/autoload.php')) {
	fwrite(STDERR, "usage: verify-extension-install.php <capture|boot|serve> <consumer-dir>\n");
	exit(2);
}
require $dir . '/vendor/autoload.php';
chdir($dir);

if ($mode === 'serve') {
	// PHP CLI does not build $_GET from QUERY_STRING; the request the application will serve
	// has to be assembled by hand.  The service id is the first parameter, as in a real URL
	// such as `/index.php?bayesian&text=cheap+pills`.
	parse_str((string) getenv('QUERY_STRING'), $_GET);
	$_REQUEST = $_GET;
}

$checks = 0;
$check = function (bool $ok, string $what) use (&$checks): void {
	$checks++;
	if (!$ok) {
		fwrite(STDERR, "FAIL: {$what}\n");
		exit(1);
	}
	fwrite(STDERR, "  ok: {$what}\n");
};

$application = new TApplication($dir . '/protected', false);
$configuration = new TApplicationConfiguration();
$configuration->captureComposerExtensions();

if ($mode === 'capture') {
	$messages = $configuration->getErrorMessages();
	$check(
		count(array_filter($messages, fn ($f) => str_ends_with($f, 'config/errorMessages.txt'))) === 1,
		'extra.prado.error-messages registered the extension message file'
	);

	$map = $configuration->getClassMap();
	$declared = json_decode(
		(string) file_get_contents($dir . '/vendor/belisoful/prado-bayesian/config/prado-bayesian-classes.json'),
		true
	);
	$check($map !== [], 'extra.prado.class-map registered a class map');
	$check(count($map) >= count($declared), 'the class map holds every declared short name');
	foreach ($declared as $short => $fqn) {
		if (($map[$short] ?? null) !== $fqn) {
			$check(false, "class map entry {$short} => {$fqn}");
		}
	}
	$check(true, 'every class-map entry maps to its declared FQN');

	$check(
		$configuration->getComposerExtensionClass('belisoful/prado-bayesian') === TBayesianModule::class,
		'extra.prado.bootstrap names TBayesianModule'
	);
	// The consumer requires only this extension; the framework has to arrive through it.
	$check(
		is_dir($dir . '/vendor/pradosoft/prado'),
		'pradosoft/prado was installed transitively, without the consumer requiring it'
	);
	$check(
		class_exists(\Prado\TApplication::class),
		'the transitively-installed framework autoloads'
	);

	foreach ($messages as $file) {
		TException::addMessageFile($file);
	}
	Prado::registerClassMap($map);

	$exception = new TInvalidDataValueException('bayesian_alpha_invalid', '0');
	$check(
		$exception->getMessage() !== 'bayesian_alpha_invalid' && str_contains($exception->getMessage(), 'alpha'),
		'a bayesian_* error code resolves to its message text'
	);

	$classifier = Prado::createComponent('TNaiveBayesClassifier');
	$check(
		$classifier instanceof \Belisoful\Prado\Util\Bayesian\Classifier\TNaiveBayesClassifier,
		'the Prado3 short name TNaiveBayesClassifier resolves through the class map'
	);
} else {
	if ($mode === 'serve') {
		// A whole request, exactly as the web entry script would serve it: the application
		// reads its own configuration, boots the module, and dispatches the service.  The
		// response writes the body to stdout on its own; the caller checks it there.
		$application->run();
	} else {
		$configuration->loadFromFile($dir . '/protected/application.xml');
		$application->applyConfiguration($configuration);
	}

	$module = $application->getModule('belisoful/prado-bayesian');
	$check($module instanceof TBayesianModule, 'the bootstrap module booted under its package id');
	$check(
		$module->getClassifier() instanceof \Belisoful\Prado\Util\Bayesian\Classifier\TComplementNaiveBayes,
		'the <classifier> element selected the configured class by short name'
	);
	$check($module->getClassifier()->getAlpha() === 0.5, 'the <classifier> element applied Alpha');
	$check($module->getStorage() !== null, 'the <storage> element created the storage backend');
	$check($module->getClassifier()->getName() === 'comment-spam', 'DefaultClassifier named the model');

	if ($mode === 'boot') {
		$check(!$module->getClassifier()->getIsTrained(), 'a model that was never saved starts untrained');
		$module->getClassifier()->trainOne('spam', 'cheap pills buy now');
		$module->getClassifier()->trainOne('ham', 'meeting agenda tomorrow');
		$module->getClassifier()->save();
		$check($module->getStorage()->exists('comment-spam'), 'the trained model was persisted');
	} else {
		$check($module->getClassifier()->getIsTrained(), 'the saved model was eagerly loaded in a new process');
		$check($module->getClassifier()->classify('cheap pills') === 'spam', 'the restored model classifies');

		$check(
			$application->getService() instanceof \Belisoful\Prado\Web\Services\TBayesianService,
			'the service registered by its class-map short name'
		);
		$check(
			$application->getResponse()->getStatusCode() === 200 &&
				$application->getResponse()->getContentType() === 'application/json',
			'the service answered 200 application/json'
		);
	}
}

fwrite(STDERR, "{$checks} checks passed ({$mode})\n");
