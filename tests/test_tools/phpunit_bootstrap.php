<?php

/**
 * Common settings for all unit tests of the PRADO Bayesian extension.
 *
 * Autoloads the framework and the extension via Composer's PSR-4 map, then registers the
 * extension's error message file (so exception codes resolve) and its Prado3 short-name class
 * map — the two things `extra.prado.error-messages` / `extra.prado.class-map` register for an
 * installed application.
 */

require_once(__DIR__ . '/../../vendor/autoload.php');

\Prado\Exceptions\TException::addMessageFile(__DIR__ . '/../../config/errorMessages.txt');
\Prado\Prado::registerClassMap(
	json_decode((string) file_get_contents(__DIR__ . '/../../config/prado-bayesian-classes.json'), true, 512, JSON_THROW_ON_ERROR)
);
