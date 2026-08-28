<?php

/**
 * BayesianTestApplication class file.
 *
 * Provides a minimal live TApplication for unit tests that exercise code paths requiring
 * Prado::getApplication() (e.g. TBayesianModule::init(), or TSqlBayesianStorage resolving a
 * TDataSourceConfig module by ConnectionID).  The application is created once per process
 * against a throw-away base path holding an empty runtime/ directory.
 */

use Prado\TApplication;

class BayesianTestApplication
{
	/** @var ?TApplication The shared application instance. */
	private static ?TApplication $_application = null;

	/**
	 * Returns the shared test application, creating it on first use.
	 * @return TApplication The application.
	 */
	public static function get(): TApplication
	{
		if (self::$_application === null) {
			$dir = sys_get_temp_dir() . '/bayesian-app-' . uniqid('', true) . '/protected';
			@mkdir($dir . '/runtime', 0o755, true);
			self::$_application = new TApplication($dir, false);
		}
		return self::$_application;
	}
}
