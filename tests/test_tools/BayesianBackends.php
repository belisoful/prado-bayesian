<?php

/**
 * Backend availability helper for the storage tests.
 *
 * The SQL and Redis backends need an extension (and sometimes a server) that a developer
 * machine may not have, so their tests skip when the backend is missing.  A skipped test
 * still exits zero, which would let a CI run go green without ever touching those backends.
 *
 * Setting the `BAYESIAN_REQUIRE_BACKENDS` environment variable turns every such skip into a
 * failure: CI declares the backends it provides, and a green run then means they were really
 * exercised.  Locally the variable is unset and the tests skip as before.
 */
class BayesianBackends
{
	/**
	 * Skips the current test when a backend is unavailable — or fails it when the environment
	 * declares that the backend must be present.
	 * @param PHPUnit\Framework\TestCase $test The running test.
	 * @param bool $available Whether the backend is available.
	 * @param string $reason Why it is unavailable, shown in the skip/failure message.
	 */
	public static function requireBackend(PHPUnit\Framework\TestCase $test, bool $available, string $reason): void
	{
		if ($available) {
			return;
		}
		if (self::isRequired()) {
			$test::fail('Required backend is unavailable: ' . $reason);
		}
		$test::markTestSkipped($reason);
	}

	/**
	 * @return bool Whether the environment declares that every backend must be present.
	 */
	public static function isRequired(): bool
	{
		$value = getenv('BAYESIAN_REQUIRE_BACKENDS');
		return $value !== false && $value !== '' && $value !== '0';
	}

	/**
	 * Returns a DSN configured through the environment (e.g. `BAYESIAN_MYSQL_DSN`).
	 * @param string $name The environment variable name.
	 * @return ?string The DSN, or null when it is not configured.
	 */
	public static function dsn(string $name): ?string
	{
		$value = getenv($name);
		return ($value === false || $value === '') ? null : $value;
	}

	/**
	 * @param string $name The environment variable name.
	 * @param string $default The value to use when it is not set.
	 * @return string The credential.
	 */
	public static function credential(string $name, string $default = ''): string
	{
		$value = getenv($name);
		return ($value === false) ? $default : $value;
	}
}
