<?php

use Prado\Data\TDbConnection;
use Prado\Data\TDbTransaction;

/**
 * A transaction that counts its OnCommit and OnRollback events and can throw from OnCommit,
 * as application code attached through TDbConnection::getCurrentTransaction() might.
 * Installed with TDbConnection::setTransactionClass().
 */
class BayesianObservedTransaction extends TDbTransaction
{
	public static int $commits = 0;
	public static int $rollbacks = 0;
	public static bool $throwOnCommit = false;

	public function __construct(TDbConnection $connection)
	{
		parent::__construct($connection);
		$this->attachEventHandler('OnCommit', static function (): void {
			self::$commits++;
			if (self::$throwOnCommit) {
				throw new \RuntimeException('after commit');
			}
		});
		$this->attachEventHandler('OnRollback', static function (): void {
			self::$rollbacks++;
		});
	}

	public static function reset(bool $throwOnCommit = false): void
	{
		self::$commits = 0;
		self::$rollbacks = 0;
		self::$throwOnCommit = $throwOnCommit;
	}

	/** @return array{0: int, 1: int} Commits raised, rollbacks raised. */
	public static function counts(): array
	{
		return [self::$commits, self::$rollbacks];
	}
}
