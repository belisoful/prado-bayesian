<?php

/**
 * TSqlBayesianStorage class file.
 *
 * @author Brad Anderson <belisoful@icloud.com>
 * @link https://github.com/belisoful/prado-bayesian
 * @license https://github.com/pradosoft/prado/blob/master/LICENSE
 */

namespace Prado\Util\Bayesian\Storage;

use Prado\Data\IDataConnection;
use Prado\Data\TDbConnection;
use Prado\Data\TDbPropertiesTrait;
use Prado\Exceptions\TConfigurationException;
use Prado\Exceptions\TDbException;
use Prado\Exceptions\TInvalidDataValueException;
use Prado\Exceptions\TInvalidOperationException;
use Prado\Prado;
use Prado\TComponent;

/**
 * TSqlBayesianStorage class.
 *
 * SQL-backed storage for SQLite, MySQL/MariaDB, and PostgreSQL through PDO.  A single table
 * holds the models, keyed by name, with the JSON-encoded payload in a large text column.  The
 * schema is created on demand (once per connection) by {@see ensureTable()} so the storage can
 * be used in any application without manual setup; the DDL is driver-aware (`VARCHAR(191)`
 * primary key and `LONGTEXT` payload on MySQL, `TEXT` on SQLite/PostgreSQL, 64-bit
 * `updated_at`).  Set {@see setAutoCreateTable AutoCreateTable} to false to manage the
 * schema yourself.
 *
 * The class uses Prado's {@see TDbConnection} and {@see \Prado\Data\TDbCommand} for all
 * database access — it does not hold a raw PDO instance.  Its connection is configured the way
 * every other database-backed component in the framework configures one, through
 * {@see \Prado\Data\TDbPropertiesTrait}: set {@see getConnectionID ConnectionID} to the id of a
 * {@see \Prado\Data\TDataSourceConfig} module and the storage shares that module's connection,
 * which is the route to use when an application already has a database configured.  Failing
 * that it falls back to its own {@see setConnectionString ConnectionString},
 * {@see setUsername Username} and {@see setPassword Password}, or to a connection handed in
 * through {@see setDbConnection()}.  {@see getHasDbConnection()} and
 * {@see deactivateDbConnection()} come with the trait and behave as they do elsewhere.
 *
 * Unlike a cache, this storage never invents a SQLite file in the runtime path when nothing is
 * configured: a trained model is not scratch data, and runtime directories get cleared.
 *
 * The upsert SQL is driver-aware: MySQL uses `ON DUPLICATE KEY UPDATE`, while SQLite and
 * PostgreSQL use `ON CONFLICT … DO UPDATE`.  A payload that cannot be JSON-encoded throws
 * `bayesian_storage_encode_failed` rather than storing an empty row.
 *
 * ```xml
 * <modules>
 *   <module id="db" class="Prado\Data\TDataSourceConfig">
 *     <database ConnectionString="mysql:host=localhost;dbname=mydb" Username="user" Password="pass" />
 *   </module>
 *   <module id="bayesian" class="Prado\Util\Bayesian\TBayesianModule">
 *     <storage class="Prado\Util\Bayesian\Storage\TSqlBayesianStorage" ConnectionID="db" />
 *   </module>
 * </modules>
 * ```
 *
 * ```php
 * return [
 *     'modules' => [
 *         'db' => [
 *             'class' => 'Prado\Data\TDataSourceConfig',
 *             // 'database', not 'properties': these configure the wrapped connection,
 *             // mirroring the <database> child element of the XML form.
 *             'database' => [
 *                 'ConnectionString' => 'mysql:host=localhost;dbname=mydb',
 *                 'Username' => 'user',
 *                 'Password' => 'pass',
 *             ],
 *         ],
 *         'bayesian' => [
 *             'class' => 'Prado\Util\Bayesian\TBayesianModule',
 *             'storage' => ['class' => 'TSqlBayesianStorage', 'ConnectionID' => 'db'],
 *         ],
 *     ],
 * ];
 * ```
 *
 * @author Brad Anderson <belisoful@icloud.com>
 * @since 0.1.0
 */
class TSqlBayesianStorage extends TComponent implements IBayesianTokenStorage
{
	use TDbPropertiesTrait {
		getDbConnection as private getTraitDbConnection;
	}

	/** The whole model is stored as one JSON payload in a single row. */
	public const MODE_PAYLOAD = 'payload';

	/** The model is stored as one row per (token, category), read a document at a time. */
	public const MODE_TOKEN = 'token';

	/**
	 * @var int How many bind parameters to put in one IN() list.  Every supported driver
	 * accepts far more (SQLite's historical floor is 999), but a document tokenized into
	 * character n-grams can be long, and chunking costs one extra round trip where exceeding
	 * the limit costs an error.
	 */
	private const TOKEN_CHUNK = 500;

	/**
	 * @var ?TDbConnection A connection handed in directly, used when no
	 * {@see getConnectionID ConnectionID} names a {@see \Prado\Data\TDataSourceConfig}.
	 */
	private ?TDbConnection $_customConnection = null;

	/** @var string The table name. */
	private string $_table = 'bayesian_models';

	/** @var string The connection string passed to TDbConnection. */
	private string $_connectionString = '';

	/** @var ?string The database user; null for drivers that do not require one (SQLite). */
	private ?string $_username = null;

	/** @var ?string The database password; null for drivers that do not require one. */
	private ?string $_password = null;

	/** @var string The connection charset. */
	private string $_charset = '';

	/** @var array<string, mixed> Extra TDbConnection attributes. */
	private array $_attributes = [];

	/** @var bool Whether {@see ensureTable()} issues CREATE TABLE IF NOT EXISTS. */
	private bool $_autoCreateTable = true;

	/** @var ?IDataConnection The connection the table was last ensured on. */
	private ?IDataConnection $_tableEnsuredOn = null;

	/** @var string Either {@see MODE_PAYLOAD} or {@see MODE_TOKEN}. */
	private string $_mode = self::MODE_PAYLOAD;

	/**
	 * Throws when the pdo extension is not loaded, since every backend driver needs it.
	 * @throws TConfigurationException When ext-pdo is missing.
	 */
	public function __construct()
	{
		if (!extension_loaded('pdo')) {
			throw new TConfigurationException('bayesian_storage_pdo_missing');
		}
		parent::__construct();
	}

	/**
	 * Activates the connection on every retrieval, as the framework's other database-backed
	 * components do: a storage instance can outlive a connection that was closed elsewhere.
	 * @return ?bool True, meaning activate each time.
	 */
	protected function getDbConnectionActivationType(): ?bool
	{
		return true;
	}

	/**
	 * Returns the connection, opening and activating it on first use.
	 *
	 * Wraps the trait's accessor so a connection that cannot be opened surfaces as this
	 * extension's `bayesian_storage_pdo_connect_failed` rather than a bare database exception —
	 * the activation happens inside the accessor, so this is the one place that can catch it
	 * for every caller.
	 * @throws TConfigurationException When the connection cannot be established or activated.
	 * @return IDataConnection The connection.
	 */
	public function getDbConnection(): IDataConnection
	{
		$connectionID = $this->getConnectionID();
		if ($connectionID !== '' && !$this->getHasDbConnection() && Prado::getApplication() === null) {
			// A ConnectionID names a module, and modules live on the application.  Without one
			// there is nothing to resolve it against; say so rather than letting the lookup
			// fail on null.  This storage is usable outside an application, but only with a
			// connection or a DSN of its own.
			throw new TConfigurationException('bayesian_storage_pdo_connect_failed', "no application to resolve ConnectionID '{$connectionID}'");
		}
		try {
			return $this->getTraitDbConnection();
		} catch (TDbException $e) {
			throw new TConfigurationException('bayesian_storage_pdo_connect_failed', $e->getMessage());
		}
	}

	/**
	 * Builds the connection to use when no {@see getConnectionID ConnectionID} was given —
	 * either one handed in through {@see setDbConnection()}, or one built from this component's
	 * own DSN properties.
	 *
	 * This is the hook {@see \Prado\Data\TDbPropertiesTrait} calls for a connection that does
	 * not come from a {@see \Prado\Data\TDataSourceConfig} module, so the standard
	 * `ConnectionID` route and the self-contained route stay in one order of precedence rather
	 * than two competing ones.
	 * @return ?TDbConnection The connection, or null when neither was configured.
	 */
	protected function getCustomDbConnection(): ?TDbConnection
	{
		if ($this->_customConnection !== null) {
			return $this->_customConnection;
		}
		if ($this->_connectionString === '') {
			return null;
		}
		$connection = new TDbConnection($this->_connectionString, $this->_username ?? '', $this->_password ?? '', $this->_charset);
		foreach ($this->_attributes as $name => $value) {
			$connection->setSubProperty($name, $value);
		}
		return $connection;
	}

	/**
	 * Sets the connection directly, bypassing both `ConnectionID` and the DSN properties.
	 * @param TDbConnection $connection The connection.
	 */
	public function setDbConnection(TDbConnection $connection): void
	{
		$this->_customConnection = $connection;
		$this->deactivateDbConnection(true);
		$this->_tableEnsuredOn = null;
	}

	/**
	 * Returns the error code raised when `ConnectionID` names something that is not a
	 * {@see \Prado\Data\TDataSourceConfig}.
	 * @return string The error code.
	 */
	protected function getConnectionInvalidExceptionKey(): string
	{
		return 'bayesian_storage_pdo_connect_failed';
	}

	/**
	 * Returns the error code raised when neither a `ConnectionID`, a connection, nor a DSN was
	 * configured.
	 * @return string The error code.
	 */
	protected function getConnectionRequiredExceptionKey(): string
	{
		return 'bayesian_storage_pdo_dsn_required';
	}

	/**
	 * Returns null: a trained model is not scratch data, so this storage never invents a SQLite
	 * file in the runtime path the way a cache may.  Runtime directories get cleared, and a
	 * model silently landing in one would look like a backend that works until the day the
	 * models vanish.  Configure a connection explicitly.
	 * @return ?string Always null.
	 */
	protected function getSqliteDatabaseName(): ?string
	{
		return null;
	}

	/**
	 * Returns whether the storage table is created automatically on first use.
	 * @return bool The flag (default true).
	 */
	public function getAutoCreateTable(): bool
	{
		return $this->_autoCreateTable;
	}

	/**
	 * Sets whether the storage table is created automatically on first use.  Disable it when
	 * the schema is managed externally (migrations) or the database user lacks DDL rights.
	 * @param bool $value The flag.
	 */
	public function setAutoCreateTable(bool $value): void
	{
		$this->_autoCreateTable = $value;
	}

	/**
	 * Ensures the connection is active and the storage table exists.
	 * @throws TConfigurationException When the connection cannot be opened.
	 * @throws TDbException When the table cannot be created.
	 */
	private function ensureConnection(): void
	{
		// getDbConnection() activates and translates failures; this only adds the schema step.
		$this->getDbConnection();
		$this->ensureTable();
	}

	/**
	 * Creates the storage table if it does not already exist.  Runs at most once per
	 * connection (DDL forces an implicit commit on MySQL, so it must not run on every call)
	 * and not at all when {@see getAutoCreateTable()} is false.
	 * @throws TDbException When the DDL fails.
	 */
	private function ensureTable(): void
	{
		$connection = $this->getDbConnection();
		if (!$this->_autoCreateTable || $this->_tableEnsuredOn === $connection) {
			return;
		}
		$driver = $connection->getDriverName();
		$connection->createCommand($this->getCreateTableSql($driver))->execute();
		if ($this->_mode === self::MODE_TOKEN) {
			foreach ($this->getCreateTokenTableSql($driver) as $sql) {
				$connection->createCommand($sql)->execute();
			}
		}
		$this->_tableEnsuredOn = $connection;
	}

	/**
	 * Returns the driver-specific `CREATE TABLE IF NOT EXISTS` statement for the storage table.
	 *
	 * MySQL cannot index a `TEXT` column without a prefix length and caps `TEXT` at 64 KB, so
	 * it gets a `VARCHAR(191)` key (utf8mb4-safe within a 767-byte index) and a `LONGTEXT`
	 * payload.  `updated_at` is 64-bit everywhere.
	 * @param string $driver The PDO driver name.
	 * @return string The DDL.
	 */
	public function getCreateTableSql(string $driver): string
	{
		switch ($driver) {
			case 'mysql':
				$columns = 'name VARCHAR(191) NOT NULL PRIMARY KEY, payload LONGTEXT NOT NULL, updated_at BIGINT NOT NULL';
				break;
			case 'pgsql':
				$columns = 'name VARCHAR(191) NOT NULL PRIMARY KEY, payload TEXT NOT NULL, updated_at BIGINT NOT NULL';
				break;
			default:
				$columns = 'name TEXT NOT NULL PRIMARY KEY, payload TEXT NOT NULL, updated_at INTEGER NOT NULL';
				break;
		}
		return sprintf('CREATE TABLE IF NOT EXISTS %s (%s)', $this->_table, $columns);
	}

	/**
	 * Returns the driver-specific DDL for the two extra tables per-token mode uses, as a list
	 * of statements.
	 *
	 * The token table is keyed `(model, token, category)` and additionally indexed on
	 * `(model, token)`, which is the shape {@see loadTokens()} queries: one `IN` list over the
	 * document's tokens, returning every category's row for each.  There is deliberately no
	 * vocabulary table — a token's corpus-wide document frequency is the sum of its
	 * per-category document counts, so it comes out of the same rows and cannot drift.
	 * @param string $driver The PDO driver name.
	 * @return string[] The DDL statements.
	 */
	public function getCreateTokenTableSql(string $driver): array
	{
		if ($driver === 'mysql') {
			$key = 'VARCHAR(191)';
			$int = 'BIGINT';
		} elseif ($driver === 'pgsql') {
			$key = 'VARCHAR(191)';
			$int = 'BIGINT';
		} else {
			$key = 'TEXT';
			$int = 'INTEGER';
		}
		$tokens = $this->_table . '_tokens';
		$categories = $this->_table . '_categories';
		return [
			sprintf(
				'CREATE TABLE IF NOT EXISTS %s (model %s NOT NULL, token %s NOT NULL, category %s NOT NULL,'
					. ' cnt %s NOT NULL, doccnt %s NOT NULL, PRIMARY KEY (model, token, category))',
				$tokens,
				$key,
				$key,
				$key,
				$int,
				$int
			),
			sprintf('CREATE INDEX IF NOT EXISTS %s_lookup ON %s (model, token)', $tokens, $tokens),
			sprintf(
				'CREATE TABLE IF NOT EXISTS %s (model %s NOT NULL, category %s NOT NULL,'
					. ' doc_count %s NOT NULL, total_tokens %s NOT NULL, PRIMARY KEY (model, category))',
				$categories,
				$key,
				$key,
				$int,
				$int
			),
		];
	}

	/**
	 * Returns whether the storage is configured to store models per token.
	 * @return bool Whether per-token mode is active.
	 */
	public function getSupportsTokenLookup(): bool
	{
		return $this->_mode === self::MODE_TOKEN;
	}

	/**
	 * Returns the storage mode.
	 * @return string Either `payload` or `token`.
	 */
	public function getMode(): string
	{
		return $this->_mode;
	}

	/**
	 * Sets the storage mode.
	 *
	 * `payload` (the default) writes the whole model as one JSON row: simple, and the right
	 * choice for a model that fits comfortably in the process.  `token` writes one row per
	 * (token, category) so a classifier can score a document by reading only that document's
	 * tokens — the model is then bounded by the database rather than by PHP's memory limit.
	 *
	 * The two are different layouts, not two views of one: a model written in one mode is not
	 * readable in the other.  Re-save a model after changing this.
	 * @param string $value Either `payload` or `token`.
	 * @throws TInvalidDataValueException When the value is neither mode.
	 */
	public function setMode(string $value): void
	{
		if ($value !== self::MODE_PAYLOAD && $value !== self::MODE_TOKEN) {
			throw new TInvalidDataValueException('bayesian_storage_mode_invalid', $value);
		}
		$this->_mode = $value;
		$this->_tableEnsuredOn = null;
	}

	/**
	 * Ensures the per-token tables exist, and that the storage is in per-token mode.
	 * @throws TInvalidOperationException When the storage is in payload mode.
	 */
	private function ensureTokenMode(): void
	{
		if ($this->_mode !== self::MODE_TOKEN) {
			throw new TInvalidOperationException('bayesian_storage_token_mode_required');
		}
		$this->ensureConnection();
	}

	/**
	 * Splits a token list into chunks small enough for one IN() list.
	 * @param string[] $tokens The tokens.
	 * @return array<int, string[]> The chunks.
	 */
	private function chunkTokens(array $tokens): array
	{
		$unique = array_values(array_unique(array_map('strval', $tokens)));
		return $unique === [] ? [] : array_chunk($unique, self::TOKEN_CHUNK);
	}

	/**
	 * Persists a payload under a model name.
	 * @param string $name The model name.
	 * @param array<string, mixed> $payload The payload.
	 * @throws TConfigurationException When the connection cannot be opened.
	 * @throws TInvalidDataValueException When the payload cannot be JSON-encoded.
	 * @throws TDbException When the statement fails.
	 */
	public function save(string $name, array $payload): void
	{
		$this->ensureConnection();
		$encoded = json_encode($payload, JSON_UNESCAPED_SLASHES | JSON_UNESCAPED_UNICODE);
		if ($encoded === false) {
			throw new TInvalidDataValueException('bayesian_storage_encode_failed', json_last_error_msg());
		}
		$driver = $this->getDbConnection()->getDriverName();
		if ($driver === 'mysql') {
			$sql = sprintf(
				'INSERT INTO %s (name, payload, updated_at) VALUES (:name, :payload, :updated_at) ON DUPLICATE KEY UPDATE payload = VALUES(payload), updated_at = VALUES(updated_at)',
				$this->_table
			);
		} else {
			$sql = sprintf(
				'INSERT INTO %s (name, payload, updated_at) VALUES (:name, :payload, :updated_at) ON CONFLICT(name) DO UPDATE SET payload = excluded.payload, updated_at = excluded.updated_at',
				$this->_table
			);
		}
		$command = $this->getDbConnection()->createCommand($sql);
		$command->bindValue(':name', $name);
		$command->bindValue(':payload', $encoded);
		$command->bindValue(':updated_at', time());
		$command->execute();
	}

	/**
	 * Loads a previously-saved payload by name, or returns null if the name is unknown or
	 * its payload is not valid JSON.
	 * @param string $name The model name.
	 * @throws TConfigurationException When the connection cannot be opened.
	 * @throws TDbException When the statement fails.
	 * @return ?array<string, mixed> The payload, or null.
	 */
	public function load(string $name): ?array
	{
		$this->ensureConnection();
		$sql = sprintf('SELECT payload FROM %s WHERE name = :name', $this->_table);
		$command = $this->getDbConnection()->createCommand($sql);
		$command->bindValue(':name', $name);
		$row = $command->queryRow();
		if ($row === false || $row === null) {
			return null;
		}
		$decoded = json_decode((string) ($row['payload'] ?? ''), true);
		if (!is_array($decoded)) {
			return null;
		}
		return $decoded;
	}

	/**
	 * Returns whether a payload with the given name is currently stored.
	 * @param string $name The model name.
	 * @throws TConfigurationException When the connection cannot be opened.
	 * @throws TDbException When the statement fails.
	 * @return bool Whether the model is present.
	 */
	public function exists(string $name): bool
	{
		$this->ensureConnection();
		$sql = sprintf('SELECT 1 FROM %s WHERE name = :name LIMIT 1', $this->_table);
		$command = $this->getDbConnection()->createCommand($sql);
		$command->bindValue(':name', $name);
		return $command->queryScalar() !== false;
	}

	/**
	 * Removes a previously-saved payload.  Removing a non-existent name is a no-op.
	 * @param string $name The model name.
	 * @throws TConfigurationException When the connection cannot be opened.
	 * @throws TDbException When the statement fails.
	 */
	public function delete(string $name): void
	{
		$this->ensureConnection();
		$sql = sprintf('DELETE FROM %s WHERE name = :name', $this->_table);
		$command = $this->getDbConnection()->createCommand($sql);
		$command->bindValue(':name', $name);
		$command->execute();
		if ($this->_mode === self::MODE_TOKEN) {
			// The metadata row is only part of a per-token model; leaving the token and
			// category rows behind would make the name reappear on the next save with counts
			// from the model that was deleted.
			$this->deleteTokenRows($name);
		}
	}

	/**
	 * Returns the names of all stored models, in alphabetical order.
	 * @throws TConfigurationException When the connection cannot be opened.
	 * @throws TDbException When the statement fails.
	 * @return string[] The model names.
	 */
	public function list(): array
	{
		$this->ensureConnection();
		$sql = sprintf('SELECT name FROM %s ORDER BY name ASC', $this->_table);
		$rows = $this->getDbConnection()->createCommand($sql)->queryAll();
		$names = [];
		foreach ($rows as $row) {
			$names[] = (string) current($row);
		}
		return $names;
	}

	/**
	 * Writes a model in per-token form, replacing whatever was stored under the name.
	 *
	 * The whole write runs in one transaction: a model that is half-replaced is not a model,
	 * and there is no later training call that could reconcile it.
	 * @param string $name The model name.
	 * @param array<string, mixed> $meta The model-level state.
	 * @param array<string, array{documentCount:int, totalTokens:int}> $categories The category scalars.
	 * @param array<string, array<string, array{count:int, docCount:int}>> $tokens The per-token statistics.
	 * @throws TInvalidOperationException When the storage is not in per-token mode.
	 * @throws TInvalidDataValueException When the metadata cannot be JSON-encoded.
	 * @throws TDbException When a statement fails.
	 */
	public function saveTokenModel(string $name, array $meta, array $categories, array $tokens): void
	{
		$this->ensureTokenMode();
		$encoded = json_encode($meta, JSON_UNESCAPED_SLASHES | JSON_UNESCAPED_UNICODE);
		if ($encoded === false) {
			throw new TInvalidDataValueException('bayesian_storage_encode_failed', json_last_error_msg());
		}
		$connection = $this->getDbConnection();
		$transaction = $connection->beginTransaction();
		try {
			$this->deleteTokenRows($name);
			$this->writeMetaRow($name, $encoded);
			foreach ($categories as $category => $stats) {
				$this->writeCategoryRow($name, (string) $category, (int) ($stats['documentCount'] ?? 0), (int) ($stats['totalTokens'] ?? 0));
			}
			$rows = [];
			foreach ($tokens as $token => $perCategory) {
				foreach ($perCategory as $category => $stats) {
					$rows[] = [(string) $token, (string) $category, (int) ($stats['count'] ?? 0), (int) ($stats['docCount'] ?? 0)];
				}
			}
			$this->insertTokenRows($name, $rows);
			$transaction->commit();
		} catch (\Throwable $e) {
			$transaction->rollBack();
			throw $e;
		}
	}

	/**
	 * Returns a model's model-level state, or null when the name is unknown.
	 * @param string $name The model name.
	 * @throws TInvalidOperationException When the storage is not in per-token mode.
	 * @return ?array<string, mixed> The metadata, or null.
	 */
	public function loadTokenMeta(string $name): ?array
	{
		$this->ensureTokenMode();
		return $this->load($name);
	}

	/**
	 * Returns a model's per-category scalars, keyed by category name.
	 * @param string $name The model name.
	 * @throws TInvalidOperationException When the storage is not in per-token mode.
	 * @return array<string, array{documentCount:int, totalTokens:int}> The category scalars.
	 */
	public function loadTokenCategories(string $name): array
	{
		$this->ensureTokenMode();
		$sql = sprintf(
			'SELECT category, doc_count, total_tokens FROM %s_categories WHERE model = :model ORDER BY category ASC',
			$this->_table
		);
		$command = $this->getDbConnection()->createCommand($sql);
		$command->bindValue(':model', $name);
		$out = [];
		foreach ($command->queryAll() as $row) {
			$out[(string) $row['category']] = [
				'documentCount' => (int) $row['doc_count'],
				'totalTokens' => (int) $row['total_tokens'],
			];
		}
		return $out;
	}

	/**
	 * Returns the per-category statistics of the given tokens, in one query per chunk.
	 * @param string $name The model name.
	 * @param string[] $tokens The tokens to fetch.
	 * @throws TInvalidOperationException When the storage is not in per-token mode.
	 * @return array<string, array<string, array{count:int, docCount:int}>> The statistics.
	 */
	public function loadTokens(string $name, array $tokens): array
	{
		$this->ensureTokenMode();
		$out = [];
		foreach ($this->chunkTokens($tokens) as $chunk) {
			$placeholders = [];
			foreach ($chunk as $index => $_) {
				$placeholders[] = ':t' . $index;
			}
			$sql = sprintf(
				'SELECT token, category, cnt, doccnt FROM %s_tokens WHERE model = :model AND token IN (%s)',
				$this->_table,
				implode(', ', $placeholders)
			);
			$command = $this->getDbConnection()->createCommand($sql);
			$command->bindValue(':model', $name);
			foreach ($chunk as $index => $token) {
				$command->bindValue(':t' . $index, $token);
			}
			foreach ($command->queryAll() as $row) {
				$out[(string) $row['token']][(string) $row['category']] = [
					'count' => (int) $row['cnt'],
					'docCount' => (int) $row['doccnt'],
				];
			}
		}
		return $out;
	}

	/**
	 * Applies one training document's deltas without rewriting the model.
	 *
	 * The token increments, the category scalars, and the metadata all move together inside one
	 * transaction, so a failure part-way leaves the model as it was rather than with counts
	 * that no later training could reconcile.
	 * @param string $name The model name.
	 * @param string $category The category the document was filed under.
	 * @param array<string, array{count:int, docCount:int}> $tokenDeltas The per-token increments.
	 * @param array<string, mixed> $meta The updated model-level state.
	 * @param array{documentCount:int, totalTokens:int} $categoryStats The category's updated scalars.
	 * @throws TInvalidOperationException When the storage is not in per-token mode.
	 * @throws TInvalidDataValueException When the metadata cannot be JSON-encoded.
	 * @throws TDbException When a statement fails.
	 */
	public function applyDeltas(string $name, string $category, array $tokenDeltas, array $meta, array $categoryStats): void
	{
		$this->ensureTokenMode();
		$encoded = json_encode($meta, JSON_UNESCAPED_SLASHES | JSON_UNESCAPED_UNICODE);
		if ($encoded === false) {
			throw new TInvalidDataValueException('bayesian_storage_encode_failed', json_last_error_msg());
		}
		$connection = $this->getDbConnection();
		$transaction = $connection->beginTransaction();
		try {
			$this->writeMetaRow($name, $encoded);
			$this->writeCategoryRow($name, $category, (int) ($categoryStats['documentCount'] ?? 0), (int) ($categoryStats['totalTokens'] ?? 0));
			// Read the affected tokens once, add the deltas in PHP, and write them back.  The
			// alternative — a driver-specific "increment or insert" — buys nothing here because
			// the row set is the document's tokens either way, and it would need a second
			// dialect for every backend.
			$existing = $this->loadTokens($name, array_keys($tokenDeltas));
			$rows = [];
			foreach ($tokenDeltas as $token => $delta) {
				$token = (string) $token;
				$current = $existing[$token][$category] ?? ['count' => 0, 'docCount' => 0];
				$rows[] = [
					$token,
					$category,
					(int) $current['count'] + (int) ($delta['count'] ?? 0),
					(int) $current['docCount'] + (int) ($delta['docCount'] ?? 0),
				];
			}
			$this->deleteTokenRowsFor($name, $category, array_keys($tokenDeltas));
			$this->insertTokenRows($name, $rows);
			$transaction->commit();
		} catch (\Throwable $e) {
			$transaction->rollBack();
			throw $e;
		}
	}

	/**
	 * Writes the model's metadata row, replacing any existing one.
	 * @param string $name The model name.
	 * @param string $encoded The JSON-encoded metadata.
	 */
	private function writeMetaRow(string $name, string $encoded): void
	{
		$driver = $this->getDbConnection()->getDriverName();
		if ($driver === 'mysql') {
			$sql = sprintf(
				'INSERT INTO %s (name, payload, updated_at) VALUES (:name, :payload, :updated_at)'
					. ' ON DUPLICATE KEY UPDATE payload = VALUES(payload), updated_at = VALUES(updated_at)',
				$this->_table
			);
		} else {
			$sql = sprintf(
				'INSERT INTO %s (name, payload, updated_at) VALUES (:name, :payload, :updated_at)'
					. ' ON CONFLICT(name) DO UPDATE SET payload = excluded.payload, updated_at = excluded.updated_at',
				$this->_table
			);
		}
		$command = $this->getDbConnection()->createCommand($sql);
		$command->bindValue(':name', $name);
		$command->bindValue(':payload', $encoded);
		$command->bindValue(':updated_at', time());
		$command->execute();
	}

	/**
	 * Writes one category's scalars, replacing any existing row.
	 * @param string $name The model name.
	 * @param string $category The category name.
	 * @param int $documentCount The document count.
	 * @param int $totalTokens The total token occurrences.
	 */
	private function writeCategoryRow(string $name, string $category, int $documentCount, int $totalTokens): void
	{
		$table = $this->_table . '_categories';
		$driver = $this->getDbConnection()->getDriverName();
		if ($driver === 'mysql') {
			$sql = sprintf(
				'INSERT INTO %s (model, category, doc_count, total_tokens) VALUES (:model, :category, :dc, :tt)'
					. ' ON DUPLICATE KEY UPDATE doc_count = VALUES(doc_count), total_tokens = VALUES(total_tokens)',
				$table
			);
		} else {
			$sql = sprintf(
				'INSERT INTO %s (model, category, doc_count, total_tokens) VALUES (:model, :category, :dc, :tt)'
					. ' ON CONFLICT(model, category) DO UPDATE SET doc_count = excluded.doc_count, total_tokens = excluded.total_tokens',
				$table
			);
		}
		$command = $this->getDbConnection()->createCommand($sql);
		$command->bindValue(':model', $name);
		$command->bindValue(':category', $category);
		$command->bindValue(':dc', $documentCount);
		$command->bindValue(':tt', $totalTokens);
		$command->execute();
	}

	/**
	 * Inserts token rows in batches.
	 * @param string $name The model name.
	 * @param array<int, array{0:string,1:string,2:int,3:int}> $rows The rows.
	 */
	private function insertTokenRows(string $name, array $rows): void
	{
		if ($rows === []) {
			return;
		}
		$table = $this->_table . '_tokens';
		foreach (array_chunk($rows, 200) as $batch) {
			$values = [];
			$bind = [];
			foreach ($batch as $index => $row) {
				$values[] = sprintf('(:m%1$d, :t%1$d, :c%1$d, :n%1$d, :d%1$d)', $index);
				$bind[':m' . $index] = $name;
				$bind[':t' . $index] = $row[0];
				$bind[':c' . $index] = $row[1];
				$bind[':n' . $index] = $row[2];
				$bind[':d' . $index] = $row[3];
			}
			$sql = sprintf('INSERT INTO %s (model, token, category, cnt, doccnt) VALUES %s', $table, implode(', ', $values));
			$command = $this->getDbConnection()->createCommand($sql);
			foreach ($bind as $key => $value) {
				$command->bindValue($key, $value);
			}
			$command->execute();
		}
	}

	/**
	 * Removes every per-token and per-category row of a model.
	 * @param string $name The model name.
	 */
	private function deleteTokenRows(string $name): void
	{
		foreach ([$this->_table . '_tokens', $this->_table . '_categories'] as $table) {
			$command = $this->getDbConnection()->createCommand(sprintf('DELETE FROM %s WHERE model = :model', $table));
			$command->bindValue(':model', $name);
			$command->execute();
		}
	}

	/**
	 * Removes the rows of specific tokens in one category, so they can be reinserted with new
	 * counts.
	 * @param string $name The model name.
	 * @param string $category The category name.
	 * @param string[] $tokens The tokens.
	 */
	private function deleteTokenRowsFor(string $name, string $category, array $tokens): void
	{
		$table = $this->_table . '_tokens';
		foreach ($this->chunkTokens($tokens) as $chunk) {
			$placeholders = [];
			foreach ($chunk as $index => $_) {
				$placeholders[] = ':t' . $index;
			}
			$sql = sprintf(
				'DELETE FROM %s WHERE model = :model AND category = :category AND token IN (%s)',
				$table,
				implode(', ', $placeholders)
			);
			$command = $this->getDbConnection()->createCommand($sql);
			$command->bindValue(':model', $name);
			$command->bindValue(':category', $category);
			foreach ($chunk as $index => $token) {
				$command->bindValue(':t' . $index, $token);
			}
			$command->execute();
		}
	}

	/**
	 * Returns the connection string.
	 * @return string The connection string.
	 */
	public function getConnectionString(): string
	{
		return $this->_connectionString;
	}

	/**
	 * Sets the connection string (DSN) for the internal TDbConnection.
	 * @param string $value The connection string.
	 */
	public function setConnectionString(string $value): void
	{
		$this->_connectionString = $value;
		$this->deactivateDbConnection(true);
		$this->_tableEnsuredOn = null;
	}

	/**
	 * Returns the database username.
	 * @return ?string The username.
	 */
	public function getUsername(): ?string
	{
		return $this->_username;
	}

	/**
	 * Sets the database username.
	 * @param ?string $value The username.
	 */
	public function setUsername(?string $value): void
	{
		$this->_username = $value;
		$this->deactivateDbConnection(true);
		$this->_tableEnsuredOn = null;
	}

	/**
	 * Returns the database password.
	 * @return ?string The password.
	 */
	public function getPassword(): ?string
	{
		return $this->_password;
	}

	/**
	 * Sets the database password.
	 * @param ?string $value The password.
	 */
	public function setPassword(?string $value): void
	{
		$this->_password = $value;
		$this->deactivateDbConnection(true);
		$this->_tableEnsuredOn = null;
	}

	/**
	 * Returns the connection charset.
	 * @return string The charset.
	 */
	public function getCharset(): string
	{
		return $this->_charset;
	}

	/**
	 * Sets the connection charset.
	 * @param string $value The charset.
	 */
	public function setCharset(string $value): void
	{
		$this->_charset = $value;
		$this->deactivateDbConnection(true);
		$this->_tableEnsuredOn = null;
	}

	/**
	 * Returns the extra TDbConnection attributes.
	 * @return array<string, mixed> The attributes.
	 */
	public function getAttributes(): array
	{
		return $this->_attributes;
	}

	/**
	 * Sets the extra TDbConnection attributes.
	 * @param array<string, mixed> $value The attributes.
	 */
	public function setAttributes(array $value): void
	{
		$this->_attributes = $value;
		$this->deactivateDbConnection(true);
		$this->_tableEnsuredOn = null;
	}

	/**
	 * Returns the table name.
	 * @return string The table name.
	 */
	public function getTable(): string
	{
		return $this->_table;
	}

	/**
	 * Sets the table name.  The name is interpolated into DDL/DML (an identifier cannot be a
	 * bound parameter), so it is validated against a strict identifier charset to keep it from
	 * becoming a SQL-injection vector.
	 * @param string $value The table name.
	 * @throws TInvalidDataValueException When the name is not a plain SQL identifier.
	 */
	public function setTable(string $value): void
	{
		if (preg_match('/^[A-Za-z_][A-Za-z0-9_]*$/', $value) !== 1) {
			throw new TInvalidDataValueException('bayesian_storage_table_invalid', $value);
		}
		$this->_table = $value;
		$this->_tableEnsuredOn = null;
	}

}
