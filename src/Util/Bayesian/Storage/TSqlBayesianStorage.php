<?php

/**
 * TSqlBayesianStorage class file.
 *
 * @author Brad Anderson <belisoful@icloud.com>
 * @link https://github.com/belisoful/prado-bayesian
 * @license https://github.com/pradosoft/prado/blob/master/LICENSE
 */

namespace Belisoful\Prado\Util\Bayesian\Storage;

use Belisoful\Prado\Util\Bayesian\TBayesianPayload;
use Belisoful\Prado\Util\Bayesian\TBayesianTokenHistogram;
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
 * **Concurrency.**  In `payload` mode a save replaces the whole row, so the last writer wins:
 * two processes that each load, train, and save the same model lose one another's documents.
 * Payload mode is a single-writer layout.  In `token` mode every training write is an atomic
 * in-database increment (`cnt = cnt + delta`), the vocabulary size is derived from the rows a
 * write actually inserted, and the model-level counters are incremented rather than rewritten,
 * so any number of web requests and background workers may train one model at once without
 * losing counts.  That is the mode to use when training happens from concurrent requests.
 *
 * A model whose metadata names histogram families (`histograms`, written by the Bernoulli and
 * Complement classifiers) also keeps those {@see TBayesianTokenHistogram} histograms in a
 * `_hist` table, moved inside the same transaction as the counts.  Such a write first locks the
 * model's counters row, so training writes to one such model run one after another — each is a
 * few statements — while models without histograms are written as concurrently as before.
 *
 * The per-token layout carries a `layoutVersion` in the model's metadata row.  A model written
 * by 0.1.0 (layout 1) is upgraded in place on first read; a layout newer than this release
 * understands is refused with `bayesian_storage_layout_unsupported` rather than misread.
 *
 * ```xml
 * <modules>
 *   <module id="db" class="Prado\Data\TDataSourceConfig">
 *     <database ConnectionString="mysql:host=localhost;dbname=mydb" Username="user" Password="pass" />
 *   </module>
 *   <module id="bayesian" class="Belisoful\Prado\Util\Bayesian\TBayesianModule">
 *     <storage class="Belisoful\Prado\Util\Bayesian\Storage\TSqlBayesianStorage" ConnectionID="db" />
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
 *             'class' => 'Belisoful\Prado\Util\Bayesian\TBayesianModule',
 *             'storage' => ['class' => 'TSqlBayesianStorage', 'ConnectionID' => 'db'],
 *         ],
 *     ],
 * ];
 * ```
 *
 * @author Brad Anderson <belisoful@icloud.com>
 * @since 0.1.0
 */
class TSqlBayesianStorage extends TComponent implements IBayesianHistogramStorage
{
	use TDbPropertiesTrait {
		getDbConnection as private getTraitDbConnection;
	}

	/** The whole model is stored as one JSON payload in a single row. */
	public const MODE_PAYLOAD = 'payload';

	/** The model is stored as one row per (token, category), read a document at a time. */
	public const MODE_TOKEN = 'token';

	/**
	 * The per-token layout this release writes.  Layout 1 (0.1.0) kept the model counters in
	 * the metadata JSON and had no vocabulary or counters table; layout 2 derives the counters
	 * from the rows.  Stored in the metadata row as `layoutVersion`.
	 * @since 0.2.0
	 */
	public const TOKEN_LAYOUT_VERSION = 2;

	/**
	 * @var int How many bind parameters to put in one IN() list.  Every supported driver
	 * accepts far more (SQLite's historical floor is 999), but a document tokenized into
	 * character n-grams can be long, and chunking costs one extra round trip where exceeding
	 * the limit costs an error.
	 */
	private const TOKEN_CHUNK = 500;

	/** @var int How many rows to put in one multi-row INSERT. */
	private const ROW_CHUNK = 200;

	/**
	 * @var int The widest token the token column holds, in characters.  MySQL and PostgreSQL
	 * store tokens in `VARCHAR(191)`; a longer token is stored under a surrogate key of this
	 * width (see {@see encodeToken()}).
	 */
	private const MAX_TOKEN_LENGTH = 191;

	/** @var int How many leading characters of an over-long token its surrogate key keeps. */
	private const TOKEN_PREFIX_LENGTH = 150;

	/**
	 * @var int The longest table name accepted.  PostgreSQL truncates identifiers at 63 bytes
	 * and MySQL refuses them past 64; the longest derived name is `<table>_tokens_lookup`.
	 */
	private const MAX_TABLE_LENGTH = 48;

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
	 * @var array<string, true> The models this instance has verified to be at
	 * {@see TOKEN_LAYOUT_VERSION}, so the check costs one read per model per instance.
	 */
	private array $_layoutChecked = [];

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
	 * Returns null: the trait hands the connection over unopened and {@see getDbConnection()}
	 * activates it itself, which is where an open failure can be translated into this
	 * extension's error code.
	 * @return ?bool Null, meaning the trait does not activate.
	 */
	protected function getDbConnectionActivationType(): ?bool
	{
		return null;
	}

	/**
	 * Returns the connection, opening and activating it on every retrieval, as the framework's
	 * other database-backed components do: a storage instance can outlive a connection that was
	 * closed elsewhere.
	 *
	 * A connection that cannot be opened surfaces as this extension's
	 * `bayesian_storage_pdo_connect_failed` rather than a bare database exception.
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
		$connection = $this->getTraitDbConnection();
		try {
			$connection->setActive(true);
		} catch (TDbException $e) {
			throw new TConfigurationException('bayesian_storage_pdo_connect_failed', $e->getMessage());
		}
		return $connection;
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
		$this->forgetSchema();
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
	 * Forgets what is known about the schema and the models on the current connection, so the
	 * next use checks again.  Called whenever the connection or the table changes.
	 */
	private function forgetSchema(): void
	{
		$this->_tableEnsuredOn = null;
		$this->_layoutChecked = [];
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
	 * Returns the driver-specific DDL for the extra tables per-token mode uses, as a list of
	 * statements.
	 *
	 * The token table is keyed `(model, token, category)` and additionally indexed on
	 * `(model, token)`, which is the shape {@see loadTokens()} queries: one `IN` list over the
	 * document's tokens, returning every category's row for each.  A token's corpus-wide
	 * document frequency is the sum of its per-category document counts, so it comes out of
	 * the same rows and cannot drift, as is the model's document total, which is the sum of the
	 * category document counts.
	 *
	 * The vocabulary table holds one row per distinct token of a model, so the number of tokens
	 * a write really added to the vocabulary is the number of rows its insert-ignore inserted;
	 * the counters table holds the resulting vocabulary size, moved by atomic increments.  Both
	 * exist so that concurrent writers add to the truth rather than each overwriting it with
	 * their own snapshot.
	 * @param string $driver The PDO driver name.
	 * @return string[] The DDL statements.
	 */
	public function getCreateTokenTableSql(string $driver): array
	{
		if ($driver === 'mysql' || $driver === 'pgsql') {
			$key = 'VARCHAR(191)';
			$int = 'BIGINT';
		} else {
			$key = 'TEXT';
			$int = 'INTEGER';
		}
		$tokens = $this->_table . '_tokens';
		$categories = $this->_table . '_categories';
		$vocabulary = $this->_table . '_vocab';
		$counters = $this->_table . '_counters';
		$histograms = $this->_table . '_hist';
		// MySQL has no CREATE INDEX IF NOT EXISTS; it declares the lookup index inline instead.
		$inlineIndex = $driver === 'mysql' ? sprintf(', INDEX %s_lookup (model, token)', $tokens) : '';
		$statements = [
			sprintf(
				'CREATE TABLE IF NOT EXISTS %s (model %s NOT NULL, token %s NOT NULL, category %s NOT NULL,'
					. ' cnt %s NOT NULL, doccnt %s NOT NULL, PRIMARY KEY (model, token, category)%s)',
				$tokens,
				$key,
				$key,
				$key,
				$int,
				$int,
				$inlineIndex
			),
		];
		if ($driver !== 'mysql') {
			$statements[] = sprintf('CREATE INDEX IF NOT EXISTS %s_lookup ON %s (model, token)', $tokens, $tokens);
		}
		$statements[] = sprintf(
			'CREATE TABLE IF NOT EXISTS %s (model %s NOT NULL, category %s NOT NULL,'
				. ' doc_count %s NOT NULL, total_tokens %s NOT NULL, PRIMARY KEY (model, category))',
			$categories,
			$key,
			$key,
			$int,
			$int
		);
		$statements[] = sprintf(
			'CREATE TABLE IF NOT EXISTS %s (model %s NOT NULL, token %s NOT NULL, PRIMARY KEY (model, token))',
			$vocabulary,
			$key,
			$key
		);
		$statements[] = sprintf(
			'CREATE TABLE IF NOT EXISTS %s (model %s NOT NULL PRIMARY KEY, vocabulary_size %s NOT NULL)',
			$counters,
			$key,
			$int
		);
		// One row per histogram cell: how many vocabulary tokens share the value `val` in the
		// family `fam` (see TBayesianTokenHistogram) for a category, or for the model when the
		// category is empty.
		$statements[] = sprintf(
			'CREATE TABLE IF NOT EXISTS %s (model %s NOT NULL, fam CHAR(1) NOT NULL, category %s NOT NULL,'
				. ' val %s NOT NULL, cnt %s NOT NULL, PRIMARY KEY (model, fam, category, val))',
			$histograms,
			$key,
			$key,
			$int,
			$int
		);
		return $statements;
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
	 * choice for a model that fits comfortably in the process and has one writer.  `token`
	 * writes one row per (token, category) so a classifier can score a document by reading only
	 * that document's tokens — the model is then bounded by the database rather than by PHP's
	 * memory limit — and training is an atomic increment, so many processes may train at once.
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
		$this->forgetSchema();
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
	 * Returns the key a token is stored under.
	 *
	 * A token up to {@see MAX_TOKEN_LENGTH} characters of valid UTF-8 is its own key.  A longer
	 * one — a URL from a regex tokenizer, a run of characters — would not fit MySQL's or
	 * PostgreSQL's `VARCHAR(191)` column, and a token that is not valid UTF-8 would be refused
	 * by PostgreSQL outright.  Those are stored under a surrogate: the first
	 * {@see TOKEN_PREFIX_LENGTH} characters, a `~`, and 40 hex digits of the token's SHA-1, so
	 * two distinct long tokens stay distinct and the same token always maps to the same row.
	 * The encoding is applied on every path (save, training, lookup), so callers never see it.
	 * @param string $token The token.
	 * @return string The storage key.
	 */
	private static function encodeToken(string $token): string
	{
		$valid = mb_check_encoding($token, 'UTF-8');
		if ($valid && strlen($token) <= self::MAX_TOKEN_LENGTH) {
			// Bytes bound characters, so this is the common case with no multibyte scan.
			return $token;
		}
		if (!$valid) {
			return '~' . sha1($token);
		}
		if (mb_strlen($token, 'UTF-8') <= self::MAX_TOKEN_LENGTH) {
			return $token;
		}
		return mb_substr($token, 0, self::TOKEN_PREFIX_LENGTH, 'UTF-8') . '~' . substr(sha1($token), 0, 40);
	}

	/**
	 * Encodes a token list into a map of storage key => original token, dropping repeats.
	 * @param string[] $tokens The tokens.
	 * @return array<string, string> The storage keys and the tokens they stand for.
	 */
	private static function encodeTokens(array $tokens): array
	{
		$map = [];
		foreach ($tokens as $token) {
			$token = (string) $token;
			$map[self::encodeToken($token)] = $token;
		}
		return $map;
	}

	/**
	 * Encodes the model-level metadata for the per-token layout.
	 *
	 * The document total and vocabulary size are not stored in the metadata: they are counters
	 * the storage derives from its own rows, and a value from a writer's snapshot would be stale
	 * the moment another writer trained.  The layout version travels with the metadata so a
	 * later release can recognize what it is reading.
	 * @param array<string, mixed> $meta The metadata.
	 * @throws TInvalidDataValueException When the metadata cannot be JSON-encoded.
	 * @return string The JSON.
	 */
	private function encodeMeta(array $meta): string
	{
		unset($meta['totalDocuments'], $meta['vocabularySize']);
		$meta['layoutVersion'] = self::TOKEN_LAYOUT_VERSION;
		$encoded = json_encode($meta, JSON_UNESCAPED_SLASHES | JSON_UNESCAPED_UNICODE);
		if ($encoded === false) {
			throw new TInvalidDataValueException('bayesian_storage_encode_failed', json_last_error_msg());
		}
		return $encoded;
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
		$this->writeMetaRow($name, $encoded);
		unset($this->_layoutChecked[$name]);
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
		if (!is_array($row)) {
			return null;
		}
		$decoded = json_decode(TBayesianPayload::string($row['payload'] ?? null), true);
		if (!is_array($decoded)) {
			return null;
		}
		return TBayesianPayload::map($decoded);
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
		unset($this->_layoutChecked[$name]);
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
	 * and there is no later training call that could reconcile it.  The counters are computed
	 * from the rows written, not taken from `$meta`.
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
		$encoded = $this->encodeMeta($meta);
		$rows = [];
		foreach ($tokens as $token => $perCategory) {
			$key = self::encodeToken((string) $token);
			foreach ($perCategory as $category => $stats) {
				$rows[] = [$key, (string) $category, max(0, (int) $stats['count']), max(0, (int) $stats['docCount'])];
			}
		}
		$connection = $this->getDbConnection();
		$transaction = $connection->beginTransaction();
		try {
			$this->deleteTokenRows($name);
			$this->writeMetaRow($name, $encoded);
			foreach ($categories as $category => $stats) {
				$this->incrementCategoryRow($name, (string) $category, max(0, (int) $stats['documentCount']), max(0, (int) $stats['totalTokens']));
			}
			$this->insertTokenRows($name, $rows);
			$this->applyHistogramDelta($name, TBayesianTokenHistogram::build($tokens, TBayesianTokenHistogram::families($meta['histograms'] ?? null)));
			$this->incrementVocabularySize($name, $this->insertVocabulary($name, array_keys(self::encodeTokens(array_keys($tokens)))));
			$transaction->commit();
		} catch (\Throwable $e) {
			$transaction->rollBack();
			throw $e;
		}
		$this->_layoutChecked[$name] = true;
	}

	/**
	 * Returns a model's model-level state, or null when the name is unknown.
	 *
	 * The document total is the sum of the category document counts and the vocabulary size
	 * comes from the counters table, not from whatever the last writer put in the metadata, so
	 * both reflect every write that has landed.
	 * @param string $name The model name.
	 * @throws TInvalidOperationException When the storage is not in per-token mode.
	 * @throws TInvalidDataValueException When the model's layout is newer than this release reads.
	 * @return ?array<string, mixed> The metadata, or null.
	 */
	public function loadTokenMeta(string $name): ?array
	{
		$this->ensureTokenMode();
		$this->ensureLayout($name);
		$meta = $this->load($name);
		if ($meta === null) {
			return null;
		}
		$counters = $this->loadCounters($name);
		$meta['totalDocuments'] = $counters['totalDocuments'];
		$meta['vocabularySize'] = $counters['vocabularySize'];
		$meta['layoutVersion'] = self::TOKEN_LAYOUT_VERSION;
		return $meta;
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
		return $this->selectTokenRows($name, self::encodeTokens($tokens), false);
	}

	/**
	 * Reads the rows of the given tokens in every category.
	 * @param string $name The model name.
	 * @param array<string, string> $map The tokens, keyed by storage key.
	 * @param bool $forUpdate Whether to read with row locks, which on MySQL also reads the
	 * latest committed rows rather than the transaction's snapshot.
	 * @return array<string, array<string, array{count:int, docCount:int}>> The statistics, keyed by token then category.
	 */
	private function selectTokenRows(string $name, array $map, bool $forUpdate): array
	{
		$suffix = $forUpdate && $this->getDbConnection()->getDriverName() !== 'sqlite' ? ' FOR UPDATE' : '';
		$out = [];
		foreach (array_chunk(array_keys($map), self::TOKEN_CHUNK) as $chunk) {
			$placeholders = [];
			foreach ($chunk as $index => $_) {
				$placeholders[] = ':t' . $index;
			}
			$sql = sprintf(
				'SELECT token, category, cnt, doccnt FROM %s_tokens WHERE model = :model AND token IN (%s)%s',
				$this->_table,
				implode(', ', $placeholders),
				$suffix
			);
			$command = $this->getDbConnection()->createCommand($sql);
			$command->bindValue(':model', $name);
			foreach ($chunk as $index => $key) {
				$command->bindValue(':t' . $index, $key);
			}
			foreach ($command->queryAll() as $row) {
				$token = $map[(string) $row['token']] ?? (string) $row['token'];
				$out[$token][(string) $row['category']] = [
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
	 * Every count moves by an in-database increment inside one transaction, so two writers
	 * applying deltas to the same rows both land, in either order.  A negative delta is
	 * accepted and clamped at zero, which is what withdrawing a document needs.  The vocabulary
	 * growth is the number of tokens the write really added, taken from the rows the vocabulary
	 * insert inserted; a token that no document contains any more after a negative delta is
	 * dropped from the vocabulary again, so the model ends as it would have without the
	 * document.
	 *
	 * The tokens are written in sorted order so two concurrent writers lock rows in the same
	 * sequence and cannot deadlock on each other.
	 * @param string $name The model name.
	 * @param string $category The category the document was filed under.
	 * @param array<string, array{count?:int, docCount?:int}> $tokenDeltas The per-token increments.
	 * @param array<string, mixed> $meta The model-level state to store alongside; `totalDocuments`
	 * and `vocabularySize` in it are ignored, and an empty array leaves the stored metadata as it is.
	 * @param array{documentCount?:int, totalTokens?:int} $categoryStats The category's increments.
	 * @throws TInvalidOperationException When the storage is not in per-token mode.
	 * @throws TInvalidDataValueException When the metadata cannot be JSON-encoded, or the model's
	 * layout is newer than this release reads.
	 * @throws TDbException When a statement fails.
	 */
	public function applyDeltas(string $name, string $category, array $tokenDeltas, array $meta, array $categoryStats): void
	{
		$this->ensureTokenMode();
		$this->ensureLayout($name);
		$encoded = $meta === [] ? null : $this->encodeMeta($meta);
		$rows = [];
		foreach ($tokenDeltas as $token => $delta) {
			$rows[self::encodeToken((string) $token)] = [(int) ($delta['count'] ?? 0), (int) ($delta['docCount'] ?? 0)];
		}
		ksort($rows, SORT_STRING);
		$documentDelta = (int) ($categoryStats['documentCount'] ?? 0);
		$tokenDelta = (int) ($categoryStats['totalTokens'] ?? 0);
		// The histograms to keep are the ones the metadata names: the metadata being written
		// when there is one, the stored metadata otherwise.  A family the writer names but the
		// store does not hold yet is rebuilt after the write rather than moved.
		$families = $meta === [] ? null : TBayesianTokenHistogram::families($meta['histograms'] ?? null);
		$stored = $families === [] ? [] : $this->storedFamilies($name);
		$families ??= $stored;
		$rebuild = array_diff($families, $stored) !== [];
		$keys = array_combine(array_keys($rows), array_keys($rows));
		$connection = $this->getDbConnection();
		$transaction = $connection->beginTransaction();
		try {
			$before = [];
			if ($families !== []) {
				$this->lockModel($name);
				$before = $rebuild ? [] : $this->selectTokenRows($name, $keys, true);
			}
			$newTokens = $this->insertVocabulary($name, array_keys($rows));
			$this->incrementTokenRows($name, $category, $rows);
			$this->incrementCategoryRow($name, $category, $documentDelta, $tokenDelta);
			$this->incrementVocabularySize($name, $newTokens - $this->pruneVocabulary($name, $rows));
			if ($rebuild) {
				$this->rebuildHistogramRows($name, $families);
			} elseif ($families !== []) {
				$this->applyHistogramDelta($name, TBayesianTokenHistogram::delta($before, $this->selectTokenRows($name, $keys, true), $families));
			}
			if ($encoded !== null) {
				$this->writeMetaRow($name, $encoded);
			}
			$transaction->commit();
		} catch (\Throwable $e) {
			$transaction->rollBack();
			throw $e;
		}
	}

	/**
	 * Returns a model's stored histograms of the given families.
	 * @param string $name The model name.
	 * @param string[] $families The family letters wanted.
	 * @throws TInvalidOperationException When the storage is not in per-token mode.
	 * @throws TDbException When the statement fails.
	 * @return array<string, array<string, array<int, int>>> The histograms, as family =>
	 * category => value => token count.
	 * @since 0.2.0
	 */
	public function loadTokenHistograms(string $name, array $families): array
	{
		$this->ensureTokenMode();
		$families = TBayesianTokenHistogram::families($families);
		if ($families === []) {
			return [];
		}
		$placeholders = [];
		foreach ($families as $index => $_) {
			$placeholders[] = ':f' . $index;
		}
		$command = $this->getDbConnection()->createCommand(sprintf(
			'SELECT fam, category, val, cnt FROM %s_hist WHERE model = :model AND fam IN (%s) AND cnt > 0',
			$this->_table,
			implode(', ', $placeholders)
		));
		$command->bindValue(':model', $name);
		foreach ($families as $index => $family) {
			$command->bindValue(':f' . $index, $family);
		}
		$out = [];
		foreach ($command->queryAll() as $row) {
			$out[(string) $row['fam']][(string) $row['category']][(int) $row['val']] = (int) $row['cnt'];
		}
		return $out;
	}

	/**
	 * Recomputes histogram families from the token rows with grouped queries, so the work
	 * happens in the database and only the histogram itself comes back.  The families are
	 * added to the ones the model's metadata already names.
	 * @param string $name The model name.
	 * @param string[] $families The family letters to build.
	 * @throws TInvalidOperationException When the storage is not in per-token mode.
	 * @throws TInvalidDataValueException When the model's layout is newer than this release reads.
	 * @throws TDbException When a statement fails.
	 * @since 0.2.0
	 */
	public function rebuildTokenHistograms(string $name, array $families): void
	{
		$this->ensureTokenMode();
		$this->ensureLayout($name);
		$meta = $this->load($name);
		if ($meta === null) {
			return;
		}
		$families = TBayesianTokenHistogram::families(array_merge($families, TBayesianTokenHistogram::families($meta['histograms'] ?? null)));
		$transaction = $this->getDbConnection()->beginTransaction();
		try {
			$this->lockModel($name);
			$this->rebuildHistogramRows($name, $families);
			// Re-read under the lock: a trainer may have rewritten the metadata meanwhile.
			$meta = $this->load($name) ?? $meta;
			$meta['histograms'] = $families;
			$this->writeMetaRow($name, $this->encodeMeta($meta));
			$transaction->commit();
		} catch (\Throwable $e) {
			$transaction->rollBack();
			throw $e;
		}
	}

	/**
	 * Returns the histogram families the stored metadata of a model names.
	 * @param string $name The model name.
	 * @return string[] The family letters.
	 */
	private function storedFamilies(string $name): array
	{
		return TBayesianTokenHistogram::families(($this->load($name) ?? [])['histograms'] ?? null);
	}

	/**
	 * Takes the model's write lock for the rest of the transaction by touching its counters
	 * row.  Moving a histogram needs a token's rows in every category to hold still between
	 * reading and writing them, and one lock per model gives that on every driver without any
	 * lock-ordering between tokens.
	 * @param string $name The model name.
	 */
	private function lockModel(string $name): void
	{
		$this->incrementVocabularySize($name, 0);
	}

	/**
	 * Replaces the rows of the given families with a recount from the token rows.  Runs inside
	 * the caller's transaction, under the model lock.
	 * @param string $name The model name.
	 * @param string[] $families The family letters.
	 */
	private function rebuildHistogramRows(string $name, array $families): void
	{
		$tokens = $this->_table . '_tokens';
		// The vocabulary is the tokens some document contains; `x` gives each its global count.
		$vocabulary = sprintf('(SELECT token, SUM(cnt) AS g FROM %s WHERE model = :m1 GROUP BY token HAVING SUM(doccnt) > 0) x', $tokens);
		$queries = [
			TBayesianTokenHistogram::FAMILY_DOCUMENTS => sprintf(
				'SELECT category, doccnt AS val, COUNT(*) AS n FROM %s WHERE model = :m1 AND doccnt > 0 GROUP BY category, doccnt',
				$tokens
			),
			TBayesianTokenHistogram::FAMILY_GLOBAL => sprintf("SELECT '' AS category, x.g AS val, COUNT(*) AS n FROM %s GROUP BY x.g", $vocabulary),
			TBayesianTokenHistogram::FAMILY_CATEGORY_GLOBAL => sprintf(
				'SELECT t.category AS category, x.g AS val, COUNT(*) AS n FROM %s t JOIN %s ON x.token = t.token'
					. ' WHERE t.model = :m2 AND t.cnt > 0 GROUP BY t.category, x.g',
				$tokens,
				$vocabulary
			),
			TBayesianTokenHistogram::FAMILY_COMPLEMENT => sprintf(
				'SELECT t.category AS category, x.g - t.cnt AS val, COUNT(*) AS n FROM %s t JOIN %s ON x.token = t.token'
					. ' WHERE t.model = :m2 AND t.cnt > 0 GROUP BY t.category, x.g - t.cnt',
				$tokens,
				$vocabulary
			),
		];
		$flat = [];
		foreach ($families as $family) {
			$command = $this->getDbConnection()->createCommand(sprintf('DELETE FROM %s_hist WHERE model = :model AND fam = :fam', $this->_table));
			$command->bindValue(':model', $name);
			$command->bindValue(':fam', $family);
			$command->execute();
			$sql = $queries[$family];
			$command = $this->getDbConnection()->createCommand($sql);
			$command->bindValue(':m1', $name);
			if (strpos($sql, ':m2') !== false) {
				$command->bindValue(':m2', $name);
			}
			foreach ($command->queryAll() as $row) {
				$flat[TBayesianTokenHistogram::key($family, (string) $row['category'], (int) $row['val'])] = (int) $row['n'];
			}
		}
		$this->applyHistogramDelta($name, $flat);
	}

	/**
	 * Adds increments to histogram cells and removes the cells that reach zero.  The cells are
	 * written in sorted order so two writers take their row locks in the same sequence.
	 * @param string $name The model name.
	 * @param array<string, int> $delta The increments, keyed by {@see TBayesianTokenHistogram::key()}.
	 */
	private function applyHistogramDelta(string $name, array $delta): void
	{
		$delta = array_filter($delta);
		if ($delta === []) {
			return;
		}
		ksort($delta, SORT_STRING);
		$table = $this->_table . '_hist';
		$update = $this->getDbConnection()->getDriverName() === 'mysql'
			? 'ON DUPLICATE KEY UPDATE cnt = cnt + VALUES(cnt)'
			: sprintf('ON CONFLICT (model, fam, category, val) DO UPDATE SET cnt = %s.cnt + excluded.cnt', $table);
		foreach (array_chunk($delta, self::ROW_CHUNK, true) as $batch) {
			$values = [];
			$bind = [];
			$index = 0;
			foreach ($batch as $key => $change) {
				$parts = TBayesianTokenHistogram::parseKey((string) $key);
				if ($parts === null) {
					continue;
				}
				$values[] = sprintf('(:m%1$d, :f%1$d, :c%1$d, :v%1$d, :n%1$d)', $index);
				$bind[':m' . $index] = $name;
				$bind[':f' . $index] = $parts[0];
				$bind[':c' . $index] = $parts[1];
				$bind[':v' . $index] = $parts[2];
				$bind[':n' . $index] = $change;
				$index++;
			}
			if ($values === []) {
				continue;
			}
			$command = $this->getDbConnection()->createCommand(sprintf('INSERT INTO %s (model, fam, category, val, cnt) VALUES %s %s', $table, implode(', ', $values), $update));
			foreach ($bind as $placeholder => $value) {
				$command->bindValue($placeholder, $value);
			}
			$command->execute();
		}
		$command = $this->getDbConnection()->createCommand(sprintf('DELETE FROM %s WHERE model = :model AND cnt <= 0', $table));
		$command->bindValue(':model', $name);
		$command->execute();
	}

	/**
	 * Checks the layout version of a model once per instance, upgrading a model written by an
	 * earlier release and refusing one written by a later release.
	 * @param string $name The model name.
	 * @throws TInvalidDataValueException When the model's layout is newer than this release reads.
	 */
	private function ensureLayout(string $name): void
	{
		if (isset($this->_layoutChecked[$name])) {
			return;
		}
		$meta = $this->load($name);
		if ($meta === null) {
			// No metadata row means no model; there is nothing to upgrade.  The check is not
			// remembered, so a model saved later under the name is examined then.
			return;
		}
		$version = TBayesianPayload::int($meta['layoutVersion'] ?? null, 1);
		if ($version > self::TOKEN_LAYOUT_VERSION) {
			throw new TInvalidDataValueException('bayesian_storage_layout_unsupported', $name, (string) $version, (string) self::TOKEN_LAYOUT_VERSION);
		}
		if ($version < self::TOKEN_LAYOUT_VERSION) {
			$this->upgradeLayout($name, $meta);
		}
		$this->_layoutChecked[$name] = true;
	}

	/**
	 * Brings a layout-1 model (0.1.0) to the current layout in one transaction: tokens too long
	 * for the column are re-keyed, the vocabulary table is filled from the distinct tokens, the
	 * counters are computed from the rows, and the metadata is stamped.  Idempotent, so a run
	 * that was interrupted is simply finished by the next.
	 * @param string $name The model name.
	 * @param array<string, mixed> $meta The model's current metadata.
	 */
	private function upgradeLayout(string $name, array $meta): void
	{
		$connection = $this->getDbConnection();
		$tokens = $this->_table . '_tokens';
		$transaction = $connection->beginTransaction();
		try {
			$command = $connection->createCommand(sprintf('SELECT DISTINCT token FROM %s WHERE model = :model AND LENGTH(token) > %d', $tokens, self::MAX_TOKEN_LENGTH));
			$command->bindValue(':model', $name);
			foreach ($command->queryAll() as $row) {
				$token = (string) $row['token'];
				$key = self::encodeToken($token);
				if ($key === $token) {
					continue;
				}
				$update = $connection->createCommand(sprintf('UPDATE %s SET token = :new WHERE model = :model AND token = :old', $tokens));
				$update->bindValue(':new', $key);
				$update->bindValue(':model', $name);
				$update->bindValue(':old', $token);
				$update->execute();
			}
			$command = $connection->createCommand($this->insertIgnoreSql(
				$this->_table . '_vocab',
				['model', 'token'],
				sprintf('SELECT DISTINCT model, token FROM %s WHERE model = :model AND doccnt > 0', $tokens)
			));
			$command->bindValue(':model', $name);
			$command->execute();

			$command = $connection->createCommand(sprintf('DELETE FROM %s_counters WHERE model = :model', $this->_table));
			$command->bindValue(':model', $name);
			$command->execute();
			$command = $connection->createCommand(sprintf(
				'INSERT INTO %1$s_counters (model, vocabulary_size) VALUES (:model, (SELECT COUNT(*) FROM %1$s_vocab WHERE model = :m2))',
				$this->_table
			));
			$command->bindValue(':model', $name);
			$command->bindValue(':m2', $name);
			$command->execute();

			$this->writeMetaRow($name, $this->encodeMeta($meta));
			$transaction->commit();
		} catch (\Throwable $e) {
			$transaction->rollBack();
			throw $e;
		}
	}

	/**
	 * Reads a model's derived counters: the document total summed from the category rows and
	 * the vocabulary size from the counters row.
	 * @param string $name The model name.
	 * @return array{totalDocuments:int, vocabularySize:int} The counters; zeros when absent.
	 */
	private function loadCounters(string $name): array
	{
		$command = $this->getDbConnection()->createCommand(sprintf('SELECT COALESCE(SUM(doc_count), 0) FROM %s_categories WHERE model = :model', $this->_table));
		$command->bindValue(':model', $name);
		$totalDocuments = TBayesianPayload::int($command->queryScalar());
		$command = $this->getDbConnection()->createCommand(sprintf('SELECT vocabulary_size FROM %s_counters WHERE model = :model', $this->_table));
		$command->bindValue(':model', $name);
		$size = $command->queryScalar();
		return ['totalDocuments' => $totalDocuments, 'vocabularySize' => TBayesianPayload::int($size)];
	}

	/**
	 * Returns the driver's insert-that-ignores-duplicates statement for a key-only table.
	 * @param string $table The table.
	 * @param string[] $columns The key columns.
	 * @param string $source Either a `VALUES (...)` list or a `SELECT` statement.
	 * @return string The SQL.
	 */
	private function insertIgnoreSql(string $table, array $columns, string $source): string
	{
		$columnList = implode(', ', $columns);
		if ($this->getDbConnection()->getDriverName() === 'mysql') {
			return sprintf('INSERT IGNORE INTO %s (%s) %s', $table, $columnList, $source);
		}
		return sprintf('INSERT INTO %s (%s) %s ON CONFLICT (%s) DO NOTHING', $table, $columnList, $source, $columnList);
	}

	/**
	 * Returns the driver's upsert statement that adds the inserted values to the existing row's
	 * counters, clamping each at zero.
	 * @param string $table The table.
	 * @param string[] $keys The key columns.
	 * @param string[] $counters The counter columns.
	 * @param string $values The `(...)` value tuples.
	 * @return string The SQL.
	 */
	private function upsertIncrementSql(string $table, array $keys, array $counters, string $values): string
	{
		$driver = $this->getDbConnection()->getDriverName();
		$columns = implode(', ', array_merge($keys, $counters));
		$updates = [];
		if ($driver === 'mysql') {
			foreach ($counters as $column) {
				$updates[] = sprintf('%1$s = GREATEST(%1$s + VALUES(%1$s), 0)', $column);
			}
			return sprintf('INSERT INTO %s (%s) VALUES %s ON DUPLICATE KEY UPDATE %s', $table, $columns, $values, implode(', ', $updates));
		}
		$greatest = $driver === 'pgsql' ? 'GREATEST' : 'MAX';
		foreach ($counters as $column) {
			$updates[] = sprintf('%1$s = %2$s(%3$s.%1$s + excluded.%1$s, 0)', $column, $greatest, $table);
		}
		return sprintf('INSERT INTO %s (%s) VALUES %s ON CONFLICT (%s) DO UPDATE SET %s', $table, $columns, $values, implode(', ', $keys), implode(', ', $updates));
	}

	/**
	 * Adds tokens to a model's vocabulary table, returning how many were new.
	 * @param string $name The model name.
	 * @param string[] $keys The token storage keys.
	 * @return int The number of tokens that were not in the vocabulary before.
	 */
	private function insertVocabulary(string $name, array $keys): int
	{
		$inserted = 0;
		foreach (array_chunk($keys, self::ROW_CHUNK) as $batch) {
			$values = [];
			$bind = [];
			foreach ($batch as $index => $key) {
				$values[] = sprintf('(:m%1$d, :t%1$d)', $index);
				$bind[':m' . $index] = $name;
				$bind[':t' . $index] = $key;
			}
			$command = $this->getDbConnection()->createCommand($this->insertIgnoreSql($this->_table . '_vocab', ['model', 'token'], 'VALUES ' . implode(', ', $values)));
			foreach ($bind as $placeholder => $value) {
				$command->bindValue($placeholder, $value);
			}
			$inserted += (int) $command->execute();
		}
		return $inserted;
	}

	/**
	 * Adds deltas to a category's token rows, inserting rows that do not exist yet.
	 * @param string $name The model name.
	 * @param string $category The category name.
	 * @param array<string, array{0:int, 1:int}> $rows The count and document-count deltas, keyed by token storage key.
	 */
	private function incrementTokenRows(string $name, string $category, array $rows): void
	{
		$table = $this->_table . '_tokens';
		foreach (array_chunk($rows, self::ROW_CHUNK, true) as $batch) {
			$values = [];
			$bind = [];
			$index = 0;
			foreach ($batch as $key => [$count, $docCount]) {
				$values[] = sprintf('(:m%1$d, :t%1$d, :c%1$d, :n%1$d, :d%1$d)', $index);
				$bind[':m' . $index] = $name;
				$bind[':t' . $index] = (string) $key;
				$bind[':c' . $index] = $category;
				// A row that does not exist yet starts at the clamped delta, never below zero.
				$bind[':n' . $index] = max(0, $count);
				$bind[':d' . $index] = max(0, $docCount);
				$index++;
			}
			$sql = $this->upsertIncrementSql($table, ['model', 'token', 'category'], ['cnt', 'doccnt'], implode(', ', $values));
			$command = $this->getDbConnection()->createCommand($sql);
			foreach ($bind as $placeholder => $value) {
				$command->bindValue($placeholder, $value);
			}
			// The bound values are the clamped inserts; the update branch adds the raw deltas.
			// Both cannot come from one bind, so rows with a negative delta are re-applied as
			// an update when the row already existed.
			$command->execute();
			$this->applyNegativeDeltas($name, $category, $batch);
		}
	}

	/**
	 * Applies the negative token deltas of a batch as plain clamped updates.
	 *
	 * The multi-row upsert binds each value once, and an insert must never create a negative
	 * row, so it binds the clamped value; the rows that already existed then still need the
	 * negative delta applied.
	 * @param string $name The model name.
	 * @param string $category The category name.
	 * @param array<string, array{0:int, 1:int}> $batch The deltas, keyed by token storage key.
	 */
	private function applyNegativeDeltas(string $name, string $category, array $batch): void
	{
		$greatest = $this->getDbConnection()->getDriverName() === 'sqlite' ? 'MAX' : 'GREATEST';
		foreach ($batch as $key => [$count, $docCount]) {
			if ($count >= 0 && $docCount >= 0) {
				continue;
			}
			$command = $this->getDbConnection()->createCommand(sprintf(
				'UPDATE %s_tokens SET cnt = %2$s(cnt + :c, 0), doccnt = %2$s(doccnt + :d, 0) WHERE model = :model AND token = :token AND category = :category',
				$this->_table,
				$greatest
			));
			$command->bindValue(':c', min(0, $count));
			$command->bindValue(':d', min(0, $docCount));
			$command->bindValue(':model', $name);
			$command->bindValue(':token', (string) $key);
			$command->bindValue(':category', $category);
			$command->execute();
		}
	}

	/**
	 * Adds deltas to one category's scalars, inserting the row when the category is new.
	 * @param string $name The model name.
	 * @param string $category The category name.
	 * @param int $documentDelta The change in document count.
	 * @param int $tokenDelta The change in total token occurrences.
	 */
	private function incrementCategoryRow(string $name, string $category, int $documentDelta, int $tokenDelta): void
	{
		$this->incrementCounterRow(
			$this->_table . '_categories',
			['model' => $name, 'category' => $category],
			['doc_count' => $documentDelta, 'total_tokens' => $tokenDelta]
		);
	}

	/**
	 * Adds to a model's vocabulary size, inserting the counters row when the model is new.
	 * @param string $name The model name.
	 * @param int $delta The change in the number of distinct tokens some document contains.
	 */
	private function incrementVocabularySize(string $name, int $delta): void
	{
		$this->incrementCounterRow($this->_table . '_counters', ['model' => $name], ['vocabulary_size' => $delta]);
	}

	/**
	 * Drops from the vocabulary table every token of the batch that no category's documents
	 * contain any more, after a negative document-count delta.
	 * @param string $name The model name.
	 * @param array<string, array{0:int, 1:int}> $rows The deltas, keyed by token storage key.
	 * @return int How many tokens left the vocabulary.
	 */
	private function pruneVocabulary(string $name, array $rows): int
	{
		$pruned = 0;
		$sql = sprintf(
			'DELETE FROM %1$s_vocab WHERE model = :model AND token = :token'
				. ' AND NOT EXISTS (SELECT 1 FROM %1$s_tokens WHERE model = :m2 AND token = :t2 AND doccnt > 0)',
			$this->_table
		);
		foreach ($rows as $key => [, $docCount]) {
			if ($docCount >= 0) {
				continue;
			}
			$command = $this->getDbConnection()->createCommand($sql);
			$command->bindValue(':model', $name);
			$command->bindValue(':token', (string) $key);
			$command->bindValue(':m2', $name);
			$command->bindValue(':t2', (string) $key);
			$pruned += (int) $command->execute();
		}
		return $pruned;
	}

	/**
	 * Adds deltas to one keyed row of counters, clamping each counter at zero.
	 * @param string $table The table.
	 * @param array<string, string> $keys The key columns and their values.
	 * @param array<string, int> $deltas The counter columns and their deltas.
	 */
	private function incrementCounterRow(string $table, array $keys, array $deltas): void
	{
		$placeholders = [];
		$bind = [];
		foreach ($keys as $column => $value) {
			$placeholders[] = ':' . $column;
			$bind[':' . $column] = $value;
		}
		foreach ($deltas as $column => $delta) {
			$placeholders[] = ':' . $column;
			$bind[':' . $column] = max(0, $delta);
		}
		$sql = $this->upsertIncrementSql($table, array_keys($keys), array_keys($deltas), '(' . implode(', ', $placeholders) . ')');
		$command = $this->getDbConnection()->createCommand($sql);
		foreach ($bind as $placeholder => $value) {
			$command->bindValue($placeholder, $value);
		}
		$command->execute();
		$negative = array_filter($deltas, static fn (int $delta): bool => $delta < 0);
		if ($negative === []) {
			return;
		}
		$greatest = $this->getDbConnection()->getDriverName() === 'sqlite' ? 'MAX' : 'GREATEST';
		$sets = [];
		$where = [];
		$bind = [];
		foreach ($negative as $column => $delta) {
			$sets[] = sprintf('%1$s = %2$s(%1$s + :%1$s, 0)', $column, $greatest);
			$bind[':' . $column] = $delta;
		}
		foreach ($keys as $column => $value) {
			$where[] = sprintf('%1$s = :%1$s', $column);
			$bind[':' . $column] = $value;
		}
		$command = $this->getDbConnection()->createCommand(sprintf('UPDATE %s SET %s WHERE %s', $table, implode(', ', $sets), implode(' AND ', $where)));
		foreach ($bind as $placeholder => $value) {
			$command->bindValue($placeholder, $value);
		}
		$command->execute();
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
		foreach (array_chunk($rows, self::ROW_CHUNK) as $batch) {
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
	 * Removes every per-token, per-category, vocabulary and counter row of a model.
	 * @param string $name The model name.
	 */
	private function deleteTokenRows(string $name): void
	{
		foreach (['_tokens', '_categories', '_vocab', '_counters', '_hist'] as $suffix) {
			$command = $this->getDbConnection()->createCommand(sprintf('DELETE FROM %s%s WHERE model = :model', $this->_table, $suffix));
			$command->bindValue(':model', $name);
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
		$this->forgetSchema();
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
		$this->forgetSchema();
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
		$this->forgetSchema();
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
		$this->forgetSchema();
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
		$this->forgetSchema();
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
	 * becoming a SQL-injection vector, and bounded so the derived per-token table names stay
	 * within every driver's identifier limit.
	 * @param string $value The table name.
	 * @throws TInvalidDataValueException When the name is not a plain SQL identifier of at most 48 characters.
	 */
	public function setTable(string $value): void
	{
		if (preg_match('/^[A-Za-z_][A-Za-z0-9_]*$/', $value) !== 1 || strlen($value) > self::MAX_TABLE_LENGTH) {
			throw new TInvalidDataValueException('bayesian_storage_table_invalid', $value);
		}
		$this->_table = $value;
		$this->forgetSchema();
	}
}
