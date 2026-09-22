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
 * `_hist` table, moved inside the same transaction as the counts.  Bernoulli's histogram
 * depends only on the row a training write already locks, so it moves without any further
 * locking and such a model is written as concurrently as a multinomial one.  Complement's
 * depends on a token's counts in every category, so a write to such a model first locks the
 * model's counters row, and those writes run one after another — each is a few statements.
 * {@see setHistogramMode() HistogramMode} trades that for freshness, per model: `deferred`
 * journals the touched tokens for a worker to fold in ({@see foldTokenHistograms()}), and
 * `periodic` leaves the histograms to a scheduled {@see rebuildTokenHistograms()}; both let
 * Complement writes run side by side, and both are served by {@see maintainTokenHistograms()}.
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
	 * Histograms move inside every training write, under the model lock when they span
	 * categories: always exact, and writes to one Complement model run one after another.
	 * @since 0.2.0
	 */
	public const HISTOGRAM_IMMEDIATE = 'immediate';

	/**
	 * Training writes only journal the tokens they touched; {@see foldTokenHistograms()}, run
	 * by one worker, folds them into the histograms afterwards.  Writes run side by side, and
	 * the histograms trail training by the worker's interval.
	 * @since 0.2.0
	 */
	public const HISTOGRAM_DEFERRED = 'deferred';

	/**
	 * Training writes leave the histograms alone entirely; {@see rebuildTokenHistograms()}, run
	 * on a schedule, recounts them.  The cheapest writes, the stalest histograms.
	 * @since 0.2.0
	 */
	public const HISTOGRAM_PERIODIC = 'periodic';

	/** The histogram family letter of the row that serves as a model's fold lock; never a real family. */
	private const FOLD_LOCK_FAMILY = 'L';

	/** How many times a write is attempted when the database reports a deadlock, and how many passes claim a vocabulary. */
	private const MAX_WRITE_ATTEMPTS = 5;

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

	/** @var string The histogram mode given to models this storage saves; one of the HISTOGRAM_* constants. */
	private string $_histogramMode = self::HISTOGRAM_IMMEDIATE;

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
		// The deferred histogram mode: the tokens written since they were last folded, with a
		// sequence that tells the folder whether a token was written again while it worked...
		$statements[] = sprintf(
			'CREATE TABLE IF NOT EXISTS %s_journal (model %s NOT NULL, token %s NOT NULL, seq %s NOT NULL, PRIMARY KEY (model, token))',
			$this->_table,
			$key,
			$key,
			$int
		);
		// ...and the token statistics as they were when last folded, which is the state the
		// histograms describe and so the state a fold takes away before adding the current one.
		$statements[] = sprintf(
			'CREATE TABLE IF NOT EXISTS %s_folded (model %s NOT NULL, token %s NOT NULL, category %s NOT NULL,'
				. ' cnt %s NOT NULL, doccnt %s NOT NULL, PRIMARY KEY (model, token, category))',
			$this->_table,
			$key,
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
		$families = TBayesianTokenHistogram::families($meta['histograms'] ?? null);
		unset($meta['histogramMode']);
		if ($families !== []) {
			// A full save is a deliberate act of a configured process: the model takes this
			// storage's mode, and every later writer follows the model.
			$meta['histogramMode'] = $this->_histogramMode;
		}
		$encoded = TBayesianPayload::encodeTokenMeta($meta, self::TOKEN_LAYOUT_VERSION);
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
			$this->applyHistogramDelta($name, TBayesianTokenHistogram::build($tokens, $families));
			if (self::effectiveMode($families, $meta) === self::HISTOGRAM_DEFERRED) {
				$this->insertTokenRows($name, $rows, '_folded');
			}
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
	 * @param string $table The table suffix to read: the token rows, or their folded copy.
	 * @return array<string, array<string, array{count:int, docCount:int}>> The statistics, keyed by token then category.
	 */
	private function selectTokenRows(string $name, array $map, bool $forUpdate, string $table = '_tokens'): array
	{
		$suffix = $forUpdate && $this->getDbConnection()->getDriverName() !== 'sqlite' ? ' FOR UPDATE' : '';
		$out = [];
		foreach (array_chunk(array_keys($map), self::TOKEN_CHUNK) as $chunk) {
			$placeholders = [];
			foreach ($chunk as $index => $_) {
				$placeholders[] = ':t' . $index;
			}
			$sql = sprintf(
				'SELECT token, category, cnt, doccnt FROM %s%s WHERE model = :model AND token IN (%s)%s',
				$this->_table,
				$table,
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
		$rows = [];
		foreach ($tokenDeltas as $token => $delta) {
			$rows[self::encodeToken((string) $token)] = [(int) ($delta['count'] ?? 0), (int) ($delta['docCount'] ?? 0)];
		}
		ksort($rows, SORT_STRING);
		$documentDelta = (int) ($categoryStats['documentCount'] ?? 0);
		$tokenDelta = (int) ($categoryStats['totalTokens'] ?? 0);
		// The histograms to keep are the ones the metadata names: the metadata being written
		// when there is one, the stored metadata otherwise.  A family the writer names but the
		// store does not hold yet is rebuilt rather than moved.
		$families = $meta === [] ? null : TBayesianTokenHistogram::families($meta['histograms'] ?? null);
		$storedMeta = $families === [] ? null : $this->load($name);
		$stored = TBayesianTokenHistogram::families(($storedMeta ?? [])['histograms'] ?? null);
		$families ??= $stored;
		$rebuild = array_diff($families, $stored) !== [];
		// How the histograms are kept is the model's own setting, not the writer's: every
		// process writing one model must treat its histograms the same way.
		$mode = self::effectiveMode($families, $storedMeta ?? ['histogramMode' => $this->_histogramMode]);
		unset($meta['histogramMode']);
		if ($meta !== [] && $families !== []) {
			$meta['histogramMode'] = self::storedMode($storedMeta ?? ['histogramMode' => $this->_histogramMode]);
		}
		$encoded = $meta === [] ? null : TBayesianPayload::encodeTokenMeta($meta, self::TOKEN_LAYOUT_VERSION);
		$plan = [
			// The document-count family depends only on the (token, category) rows this write
			// locks anyway, so it moves from their own transitions.
			'rowLocal' => !$rebuild && $families === [TBayesianTokenHistogram::FAMILY_DOCUMENTS],
			// Any other family depends on a token's rows in every category, which only hold
			// still under the model lock — unless the model leaves them to maintenance.
			'locked' => $mode === self::HISTOGRAM_IMMEDIATE && $families !== [] && ($rebuild || $families !== [TBayesianTokenHistogram::FAMILY_DOCUMENTS]),
			'rebuild' => $rebuild && $mode === self::HISTOGRAM_IMMEDIATE,
			'journal' => $mode === self::HISTOGRAM_DEFERRED,
			'families' => $families,
			// A write that takes documents away may empty a token out of the vocabulary, which
			// is a decision about the token's rows in every category: it holds the token
			// exclusively.
			'withdraws' => false,
		];
		foreach ($rows as [, $docCount]) {
			$plan['withdraws'] = $plan['withdraws'] || $docCount < 0;
		}
		$this->transactional(function () use ($name, $category, $rows, $plan, $documentDelta, $tokenDelta, $encoded): void {
			$this->applyDeltasInTransaction($name, $category, $rows, $plan, $documentDelta, $tokenDelta, $encoded);
		});
		if ($rebuild && $mode !== self::HISTOGRAM_IMMEDIATE) {
			// The model leaves its histograms to maintenance and has none yet: count them now,
			// outside the write, the way maintenance would.
			$this->rebuildTokenHistograms($name, $families);
		}
	}

	/**
	 * The body of {@see applyDeltas()}, inside its transaction.
	 *
	 * Locks are always taken in one order — the model's counters row when the histograms need
	 * it, then the vocabulary rows of the document's tokens in sorted order, then the token
	 * rows in sorted order, the category row, the counters row, and the histogram cells in
	 * sorted order — so writers queue behind one another instead of deadlocking.
	 * @param string $name The model name.
	 * @param string $category The category the document was filed under.
	 * @param array<string, array{0:int, 1:int}> $rows The count and document-count deltas, keyed by token storage key.
	 * @param array{rowLocal:bool, locked:bool, rebuild:bool, journal:bool, families:string[], withdraws:bool} $plan How the
	 * histograms are kept for this write: moved from the written rows' own transitions, moved
	 * under the model lock, recounted under it, or journalled for the folder; and whether any
	 * document count goes down.
	 * @param int $documentDelta The change in the category's document count.
	 * @param int $tokenDelta The change in the category's token total.
	 * @param ?string $encoded The JSON-encoded metadata to write, or null to leave it.
	 */
	private function applyDeltasInTransaction(string $name, string $category, array $rows, array $plan, int $documentDelta, int $tokenDelta, ?string $encoded): void
	{
		$keys = array_combine(array_map('strval', array_keys($rows)), array_map('strval', array_keys($rows)));
		$before = [];
		if ($plan['locked']) {
			$this->lockModel($name);
		}
		$newTokens = $this->claimVocabulary($name, array_keys($rows), $plan['withdraws']);
		if ($plan['locked'] && !$plan['rebuild']) {
			$before = $this->selectTokenRows($name, $keys, true);
		}
		$transitions = $this->incrementTokenRows($name, $category, $rows, $plan['rowLocal']);
		$this->incrementCategoryRow($name, $category, $documentDelta, $tokenDelta);
		$this->incrementVocabularySize($name, $newTokens - $this->pruneVocabulary($name, $rows));
		if ($plan['rebuild']) {
			$this->rebuildHistogramRows($name, $plan['families']);
		} elseif ($plan['rowLocal']) {
			$this->applyHistogramDelta($name, self::documentCountDelta($category, $transitions));
		} elseif ($plan['locked']) {
			$this->applyHistogramDelta($name, TBayesianTokenHistogram::delta($before, $this->selectTokenRows($name, $keys, true), $plan['families']));
		} elseif ($plan['journal']) {
			// After the token rows, never before: the folder takes token rows and then journal
			// rows too, and a mark made before the write could be consumed before the write
			// it announces becomes visible.
			$this->markJournal($name, array_keys($keys));
		}
		if ($encoded !== null) {
			$this->writeMetaRow($name, $encoded);
		}
	}

	/**
	 * Returns whether a failure is the database refusing a transaction over locks — a deadlock
	 * or a serialization failure — which is safe to answer by running the transaction again.
	 * @param \Throwable $e The failure.
	 * @return bool Whether to retry.
	 */
	private static function isLockFailure(\Throwable $e): bool
	{
		for ($error = $e; $error !== null; $error = $error->getPrevious()) {
			if (preg_match('/SQLSTATE\[(40001|40P01)\]/', $error->getMessage()) === 1) {
				return true;
			}
		}
		return false;
	}

	/**
	 * Adds the document's tokens to the vocabulary and takes this transaction's hold on their
	 * vocabulary rows, returning how many tokens were new.
	 *
	 * The vocabulary row is a token's lock.  A write that only adds documents holds it shared,
	 * so trainers of different categories run side by side; a write that takes documents away
	 * holds it exclusively, because whether the token leaves the vocabulary depends on its rows
	 * in every category, and those only hold still while no other writer is inside the token.
	 * Without it, two writers withdrawing a token's last documents from different categories
	 * each see the other's row still alive and neither removes the token (or, on MySQL, each
	 * waits for the other's row and one is killed).
	 *
	 * Existing rows are locked before the insert when the hold is exclusive, so that a lock is
	 * never upgraded from the shared one MySQL's insert-ignore leaves on a duplicate.  A row
	 * that another writer removed between the insert and the lock is inserted again.
	 * @param string $name The model name.
	 * @param array<int, int|string> $keys The token storage keys, in sorted order.
	 * @param bool $exclusive Whether to hold the tokens exclusively.
	 * @return int The number of tokens that were not in the vocabulary before.
	 */
	private function claimVocabulary(string $name, array $keys, bool $exclusive): int
	{
		$keys = array_map('strval', $keys);
		if ($exclusive) {
			$this->lockVocabularyRows($name, $keys, true);
		}
		$inserted = 0;
		$pending = $keys;
		for ($pass = 0; $pending !== [] && $pass < self::MAX_WRITE_ATTEMPTS; $pass++) {
			$inserted += $this->insertVocabulary($name, $pending);
			$pending = array_values(array_diff($pending, $this->lockVocabularyRows($name, $pending, $exclusive)));
		}
		return $inserted;
	}

	/**
	 * Locks the vocabulary rows of the given tokens for the rest of the transaction, in token
	 * order.  SQLite has no row locks — its writers are serialized by the database lock the
	 * first write takes — so there every key counts as held.
	 * @param string $name The model name.
	 * @param string[] $keys The token storage keys.
	 * @param bool $exclusive Whether to lock for update rather than for share.
	 * @return string[] The keys whose rows exist and are now held.
	 */
	private function lockVocabularyRows(string $name, array $keys, bool $exclusive): array
	{
		$driver = $this->getDbConnection()->getDriverName();
		if ($driver === 'sqlite') {
			return $keys;
		}
		$mode = $exclusive ? 'FOR UPDATE' : ($driver === 'mysql' ? 'LOCK IN SHARE MODE' : 'FOR SHARE');
		$held = [];
		foreach (array_chunk($keys, self::TOKEN_CHUNK) as $chunk) {
			$placeholders = [];
			foreach ($chunk as $index => $_) {
				$placeholders[] = ':t' . $index;
			}
			$command = $this->getDbConnection()->createCommand(sprintf(
				'SELECT token FROM %s_vocab WHERE model = :model AND token IN (%s) ORDER BY token %s',
				$this->_table,
				implode(', ', $placeholders),
				$mode
			));
			$command->bindValue(':model', $name);
			foreach ($chunk as $index => $key) {
				$command->bindValue(':t' . $index, $key);
			}
			foreach ($command->queryAll() as $row) {
				$held[] = (string) $row['token'];
			}
		}
		return $held;
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
		$this->transactional(function () use ($name, $families, $meta): void {
			$this->rebuildInTransaction($name, $families, $meta);
		});
	}

	/**
	 * The body of {@see rebuildTokenHistograms()}, inside its transaction.
	 *
	 * An immediate model is recounted under the model lock, which its writers hold too.  A
	 * model that leaves its histograms to maintenance is recounted under the fold lock, which
	 * its writers never take, so training goes on meanwhile; a deferred one first copies the
	 * token rows into their folded copy and counts that, so the histograms describe exactly
	 * the state the next fold will take away.
	 * @param string $name The model name.
	 * @param string[] $families The family letters to build.
	 * @param array<string, mixed> $meta The model's metadata as read before the transaction.
	 * @return int The number of histogram cells written.
	 */
	private function rebuildInTransaction(string $name, array $families, array $meta): int
	{
		$mode = self::effectiveMode($families, $meta);
		if ($mode === self::HISTOGRAM_IMMEDIATE) {
			$this->lockModel($name);
		} else {
			$this->lockFold($name);
		}
		if ($mode === self::HISTOGRAM_DEFERRED) {
			$this->refreshFoldedRows($name);
			$cells = $this->rebuildHistogramRows($name, $families, '_folded');
		} else {
			$cells = $this->rebuildHistogramRows($name, $families);
		}
		if ($mode === self::HISTOGRAM_IMMEDIATE) {
			// Nothing journals or folds an immediate model; whatever a writer that started
			// under an earlier mode left behind is covered by this recount.
			$this->deleteModelRows($name, ['_journal', '_folded']);
		}
		// Re-read inside the transaction: a trainer may have rewritten the metadata meanwhile.
		$meta = $this->load($name) ?? $meta;
		$meta['histograms'] = $families;
		$this->writeMetaRow($name, TBayesianPayload::encodeTokenMeta($meta, self::TOKEN_LAYOUT_VERSION));
		return $cells;
	}

	/**
	 * Folds journalled tokens into the histograms of a model in the deferred mode: for each,
	 * the cells its folded statistics counted towards lose one, the cells its current
	 * statistics count towards gain one, and the current statistics become the folded ones.
	 *
	 * Run it from one worker — a cron job, a queue consumer, a loop in a console command.  Two
	 * at once are safe (they take turns on the model's fold lock), just pointless.  Training
	 * writes run alongside: a fold holds the rows of the tokens it is folding only for the
	 * length of one small transaction, and a token written again while it was being folded
	 * stays in the journal for the next fold.
	 * @param string $name The model name.
	 * @param int $limit The most tokens to fold in this call, and so in one transaction.
	 * @throws TInvalidOperationException When the storage is not in per-token mode.
	 * @throws TInvalidDataValueException When the model's layout is newer than this release reads.
	 * @throws TDbException When a statement fails.
	 * @return int The number of tokens folded; 0 when the journal is empty, the model is
	 * unknown, or the model is not in the deferred mode.
	 * @since 0.2.0
	 */
	public function foldTokenHistograms(string $name, int $limit = 1000): int
	{
		$this->ensureTokenMode();
		$this->ensureLayout($name);
		$meta = $this->load($name);
		$families = TBayesianTokenHistogram::families(($meta ?? [])['histograms'] ?? null);
		if ($meta === null || self::effectiveMode($families, $meta) !== self::HISTOGRAM_DEFERRED) {
			return 0;
		}
		$limit = max(1, $limit);
		return $this->transactional(function () use ($name, $families, $limit): int {
			$this->lockFold($name);
			$command = $this->getDbConnection()->createCommand(sprintf(
				'SELECT token, seq FROM %s_journal WHERE model = :model ORDER BY token LIMIT %d',
				$this->_table,
				$limit
			));
			$command->bindValue(':model', $name);
			$batch = [];
			foreach ($command->queryAll() as $row) {
				$batch[(string) $row['token']] = (int) $row['seq'];
			}
			if ($batch === []) {
				return 0;
			}
			$keys = array_combine(array_map('strval', array_keys($batch)), array_map('strval', array_keys($batch)));
			// Token rows first, journal rows last: the order training writes take them in.
			$live = $this->selectTokenRows($name, $keys, true);
			$folded = $this->selectTokenRows($name, $keys, false, '_folded');
			$this->applyHistogramDelta($name, TBayesianTokenHistogram::delta($folded, $live, $families));
			$this->replaceFoldedRows($name, array_keys($keys), $live);
			$this->deleteJournalRows($name, $batch);
			return count($batch);
		});
	}

	/**
	 * Returns how many tokens of a model are journalled and not yet folded.
	 * @param string $name The model name.
	 * @throws TInvalidOperationException When the storage is not in per-token mode.
	 * @throws TDbException When the statement fails.
	 * @return int The number of tokens waiting for {@see foldTokenHistograms()}.
	 * @since 0.2.0
	 */
	public function getPendingTokenCount(string $name): int
	{
		$this->ensureTokenMode();
		$command = $this->getDbConnection()->createCommand(sprintf('SELECT COUNT(*) FROM %s_journal WHERE model = :model', $this->_table));
		$command->bindValue(':model', $name);
		return TBayesianPayload::int($command->queryScalar());
	}

	/**
	 * Brings a model's histograms up to date in whatever way its mode calls for: the one call
	 * a maintenance worker needs.  A deferred model has its journal folded until it is empty;
	 * a periodic model is recounted; an immediate model needs nothing, unless a writer that
	 * started under an earlier mode left journal entries behind, in which case it is recounted
	 * once.
	 * @param string $name The model name.
	 * @param int $batch The most tokens to fold per transaction, for a deferred model.
	 * @throws TInvalidOperationException When the storage is not in per-token mode.
	 * @throws TInvalidDataValueException When the model's layout is newer than this release reads.
	 * @throws TDbException When a statement fails.
	 * @return int The tokens folded (deferred) or the histogram cells written (periodic, or an
	 * immediate model that was repaired); 0 when there was nothing to do.
	 * @since 0.2.0
	 */
	public function maintainTokenHistograms(string $name, int $batch = 1000): int
	{
		$this->ensureTokenMode();
		$this->ensureLayout($name);
		$meta = $this->load($name);
		$families = TBayesianTokenHistogram::families(($meta ?? [])['histograms'] ?? null);
		if ($meta === null || $families === []) {
			return 0;
		}
		$mode = self::effectiveMode($families, $meta);
		if ($mode === self::HISTOGRAM_DEFERRED) {
			$total = 0;
			while (($folded = $this->foldTokenHistograms($name, $batch)) > 0) {
				$total += $folded;
			}
			return $total;
		}
		if ($mode === self::HISTOGRAM_IMMEDIATE && $this->getPendingTokenCount($name) === 0) {
			return 0;
		}
		return $this->transactional(fn (): int => $this->rebuildInTransaction($name, $families, $meta));
	}

	/**
	 * Changes how an existing model keeps its histograms, recounting them on the way so that
	 * the model starts its new mode exact.
	 *
	 * Writers read a model's mode just before they write, so switch while training is paused
	 * (a deploy, a maintenance window).  A writer caught mid-switch is not lost — its counts
	 * land as always — but its histogram change may be missed; {@see maintainTokenHistograms()}
	 * notices the journal entry such a writer leaves and recounts.
	 * @param string $name The model name.
	 * @param string $mode One of the HISTOGRAM_* constants.
	 * @throws TInvalidDataValueException When the mode is not one of the constants.
	 * @throws TInvalidOperationException When the storage is not in per-token mode.
	 * @throws TDbException When a statement fails.
	 * @since 0.2.0
	 */
	public function setTokenHistogramMode(string $name, string $mode): void
	{
		self::assertHistogramMode($mode);
		$this->ensureTokenMode();
		$this->ensureLayout($name);
		$meta = $this->load($name);
		if ($meta === null) {
			return;
		}
		$families = TBayesianTokenHistogram::families($meta['histograms'] ?? null);
		if (self::effectiveMode($families, $meta) === self::HISTOGRAM_DEFERRED) {
			$this->maintainTokenHistograms($name);
		}
		$meta = $this->load($name) ?? $meta;
		$meta['histogramMode'] = $mode;
		$this->transactional(function () use ($name, $families, $meta): void {
			$this->writeMetaRow($name, TBayesianPayload::encodeTokenMeta($meta, self::TOKEN_LAYOUT_VERSION));
			if ($families !== []) {
				$this->rebuildInTransaction($name, $families, $meta);
			}
		});
	}

	/**
	 * Returns the histogram mode this storage gives the models it saves.
	 * @return string One of the HISTOGRAM_* constants.
	 * @since 0.2.0
	 */
	public function getHistogramMode(): string
	{
		return $this->_histogramMode;
	}

	/**
	 * Sets how the models this storage saves keep histograms that span categories — in
	 * practice, how Complement Naive Bayes models are trained through per-token storage:
	 *
	 * - `immediate` (the default): every training write moves them, under a per-model lock.
	 *   Always exact; writes to one model run one after another.
	 * - `deferred`: training writes journal the tokens they touched and a worker calling
	 *   {@see maintainTokenHistograms()} folds them in.  Writes run side by side; scores trail
	 *   training by the worker's interval and are exact once the journal is empty.
	 * - `periodic`: training writes do nothing extra and a scheduled
	 *   {@see maintainTokenHistograms()} recounts the histograms in the database.  The
	 *   cheapest writes; scores trail training by the schedule.
	 *
	 * The mode is recorded in a model's metadata when the model is saved, and from then on
	 * every writer follows the model, whatever its own storage is configured with; change an
	 * existing model with {@see setTokenHistogramMode()}.  Models whose histograms need no lock
	 * (Bernoulli) or that keep none (multinomial) are unaffected.
	 * @param string $value One of `immediate`, `deferred`, `periodic`.
	 * @throws TInvalidDataValueException When the value is none of them.
	 * @since 0.2.0
	 */
	public function setHistogramMode(string $value): void
	{
		self::assertHistogramMode($value);
		$this->_histogramMode = $value;
	}

	/**
	 * Throws unless the value is a histogram mode.
	 * @param string $value The value.
	 * @throws TInvalidDataValueException When it is not.
	 */
	private static function assertHistogramMode(string $value): void
	{
		if (!in_array($value, [self::HISTOGRAM_IMMEDIATE, self::HISTOGRAM_DEFERRED, self::HISTOGRAM_PERIODIC], true)) {
			throw new TInvalidDataValueException('bayesian_storage_histogram_mode_invalid', $value);
		}
	}

	/**
	 * Returns the histogram mode a model's metadata records; a model that records none, or an
	 * unknown one, is immediate.
	 * @param array<string, mixed> $meta The model's metadata.
	 * @return string One of the HISTOGRAM_* constants.
	 */
	private static function storedMode(array $meta): string
	{
		$mode = $meta['histogramMode'] ?? null;
		return $mode === self::HISTOGRAM_DEFERRED || $mode === self::HISTOGRAM_PERIODIC ? $mode : self::HISTOGRAM_IMMEDIATE;
	}

	/**
	 * Returns the mode that governs a model's writes.  Only histograms that span categories
	 * have anything to gain from being deferred: a model that keeps none, or only the
	 * document-count family, is immediate whatever its metadata says.
	 * @param string[] $families The histogram families the model keeps.
	 * @param array<string, mixed> $meta The model's metadata.
	 * @return string One of the HISTOGRAM_* constants.
	 */
	private static function effectiveMode(array $families, array $meta): string
	{
		if ($families === [] || $families === [TBayesianTokenHistogram::FAMILY_DOCUMENTS]) {
			return self::HISTOGRAM_IMMEDIATE;
		}
		return self::storedMode($meta);
	}

	/**
	 * Runs a unit of work in a transaction, running it again when the database refuses the
	 * transaction over locks.  The work must be safe to repeat after a rollback: increments,
	 * or a computation from what it reads inside the transaction.
	 * @template T
	 * @param callable(): T $work The work.
	 * @throws \Throwable Whatever the work throws, once it is not a lock failure or the attempts are used up.
	 * @return T What the work returned.
	 */
	private function transactional(callable $work)
	{
		for ($attempt = 1; ; $attempt++) {
			$transaction = $this->getDbConnection()->beginTransaction();
			try {
				$result = $work();
				$transaction->commit();
				return $result;
			} catch (\Throwable $e) {
				$transaction->rollBack();
				if ($attempt >= self::MAX_WRITE_ATTEMPTS || !self::isLockFailure($e)) {
					throw $e;
				}
				// The database chose this transaction as a deadlock victim.  Nothing of the
				// attempt survived the rollback, so the work is simply done again, after a
				// short, growing, jittered pause.
				usleep(random_int(1000, 5000) * $attempt);
			}
		}
	}

	/**
	 * Takes a model's fold lock for the rest of the transaction: a row of the histogram table
	 * that is not a histogram cell.  Folds and recounts of a model that leaves its histograms
	 * to maintenance take turns on it.  It is not the counters row, because every training
	 * write touches that one after its token rows, and a fold needs token rows after its lock.
	 * @param string $name The model name.
	 */
	private function lockFold(string $name): void
	{
		$table = $this->_table . '_hist';
		$update = $this->getDbConnection()->getDriverName() === 'mysql'
			? 'ON DUPLICATE KEY UPDATE cnt = cnt'
			: sprintf('ON CONFLICT (model, fam, category, val) DO UPDATE SET cnt = %s.cnt', $table);
		$command = $this->getDbConnection()->createCommand(sprintf("INSERT INTO %s (model, fam, category, val, cnt) VALUES (:model, :fam, '', 0, 0) %s", $table, $update));
		$command->bindValue(':model', $name);
		$command->bindValue(':fam', self::FOLD_LOCK_FAMILY);
		$command->execute();
	}

	/**
	 * Records that tokens were written, for the folder.  Each mark bumps the token's sequence,
	 * so a fold that read the journal before the mark leaves the entry in place.
	 * @param string $name The model name.
	 * @param string[] $keys The token storage keys, in sorted order.
	 */
	private function markJournal(string $name, array $keys): void
	{
		$table = $this->_table . '_journal';
		$update = $this->getDbConnection()->getDriverName() === 'mysql'
			? 'ON DUPLICATE KEY UPDATE seq = seq + 1'
			: sprintf('ON CONFLICT (model, token) DO UPDATE SET seq = %s.seq + 1', $table);
		foreach (array_chunk($keys, self::ROW_CHUNK) as $batch) {
			$values = [];
			$bind = [];
			foreach ($batch as $index => $key) {
				$values[] = sprintf('(:m%1$d, :t%1$d, 1)', $index);
				$bind[':m' . $index] = $name;
				$bind[':t' . $index] = $key;
			}
			$command = $this->getDbConnection()->createCommand(sprintf('INSERT INTO %s (model, token, seq) VALUES %s %s', $table, implode(', ', $values), $update));
			foreach ($bind as $placeholder => $value) {
				$command->bindValue($placeholder, $value);
			}
			$command->execute();
		}
	}

	/**
	 * Removes the journal entries a fold has consumed — those whose sequence is still the one
	 * the fold read.  An entry marked again since then stays for the next fold.
	 * @param string $name The model name.
	 * @param array<string, int> $batch The sequence read for each token storage key.
	 */
	private function deleteJournalRows(string $name, array $batch): void
	{
		foreach (array_chunk($batch, self::ROW_CHUNK, true) as $chunk) {
			$conditions = [];
			$bind = [];
			$index = 0;
			foreach ($chunk as $key => $sequence) {
				$conditions[] = sprintf('(token = :t%1$d AND seq = :s%1$d)', $index);
				$bind[':t' . $index] = (string) $key;
				$bind[':s' . $index] = $sequence;
				$index++;
			}
			$command = $this->getDbConnection()->createCommand(sprintf('DELETE FROM %s_journal WHERE model = :model AND (%s)', $this->_table, implode(' OR ', $conditions)));
			$command->bindValue(':model', $name);
			foreach ($bind as $placeholder => $value) {
				$command->bindValue($placeholder, $value);
			}
			$command->execute();
		}
	}

	/**
	 * Makes the given tokens' folded statistics equal to the statistics just folded.
	 * @param string $name The model name.
	 * @param string[] $keys The token storage keys.
	 * @param array<string, array<string, array{count:int, docCount:int}>> $live The statistics folded, keyed by token storage key then category.
	 */
	private function replaceFoldedRows(string $name, array $keys, array $live): void
	{
		foreach (array_chunk($keys, self::TOKEN_CHUNK) as $chunk) {
			$placeholders = [];
			foreach ($chunk as $index => $_) {
				$placeholders[] = ':t' . $index;
			}
			$command = $this->getDbConnection()->createCommand(sprintf('DELETE FROM %s_folded WHERE model = :model AND token IN (%s)', $this->_table, implode(', ', $placeholders)));
			$command->bindValue(':model', $name);
			foreach ($chunk as $index => $key) {
				$command->bindValue(':t' . $index, $key);
			}
			$command->execute();
		}
		$rows = [];
		foreach ($live as $key => $categories) {
			foreach ($categories as $category => $stats) {
				if ($stats['count'] > 0 || $stats['docCount'] > 0) {
					$rows[] = [(string) $key, (string) $category, $stats['count'], $stats['docCount']];
				}
			}
		}
		$this->insertTokenRows($name, $rows, '_folded');
	}

	/**
	 * Replaces a model's folded statistics with a copy of its token rows, inside the database.
	 * @param string $name The model name.
	 */
	private function refreshFoldedRows(string $name): void
	{
		$this->deleteModelRows($name, ['_folded']);
		$command = $this->getDbConnection()->createCommand(sprintf(
			'INSERT INTO %1$s_folded (model, token, category, cnt, doccnt)'
				. ' SELECT model, token, category, cnt, doccnt FROM %1$s_tokens WHERE model = :model AND (cnt > 0 OR doccnt > 0)',
			$this->_table
		));
		$command->bindValue(':model', $name);
		$command->execute();
	}

	/**
	 * Removes a model's rows from the given per-token tables.
	 * @param string $name The model name.
	 * @param string[] $suffixes The table suffixes.
	 */
	private function deleteModelRows(string $name, array $suffixes): void
	{
		foreach ($suffixes as $suffix) {
			$command = $this->getDbConnection()->createCommand(sprintf('DELETE FROM %s%s WHERE model = :model', $this->_table, $suffix));
			$command->bindValue(':model', $name);
			$command->execute();
		}
	}

	/**
	 * Takes the model's write lock for the rest of the transaction by touching its counters
	 * row.  Moving a histogram that spans categories needs a token's rows in every category to
	 * hold still between reading and writing them, and one lock per model gives that on every
	 * driver without any lock-ordering between tokens.  The document-count family does not
	 * need it (see {@see applyDeltas()}).
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
	 * @param string $source The table suffix to count: the token rows, or their folded copy.
	 * @return int The number of histogram cells written.
	 */
	private function rebuildHistogramRows(string $name, array $families, string $source = '_tokens'): int
	{
		$tokens = $this->_table . $source;
		$bind = [];
		$model = static function () use (&$bind, $name): string {
			$placeholder = ':m' . count($bind);
			$bind[$placeholder] = $name;
			return $placeholder;
		};
		// The vocabulary is the tokens some document contains; `x` gives each its global count.
		$vocabulary = static fn (): string => sprintf('(SELECT token, SUM(cnt) AS g FROM %s WHERE model = %s GROUP BY token HAVING SUM(doccnt) > 0) x', $tokens, $model());
		$parts = [];
		foreach ($families as $family) {
			$literal = "'" . $family . "'";
			if ($family === TBayesianTokenHistogram::FAMILY_DOCUMENTS) {
				$parts[] = sprintf('SELECT %s AS fam, category, doccnt AS val, COUNT(*) AS n FROM %s WHERE model = %s AND doccnt > 0 GROUP BY category, doccnt', $literal, $tokens, $model());
			} elseif ($family === TBayesianTokenHistogram::FAMILY_GLOBAL) {
				$parts[] = sprintf("SELECT %s AS fam, '' AS category, x.g AS val, COUNT(*) AS n FROM %s GROUP BY x.g", $literal, $vocabulary());
			} else {
				$value = $family === TBayesianTokenHistogram::FAMILY_COMPLEMENT ? 'x.g - t.cnt' : 'x.g';
				$parts[] = sprintf(
					'SELECT %1$s AS fam, t.category AS category, %2$s AS val, COUNT(*) AS n FROM %3$s t JOIN %4$s ON x.token = t.token'
						. ' WHERE t.model = %5$s AND t.cnt > 0 GROUP BY t.category, %2$s',
					$literal,
					$value,
					$tokens,
					$vocabulary(),
					$model()
				);
			}
		}
		if ($parts === []) {
			return 0;
		}
		// One statement, so that every family is counted from the same snapshot even while
		// training writes go on: the complement counts are merged from three of them.
		$command = $this->getDbConnection()->createCommand(implode(' UNION ALL ', $parts));
		foreach ($bind as $placeholder => $value) {
			$command->bindValue($placeholder, $value);
		}
		$flat = [];
		foreach ($command->queryAll() as $row) {
			$flat[TBayesianTokenHistogram::key((string) $row['fam'], (string) $row['category'], (int) $row['val'])] = (int) $row['n'];
		}
		foreach ($families as $family) {
			$command = $this->getDbConnection()->createCommand(sprintf('DELETE FROM %s_hist WHERE model = :model AND fam = :fam', $this->_table));
			$command->bindValue(':model', $name);
			$command->bindValue(':fam', $family);
			$command->execute();
		}
		$this->applyHistogramDelta($name, $flat);
		return count($flat);
	}

	/**
	 * Adds increments to histogram cells and removes the cells that reach zero.  The cells are
	 * written in sorted order so two writers take their row locks in the same sequence, and
	 * only the cells this call decremented are candidates for removal, addressed by their full
	 * key: a delete ranging over the model's cells would lock rows other writers hold.
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
		$emptied = [];
		foreach ($delta as $key => $change) {
			$parts = $change < 0 ? TBayesianTokenHistogram::parseKey((string) $key) : null;
			if ($parts !== null) {
				$emptied[$parts[0]][$parts[1]][] = $parts[2];
			}
		}
		foreach ($emptied as $family => $categories) {
			foreach ($categories as $category => $values) {
				foreach (array_chunk($values, self::TOKEN_CHUNK) as $chunk) {
					$placeholders = [];
					foreach ($chunk as $index => $_) {
						$placeholders[] = ':v' . $index;
					}
					$command = $this->getDbConnection()->createCommand(sprintf(
						'DELETE FROM %s WHERE model = :model AND fam = :fam AND category = :category AND val IN (%s) AND cnt <= 0',
						$table,
						implode(', ', $placeholders)
					));
					$command->bindValue(':model', $name);
					$command->bindValue(':fam', (string) $family);
					$command->bindValue(':category', (string) $category);
					foreach ($chunk as $index => $value) {
						$command->bindValue(':v' . $index, $value);
					}
					$command->execute();
				}
			}
		}
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

			$this->writeMetaRow($name, TBayesianPayload::encodeTokenMeta($meta, self::TOKEN_LAYOUT_VERSION));
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
	 * @param bool $trackDocuments Whether to report each row's document count before and after.
	 * @return array<string, array{0:int, 1:int}> The document count before and after the write,
	 * keyed by token storage key; empty unless asked for.
	 */
	private function incrementTokenRows(string $name, string $category, array $rows, bool $trackDocuments = false): array
	{
		$table = $this->_table . '_tokens';
		$transitions = [];
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
			if ($trackDocuments) {
				// The upsert holds this transaction's lock on every row of the batch, and has
				// added the positive deltas but not the negative ones.  Read now, the rows give
				// both ends of each transition exactly, whatever other writers are doing.
				foreach ($this->selectLockedDocumentCounts($name, $category, array_keys($batch)) as $key => $current) {
					$docCount = $batch[$key][1];
					$transitions[$key] = $docCount >= 0 ? [$current - $docCount, $current] : [$current, max(0, $current + $docCount)];
				}
			}
			$this->applyNegativeDeltas($name, $category, $batch);
		}
		return $transitions;
	}

	/**
	 * Reads the document counts of rows this transaction has already locked, in one category.
	 *
	 * The read names the category so that it touches only those rows: reaching for the same
	 * tokens' rows in other categories would take locks another writer may hold while waiting
	 * for these.  It is a locking read so that MySQL returns the latest committed values rather
	 * than the transaction's snapshot.
	 * @param string $name The model name.
	 * @param string $category The category name.
	 * @param array<int, int|string> $keys The token storage keys.
	 * @return array<string, int> The document counts, keyed by token storage key.
	 */
	private function selectLockedDocumentCounts(string $name, string $category, array $keys): array
	{
		$placeholders = [];
		foreach ($keys as $index => $_) {
			$placeholders[] = ':t' . $index;
		}
		$command = $this->getDbConnection()->createCommand(sprintf(
			'SELECT token, doccnt FROM %s_tokens WHERE model = :model AND category = :category AND token IN (%s)%s',
			$this->_table,
			implode(', ', $placeholders),
			$this->getDbConnection()->getDriverName() === 'sqlite' ? '' : ' FOR UPDATE'
		));
		$command->bindValue(':model', $name);
		$command->bindValue(':category', $category);
		foreach ($keys as $index => $key) {
			$command->bindValue(':t' . $index, (string) $key);
		}
		$out = [];
		foreach ($command->queryAll() as $row) {
			$out[(string) $row['token']] = (int) $row['doccnt'];
		}
		return $out;
	}

	/**
	 * Returns the document-count histogram increments of a set of row transitions: each row
	 * leaves the cell of its old document count and enters the cell of its new one, and a count
	 * of zero has no cell.
	 * @param string $category The category name.
	 * @param array<string, array{0:int, 1:int}> $transitions The document count before and after, per row.
	 * @return array<string, int> The increments, keyed by {@see TBayesianTokenHistogram::key()}.
	 */
	private static function documentCountDelta(string $category, array $transitions): array
	{
		$delta = [];
		foreach ($transitions as [$before, $after]) {
			if ($before > 0) {
				$key = TBayesianTokenHistogram::key(TBayesianTokenHistogram::FAMILY_DOCUMENTS, $category, $before);
				$delta[$key] = ($delta[$key] ?? 0) - 1;
			}
			if ($after > 0) {
				$key = TBayesianTokenHistogram::key(TBayesianTokenHistogram::FAMILY_DOCUMENTS, $category, $after);
				$delta[$key] = ($delta[$key] ?? 0) + 1;
			}
		}
		return array_filter($delta);
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
	 * @param string $suffix The table to insert into: the token rows, or their folded copy.
	 */
	private function insertTokenRows(string $name, array $rows, string $suffix = '_tokens'): void
	{
		if ($rows === []) {
			return;
		}
		$table = $this->_table . $suffix;
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
		foreach (['_tokens', '_categories', '_vocab', '_counters', '_hist', '_journal', '_folded'] as $suffix) {
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
