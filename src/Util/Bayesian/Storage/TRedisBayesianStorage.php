<?php

/**
 * TRedisBayesianStorage class file.
 *
 * @author Brad Anderson <belisoful@icloud.com>
 * @link https://github.com/belisoful/prado-bayesian
 * @license https://github.com/pradosoft/prado/blob/master/LICENSE
 */

namespace Belisoful\Prado\Util\Bayesian\Storage;

use Prado\Exceptions\TConfigurationException;
use Prado\Exceptions\TInvalidDataValueException;
use Prado\Exceptions\TInvalidOperationException;
use Prado\TComponent;
use Redis;

/**
 * TRedisBayesianStorage class.
 *
 * Redis-backed storage for shared hosts and for models shared across processes.  The class
 * refuses to construct itself when the `redis` extension is absent, and surfaces connect
 * failures as configuration exceptions.
 *
 * Two layouts, chosen by {@see setMode Mode}:
 *
 * - **`payload`** (default): the whole model is one JSON string under the model key, with a
 *   Redis set holding the index of model names.  Simple, and right for a model that fits.
 * - **`token`**: the model is spread across a metadata string, a categories hash, and one hash
 *   per token — so a classifier can score a document by reading only that document's tokens
 *   through {@see \Belisoful\Prado\Util\Bayesian\TLazyBayesianVocabulary}.  The model is then
 *   bounded by Redis rather than by PHP's memory limit, and incremental training uses `HINCRBY`
 *   so a document's counts land atomically with no read-modify-write.  Redis still holds the
 *   whole model in RAM, so this raises the per-process ceiling, not the machine's.
 *
 * The connection is opened lazily on the first call.  Callers can supply a fully configured
 * {@see Redis} instance via {@see setRedis()}, or the host/port/password triple via the matching
 * setters.
 *
 * @author Brad Anderson <belisoful@icloud.com>
 * @since 0.1.0
 */
class TRedisBayesianStorage extends TComponent implements IBayesianTokenStorage
{
	/** The whole model is stored as one JSON string under the model key. */
	public const MODE_PAYLOAD = 'payload';

	/** The model is stored across per-token hashes, read a document at a time. */
	public const MODE_TOKEN = 'token';

	/** @var string Either {@see MODE_PAYLOAD} or {@see MODE_TOKEN}. */
	private string $_mode = self::MODE_PAYLOAD;

	/** @var ?Redis The connection. */
	private ?Redis $_redis = null;

	/** @var string The key prefix; the model name is appended to form the storage key. */
	private string $_keyPrefix = 'bayesian:model:';

	/** @var string The Redis set key holding the index of model names. */
	private string $_indexKey = 'bayesian:models';

	/** @var string The Redis host. */
	private string $_host = '127.0.0.1';

	/** @var int The Redis port. */
	private int $_port = 6379;

	/** @var float The connect timeout, in seconds. */
	private float $_timeout = 2.5;

	/** @var ?string The optional Redis password. */
	private ?string $_password = null;

	/** @var int The optional Redis database number. */
	private int $_database = 0;

	/**
	 * Throws when the redis extension is not loaded.
	 * @throws TConfigurationException When ext-redis is missing.
	 */
	public function __construct()
	{
		if (!extension_loaded('redis')) {
			throw new TConfigurationException('bayesian_storage_redis_missing');
		}
		parent::__construct();
	}

	/**
	 * Returns the active Redis connection, opening it lazily from the configured host/port.
	 * @throws TConfigurationException When the connection cannot be established.
	 * @return Redis The connection.
	 */
	private function redis(): Redis
	{
		if ($this->_redis !== null) {
			return $this->_redis;
		}
		$redis = new Redis();
		try {
			$connected = $redis->connect($this->_host, $this->_port, $this->_timeout);
		} catch (\RedisException $e) {
			throw new TConfigurationException('bayesian_storage_redis_connect_failed', $e->getMessage());
		}
		if (!$connected) {
			throw new TConfigurationException('bayesian_storage_redis_connect_failed', $this->_host . ':' . $this->_port);
		}
		// phpredis returns false (rather than throwing) for AUTH/SELECT error replies; a
		// silently failed AUTH would make every later command fail with NOAUTH.
		try {
			if ($this->_password !== null && $this->_password !== '' && !$redis->auth($this->_password)) {
				throw new TConfigurationException('bayesian_storage_redis_connect_failed', $this->_host . ':' . $this->_port . ' (AUTH failed)');
			}
			if ($this->_database !== 0 && !$redis->select($this->_database)) {
				throw new TConfigurationException('bayesian_storage_redis_connect_failed', $this->_host . ':' . $this->_port . ' (SELECT ' . $this->_database . ' failed)');
			}
		} catch (\RedisException $e) {
			throw new TConfigurationException('bayesian_storage_redis_connect_failed', $e->getMessage());
		}
		$redis->setOption(Redis::OPT_SERIALIZER, (string) Redis::SERIALIZER_NONE);
		$this->_redis = $redis;
		return $redis;
	}

	/**
	 * Returns the Redis key backing a model: {@see getKeyPrefix KeyPrefix} followed by the
	 * name.  Unlike the file backend a path separator is harmless here, so only empty names
	 * and null bytes (which would truncate the key) are rejected.
	 * @param string $name The model name.
	 * @throws TInvalidDataValueException When the name is empty or contains a null byte.
	 * @return string The storage key.
	 */
	private function key(string $name): string
	{
		if ($name === '' || strpbrk($name, "\0") !== false) {
			throw new TInvalidDataValueException('bayesian_storage_name_invalid', $name);
		}
		return $this->_keyPrefix . $name;
	}

	/**
	 * Returns the hash key holding one token's per-category counts for a model.
	 *
	 * The token is part of the Redis key.  Redis keys are binary-safe, so a token needs no
	 * escaping here — unlike the file backend, where a token in the path could escape the
	 * directory.
	 * @param string $name The model name.
	 * @param string $token The token.
	 * @return string The Redis key.
	 */
	private function tokenKey(string $name, string $token): string
	{
		return $this->key($name) . ':__t:' . $token;
	}

	/**
	 * Returns the hash key holding a model's per-category scalars.
	 * @param string $name The model name.
	 * @return string The Redis key.
	 */
	private function categoriesKey(string $name): string
	{
		return $this->key($name) . ':__cat';
	}

	/**
	 * Returns the set key holding a model's distinct tokens, so a delete can find every
	 * per-token hash without scanning the keyspace.
	 * @param string $name The model name.
	 * @return string The Redis key.
	 */
	private function tokenSetKey(string $name): string
	{
		return $this->key($name) . ':__toks';
	}

	/**
	 * Returns the hash field a token's count for a category is stored under.
	 *
	 * The field is one type byte (`c` count, `d` document count) followed by the raw category
	 * name.  A single fixed-width prefix keeps `c`/`d` from ever colliding and keeps the
	 * category unescaped, so any category name round-trips.
	 * @param string $type Either `c` or `d`.
	 * @param string $category The category name.
	 * @return string The hash field.
	 */
	private static function tokenField(string $type, string $category): string
	{
		return $type . $category;
	}

	/**
	 * Parses one token's `HGETALL` hash back into per-category counts.
	 *
	 * The inverse of the {@see tokenField()} encoding: fields are `c<category>` and
	 * `d<category>`, and this regroups them by category into the `{count, docCount}` shape the
	 * lazy vocabulary reads.  A category present with only one of the two fields still yields a
	 * complete pair, the missing half defaulting to zero.
	 * @param array<string, mixed> $hash The raw hash from Redis.
	 * @return array<string, array{count:int, docCount:int}> The per-category counts.
	 */
	private static function parseTokenHash(array $hash): array
	{
		$out = [];
		foreach ($hash as $field => $value) {
			$field = (string) $field;
			if ($field === '') {
				continue;
			}
			$type = $field[0];
			$category = substr($field, 1);
			if ($type !== 'c' && $type !== 'd') {
				continue;
			}
			if (!isset($out[$category])) {
				$out[$category] = ['count' => 0, 'docCount' => 0];
			}
			$out[$category][$type === 'c' ? 'count' : 'docCount'] = (int) $value;
		}
		return $out;
	}

	/**
	 * Packs a category's scalars into the value stored under its field in the categories hash.
	 * @param int $documentCount The document count.
	 * @param int $totalTokens The total token occurrences.
	 * @return string The packed value.
	 */
	private static function packCategoryScalars(int $documentCount, int $totalTokens): string
	{
		return $documentCount . ':' . $totalTokens;
	}

	/**
	 * Unpacks the value written by {@see packCategoryScalars()}.
	 * @param string $value The packed value.
	 * @return array{documentCount:int, totalTokens:int} The scalars.
	 */
	private static function unpackCategoryScalars(string $value): array
	{
		$parts = explode(':', $value, 2);
		return [
			'documentCount' => (int) ($parts[0] ?? 0),
			'totalTokens' => (int) ($parts[1] ?? 0),
		];
	}

	/**
	 * Persists a payload under a model name.
	 * @param string $name The model name.
	 * @param array<string, mixed> $payload The payload.
	 * @throws TConfigurationException When the connection cannot be established.
	 * @throws TInvalidDataValueException When the name is invalid or the payload cannot be JSON-encoded.
	 * @throws TInvalidOperationException When Redis rejects the write.
	 * @throws \RedisException On a connection failure mid-command.
	 */
	public function save(string $name, array $payload): void
	{
		$key = $this->key($name);
		$encoded = json_encode($payload, JSON_UNESCAPED_SLASHES | JSON_UNESCAPED_UNICODE);
		if ($encoded === false) {
			throw new TInvalidDataValueException('bayesian_storage_encode_failed', json_last_error_msg());
		}
		$redis = $this->redis();
		if ($redis->set($key, $encoded) !== true) {
			throw new TInvalidOperationException('bayesian_storage_redis_write_failed', $key);
		}
		if ($redis->sAdd($this->_indexKey, $name) === false) {
			throw new TInvalidOperationException('bayesian_storage_redis_write_failed', $this->_indexKey);
		}
	}

	/**
	 * Loads a payload by name; null when the key is absent or does not hold valid JSON.
	 * @param string $name The model name.
	 * @throws TConfigurationException When the connection cannot be established.
	 * @throws TInvalidDataValueException When the name is invalid.
	 * @throws \RedisException On a connection failure mid-command.
	 * @return ?array<string, mixed> The payload, or null.
	 */
	public function load(string $name): ?array
	{
		$raw = $this->redis()->get($this->key($name));
		if ($raw === false || $raw === null) {
			return null;
		}
		$decoded = json_decode((string) $raw, true);
		if (!is_array($decoded)) {
			return null;
		}
		return $decoded;
	}

	/**
	 * Returns whether a payload with the given name is stored.
	 * @param string $name The model name.
	 * @throws TConfigurationException When the connection cannot be established.
	 * @throws TInvalidDataValueException When the name is invalid.
	 * @throws \RedisException On a connection failure mid-command.
	 * @return bool Whether the model is present.
	 */
	public function exists(string $name): bool
	{
		return (bool) $this->redis()->exists($this->key($name));
	}

	/**
	 * Removes a payload.  Removing a non-existent name is a no-op.
	 * @param string $name The model name.
	 * @throws TConfigurationException When the connection cannot be established.
	 * @throws TInvalidDataValueException When the name is invalid.
	 * @throws TInvalidOperationException When Redis rejects the delete.
	 * @throws \RedisException On a connection failure mid-command.
	 */
	public function delete(string $name): void
	{
		$key = $this->key($name);
		$redis = $this->redis();
		if ($this->_mode === self::MODE_TOKEN) {
			// The model is spread across the meta key, the categories hash, one hash per token,
			// and the token set.  Leaving any of those behind would resurrect part of the model
			// on the next save under the same name, so all of them go.
			$tokens = $redis->sMembers($this->tokenSetKey($name)) ?: [];
			$keys = [$key, $this->categoriesKey($name), $this->tokenSetKey($name)];
			foreach ($tokens as $token) {
				$keys[] = $this->tokenKey($name, (string) $token);
			}
			if ($redis->del($keys) === false) {
				throw new TInvalidOperationException('bayesian_storage_redis_write_failed', $key);
			}
		} elseif ($redis->del($key) === false) {
			throw new TInvalidOperationException('bayesian_storage_redis_write_failed', $key);
		}
		if ($redis->sRem($this->_indexKey, $name) === false) {
			throw new TInvalidOperationException('bayesian_storage_redis_write_failed', $this->_indexKey);
		}
	}

	/**
	 * Returns whether the storage is configured for per-token lookup.
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
	 * `payload` (the default) stores the whole model as one JSON string; `token` spreads it
	 * across per-token hashes so a classifier can score a document by reading only that
	 * document's tokens, and the model is bounded by Redis rather than by PHP's memory limit.
	 * The two are different layouts of the same model — one is not readable in the other mode,
	 * so re-save a model after changing this.
	 * @param string $value Either `payload` or `token`.
	 * @throws TInvalidDataValueException When the value is neither mode.
	 */
	public function setMode(string $value): void
	{
		if ($value !== self::MODE_PAYLOAD && $value !== self::MODE_TOKEN) {
			throw new TInvalidDataValueException('bayesian_storage_mode_invalid', $value);
		}
		$this->_mode = $value;
	}

	/**
	 * Ensures the storage is in per-token mode, returning the live connection.
	 * @throws TInvalidOperationException When the storage is in payload mode.
	 * @return Redis The connection.
	 */
	private function requireTokenMode(): Redis
	{
		if ($this->_mode !== self::MODE_TOKEN) {
			throw new TInvalidOperationException('bayesian_storage_token_mode_required');
		}
		return $this->redis();
	}

	/**
	 * Writes a model in per-token form, replacing whatever was stored under the name.
	 *
	 * The whole write is one Redis transaction, so a model is never left half-replaced.  The
	 * old model's token hashes are read first (outside the transaction, which cannot read) and
	 * deleted inside it, since the new model may not cover every token the old one did.
	 * @param string $name The model name.
	 * @param array<string, mixed> $meta The model-level state.
	 * @param array<string, array{documentCount:int, totalTokens:int}> $categories The category scalars.
	 * @param array<string, array<string, array{count:int, docCount:int}>> $tokens The per-token statistics.
	 * @throws TInvalidOperationException When the storage is not in per-token mode, or Redis rejects the write.
	 * @throws TInvalidDataValueException When the metadata cannot be JSON-encoded.
	 */
	public function saveTokenModel(string $name, array $meta, array $categories, array $tokens): void
	{
		$redis = $this->requireTokenMode();
		$encoded = json_encode($meta, JSON_UNESCAPED_SLASHES | JSON_UNESCAPED_UNICODE);
		if ($encoded === false) {
			throw new TInvalidDataValueException('bayesian_storage_encode_failed', json_last_error_msg());
		}
		$metaKey = $this->key($name);
		$categoriesKey = $this->categoriesKey($name);
		$tokenSetKey = $this->tokenSetKey($name);
		$staleTokens = $redis->sMembers($tokenSetKey) ?: [];

		$tx = $redis->multi();
		$tx->del([$metaKey, $categoriesKey, $tokenSetKey]);
		foreach ($staleTokens as $token) {
			$tx->del($this->tokenKey($name, (string) $token));
		}
		$tx->set($metaKey, $encoded);
		$tx->sAdd($this->_indexKey, $name);
		foreach ($categories as $category => $stats) {
			$tx->hSet($categoriesKey, (string) $category, self::packCategoryScalars((int) ($stats['documentCount'] ?? 0), (int) ($stats['totalTokens'] ?? 0)));
		}
		foreach ($tokens as $token => $perCategory) {
			$token = (string) $token;
			$tokenKey = $this->tokenKey($name, $token);
			foreach ($perCategory as $category => $stats) {
				$category = (string) $category;
				$tx->hSet($tokenKey, self::tokenField('c', $category), (string) (int) ($stats['count'] ?? 0));
				$tx->hSet($tokenKey, self::tokenField('d', $category), (string) (int) ($stats['docCount'] ?? 0));
			}
			$tx->sAdd($tokenSetKey, $token);
		}
		$tx->exec();
	}

	/**
	 * Returns a model's model-level state, or null when the name is unknown.
	 * @param string $name The model name.
	 * @throws TInvalidOperationException When the storage is not in per-token mode.
	 * @return ?array<string, mixed> The metadata, or null.
	 */
	public function loadTokenMeta(string $name): ?array
	{
		$this->requireTokenMode();
		$raw = $this->redis()->get($this->key($name));
		if ($raw === false || $raw === null) {
			return null;
		}
		$decoded = json_decode((string) $raw, true);
		return is_array($decoded) ? $decoded : null;
	}

	/**
	 * Returns a model's per-category scalars, keyed by category name and sorted by it, matching
	 * the SQL backend so both order categories the same way.
	 * @param string $name The model name.
	 * @throws TInvalidOperationException When the storage is not in per-token mode.
	 * @return array<string, array{documentCount:int, totalTokens:int}> The category scalars.
	 */
	public function loadTokenCategories(string $name): array
	{
		$redis = $this->requireTokenMode();
		$hash = $redis->hGetAll($this->categoriesKey($name)) ?: [];
		$out = [];
		foreach ($hash as $category => $packed) {
			$out[(string) $category] = self::unpackCategoryScalars((string) $packed);
		}
		ksort($out);
		return $out;
	}

	/**
	 * Returns the per-category statistics of the given tokens, in one pipelined round trip.
	 *
	 * Redis has no bind-parameter limit, so the whole document goes in one pipeline of
	 * `HGETALL`s rather than being chunked the way the SQL `IN` list is.
	 * @param string $name The model name.
	 * @param string[] $tokens The tokens to fetch.
	 * @throws TInvalidOperationException When the storage is not in per-token mode.
	 * @return array<string, array<string, array{count:int, docCount:int}>> The statistics.
	 */
	public function loadTokens(string $name, array $tokens): array
	{
		$redis = $this->requireTokenMode();
		$unique = array_values(array_unique(array_map('strval', $tokens)));
		if ($unique === []) {
			return [];
		}
		$pipe = $redis->multi(Redis::PIPELINE);
		foreach ($unique as $token) {
			$pipe->hGetAll($this->tokenKey($name, $token));
		}
		$results = $pipe->exec();
		$out = [];
		foreach ($unique as $i => $token) {
			$hash = $results[$i] ?? [];
			if (!is_array($hash) || $hash === []) {
				continue;
			}
			$out[$token] = self::parseTokenHash($hash);
		}
		return $out;
	}

	/**
	 * Applies one training document's deltas without rewriting the model.
	 *
	 * The increments use `HINCRBY`, which is atomic per field, so this needs no read of the old
	 * counts — the race a read-modify-write would have is gone, and the whole document still
	 * lands in one transaction.
	 * @param string $name The model name.
	 * @param string $category The category the document was filed under.
	 * @param array<string, array{count:int, docCount:int}> $tokenDeltas The per-token increments.
	 * @param array<string, mixed> $meta The updated model-level state.
	 * @param array{documentCount:int, totalTokens:int} $categoryStats The category's updated scalars.
	 * @throws TInvalidOperationException When the storage is not in per-token mode.
	 * @throws TInvalidDataValueException When the metadata cannot be JSON-encoded.
	 */
	public function applyDeltas(string $name, string $category, array $tokenDeltas, array $meta, array $categoryStats): void
	{
		$redis = $this->requireTokenMode();
		$encoded = json_encode($meta, JSON_UNESCAPED_SLASHES | JSON_UNESCAPED_UNICODE);
		if ($encoded === false) {
			throw new TInvalidDataValueException('bayesian_storage_encode_failed', json_last_error_msg());
		}
		$countField = self::tokenField('c', $category);
		$docField = self::tokenField('d', $category);
		$tokenSetKey = $this->tokenSetKey($name);

		$tx = $redis->multi();
		$tx->set($this->key($name), $encoded);
		$tx->sAdd($this->_indexKey, $name);
		$tx->hSet($this->categoriesKey($name), $category, self::packCategoryScalars((int) ($categoryStats['documentCount'] ?? 0), (int) ($categoryStats['totalTokens'] ?? 0)));
		foreach ($tokenDeltas as $token => $delta) {
			$token = (string) $token;
			$tokenKey = $this->tokenKey($name, $token);
			$tx->hIncrBy($tokenKey, $countField, (int) ($delta['count'] ?? 0));
			$tx->hIncrBy($tokenKey, $docField, (int) ($delta['docCount'] ?? 0));
			$tx->sAdd($tokenSetKey, $token);
		}
		$tx->exec();
	}

	/**
	 * Returns the names of all stored models, in sorted order.
	 * @throws TConfigurationException When the connection cannot be established.
	 * @throws \RedisException On a connection failure mid-command.
	 * @return string[] The model names, in sorted order.
	 */
	public function list(): array
	{
		$members = $this->redis()->sMembers($this->_indexKey) ?: [];
		$names = array_map('strval', $members);
		sort($names);
		return $names;
	}

	/**
	 * Returns the Redis host.
	 * @return string The host.
	 */
	public function getHost(): string
	{
		return $this->_host;
	}

	/**
	 * Sets the Redis host.
	 * @param string $value The host.
	 */
	public function setHost(string $value): void
	{
		$this->_host = $value;
		$this->_redis = null;
	}

	/**
	 * Returns the Redis port.
	 * @return int The port.
	 */
	public function getPort(): int
	{
		return $this->_port;
	}

	/**
	 * Sets the Redis port.
	 * @param int $value The port.
	 */
	public function setPort(int $value): void
	{
		$this->_port = $value;
		$this->_redis = null;
	}

	/**
	 * Returns the connect timeout in seconds.
	 * @return float The timeout.
	 */
	public function getTimeout(): float
	{
		return $this->_timeout;
	}

	/**
	 * Sets the connect timeout in seconds.
	 * @param float $value The timeout.
	 */
	public function setTimeout(float $value): void
	{
		$this->_timeout = $value;
		$this->_redis = null;
	}

	/**
	 * Returns the optional password.
	 * @return ?string The password.
	 */
	public function getPassword(): ?string
	{
		return $this->_password;
	}

	/**
	 * Sets the password.
	 * @param ?string $value The password.
	 */
	public function setPassword(?string $value): void
	{
		$this->_password = $value;
		$this->_redis = null;
	}

	/**
	 * Returns the database number.
	 * @return int The database number.
	 */
	public function getDatabase(): int
	{
		return $this->_database;
	}

	/**
	 * Sets the database number.
	 * @param int $value The database number.
	 */
	public function setDatabase(int $value): void
	{
		$this->_database = $value;
		$this->_redis = null;
	}

	/**
	 * Returns the key prefix.
	 * @return string The key prefix.
	 */
	public function getKeyPrefix(): string
	{
		return $this->_keyPrefix;
	}

	/**
	 * Sets the key prefix.
	 * @param string $value The key prefix.
	 */
	public function setKeyPrefix(string $value): void
	{
		$this->_keyPrefix = $value;
	}

	/**
	 * Returns the index set key.
	 * @return string The index key.
	 */
	public function getIndexKey(): string
	{
		return $this->_indexKey;
	}

	/**
	 * Sets the index set key.
	 * @param string $value The index key.
	 */
	public function setIndexKey(string $value): void
	{
		$this->_indexKey = $value;
	}

	/**
	 * Sets the Redis connection directly (bypasses host/port-based construction).
	 * @param Redis $redis The connection.
	 */
	public function setRedis(Redis $redis): void
	{
		$this->_redis = $redis;
	}

	/**
	 * Returns the active Redis connection, opening it lazily if needed.
	 * @throws TConfigurationException When the connection cannot be established.
	 * @return Redis The connection.
	 */
	public function getRedis(): Redis
	{
		return $this->redis();
	}
}
