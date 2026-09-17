<?php

/**
 * TRedisBayesianStorage class file.
 *
 * @author Brad Anderson <belisoful@icloud.com>
 * @link https://github.com/belisoful/prado-bayesian
 * @license https://github.com/pradosoft/prado/blob/master/LICENSE
 */

namespace Belisoful\Prado\Util\Bayesian\Storage;

use Belisoful\Prado\Util\Bayesian\TBayesianPayload;
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
 *   Redis set holding the index of model names.  Simple, and right for a model that fits and
 *   has one writer: a save replaces the whole value, so two processes that each load, train
 *   and save the same model lose one another's documents.
 * - **`token`**: the model is spread across a metadata string, two category hashes, a token
 *   set, and one hash per token — so a classifier can score a document by reading only that
 *   document's tokens through {@see \Belisoful\Prado\Util\Bayesian\TLazyBayesianVocabulary}.
 *   The model is then bounded by Redis rather than by PHP's memory limit, and every training
 *   write is one Lua script applying `HINCRBY` increments, so any number of processes may
 *   train one model at once without losing counts.  Redis still holds the whole model in RAM,
 *   so this raises the per-process ceiling, not the machine's.
 *
 * The per-token layout carries a `layoutVersion` in the metadata.  A model written by 0.1.0
 * (layout 1) is upgraded in place on first read; a newer layout than this release understands
 * is refused with `bayesian_storage_layout_unsupported`.  Scripts touch keys they derive from
 * the model name, which a Redis Cluster does not permit; use a single Redis instance (or
 * Sentinel-managed replicas) for this storage.
 *
 * Keys are `KeyPrefix` followed by the model name, and the per-token layout appends `:__t:`,
 * `:__catdocs`, `:__cattoks` and `:__toks` to that.  A model name may therefore contain a
 * colon but not the sequence `:__`, and two applications sharing one Redis must set both
 * `KeyPrefix` and `IndexKey` to keep their models apart.
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

	/**
	 * The per-token layout this release writes.  Layout 1 (0.1.0) packed each category's
	 * scalars as one `docs:tokens` string in a single hash and kept the model totals in the
	 * metadata; layout 2 keeps the scalars as integer hash fields so they can be incremented
	 * and derives the totals from the keys.
	 * @since 0.2.0
	 */
	public const TOKEN_LAYOUT_VERSION = 2;

	/** @var string The sequence the per-token sub-keys start with; reserved in model names. */
	private const RESERVED = ':__';

	/** @var int How many tokens one training script call carries. */
	private const TOKEN_CHUNK = 1000;

	/**
	 * @var string Applies one document's token deltas.  KEYS[1] is the token set and KEYS[2..]
	 * the token hashes; ARGV[1] is the category, then (token, count, docCount) per hash.  Every
	 * counter is clamped at zero after its increment, and a token that no category's documents
	 * contain any more after a negative document delta is deleted and leaves the token set.
	 * Returns how many tokens were new.
	 */
	private const TOKEN_DELTA_SCRIPT = <<<'LUA'
		local set = KEYS[1]
		local cat = ARGV[1]
		local added = 0
		for i = 1, #KEYS - 1 do
			local key = KEYS[i + 1]
			local base = 1 + (i - 1) * 3
			if redis.call('SADD', set, ARGV[base + 1]) == 1 then
				added = added + 1
			end
			if redis.call('HINCRBY', key, 'c' .. cat, tonumber(ARGV[base + 2])) < 0 then
				redis.call('HSET', key, 'c' .. cat, 0)
			end
			local docs = tonumber(ARGV[base + 3])
			if redis.call('HINCRBY', key, 'd' .. cat, docs) < 0 then
				redis.call('HSET', key, 'd' .. cat, 0)
			end
			if docs < 0 then
				local alive = false
				local fields = redis.call('HGETALL', key)
				for j = 1, #fields, 2 do
					if string.sub(fields[j], 1, 1) == 'd' and tonumber(fields[j + 1]) > 0 then
						alive = true
						break
					end
				end
				if not alive then
					redis.call('DEL', key)
					redis.call('SREM', set, ARGV[base + 1])
				end
			end
		end
		return added
		LUA;

	/**
	 * @var string Applies one document's category deltas.  KEYS[1] is the document-count hash
	 * and KEYS[2] the token-total hash; ARGV[1] is the category, ARGV[2] and ARGV[3] the
	 * deltas.  Both counters are clamped at zero.
	 */
	private const CATEGORY_DELTA_SCRIPT = <<<'LUA'
		if redis.call('HINCRBY', KEYS[1], ARGV[1], tonumber(ARGV[2])) < 0 then
			redis.call('HSET', KEYS[1], ARGV[1], 0)
		end
		if redis.call('HINCRBY', KEYS[2], ARGV[1], tonumber(ARGV[3])) < 0 then
			redis.call('HSET', KEYS[2], ARGV[1], 0)
		end
		return 1
		LUA;

	/**
	 * @var string Deletes a per-token model atomically, so a trainer racing the delete cannot
	 * leave orphan token hashes.  KEYS are the metadata, category, legacy category, token-set
	 * and index keys; ARGV[1] is the model name and ARGV[2] the token-key prefix.
	 */
	private const DELETE_SCRIPT = <<<'LUA'
		local tokens = redis.call('SMEMBERS', KEYS[5])
		for _, token in ipairs(tokens) do
			redis.call('DEL', ARGV[2] .. token)
		end
		redis.call('DEL', KEYS[1], KEYS[2], KEYS[3], KEYS[4], KEYS[5])
		redis.call('SREM', KEYS[6], ARGV[1])
		return #tokens
		LUA;

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

	/** @var array<string, true> The models this instance has verified to be at the current layout. */
	private array $_layoutChecked = [];

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
	 * name.  A path separator is harmless here, so a name is rejected only when it is empty,
	 * contains a null byte (which would truncate the key), or contains `:__`, the sequence the
	 * per-token sub-keys start with — a model named `m:__t:x` would share a key with token `x`
	 * of model `m`.
	 * @param string $name The model name.
	 * @throws TInvalidDataValueException When the name is invalid.
	 * @return string The storage key.
	 */
	private function key(string $name): string
	{
		if ($name === '' || strpbrk($name, "\0") !== false || str_contains($name, self::RESERVED)) {
			throw new TInvalidDataValueException('bayesian_storage_name_invalid', $name);
		}
		return $this->_keyPrefix . $name;
	}

	/**
	 * Returns the prefix of a model's per-token hash keys; the raw token follows it.  Redis
	 * keys are binary-safe, so a token needs no escaping — unlike the file backend, where a
	 * token in a path could escape the directory.
	 * @param string $name The model name.
	 * @return string The key prefix.
	 */
	private function tokenKeyPrefix(string $name): string
	{
		return $this->key($name) . self::RESERVED . 't:';
	}

	/**
	 * Returns the hash key holding one token's per-category counts for a model.
	 * @param string $name The model name.
	 * @param string $token The token.
	 * @return string The Redis key.
	 */
	private function tokenKey(string $name, string $token): string
	{
		return $this->tokenKeyPrefix($name) . $token;
	}

	/**
	 * Returns the hash key holding a model's per-category document counts.
	 * @param string $name The model name.
	 * @return string The Redis key.
	 */
	private function categoryDocumentsKey(string $name): string
	{
		return $this->key($name) . self::RESERVED . 'catdocs';
	}

	/**
	 * Returns the hash key holding a model's per-category token totals.
	 * @param string $name The model name.
	 * @return string The Redis key.
	 */
	private function categoryTokensKey(string $name): string
	{
		return $this->key($name) . self::RESERVED . 'cattoks';
	}

	/**
	 * Returns the hash key layout 1 kept a model's packed category scalars under.
	 * @param string $name The model name.
	 * @return string The Redis key.
	 */
	private function legacyCategoriesKey(string $name): string
	{
		return $this->key($name) . self::RESERVED . 'cat';
	}

	/**
	 * Returns the set key holding a model's distinct tokens.  Its cardinality is the
	 * vocabulary size, and a delete finds every per-token hash through it without scanning
	 * the keyspace.
	 * @param string $name The model name.
	 * @return string The Redis key.
	 */
	private function tokenSetKey(string $name): string
	{
		return $this->key($name) . self::RESERVED . 'toks';
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
			$out[$category][$type === 'c' ? 'count' : 'docCount'] = TBayesianPayload::int($value);
		}
		return $out;
	}

	/**
	 * Unpacks a layout-1 category value, `<documentCount>:<totalTokens>`.
	 * @param string $value The packed value.
	 * @return array{documentCount:int, totalTokens:int} The scalars.
	 */
	private static function unpackCategoryScalars(string $value): array
	{
		$parts = explode(':', $value, 2);
		return [
			'documentCount' => (int) $parts[0],
			'totalTokens' => (int) ($parts[1] ?? 0),
		];
	}

	/**
	 * Encodes the model-level metadata for the per-token layout: the totals are dropped (the
	 * storage derives them from its keys) and the layout version is added.
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
	 * Reads and decodes a model's metadata string.
	 * @param string $name The model name.
	 * @return ?array<string, mixed> The metadata, or null when absent or not JSON.
	 */
	private function readMeta(string $name): ?array
	{
		$raw = $this->redis()->get($this->key($name));
		if (!is_string($raw)) {
			return null;
		}
		$decoded = json_decode($raw, true);
		return is_array($decoded) ? TBayesianPayload::map($decoded) : null;
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
		unset($this->_layoutChecked[$name]);
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
		return $this->readMeta($name);
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
	 *
	 * In per-token mode the model is spread across the metadata key, the category hashes, one
	 * hash per token, and the token set; one Lua script removes all of them, so a trainer
	 * racing the delete cannot leave part of the model behind to resurrect on the next save.
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
			$result = $redis->eval(self::DELETE_SCRIPT, [
				$key,
				$this->categoryDocumentsKey($name),
				$this->categoryTokensKey($name),
				$this->legacyCategoriesKey($name),
				$this->tokenSetKey($name),
				$this->_indexKey,
				$name,
				$this->tokenKeyPrefix($name),
			], 6);
			if ($result === false) {
				throw new TInvalidOperationException('bayesian_storage_redis_write_failed', $key);
			}
			unset($this->_layoutChecked[$name]);
			return;
		}
		if ($redis->del($key) === false) {
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
	 * document's tokens, the model is bounded by Redis rather than by PHP's memory limit, and
	 * many processes may train it at once.  The two are different layouts of the same model —
	 * one is not readable in the other mode, so re-save a model after changing this.
	 * @param string $value Either `payload` or `token`.
	 * @throws TInvalidDataValueException When the value is neither mode.
	 */
	public function setMode(string $value): void
	{
		if ($value !== self::MODE_PAYLOAD && $value !== self::MODE_TOKEN) {
			throw new TInvalidDataValueException('bayesian_storage_mode_invalid', $value);
		}
		$this->_mode = $value;
		$this->_layoutChecked = [];
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
	 * Checks the layout version of a model once per instance, upgrading a model written by an
	 * earlier release and refusing one written by a later release.
	 *
	 * A model with no metadata but a layout-1 category hash (only a hand-built fixture looks
	 * like that) is upgraded too, so the category read never has to guess at the packing.
	 * @param string $name The model name.
	 * @throws TInvalidDataValueException When the model's layout is newer than this release reads.
	 */
	private function ensureLayout(string $name): void
	{
		if (isset($this->_layoutChecked[$name])) {
			return;
		}
		$redis = $this->redis();
		$meta = $this->readMeta($name);
		if ($meta === null) {
			if (!$redis->exists($this->legacyCategoriesKey($name))) {
				return;
			}
			$version = 1;
		} else {
			$version = TBayesianPayload::int($meta['layoutVersion'] ?? null, 1);
		}
		if ($version > self::TOKEN_LAYOUT_VERSION) {
			throw new TInvalidDataValueException('bayesian_storage_layout_unsupported', $name, (string) $version, (string) self::TOKEN_LAYOUT_VERSION);
		}
		if ($version < self::TOKEN_LAYOUT_VERSION) {
			$this->upgradeLayout($name, $meta);
		}
		$this->_layoutChecked[$name] = true;
	}

	/**
	 * Brings a layout-1 model to the current layout in one transaction: the packed category
	 * hash is split into the two integer hashes and the metadata is stamped.
	 * @param string $name The model name.
	 * @param ?array<string, mixed> $meta The model's metadata, when it has any.
	 */
	private function upgradeLayout(string $name, ?array $meta): void
	{
		$redis = $this->redis();
		$legacyKey = $this->legacyCategoriesKey($name);
		$packed = $redis->hGetAll($legacyKey);
		$tx = $redis->multi();
		if (is_array($packed)) {
			foreach ($packed as $category => $value) {
				$scalars = self::unpackCategoryScalars((string) $value);
				$tx->hSet($this->categoryDocumentsKey($name), (string) $category, (string) $scalars['documentCount']);
				$tx->hSet($this->categoryTokensKey($name), (string) $category, (string) $scalars['totalTokens']);
			}
		}
		$tx->del($legacyKey);
		if ($meta !== null) {
			$tx->set($this->key($name), $this->encodeMeta($meta));
		}
		$tx->exec();
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
		$encoded = $this->encodeMeta($meta);
		$metaKey = $this->key($name);
		$documentsKey = $this->categoryDocumentsKey($name);
		$totalsKey = $this->categoryTokensKey($name);
		$tokenSetKey = $this->tokenSetKey($name);
		$staleTokens = $redis->sMembers($tokenSetKey);

		$tx = $redis->multi();
		$tx->del([$metaKey, $documentsKey, $totalsKey, $this->legacyCategoriesKey($name), $tokenSetKey]);
		if (is_array($staleTokens)) {
			foreach ($staleTokens as $token) {
				$tx->del($this->tokenKey($name, (string) $token));
			}
		}
		$tx->set($metaKey, $encoded);
		$tx->sAdd($this->_indexKey, $name);
		foreach ($categories as $category => $stats) {
			$tx->hSet($documentsKey, (string) $category, (string) max(0, (int) $stats['documentCount']));
			$tx->hSet($totalsKey, (string) $category, (string) max(0, (int) $stats['totalTokens']));
		}
		foreach ($tokens as $token => $perCategory) {
			$token = (string) $token;
			$tokenKey = $this->tokenKey($name, $token);
			foreach ($perCategory as $category => $stats) {
				$category = (string) $category;
				$tx->hSet($tokenKey, self::tokenField('c', $category), (string) max(0, (int) $stats['count']));
				$tx->hSet($tokenKey, self::tokenField('d', $category), (string) max(0, (int) $stats['docCount']));
			}
			$tx->sAdd($tokenSetKey, $token);
		}
		$tx->exec();
		$this->_layoutChecked[$name] = true;
	}

	/**
	 * Returns a model's model-level state, or null when the name is unknown.
	 *
	 * The document total is the sum of the category document counts and the vocabulary size
	 * the cardinality of the token set, so both reflect every write that has landed.
	 * @param string $name The model name.
	 * @throws TInvalidOperationException When the storage is not in per-token mode.
	 * @throws TInvalidDataValueException When the model's layout is newer than this release reads.
	 * @return ?array<string, mixed> The metadata, or null.
	 */
	public function loadTokenMeta(string $name): ?array
	{
		$redis = $this->requireTokenMode();
		$this->ensureLayout($name);
		$meta = $this->readMeta($name);
		if ($meta === null) {
			return null;
		}
		$total = 0;
		$documents = $redis->hGetAll($this->categoryDocumentsKey($name));
		if (is_array($documents)) {
			foreach ($documents as $count) {
				$total += TBayesianPayload::int($count);
			}
		}
		$meta['totalDocuments'] = $total;
		$meta['vocabularySize'] = (int) $redis->sCard($this->tokenSetKey($name));
		$meta['layoutVersion'] = self::TOKEN_LAYOUT_VERSION;
		return $meta;
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
		$this->ensureLayout($name);
		$documents = $redis->hGetAll($this->categoryDocumentsKey($name));
		$totals = $redis->hGetAll($this->categoryTokensKey($name));
		$out = [];
		if (is_array($documents)) {
			foreach ($documents as $category => $count) {
				$out[(string) $category] = ['documentCount' => (int) $count, 'totalTokens' => 0];
			}
		}
		if (is_array($totals)) {
			foreach ($totals as $category => $count) {
				$category = (string) $category;
				if (!isset($out[$category])) {
					$out[$category] = ['documentCount' => 0, 'totalTokens' => 0];
				}
				$out[$category]['totalTokens'] = (int) $count;
			}
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
		if (!is_array($results)) {
			return [];
		}
		foreach ($unique as $i => $token) {
			$hash = $results[$i] ?? [];
			if (!is_array($hash) || $hash === []) {
				continue;
			}
			$out[$token] = self::parseTokenHash(TBayesianPayload::map($hash));
		}
		return $out;
	}

	/**
	 * Applies one training document's deltas without rewriting the model.
	 *
	 * The token increments, the category increments and the metadata write run inside one
	 * `MULTI`, and the increments themselves are Lua scripts using `HINCRBY`, so nothing is
	 * read, added in PHP and written back: two writers training at once both land, in either
	 * order.  A negative delta is accepted and clamped at zero, and a token no document contains
	 * any more leaves the vocabulary, so the model ends as it would have without the document.
	 * @param string $name The model name.
	 * @param string $category The category the document was filed under.
	 * @param array<string, array{count?:int, docCount?:int}> $tokenDeltas The per-token increments.
	 * @param array<string, mixed> $meta The model-level state to store alongside; `totalDocuments`
	 * and `vocabularySize` in it are ignored, and an empty array leaves the stored metadata as it is.
	 * @param array{documentCount?:int, totalTokens?:int} $categoryStats The category's increments.
	 * @throws TInvalidOperationException When the storage is not in per-token mode.
	 * @throws TInvalidDataValueException When the metadata cannot be JSON-encoded, or the model's
	 * layout is newer than this release reads.
	 */
	public function applyDeltas(string $name, string $category, array $tokenDeltas, array $meta, array $categoryStats): void
	{
		$redis = $this->requireTokenMode();
		$this->ensureLayout($name);
		$encoded = $meta === [] ? null : $this->encodeMeta($meta);
		$tokenSetKey = $this->tokenSetKey($name);

		$tx = $redis->multi();
		foreach (array_chunk($tokenDeltas, self::TOKEN_CHUNK, true) as $chunk) {
			$keys = [$tokenSetKey];
			$args = [$category];
			foreach ($chunk as $token => $delta) {
				$token = (string) $token;
				$keys[] = $this->tokenKey($name, $token);
				$args[] = $token;
				$args[] = (string) (int) ($delta['count'] ?? 0);
				$args[] = (string) (int) ($delta['docCount'] ?? 0);
			}
			$tx->eval(self::TOKEN_DELTA_SCRIPT, array_merge($keys, $args), count($keys));
		}
		$tx->eval(self::CATEGORY_DELTA_SCRIPT, [
			$this->categoryDocumentsKey($name),
			$this->categoryTokensKey($name),
			$category,
			(string) (int) ($categoryStats['documentCount'] ?? 0),
			(string) (int) ($categoryStats['totalTokens'] ?? 0),
		], 2);
		$tx->sAdd($this->_indexKey, $name);
		if ($encoded !== null) {
			$tx->set($this->key($name), $encoded);
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
		$members = $this->redis()->sMembers($this->_indexKey);
		$names = [];
		if (is_array($members)) {
			foreach ($members as $member) {
				$names[] = (string) $member;
			}
		}
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
	 * Sets the key prefix.  Set {@see setIndexKey IndexKey} alongside it: the index of model
	 * names is a key of its own and does not follow the prefix.
	 * @param string $value The key prefix.
	 */
	public function setKeyPrefix(string $value): void
	{
		$this->_keyPrefix = $value;
		$this->_layoutChecked = [];
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
		$this->_layoutChecked = [];
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
