<?php

/**
 * TRedisBayesianStorage class file.
 *
 * @author Brad Anderson <belisoful@icloud.com>
 * @link https://github.com/belisoful/prado-bayesian
 * @license https://github.com/pradosoft/prado/blob/master/LICENSE
 */

namespace Prado\Util\Bayesian\Storage;

use Prado\Exceptions\TConfigurationException;
use Prado\Exceptions\TInvalidDataValueException;
use Prado\Exceptions\TInvalidOperationException;
use Prado\TComponent;
use Redis;

/**
 * TRedisBayesianStorage class.
 *
 * Redis-backed storage for shared hosts: one Redis key per model, with the JSON-encoded
 * payload as the value and a Redis set holding the index of model names.  The class refuses
 * to construct itself when the `redis` extension is absent, and surfaces connect failures as
 * configuration exceptions.
 *
 * The connection is opened lazily on the first save/load call.  Callers can supply a fully
 * configured {@see Redis} instance via {@see setRedis()}, or the host/port/password triple
 * via the matching setters.
 *
 * @author Brad Anderson <belisoful@icloud.com>
 * @since 0.1.0
 */
class TRedisBayesianStorage extends TComponent implements IBayesianStorage
{
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
		if ($redis->del($key) === false) {
			throw new TInvalidOperationException('bayesian_storage_redis_write_failed', $key);
		}
		if ($redis->sRem($this->_indexKey, $name) === false) {
			throw new TInvalidOperationException('bayesian_storage_redis_write_failed', $this->_indexKey);
		}
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
