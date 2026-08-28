<?php

/**
 * TMemoryBayesianStorage class file.
 *
 * @author Brad Anderson <belisoful@icloud.com>
 * @link https://github.com/belisoful/prado-bayesian
 * @license https://github.com/pradosoft/prado/blob/master/LICENSE
 */

namespace Prado\Util\Bayesian\Storage;

use Prado\TComponent;

/**
 * TMemoryBayesianStorage class.
 *
 * Process-local in-memory storage.  The default storage backend when no persistence is
 * configured; useful for unit tests, request-scoped classifiers, and any short-lived
 * classification that does not need to survive the process.
 *
 * The storage does no I/O.  Models are lost when the process exits, so production use should
 * set one of the persistent backends ({@see TFileBayesianStorage},
 * {@see TSqlBayesianStorage}, or {@see TRedisBayesianStorage}) on the classifier.
 *
 * @author Brad Anderson <belisoful@icloud.com>
 * @since 0.1.0
 */
class TMemoryBayesianStorage extends TComponent implements IBayesianStorage
{
	/** @var array<string, array<string, mixed>> The stored models, keyed by name. */
	private array $_models = [];

	/**
	 * Persists a payload under a model name, replacing any payload already stored under it.
	 * The array is held as-is; no encoding round-trip happens, so this backend cannot fail on
	 * a payload that is not JSON-encodable the way the persistent backends do.
	 * @param string $name The model name.
	 * @param array<string, mixed> $payload The payload to store.
	 */
	public function save(string $name, array $payload): void
	{
		$this->_models[$name] = $payload;
	}

	/**
	 * Loads a previously-saved payload by name.
	 * @param string $name The model name.
	 * @return ?array<string, mixed> The payload, or null when the name is unknown.
	 */
	public function load(string $name): ?array
	{
		return $this->_models[$name] ?? null;
	}

	/**
	 * Returns whether a payload with the given name is currently stored.
	 * @param string $name The model name.
	 * @return bool Whether the model is present.
	 */
	public function exists(string $name): bool
	{
		return isset($this->_models[$name]);
	}

	/**
	 * Removes a previously-saved payload.  Removing a non-existent name is a no-op.
	 * @param string $name The model name.
	 */
	public function delete(string $name): void
	{
		unset($this->_models[$name]);
	}

	/**
	 * Returns the names of all stored models, in ascending order by name.
	 * @return string[] The model names, sorted ascending.
	 */
	public function list(): array
	{
		$names = array_keys($this->_models);
		sort($names);
		return $names;
	}
}
