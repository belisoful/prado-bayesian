<?php

/**
 * IBayesianStorage interface file.
 *
 * @author Brad Anderson <belisoful@icloud.com>
 * @link https://github.com/belisoful/prado-bayesian
 * @license https://github.com/pradosoft/prado/blob/master/LICENSE
 */

namespace Belisoful\Prado\Util\Bayesian\Storage;

/**
 * IBayesianStorage interface.
 *
 * The seam that persists a trained classifier.  The classifier hands the storage a name and a
 * payload (a JSON-serializable array describing its vocabulary, configuration, and training
 * data); the storage decides how to record it.  {@see TMemoryBayesianStorage} keeps the model
 * in process, {@see TFileBayesianStorage} writes JSON to a directory,
 * {@see TSqlBayesianStorage} uses a relational database, and {@see TRedisBayesianStorage}
 * uses Redis.
 *
 * The payload is opaque to the storage — it is a class-defined blob the classifier produced
 * and is responsible for parsing back.  This lets each storage backend pick the encoding that
 * fits it best (a single JSON blob for file/memory, columns for SQL, a string for Redis).
 *
 * @author Brad Anderson <belisoful@icloud.com>
 * @since 0.1.0
 */
interface IBayesianStorage
{
	/**
	 * Persists a payload under a name, overwriting any previous payload of the same name.
	 * @param string $name The model name.
	 * @param array<string, mixed> $payload The payload to store.
	 */
	public function save(string $name, array $payload): void;

	/**
	 * Loads a previously-saved payload by name, or returns null if the name is unknown.
	 * @param string $name The model name.
	 * @return ?array<string, mixed> The payload, or null.
	 */
	public function load(string $name): ?array;

	/**
	 * Returns whether a payload with the given name is currently stored.
	 * @param string $name The model name.
	 * @return bool Whether the model is present.
	 */
	public function exists(string $name): bool;

	/**
	 * Removes a previously-saved payload.  Removing a non-existent name is a no-op.
	 * @param string $name The model name.
	 */
	public function delete(string $name): void;

	/**
	 * Returns the names of all stored models, in ascending order by name.
	 *
	 * Every backend sorts, so a caller can compare or display the list without knowing which
	 * storage is configured.  The sort is by the backend's natural string ordering, which is
	 * byte order for the in-process, file, and Redis backends and the column collation for
	 * {@see TSqlBayesianStorage} — the two can disagree on case and accents, so callers that
	 * need one exact ordering across backends should re-sort the returned names themselves.
	 * @return string[] The model names, sorted ascending.
	 */
	public function list(): array;
}
