<?php

/**
 * TFileBayesianStorage class file.
 *
 * @author Brad Anderson <belisoful@icloud.com>
 * @link https://github.com/belisoful/prado-bayesian
 * @license https://github.com/pradosoft/prado/blob/master/LICENSE
 */

namespace Belisoful\Prado\Util\Bayesian\Storage;

use Prado\Exceptions\TConfigurationException;
use Prado\Exceptions\TIOException;
use Prado\Exceptions\TInvalidDataValueException;
use Prado\TComponent;

/**
 * TFileBayesianStorage class.
 *
 * Persists models to a directory as one JSON file per model, named "<name>.json".  The
 * directory is created on first save if it does not exist; the storage refuses to operate
 * when the directory is unset or unwritable, surfacing the failure as a configuration
 * exception.
 *
 * The JSON encoding uses {@see JSON_UNESCAPED_SLASHES} and {@see JSON_UNESCAPED_UNICODE} so
 * the on-disk files stay human-readable for non-ASCII training data and URLs in token sets.
 * Atomicity is provided by writing to a sibling `<name>.json.tmp` and renaming on success.
 *
 * @author Brad Anderson <belisoful@icloud.com>
 * @since 0.1.0
 */
class TFileBayesianStorage extends TComponent implements IBayesianStorage
{
	/** @var ?string The directory holding the model files. */
	private ?string $_directory = null;

	/** @var int The JSON encoding flags. */
	private const JSON_FLAGS = JSON_UNESCAPED_SLASHES | JSON_UNESCAPED_UNICODE | JSON_PRETTY_PRINT;

	/**
	 * Ensures the storage directory exists; throws when it is missing/unwritable.
	 * @throws TConfigurationException When the directory is unset or cannot be created.
	 */
	private function ensureDirectory(): void
	{
		if ($this->_directory === null || $this->_directory === '') {
			throw new TConfigurationException('bayesian_storage_directory_required');
		}
		if (!is_dir($this->_directory) && !@mkdir($this->_directory, 0o755, true) && !is_dir($this->_directory)) {
			throw new TConfigurationException('bayesian_storage_directory_unwritable', $this->_directory);
		}
		if (!is_writable($this->_directory)) {
			throw new TConfigurationException('bayesian_storage_directory_unwritable', $this->_directory);
		}
	}

	/**
	 * Returns the absolute path of the file backing a model.  The name is validated first: a
	 * path separator or null byte in the name could otherwise escape the storage directory
	 * (e.g. "../../etc/passwd"), so such names are rejected rather than resolved.
	 * @param string $name The model name.
	 * @throws TInvalidDataValueException When the name is empty or contains a path separator or null byte.
	 * @return string The file path.
	 */
	private function path(string $name): string
	{
		if ($name === '' || strpbrk($name, "/\\\0") !== false) {
			throw new TInvalidDataValueException('bayesian_storage_name_invalid', $name);
		}
		return rtrim($this->_directory ?? '', DIRECTORY_SEPARATOR) . DIRECTORY_SEPARATOR . $name . '.json';
	}

	/**
	 * Persists a payload under a model name.
	 * @param string $name The model name.
	 * @param array<string, mixed> $payload The payload.
	 * @throws TConfigurationException When the directory is unset or unwritable.
	 * @throws TIOException When the JSON encoding or file write fails.
	 */
	public function save(string $name, array $payload): void
	{
		$this->ensureDirectory();
		$path = $this->path($name);
		$encoded = json_encode($payload, self::JSON_FLAGS);
		if ($encoded === false) {
			throw new TIOException('bayesian_storage_encode_failed', json_last_error_msg());
		}
		// Write to a per-call unique temp file in the same directory, then rename it into
		// place: rename() is atomic on the same filesystem, so readers never see a partial
		// file and two concurrent saves of the same model cannot interleave in one temp file.
		$tmp = @tempnam(dirname($path), '.' . basename($path) . '.');
		if ($tmp === false) {
			throw new TIOException('bayesian_storage_save_failed', $path);
		}
		if (@file_put_contents($tmp, $encoded, LOCK_EX) === false) {
			@unlink($tmp);
			throw new TIOException('bayesian_storage_save_failed', $tmp);
		}
		@chmod($tmp, 0o644);
		if (!@rename($tmp, $path)) {
			@unlink($tmp);
			throw new TIOException('bayesian_storage_save_failed', $path);
		}
	}

	/**
	 * Loads a payload by name; null when the directory is unset, the file is missing, or it
	 * does not hold valid JSON.
	 * @param string $name The model name.
	 * @throws TInvalidDataValueException When the name is invalid.
	 * @return ?array<string, mixed> The payload, or null.
	 */
	public function load(string $name): ?array
	{
		if (!$this->hasDirectory()) {
			return null;
		}
		$path = $this->path($name);
		if (!is_file($path)) {
			return null;
		}
		$contents = @file_get_contents($path);
		if ($contents === false) {
			return null;
		}
		$decoded = json_decode($contents, true);
		if (!is_array($decoded)) {
			return null;
		}
		return $decoded;
	}

	/**
	 * Returns whether a payload with the given name is stored.
	 * @param string $name The model name.
	 * @throws TInvalidDataValueException When the name is invalid.
	 * @return bool Whether the model is present.
	 */
	public function exists(string $name): bool
	{
		if (!$this->hasDirectory()) {
			return false;
		}
		return is_file($this->path($name));
	}

	/**
	 * Removes a payload.  Removing a non-existent name is a no-op.
	 * @param string $name The model name.
	 * @throws TInvalidDataValueException When the name is invalid.
	 */
	public function delete(string $name): void
	{
		if (!$this->hasDirectory()) {
			return;
		}
		@unlink($this->path($name));
	}

	/**
	 * Returns the names of all stored models, in ascending order by name.  An unset directory
	 * yields an empty list rather than an error, so {@see list()} is safe to call before any
	 * model has been saved.
	 * @return string[] The model names, sorted ascending.
	 */
	public function list(): array
	{
		if (!$this->hasDirectory()) {
			return [];
		}
		$files = glob(rtrim((string) $this->_directory, DIRECTORY_SEPARATOR) . DIRECTORY_SEPARATOR . '*.json') ?: [];
		$names = [];
		foreach ($files as $file) {
			$names[] = basename($file, '.json');
		}
		sort($names);
		return $names;
	}

	/**
	 * Returns whether a non-empty directory is configured.  An empty string is treated as
	 * unset so reads and deletes never resolve to the filesystem root.
	 * @return bool Whether a directory is set.
	 */
	private function hasDirectory(): bool
	{
		return $this->_directory !== null && $this->_directory !== '';
	}

	/**
	 * Returns the storage directory.
	 * @return ?string The directory.
	 */
	public function getDirectory(): ?string
	{
		return $this->_directory;
	}

	/**
	 * Sets the storage directory.
	 * @param string $value The directory.
	 */
	public function setDirectory(string $value): void
	{
		$this->_directory = $value;
	}
}
