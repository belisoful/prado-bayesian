<?php

/**
 * TBayesianService class file.
 *
 * @author Brad Anderson <belisoful@icloud.com>
 * @link https://github.com/belisoful/prado-bayesian
 * @license https://github.com/pradosoft/prado/blob/master/LICENSE
 */

namespace Belisoful\Prado\Web\Services;

use Belisoful\Prado\Util\Bayesian\Classifier\IBayesianClassifier;
use Belisoful\Prado\Util\Bayesian\IBayesianRecommender;
use Belisoful\Prado\Util\Bayesian\TBayesianModule;
use Belisoful\Prado\Util\Bayesian\TBayesianRecommender;
use Prado\Exceptions\TConfigurationException;
use Prado\Exceptions\TInvalidDataValueException;
use Prado\Exceptions\TInvalidOperationException;
use Prado\TService;

/**
 * TBayesianService class.
 *
 * A PRADO service that exposes the configured default classifier over an HTTP request.  The
 * framework dispatches the active service by calling {@see run()}, which reads the request
 * parameters, runs the requested action, and writes a JSON response.  Two actions are
 * supported:
 *
 * - `classify` (default): scores a `text` parameter (and optional `category` for the spam-filter
 *   shortcut).  Returns `{"category": "...", "scores": {...}}` plus `"isSpam": bool` when
 *   `category` was given.
 * - `recommend`: scores a list of `candidates[]` against a `context[]` list.  Returns
 *   `{"scores": {...}}` with candidates ordered by P(positive); the map is always encoded as a
 *   JSON object even when every candidate identifier is numeric.
 *
 * Errors are JSON too — `{"error": "<code>", "message": "..."}` — with an HTTP status: 400 for
 * a bad request (missing or non-string `text`, array-valued scalar parameters, no candidates,
 * an unknown `action`), 413 when `text` exceeds {@see setMaxTextLength() MaxTextLength}, and 503
 * when the classifier has not been trained yet.  A server-side misconfiguration (no classifier
 * resolvable) propagates to the framework's error handler.  Responses carry
 * `X-Content-Type-Options: nosniff` and never contain invalid UTF-8.
 *
 * The service is read-only: it classifies and recommends but exposes no training, saving, or
 * deletion over HTTP.
 *
 * The service backs onto the default classifier of a {@see TBayesianModule}.  Set one explicitly
 * with {@see setClassifier()}, or let the service resolve it from the configured module: by
 * {@see setModuleID() ModuleID} when given, otherwise the first {@see TBayesianModule} registered
 * in the application.  So a single configuration wires up the bootstrap module and the request
 * surface:
 *
 * ```xml
 * <services>
 *     <service id="bayesian" class="TBayesianService" ModuleID="bayesian" />
 * </services>
 * ```
 *
 * ```php
 * return [
 *     'services' => [
 *         'bayesian' => [
 *             'class' => 'TBayesianService',
 *             'properties' => ['ModuleID' => 'bayesian'],
 *         ],
 *     ],
 * ];
 * ```
 *
 * The short class name comes from the extension's `extra.prado.class-map`; PRADO resolves
 * `<service class="...">` through {@see \Prado\Prado::usingClass()}, which does not autoload a
 * fully-qualified extension class that has not been loaded yet.
 *
 * @author Brad Anderson <belisoful@icloud.com>
 * @since 0.1.0
 */
class TBayesianService extends TService
{
	/** @var ?IBayesianClassifier The classifier to use (set explicitly or resolved from the module). */
	private ?IBayesianClassifier $_classifier = null;

	/** @var ?IBayesianRecommender The recommender to use (created lazily). */
	private ?IBayesianRecommender $_recommender = null;

	/** @var ?string The id of the TBayesianModule to source the classifier from; null = auto-detect. */
	private ?string $_moduleID = null;

	/** @var int The maximum accepted byte length of `text`; 0 disables the limit. */
	private int $_maxTextLength = 65536;

	/** @var int The JSON encoding flags for responses. */
	private const JSON_FLAGS = JSON_UNESCAPED_SLASHES | JSON_UNESCAPED_UNICODE | JSON_INVALID_UTF8_SUBSTITUTE | JSON_PRESERVE_ZERO_FRACTION;

	/**
	 * Runs the service.  Invoked automatically by the application: reads the request parameters,
	 * dispatches the action, and writes the JSON-encoded result (or JSON error) to the response.
	 * @return void
	 */
	public function run()
	{
		$status = 200;
		try {
			$result = $this->runService($this->collectParameters());
		} catch (TInvalidDataValueException $e) {
			$status = $e->getErrorCode() === 'bayesian_service_text_too_long' ? 413 : 400;
			$result = ['error' => $e->getErrorCode(), 'message' => $e->getErrorMessage()];
		} catch (TInvalidOperationException $e) {
			$status = 503;
			$result = ['error' => $e->getErrorCode(), 'message' => $e->getErrorMessage()];
		}
		$response = $this->getResponse();
		$response->setStatusCode($status);
		$response->setContentType('application/json');
		$response->setCharset('UTF-8');
		$response->appendHeader('X-Content-Type-Options: nosniff');
		$response->write($this->encode($result));
	}

	/**
	 * JSON-encodes a service result.  Recommend `scores` are forced to an object so numeric
	 * candidate identifiers are not flattened into a list.
	 * @param array<string, mixed> $result The result.
	 * @return string The JSON.
	 */
	protected function encode(array $result): string
	{
		if (isset($result['scores']) && is_array($result['scores'])) {
			$result['scores'] = $result['scores'] === [] ? new \stdClass() : (object) $result['scores'];
		}
		$json = json_encode($result, self::JSON_FLAGS);
		return $json === false ? '{"error":"encode_failed"}' : $json;
	}

	/**
	 * Collects the recognized service parameters from the current request.
	 * @return array<string, mixed> The parameters present in the request.
	 */
	protected function collectParameters(): array
	{
		$request = $this->getRequest();
		$params = [];
		foreach (['action', 'text', 'category', 'context', 'candidates'] as $key) {
			if ($request->contains($key)) {
				$params[$key] = $request->itemAt($key);
			}
		}
		return $params;
	}

	/**
	 * Dispatches the `action` parameter to the right handler.  When the action is omitted it
	 * defaults to `classify`.
	 * @param array<string, mixed> $params The service parameters.
	 * @throws TInvalidDataValueException When the action is unknown or not a string, or a
	 * handler rejects its parameters.
	 * @throws TInvalidOperationException When the classifier has not been trained.
	 * @throws TConfigurationException When no classifier can be resolved.
	 * @return array<string, mixed> The JSON-serializable response.
	 */
	public function runService($params)
	{
		$params = is_array($params) ? $params : [];
		$action = $this->scalar($params, 'action') ?? 'classify';
		switch ($action) {
			case 'classify':
				return $this->runClassify($params);
			case 'recommend':
				return $this->runRecommend($params);
			default:
				throw new TInvalidDataValueException('bayesian_service_action_unknown', $action);
		}
	}

	/**
	 * Returns a scalar request parameter as a string, or null when absent.  An array-valued
	 * parameter (e.g. `?text[]=x`) is a client error rather than an "Array to string" notice.
	 * @param array<string, mixed> $params The service parameters.
	 * @param string $key The parameter name.
	 * @throws TInvalidDataValueException When the parameter is not a scalar.
	 * @return ?string The value, or null.
	 */
	private function scalar(array $params, string $key): ?string
	{
		if (!array_key_exists($key, $params) || $params[$key] === null) {
			return null;
		}
		$value = $params[$key];
		if (!is_scalar($value)) {
			throw new TInvalidDataValueException('bayesian_service_parameter_invalid', $key);
		}
		return (string) $value;
	}

	/**
	 * Returns a list request parameter as a list of strings, or an empty list when absent.
	 * Nested arrays are a client error.
	 * @param array<string, mixed> $params The service parameters.
	 * @param string $key The parameter name.
	 * @throws TInvalidDataValueException When an entry is not a scalar.
	 * @return string[] The values.
	 */
	private function stringList(array $params, string $key): array
	{
		if (!isset($params[$key])) {
			return [];
		}
		$value = is_array($params[$key]) ? $params[$key] : [$params[$key]];
		$out = [];
		foreach ($value as $entry) {
			if (!is_scalar($entry)) {
				throw new TInvalidDataValueException('bayesian_service_parameter_invalid', $key);
			}
			$out[] = (string) $entry;
		}
		return $out;
	}

	/**
	 * Handles the classify action.  Requires a `text` parameter; returns the predicted
	 * category and the full probability distribution (normalized to sum to 1).
	 * @param array<string, mixed> $params The service parameters.
	 * @throws TInvalidDataValueException When `text` is missing, not a string, or too long.
	 * @throws TInvalidOperationException When the classifier has not been trained.
	 * @return array<string, mixed> The response.
	 */
	private function runClassify(array $params): array
	{
		$text = $this->scalar($params, 'text');
		if ($text === null) {
			throw new TInvalidDataValueException('bayesian_service_text_required');
		}
		if ($this->_maxTextLength > 0 && strlen($text) > $this->_maxTextLength) {
			throw new TInvalidDataValueException('bayesian_service_text_too_long', (string) $this->_maxTextLength);
		}
		$category = $this->scalar($params, 'category');
		$classifier = $this->getClassifier();
		$predicted = $classifier->classify($text);
		$response = [
			'category' => $predicted,
			'scores' => $classifier->score($text),
		];
		if ($category !== null) {
			$response['isSpam'] = $predicted === $category;
		}
		return $response;
	}

	/**
	 * Handles the recommend action.  Takes `context[]` and `candidates[]` lists.
	 * @param array<string, mixed> $params The service parameters.
	 * @throws TInvalidDataValueException When the lists are malformed or `candidates` is empty.
	 * @throws TInvalidOperationException When the classifier has not been trained.
	 * @return array<string, mixed> The response.
	 */
	private function runRecommend(array $params): array
	{
		$context = $this->stringList($params, 'context');
		$candidates = $this->stringList($params, 'candidates');
		return [
			'scores' => $this->getRecommender()->recommend($context, $candidates),
		];
	}

	/**
	 * Sets the classifier explicitly, overriding module resolution.  Injecting one here means
	 * the service never consults {@see getModuleID ModuleID}.
	 * @param IBayesianClassifier $value The classifier to back the service.
	 */
	public function setClassifier(IBayesianClassifier $value): void
	{
		$this->_classifier = $value;
	}

	/**
	 * Returns the classifier, resolving it from the configured {@see TBayesianModule} on first
	 * use when one was not set explicitly.
	 * @throws TConfigurationException When no classifier is set and none can be resolved.
	 * @return IBayesianClassifier The classifier.
	 */
	public function getClassifier(): IBayesianClassifier
	{
		if ($this->_classifier === null) {
			$this->_classifier = $this->resolveModuleClassifier();
		}
		if ($this->_classifier === null) {
			throw new TConfigurationException('bayesian_service_classifier_missing');
		}
		return $this->_classifier;
	}

	/**
	 * Resolves the default classifier from the application's {@see TBayesianModule}.
	 * @return ?IBayesianClassifier The module's classifier, or null when no module is available.
	 */
	protected function resolveModuleClassifier(): ?IBayesianClassifier
	{
		$application = $this->getApplication();
		if ($application === null) {
			return null;
		}
		if ($this->_moduleID !== null && $this->_moduleID !== '') {
			$module = $application->getModule($this->_moduleID);
			return $module instanceof TBayesianModule ? $module->getClassifier() : null;
		}
		foreach ($application->getModulesByType(TBayesianModule::class) as $id => $module) {
			$module ??= $application->getModule($id);
			if ($module instanceof TBayesianModule) {
				return $module->getClassifier();
			}
		}
		return null;
	}

	/**
	 * Sets the recommender explicitly, overriding the one {@see getRecommender()} would build
	 * from the service's classifier.
	 * @param IBayesianRecommender $value The recommender to back the service.
	 */
	public function setRecommender(IBayesianRecommender $value): void
	{
		$this->_recommender = $value;
	}

	/**
	 * Returns the recommender backing the `recommend` action, wrapping the service's classifier
	 * in a {@see TBayesianRecommender} on first use when none was injected.
	 * @return IBayesianRecommender The recommender, created on first use from the classifier.
	 */
	public function getRecommender(): IBayesianRecommender
	{
		if ($this->_recommender === null) {
			$recommender = new TBayesianRecommender();
			$recommender->setClassifier($this->getClassifier());
			$this->_recommender = $recommender;
		}
		return $this->_recommender;
	}

	/**
	 * Returns the cap on the `text` request parameter, in bytes.
	 * @return int The maximum accepted byte length of the `text` parameter; 0 = unlimited.
	 */
	public function getMaxTextLength(): int
	{
		return $this->_maxTextLength;
	}

	/**
	 * Sets the maximum accepted byte length of the `text` parameter.  Classification cost grows
	 * with input size and the endpoint is unauthenticated, so a cap (default 65536) bounds the
	 * work a single request can demand.  Set 0 to disable.
	 * @param int $value The maximum length in bytes; values < 0 are treated as 0.
	 */
	public function setMaxTextLength(int $value): void
	{
		$this->_maxTextLength = $value < 0 ? 0 : $value;
	}

	/**
	 * Returns the id of the {@see TBayesianModule} the default classifier is sourced from.
	 * @return ?string The id of the TBayesianModule the classifier is sourced from, or null.
	 */
	public function getModuleID(): ?string
	{
		return $this->_moduleID;
	}

	/**
	 * Sets the id of the {@see TBayesianModule} to source the default classifier from.  When
	 * unset, the first registered TBayesianModule is used.
	 * @param ?string $value The module id.
	 */
	public function setModuleID(?string $value): void
	{
		$this->_moduleID = $value === '' ? null : $value;
	}
}
