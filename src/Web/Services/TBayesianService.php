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
use Belisoful\Prado\Util\Bayesian\Classifier\TNaiveBayesClassifier;
use Belisoful\Prado\Util\Bayesian\IBayesianRecommender;
use Belisoful\Prado\Util\Bayesian\IBayesianTagger;
use Belisoful\Prado\Util\Bayesian\TBayesianModule;
use Belisoful\Prado\Util\Bayesian\TBayesianPayload;
use Belisoful\Prado\Util\Bayesian\TBayesianRecommender;
use Belisoful\Prado\Util\Bayesian\TBayesianTagger;
use Prado\Exceptions\TConfigurationException;
use Prado\Exceptions\THttpException;
use Prado\Exceptions\TInvalidDataValueException;
use Prado\Exceptions\TInvalidOperationException;
use Prado\Security\IUser;
use Prado\Security\Permissions\IPermissions;
use Prado\Security\Permissions\TPermissionEvent;
use Prado\Security\TAuthorizationRule;
use Prado\Security\TAuthorizationRuleCollection;
use Prado\TService;
use Prado\Xml\TXmlElement;

/**
 * TBayesianService class.
 *
 * A PRADO service that exposes the configured default classifier over an HTTP request.  The
 * framework dispatches the active service by calling {@see run()}, which reads the request
 * parameters, runs the requested action, and writes a JSON response.  Two actions are
 * supported:
 *
 * - `classify` (default): scores a `text` parameter (and optional `category` for the spam-filter
 *   shortcut).  Returns `{"category": "...", "scores": {...}, "calibrated": bool}` plus
 *   `"isSpam": bool` when `category` was given.  `calibrated` says whether the scores are
 *   calibrated probabilities (the classifier has a fitted calibration) or the plain normalized
 *   Naive Bayes scores — a ranking that sums to one.
 * - `recommend`: scores a list of `candidates[]` against a `context[]` list.  Returns
 *   `{"scores": {...}}` with candidates ordered by P(positive); the map is always encoded as a
 *   JSON object even when every candidate identifier is numeric.
 * - `tag`: multi-label tagging of a `text` parameter through a {@see TBayesianTagger} over the
 *   service's classifier.  Returns `{"tags": {...}, "calibrated": bool}` with the labels whose
 *   probability reaches {@see setTagThreshold() TagThreshold}, highest first, at most
 *   {@see setMaxTags() MaxTags} of them.  The classifier must be a {@see TNaiveBayesClassifier}
 *   trained through a tagger (see that class).
 *
 * Errors are JSON too — `{"error": "<code>", "message": "..."}` — with an HTTP status: 400 for
 * a bad request (missing or non-string `text`, array-valued scalar parameters, no candidates,
 * an unknown `action`), 413 when `text`, the joined `context` or a candidate exceeds
 * {@see setMaxTextLength() MaxTextLength} or more than {@see setMaxCandidates() MaxCandidates}
 * candidates are sent, 401/403 when access is refused, and 503 when the classifier has not been
 * trained yet.  A server-side misconfiguration (no classifier resolvable) propagates to the
 * framework's error handler.  Responses carry `X-Content-Type-Options: nosniff` and never
 * contain invalid UTF-8.
 *
 * The service is read-only: it classifies and recommends but exposes no training, saving, or
 * deletion over HTTP.
 *
 * **Access control.**  The service enforces nothing by default, so every request that reaches
 * it is answered; it must not be exposed to the public unrestricted, because each request
 * costs a classification and the scores describe the model.  Two opt-in mechanisms, both
 * PRADO's own, restrict it:
 *
 * - **Authorization rules**, the same `<allow>`/`<deny>` rules a page uses, declared inside an
 *   `<authorization>` element of the service (or an `authorization` list in PHP
 *   configuration) or added to {@see getAuthorizationRules()} from code.  They are evaluated
 *   by {@see run()} against the application user before any work is done.  A guest that is
 *   refused gets 401 and an authenticated user 403, both as JSON.  Rules need a user: when
 *   rules exist but no authentication module has put a user on the application, the request is
 *   refused with 401 `bayesian_service_user_required` rather than answered.
 * - **Permissions**: the service implements {@see IPermissions} and declares
 *   {@see PERM_CLASSIFY} and {@see PERM_RECOMMEND}.  When a
 *   {@see \Prado\Security\Permissions\TPermissionsManager} module is configured, it attaches
 *   its behavior to the service and each action first checks the current user's permission
 *   through the `dyClassify` / `dyRecommend` dynamic events; a user without it gets 401/403
 *   `bayesian_service_permission_denied`.  Without a permissions manager the events are inert.
 *
 * ```xml
 * <services>
 *     <service id="bayesian" class="TBayesianService" ModuleID="bayesian" MaxCandidates="50">
 *         <authorization>
 *             <allow roles="editor" />
 *             <deny users="*" />
 *         </authorization>
 *     </service>
 * </services>
 * ```
 *
 * ```php
 * return [
 *     'services' => [
 *         'bayesian' => [
 *             'class' => 'TBayesianService',
 *             'properties' => ['ModuleID' => 'bayesian', 'MaxCandidates' => 50],
 *             'authorization' => [
 *                 ['action' => 'allow', 'roles' => 'editor'],
 *                 ['action' => 'deny', 'users' => '*'],
 *             ],
 *         ],
 *     ],
 * ];
 * ```
 *
 * The service backs onto the default classifier of a {@see TBayesianModule}.  Set one explicitly
 * with {@see setClassifier()}, or let the service resolve it from the configured module: by
 * {@see setModuleID() ModuleID} when given, otherwise the first {@see TBayesianModule} registered
 * in the application.
 *
 * The short class name comes from the extension's `extra.prado.class-map`; PRADO resolves
 * `<service class="...">` through {@see \Prado\Prado::usingClass()}, which does not autoload a
 * fully-qualified extension class that has not been loaded yet.
 *
 * @author Brad Anderson <belisoful@icloud.com>
 * @since 0.1.0
 * @method bool dyClassify(bool $denied) Raised before a `classify` action; a handler returning true refuses it.
 * @method bool dyRecommend(bool $denied) Raised before a `recommend` action; a handler returning true refuses it.
 * @method bool dyTag(bool $denied) Raised before a `tag` action; a handler returning true refuses it.
 */
class TBayesianService extends TService implements IPermissions
{
	/**
	 * The permission checked before a `classify` action.
	 * @since 0.2.0
	 */
	public const PERM_CLASSIFY = 'bayesian_classify';

	/**
	 * The permission checked before a `recommend` action.
	 * @since 0.2.0
	 */
	public const PERM_RECOMMEND = 'bayesian_recommend';

	/**
	 * The permission checked before a `tag` action.
	 * @since 0.2.0
	 */
	public const PERM_TAG = 'bayesian_tag';

	/** @var ?IBayesianClassifier The classifier to use (set explicitly or resolved from the module). */
	private ?IBayesianClassifier $_classifier = null;

	/** @var ?IBayesianRecommender The recommender to use (created lazily). */
	private ?IBayesianRecommender $_recommender = null;

	/** @var ?IBayesianTagger The tagger to use (created lazily). */
	private ?IBayesianTagger $_tagger = null;

	/** @var float The probability a label needs to be returned by the `tag` action. */
	private float $_tagThreshold = 0.5;

	/** @var int The most tags the `tag` action returns; 0 for no limit. */
	private int $_maxTags = 0;

	/** @var ?string The id of the TBayesianModule to source the classifier from; null = auto-detect. */
	private ?string $_moduleID = null;

	/** @var int The maximum accepted byte length of `text`, the joined `context`, and each candidate; 0 disables the limit. */
	private int $_maxTextLength = 65536;

	/** @var int The maximum number of `candidates` entries accepted; 0 disables the limit. */
	private int $_maxCandidates = 100;

	/** @var ?TAuthorizationRuleCollection The authorization rules, created on first use. */
	private ?TAuthorizationRuleCollection $_rules = null;

	/** @var int The JSON encoding flags for responses. */
	private const JSON_FLAGS = JSON_UNESCAPED_SLASHES | JSON_UNESCAPED_UNICODE | JSON_INVALID_UTF8_SUBSTITUTE | JSON_PRESERVE_ZERO_FRACTION;

	/**
	 * Initializes the service, reading any `<authorization>` rules from its configuration.
	 *
	 * In XML the rules are `<allow>` and `<deny>` elements inside an `<authorization>` child of
	 * the service element, each with the optional `users`, `roles`, `verb`, `ips` and
	 * `priority` attributes of {@see TAuthorizationRule}.  In PHP configuration they are an
	 * `authorization` list of maps with an `action` of `allow` or `deny` and the same keys.
	 * @param null|array<string, mixed>|TXmlElement $config The service configuration.
	 * @throws TConfigurationException When a rule is neither allow nor deny.
	 * @return void
	 * @since 0.2.0 Reads authorization rules.
	 */
	public function init($config)
	{
		if ($config instanceof TXmlElement) {
			$authorization = $config->getElementByTagName('authorization');
			if ($authorization instanceof TXmlElement) {
				foreach ($authorization->getElements() as $element) {
					/** @var TXmlElement $element */
					$this->addRule(
						$element->getTagName(),
						TBayesianPayload::string($element->getAttribute('users')),
						TBayesianPayload::string($element->getAttribute('roles')),
						TBayesianPayload::string($element->getAttribute('verb')),
						TBayesianPayload::string($element->getAttribute('ips')),
						$element->getAttribute('priority')
					);
				}
			}
		} elseif (is_array($config) && isset($config['authorization']) && is_array($config['authorization'])) {
			foreach ($config['authorization'] as $rule) {
				if (!is_array($rule)) {
					throw new TConfigurationException('bayesian_service_rule_invalid', is_scalar($rule) ? (string) $rule : gettype($rule));
				}
				$this->addRule(
					TBayesianPayload::string($rule['action'] ?? null),
					TBayesianPayload::string($rule['users'] ?? null),
					TBayesianPayload::string($rule['roles'] ?? null),
					TBayesianPayload::string($rule['verb'] ?? null),
					TBayesianPayload::string($rule['ips'] ?? null),
					$rule['priority'] ?? null
				);
			}
		}
		parent::init($config);
	}

	/**
	 * Adds one configured authorization rule.
	 * @param string $action `allow` or `deny`, in any case.
	 * @param string $users The comma-separated users.
	 * @param string $roles The comma-separated roles.
	 * @param string $verb The verb, `get`, `post` or empty for both.
	 * @param string $ips The comma-separated IP rules.
	 * @param mixed $priority The rule priority, or null for the default.
	 * @throws TConfigurationException When the action is neither allow nor deny.
	 */
	private function addRule(string $action, string $users, string $roles, string $verb, string $ips, $priority): void
	{
		$normalized = strtolower(trim($action));
		if ($normalized !== 'allow' && $normalized !== 'deny') {
			throw new TConfigurationException('bayesian_service_rule_invalid', $action);
		}
		$this->getAuthorizationRules()->add(new TAuthorizationRule($normalized, $users, $roles, $verb, $ips, is_numeric($priority) ? $priority : null));
	}

	/**
	 * Returns the authorization rules the service enforces in {@see run()}, creating the
	 * collection on first use so rules can also be added from code.  An empty collection
	 * enforces nothing.
	 * @return TAuthorizationRuleCollection The rules.
	 * @since 0.2.0
	 */
	public function getAuthorizationRules(): TAuthorizationRuleCollection
	{
		if ($this->_rules === null) {
			$this->_rules = new TAuthorizationRuleCollection();
		}
		return $this->_rules;
	}

	/**
	 * Returns the permissions this service registers with a
	 * {@see \Prado\Security\Permissions\TPermissionsManager}: one per action, each tied to the
	 * dynamic event that action raises first.
	 * @param mixed $manager The permissions manager.
	 * @return TPermissionEvent[] The permission events.
	 * @since 0.2.0
	 */
	public function getPermissions($manager)
	{
		return [
			new TPermissionEvent(static::PERM_CLASSIFY, 'Classifies text through TBayesianService.', ['dyClassify']),
			new TPermissionEvent(static::PERM_RECOMMEND, 'Ranks candidates through TBayesianService.', ['dyRecommend']),
			new TPermissionEvent(static::PERM_TAG, 'Tags text through TBayesianService.', ['dyTag']),
		];
	}

	/**
	 * Runs the service.  Invoked automatically by the application: checks the authorization
	 * rules, reads the request parameters, dispatches the action, and writes the JSON-encoded
	 * result (or JSON error) to the response.
	 * @return void
	 */
	public function run()
	{
		$status = 200;
		try {
			$this->authorize();
			$result = $this->runService($this->collectParameters());
		} catch (THttpException $e) {
			$status = $e->getStatusCode();
			$result = ['error' => $e->getErrorCode(), 'message' => $e->getErrorMessage()];
		} catch (TInvalidDataValueException $e) {
			$status = in_array($e->getErrorCode(), ['bayesian_service_text_too_long', 'bayesian_service_candidates_too_many'], true) ? 413 : 400;
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
	 * Enforces the authorization rules against the application user, when any are configured.
	 *
	 * Rules cannot be evaluated without a user, and a missing user means no authentication
	 * module is configured, so that case is refused rather than let through: a service that was
	 * meant to be restricted must never fall open because of a second configuration mistake.
	 * @throws THttpException 401 when there is no user or a guest is refused; 403 when an
	 * authenticated user is refused.
	 * @since 0.2.0
	 */
	protected function authorize(): void
	{
		if ($this->_rules === null || $this->_rules->getCount() === 0) {
			return;
		}
		$application = $this->getApplication();
		$user = $this->currentUser();
		if ($user === null) {
			throw new THttpException(401, 'bayesian_service_user_required');
		}
		$request = $application->getRequest();
		if (!$this->_rules->isUserAllowed($user, (string) $request->getRequestType(), (string) $request->getUserHostAddress())) {
			throw new THttpException($user->getIsGuest() ? 401 : 403, 'bayesian_service_unauthorized', $user->getName());
		}
	}

	/**
	 * Returns the application user, or null when no authentication module has set one.
	 *
	 * The framework declares {@see \Prado\TApplication::getUser()} as returning an
	 * {@see IUser}, but it returns null until an authentication module sets one, which is
	 * exactly the case this method exists to detect; hence the analysis exemptions.
	 * @return ?IUser The user, or null.
	 */
	private function currentUser(): ?IUser // @phpstan-ignore return.unusedType (null before authentication, see above)
	{
		$user = $this->getApplication()->getUser();
		return $user instanceof IUser ? $user : null; // @phpstan-ignore instanceof.alwaysTrue (null before authentication, see above)
	}

	/**
	 * Refuses an action whose permission check returned true.
	 * @param string $permission The permission name.
	 * @param mixed $denied The result of the action's dynamic event.
	 * @throws THttpException 401 for an anonymous caller, 403 for an authenticated one.
	 */
	private function requirePermission(string $permission, $denied): void
	{
		if ($denied !== true) {
			return;
		}
		$user = $this->currentUser();
		throw new THttpException($user !== null && !$user->getIsGuest() ? 403 : 401, 'bayesian_service_permission_denied', $permission);
	}

	/**
	 * JSON-encodes a service result.  Recommend `scores` are forced to an object so numeric
	 * candidate identifiers are not flattened into a list.
	 * @param array<string, mixed> $result The result.
	 * @return string The JSON.
	 */
	protected function encode(array $result): string
	{
		foreach (['scores', 'tags'] as $map) {
			if (isset($result[$map]) && is_array($result[$map])) {
				$result[$map] = $result[$map] === [] ? new \stdClass() : (object) $result[$map];
			}
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
	 * defaults to `classify`.  Each handler raises its permission event first.
	 * @param array<string, mixed> $params The service parameters.
	 * @throws TInvalidDataValueException When the action is unknown or not a string, or a
	 * handler rejects its parameters.
	 * @throws TInvalidOperationException When the classifier has not been trained.
	 * @throws TConfigurationException When no classifier can be resolved.
	 * @throws THttpException When the action's permission is denied.
	 * @return array<string, mixed> The JSON-serializable response.
	 */
	public function runService(array $params)
	{
		$action = $this->scalar($params, 'action') ?? 'classify';
		switch ($action) {
			case 'classify':
				$this->requirePermission(static::PERM_CLASSIFY, $this->dyClassify(false));
				return $this->runClassify($params);
			case 'recommend':
				$this->requirePermission(static::PERM_RECOMMEND, $this->dyRecommend(false));
				return $this->runRecommend($params);
			case 'tag':
				$this->requirePermission(static::PERM_TAG, $this->dyTag(false));
				return $this->runTag($params);
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
	 * Refuses input longer than {@see getMaxTextLength MaxTextLength}.
	 * @param string $parameter The parameter name, for the message.
	 * @param int $length The input length in bytes.
	 * @throws TInvalidDataValueException When the length exceeds the cap.
	 */
	private function assertLength(string $parameter, int $length): void
	{
		if ($this->_maxTextLength > 0 && $length > $this->_maxTextLength) {
			throw new TInvalidDataValueException('bayesian_service_text_too_long', (string) $this->_maxTextLength, $parameter);
		}
	}

	/**
	 * Handles the classify action.  Requires a `text` parameter; returns the predicted
	 * category and the normalized scores of every category (summing to 1).
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
		$this->assertLength('text', strlen($text));
		$category = $this->scalar($params, 'category');
		$classifier = $this->getClassifier();
		$predicted = $classifier->classify($text);
		$response = [
			'category' => $predicted,
			'scores' => $classifier->score($text),
			'calibrated' => $classifier instanceof TNaiveBayesClassifier && $classifier->getIsCalibrated(),
		];
		if ($category !== null) {
			$response['isSpam'] = $predicted === $category;
		}
		return $response;
	}

	/**
	 * Handles the recommend action.  Takes `context[]` and `candidates[]` lists; the joined
	 * context and each candidate are bounded by {@see getMaxTextLength MaxTextLength} and the
	 * candidate count by {@see getMaxCandidates MaxCandidates}, since every candidate costs a
	 * classification of the context plus that candidate.
	 * @param array<string, mixed> $params The service parameters.
	 * @throws TInvalidDataValueException When the lists are malformed, `candidates` is empty
	 * or too long, or an entry is too long.
	 * @throws TInvalidOperationException When the classifier has not been trained.
	 * @return array<string, mixed> The response.
	 */
	private function runRecommend(array $params): array
	{
		$context = $this->stringList($params, 'context');
		$candidates = $this->stringList($params, 'candidates');
		if ($this->_maxCandidates > 0 && count($candidates) > $this->_maxCandidates) {
			throw new TInvalidDataValueException('bayesian_service_candidates_too_many', (string) $this->_maxCandidates);
		}
		$this->assertLength('context', strlen(implode(' ', $context)));
		foreach ($candidates as $candidate) {
			$this->assertLength('candidates', strlen($candidate));
		}
		return [
			'scores' => $this->getRecommender()->recommend($context, $candidates),
		];
	}

	/**
	 * Handles the tag action.  Requires a `text` parameter; returns the labels whose
	 * probability reaches {@see getTagThreshold TagThreshold}, highest first.
	 * @param array<string, mixed> $params The service parameters.
	 * @throws TInvalidDataValueException When `text` is missing, not a string, or too long.
	 * @throws TInvalidOperationException When the tagger has not been trained.
	 * @throws TConfigurationException When the classifier cannot back a tagger.
	 * @return array<string, mixed> The response.
	 */
	private function runTag(array $params): array
	{
		$text = $this->scalar($params, 'text');
		if ($text === null) {
			throw new TInvalidDataValueException('bayesian_service_text_required');
		}
		$this->assertLength('text', strlen($text));
		$tagger = $this->getTagger();
		return [
			'tags' => $tagger->tag($text),
			'calibrated' => $tagger instanceof TBayesianTagger && $tagger->getIsCalibrated(),
		];
	}

	/**
	 * Sets the tagger explicitly, overriding the one {@see getTagger()} would build from the
	 * service's classifier.
	 * @param IBayesianTagger $value The tagger to back the `tag` action.
	 * @since 0.2.0
	 */
	public function setTagger(IBayesianTagger $value): void
	{
		$this->_tagger = $value;
	}

	/**
	 * Returns the tagger backing the `tag` action, wrapping the service's classifier in a
	 * {@see TBayesianTagger} with this service's {@see getTagThreshold TagThreshold} and
	 * {@see getMaxTags MaxTags} on first use when none was injected.
	 * @throws TConfigurationException When the classifier is not a {@see TNaiveBayesClassifier}.
	 * @return IBayesianTagger The tagger.
	 * @since 0.2.0
	 */
	public function getTagger(): IBayesianTagger
	{
		if ($this->_tagger === null) {
			$classifier = $this->getClassifier();
			if (!($classifier instanceof TNaiveBayesClassifier)) {
				throw new TConfigurationException('bayesian_tagger_classifier_invalid', $classifier::class);
			}
			$tagger = new TBayesianTagger();
			$tagger->setClassifier($classifier);
			$tagger->setThreshold($this->_tagThreshold);
			$tagger->setMaxTags($this->_maxTags);
			$this->_tagger = $tagger;
		}
		return $this->_tagger;
	}

	/**
	 * Returns the probability a label needs to be returned by the `tag` action.
	 * @return float The threshold.
	 * @since 0.2.0
	 */
	public function getTagThreshold(): float
	{
		return $this->_tagThreshold;
	}

	/**
	 * Sets the probability a label needs to be returned by the `tag` action (default 0.5).
	 * Applies to the tagger the service builds; an injected tagger keeps its own.
	 * @param float $value The threshold, in [0, 1].
	 * @since 0.2.0
	 */
	public function setTagThreshold(float $value): void
	{
		$this->_tagThreshold = is_nan($value) ? 0.5 : min(1.0, max(0.0, $value));
	}

	/**
	 * Returns the most tags the `tag` action returns; 0 for no limit.
	 * @return int The limit.
	 * @since 0.2.0
	 */
	public function getMaxTags(): int
	{
		return $this->_maxTags;
	}

	/**
	 * Sets the most tags the `tag` action returns (default 0, no limit).  Applies to the tagger
	 * the service builds; an injected tagger keeps its own.
	 * @param int $value The limit; values below 1 mean no limit.
	 * @since 0.2.0
	 */
	public function setMaxTags(int $value): void
	{
		$this->_maxTags = max(0, $value);
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
		// The framework declares getApplication() as non-null, but a service constructed outside
		// a running application (a script, a test double) has none; resolve to "no module" then.
		$application = $this->getApplication();
		if (!is_object($application)) { // @phpstan-ignore function.alreadyNarrowedType (no application outside a request, see above)
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
	 * Sets the maximum accepted byte length of the `text` parameter, of the joined `context`
	 * list, and of each `candidates` entry.  Classification cost grows with input size, so a
	 * cap (default 65536) bounds the work a single request can demand.  Set 0 to disable.
	 * @param int $value The maximum length in bytes; values < 0 are treated as 0.
	 */
	public function setMaxTextLength(int $value): void
	{
		$this->_maxTextLength = $value < 0 ? 0 : $value;
	}

	/**
	 * Returns the cap on the number of `candidates` entries a recommend request may send.
	 * @return int The maximum number of candidates; 0 = unlimited.
	 * @since 0.2.0
	 */
	public function getMaxCandidates(): int
	{
		return $this->_maxCandidates;
	}

	/**
	 * Sets the maximum number of `candidates` entries a recommend request may send.  Every
	 * candidate costs one classification of the context plus the candidate, so the cap (default
	 * 100) bounds the work a single request can demand.  Set 0 to disable.
	 * @param int $value The maximum number of candidates; values < 0 are treated as 0.
	 * @since 0.2.0
	 */
	public function setMaxCandidates(int $value): void
	{
		$this->_maxCandidates = $value < 0 ? 0 : $value;
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
