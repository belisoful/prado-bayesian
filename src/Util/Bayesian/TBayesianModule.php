<?php

/**
 * TBayesianModule class file.
 *
 * @author Brad Anderson <belisoful@icloud.com>
 * @link https://github.com/belisoful/prado-bayesian
 * @license https://github.com/pradosoft/prado/blob/master/LICENSE
 */

namespace Belisoful\Prado\Util\Bayesian;

use Belisoful\Prado\Util\Bayesian\Classifier\IBayesianClassifier;
use Belisoful\Prado\Util\Bayesian\Classifier\TNaiveBayesClassifier;
use Belisoful\Prado\Util\Bayesian\Storage\IBayesianStorage;
use Belisoful\Prado\Util\Bayesian\Tokenizer\IBayesianTokenizer;
use Prado\Exceptions\TConfigurationException;
use Prado\Prado;
use Prado\TComponent;
use Prado\TModule;
use Prado\Xml\TXmlElement;

/**
 * TBayesianModule class.
 *
 * The bootstrap module of the PRADO Bayesian extension, named in `extra.prado.bootstrap` of
 * the package's composer.json.  The `extra.prado.error-messages` field registers
 * `config/errorMessages.txt` system-wide (via {@see \Prado\TApplicationConfiguration}),
 * so the `bayesian_*` codes resolve in a PRADO application without any further wiring.
 * The `extra.prado.class-map` field registers the short class names (e.g.
 * `TNaiveBayesClassifier`) to their PHP FQNs for Prado3-style class resolution.
 *
 * The module also owns a default {@see IBayesianClassifier classifier} and a default
 * {@see IBayesianStorage storage} backend.  Configure storage as a `<storage>` child element
 * to make trained models persist across requests; an optional `<classifier>` child element
 * picks the classifier class and its properties (a {@see TNaiveBayesClassifier} is used when
 * omitted).  When `DefaultClassifier` names a model, the classifier takes that name and, if
 * the storage already holds a model of that name, it is loaded eagerly during {@see init()}
 * so the module is ready to classify as soon as the application starts.  A model that does
 * not exist yet is simply not loaded (the classifier stays empty until trained and saved);
 * any other storage failure (unreachable database, unwritable directory) propagates as a
 * configuration error rather than being swallowed.  Application code reaches the default
 * classifier through {@see getClassifier()}, and the storage through {@see getStorage()}.
 *
 * ```xml
 * <modules>
 *     <module id="bayesian" class="Belisoful\Prado\Util\Bayesian\TBayesianModule" DefaultClassifier="comment-spam">
 *         <classifier class="Belisoful\Prado\Util\Bayesian\Classifier\TNaiveBayesClassifier" Alpha="0.5" />
 *         <storage class="Belisoful\Prado\Util\Bayesian\Storage\TFileBayesianStorage" Directory="/var/lib/myapp/bayesian" />
 *     </module>
 * </modules>
 * <services>
 *     <service id="bayesian" class="TBayesianService" />
 * </services>
 * ```
 *
 * The same in PHP configuration.  A module attribute is a property and goes under
 * `properties`; the `<classifier>`/`<storage>` child elements become sibling keys of `class`:
 *
 * ```php
 * return [
 *     'modules' => [
 *         'bayesian' => [
 *             'class' => 'Belisoful\Prado\Util\Bayesian\TBayesianModule',
 *             'properties' => ['DefaultClassifier' => 'comment-spam'],
 *             'classifier' => ['class' => 'TNaiveBayesClassifier', 'Alpha' => 0.5],
 *             'storage' => ['class' => 'TFileBayesianStorage', 'Directory' => '/var/lib/myapp/bayesian'],
 *         ],
 *     ],
 *     'services' => [
 *         'bayesian' => ['class' => 'TBayesianService'],
 *     ],
 * ];
 * ```
 *
 * The module may equally be registered as `<module id="belisoful/prado-bayesian" .../>` (the
 * class then comes from `extra.prado.bootstrap`).  The service is registered by its class-map
 * short name because `<service class="...">` is resolved through {@see \Prado\Prado::usingClass()},
 * which does not autoload a fully-qualified extension class that is not loaded yet.
 *
 * @author Brad Anderson <belisoful@icloud.com>
 * @since 0.1.0
 */
class TBayesianModule extends TModule
{
	/** @var ?string The name of the classifier to load from storage on initialization. */
	private ?string $_defaultClassifier = null;

	/** @var ?IBayesianClassifier The configured default classifier. */
	private ?IBayesianClassifier $_classifier = null;

	/**
	 * @var array<string, IBayesianClassifier> The classifiers configured with an id, keyed by
	 * it.  One application often wants several models — a spam filter and a language detector,
	 * say — over one storage backend, each with its own tokenizer and its own model name.
	 */
	private array $_classifiers = [];

	/** @var ?string The id of the classifier {@see getClassifier()} returns when asked for none. */
	private ?string $_defaultClassifierID = null;

	/** @var ?IBayesianStorage The configured storage backend. */
	private ?IBayesianStorage $_storage = null;

	/**
	 * Initializes the module, creating the configured classifier and storage from
	 * `<classifier>` and `<storage>` child elements.  When `DefaultClassifier` is set and a
	 * storage is configured, the classifier is named after it and the model is loaded eagerly
	 * if the storage already holds it.
	 * @param null|array|TXmlElement $config The module configuration.
	 * @throws TConfigurationException When a configured component class is missing or invalid,
	 * or the storage cannot be reached.
	 * @return void
	 */
	public function init($config)
	{
		// Storage first: a classifier is wired to it as it is created, so several classifiers
		// configured together all share the one backend.
		if ($config instanceof TXmlElement) {
			foreach ($config->getElementsByTagName('storage') as $element) {
				$this->createStorage($element->getAttributes()->toArray());
			}
			foreach ($config->getElementsByTagName('classifier') as $element) {
				$this->createClassifier($element->getAttributes()->toArray(), $element);
			}
		} elseif (is_array($config)) {
			if (isset($config['storage']) && is_array($config['storage'])) {
				$this->createStorage($config['storage']);
			}
			foreach ($this->classifierConfigs($config) as $properties) {
				$this->createClassifier($properties, null);
			}
		}
		parent::init($config);
		$this->loadConfiguredModels();
	}

	/**
	 * Normalizes the PHP-configuration forms of `classifier` into a list.
	 *
	 * A single classifier may be given as one map; several are given as a map of id => map, so
	 * both `'classifier' => ['class' => ...]` and
	 * `'classifier' => ['spam' => ['class' => ...], 'lang' => [...]]` are accepted.
	 * @param array<string, mixed> $config The module configuration.
	 * @return array<int, array<string, mixed>> The per-classifier property maps.
	 */
	private function classifierConfigs(array $config): array
	{
		$classifier = $config['classifier'] ?? null;
		if (!is_array($classifier) || $classifier === []) {
			return [];
		}
		if (isset($classifier['class'])) {
			return [$classifier];
		}
		$out = [];
		foreach ($classifier as $id => $properties) {
			if (is_array($properties)) {
				$properties['id'] ??= (string) $id;
				$out[] = $properties;
			}
		}
		return $out;
	}

	/**
	 * Names each configured classifier after its model and loads the model when the storage
	 * already holds it.
	 *
	 * Only a model that is actually present is loaded; a missing model is the normal "not
	 * trained yet" state.  Storage errors (unreachable database, bad directory) propagate so a
	 * misconfiguration is not mistaken for an empty model.
	 */
	private function loadConfiguredModels(): void
	{
		$targets = $this->_classifiers;
		if ($targets === []) {
			$name = $this->_defaultClassifier;
			if ($name === null || $name === '') {
				return;
			}
			$classifier = $this->getClassifier();
			if ($classifier->getName() === null) {
				$classifier->setName($name);
			}
			$targets = ['' => $classifier];
		} elseif ($this->_defaultClassifier !== null && $this->_defaultClassifier !== '') {
			// A module-level DefaultClassifier still names the default classifier's model when
			// that classifier did not carry a Model of its own.
			$default = $this->getClassifier();
			if ($default->getName() === null) {
				$default->setName($this->_defaultClassifier);
			}
		}
		foreach ($targets as $classifier) {
			$name = $classifier->getName();
			if ($name === null || $name === '' || $this->_storage === null) {
				continue;
			}
			if ($this->_storage->exists($name)) {
				$classifier->load($name);
			}
		}
	}

	/**
	 * Resolves a configured class name (PHP FQN or Prado3 dotted/short name via the class map)
	 * and checks that it implements the required interface before it is instantiated.
	 * @param mixed $class The configured class name.
	 * @param string $interface The interface the class must implement.
	 * @param string $errorCode The error code to throw with.
	 * @throws TConfigurationException When the class is absent, unresolvable, or of the wrong type.
	 * @return string The resolved PHP class name.
	 */
	private function resolveClass($class, string $interface, string $errorCode): string
	{
		if (!is_string($class) || $class === '') {
			throw new TConfigurationException($errorCode, '');
		}
		// A PHP FQN autoloads directly; a Prado3 dotted or short (class-map) name resolves
		// through Prado::usingClass().
		$resolved = ltrim(str_replace('.', '\\', $class), '\\');
		if (!class_exists($resolved)) {
			$resolved = Prado::usingClass($class);
		}
		if (!is_string($resolved) || !class_exists($resolved) || !is_a($resolved, $interface, true) || !is_a($resolved, TComponent::class, true)) {
			throw new TConfigurationException($errorCode, $class);
		}
		return $resolved;
	}

	/**
	 * Creates a classifier from a configuration map and registers it.
	 *
	 * A classifier given an `id` joins {@see getClassifiers()} under it, so one module can own
	 * several models over one storage backend; one without an id is the module's single default,
	 * which is what a one-model configuration wants.
	 * @param array<string, mixed> $properties The classifier properties, including its `class`.
	 * @param ?TXmlElement $element The configuration element, when the configuration is XML, so
	 * a `<tokenizer>` child can be read from it.
	 * @throws TConfigurationException When the class is absent or not an {@see IBayesianClassifier}.
	 */
	protected function createClassifier(array $properties, ?TXmlElement $element = null): void
	{
		$class = $properties['class'] ?? null;
		$id = isset($properties['id']) ? (string) $properties['id'] : null;
		// `Model` is the storage key this classifier reads and writes; it is not a property of
		// the classifier, which calls the same thing `Name`.
		$model = isset($properties['Model']) ? (string) $properties['Model'] : null;
		unset($properties['class'], $properties['id'], $properties['Model'], $properties['tokenizer']);
		$resolved = $this->resolveClass($class, IBayesianClassifier::class, 'bayesian_classifier_class_invalid');
		/** @var IBayesianClassifier&TComponent $classifier */
		$classifier = Prado::createComponent($resolved);
		foreach ($properties as $name => $value) {
			$classifier->setSubProperty($name, $value);
		}
		if ($model !== null && $model !== '') {
			$classifier->setName($model);
		}
		$tokenizer = $element instanceof TXmlElement
			? $element->getElementByTagName('tokenizer')
			: null;
		if ($tokenizer instanceof TXmlElement) {
			$this->createTokenizer($classifier, $tokenizer->getAttributes()->toArray());
		}
		if ($id !== null && $id !== '') {
			$this->_classifiers[$id] = $classifier;
			// The first classifier configured is the default unless DefaultClassifierID names
			// another, so a single <classifier id="..."> behaves like one without an id.
			if ($this->_defaultClassifierID === null && $this->_classifier === null) {
				$this->_classifier = $classifier;
				if ($this->_storage !== null) {
					$classifier->setStorage($this->_storage);
				}
			} elseif ($this->_storage !== null) {
				$classifier->setStorage($this->_storage);
			}
			return;
		}
		$this->setClassifier($classifier);
	}

	/**
	 * Builds a classifier's tokenizer from a `<tokenizer>` child element.
	 *
	 * Configuring one matters most when several models share a module: a spam filter wants word
	 * tokens and a language detector wants character n-grams, and neither should have to be
	 * wired up in code.  A tokenizer set here is used for training; a model loaded from storage
	 * brings back the tokenizer it was trained with, which takes precedence.
	 * @param IBayesianClassifier $classifier The classifier to configure.
	 * @param array<string, mixed> $properties The tokenizer properties, including its `class`.
	 * @throws TConfigurationException When the class is absent or not an {@see IBayesianTokenizer}.
	 */
	protected function createTokenizer(IBayesianClassifier $classifier, array $properties): void
	{
		$class = $properties['class'] ?? null;
		unset($properties['class'], $properties['id']);
		$resolved = $this->resolveClass($class, IBayesianTokenizer::class, 'bayesian_tokenizer_class_invalid');
		/** @var IBayesianTokenizer&TComponent $tokenizer */
		$tokenizer = Prado::createComponent($resolved);
		foreach ($properties as $name => $value) {
			$tokenizer->setSubProperty($name, $value);
		}
		$classifier->setTokenizer($tokenizer);
	}

	/**
	 * Creates a storage backend from a configuration map and sets it.
	 * @param array<string, mixed> $properties The storage properties, including its `class`.
	 * @throws TConfigurationException When the class is absent or not an {@see IBayesianStorage}.
	 */
	protected function createStorage(array $properties): void
	{
		$class = $properties['class'] ?? null;
		unset($properties['class'], $properties['id']);
		$resolved = $this->resolveClass($class, IBayesianStorage::class, 'bayesian_storage_class_invalid');
		/** @var IBayesianStorage&TComponent $storage */
		$storage = Prado::createComponent($resolved);
		foreach ($properties as $name => $value) {
			$storage->setSubProperty($name, $value);
		}
		$this->setStorage($storage);
	}

	/**
	 * Returns a configured classifier by id, or the default when asked for none.
	 *
	 * The default is the classifier named by {@see getDefaultClassifierID() DefaultClassifierID},
	 * or the first one configured, or — when none was configured at all — a
	 * {@see TNaiveBayesClassifier} created on first use and wired to the module's storage.
	 * @param ?string $id The classifier id, or null for the default.
	 * @throws TConfigurationException When an id is given that was never configured.
	 * @return IBayesianClassifier The classifier.
	 */
	public function getClassifier(?string $id = null): IBayesianClassifier
	{
		if ($id !== null && $id !== '') {
			if (!isset($this->_classifiers[$id])) {
				throw new TConfigurationException('bayesian_classifier_id_unknown', $id, implode(', ', array_keys($this->_classifiers)));
			}
			return $this->_classifiers[$id];
		}
		if ($this->_defaultClassifierID !== null && isset($this->_classifiers[$this->_defaultClassifierID])) {
			return $this->_classifiers[$this->_defaultClassifierID];
		}
		if ($this->_classifier === null) {
			$this->_classifier = new TNaiveBayesClassifier();
			if ($this->_storage !== null) {
				$this->_classifier->setStorage($this->_storage);
			}
		}
		return $this->_classifier;
	}

	/**
	 * Returns every classifier configured with an id, keyed by it.
	 * @return array<string, IBayesianClassifier> The classifiers.
	 */
	public function getClassifiers(): array
	{
		return $this->_classifiers;
	}

	/**
	 * Returns whether a classifier is registered under the id.
	 * @param string $id The classifier id.
	 * @return bool Whether it exists.
	 */
	public function hasClassifier(string $id): bool
	{
		return isset($this->_classifiers[$id]);
	}

	/**
	 * Registers a classifier under an id, wiring it to the module's storage.
	 * @param string $id The id to register it under.
	 * @param IBayesianClassifier $classifier The classifier.
	 */
	public function addClassifier(string $id, IBayesianClassifier $classifier): void
	{
		$this->_classifiers[$id] = $classifier;
		if ($this->_storage !== null) {
			$classifier->setStorage($this->_storage);
		}
	}

	/**
	 * Returns the id of the classifier {@see getClassifier()} returns when asked for none.
	 * @return ?string The id, or null when the first configured classifier is the default.
	 */
	public function getDefaultClassifierID(): ?string
	{
		return $this->_defaultClassifierID;
	}

	/**
	 * Sets which configured classifier is the default.  Without it the first `<classifier>`
	 * is, which is what a single-model configuration wants.
	 * @param ?string $value The classifier id.
	 */
	public function setDefaultClassifierID(?string $value): void
	{
		$this->_defaultClassifierID = ($value === '') ? null : $value;
	}

	/**
	 * Sets the default classifier; the storage (if any) is wired onto it.
	 * @param IBayesianClassifier $value The classifier.
	 */
	public function setClassifier(IBayesianClassifier $value): void
	{
		$this->_classifier = $value;
		if ($this->_storage !== null) {
			$value->setStorage($this->_storage);
		}
	}

	/**
	 * Returns the configured storage backend; null when none is configured.
	 * @return ?IBayesianStorage The storage, or null.
	 */
	public function getStorage(): ?IBayesianStorage
	{
		return $this->_storage;
	}

	/**
	 * Sets the storage backend; the configured classifier (if any) is wired to it.
	 * @param IBayesianStorage $value The storage.
	 */
	public function setStorage(IBayesianStorage $value): void
	{
		$this->_storage = $value;
		if ($this->_classifier !== null) {
			$this->_classifier->setStorage($value);
		}
		// Every named classifier shares the module's storage; that is the point of configuring
		// several of them together.
		foreach ($this->_classifiers as $classifier) {
			$classifier->setStorage($value);
		}
	}

	/**
	 * Returns the name of the model eagerly loaded from storage during module initialization.
	 * @return ?string The configured default classifier name, or null when none is configured.
	 */
	public function getDefaultClassifier(): ?string
	{
		return $this->_defaultClassifier;
	}

	/**
	 * Sets the name of the model to load from storage on initialization.
	 * @param ?string $value The model name.
	 */
	public function setDefaultClassifier(?string $value): void
	{
		$this->_defaultClassifier = $value === '' ? null : $value;
	}
}
