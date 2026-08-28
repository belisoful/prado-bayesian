<?php

/**
 * TBayesianModelConverter class file.
 *
 * @author Brad Anderson <belisoful@icloud.com>
 * @link https://github.com/belisoful/prado-bayesian
 * @license https://github.com/pradosoft/prado/blob/master/LICENSE
 */

namespace Belisoful\Prado\Util\Bayesian;

use Belisoful\Prado\Util\Bayesian\Classifier\IBayesianClassifier;
use Belisoful\Prado\Util\Bayesian\Classifier\TBernoulliNaiveBayes;
use Belisoful\Prado\Util\Bayesian\Classifier\TComplementNaiveBayes;
use Belisoful\Prado\Util\Bayesian\Classifier\TMultinomialNaiveBayes;
use Belisoful\Prado\Util\Bayesian\Classifier\TNaiveBayesClassifier;
use Belisoful\Prado\Util\Bayesian\Storage\IBayesianStorage;
use Belisoful\Prado\Util\Bayesian\Storage\IBayesianTokenStorage;
use Prado\Exceptions\TConfigurationException;
use Prado\Exceptions\TInvalidDataValueException;

/**
 * TBayesianModelConverter class.
 *
 * Rewrites a whole-payload model into the per-token layout, so a model that has outgrown the
 * process it must be classified in can move to a backend that reads it a document at a time
 * without being retrained.
 *
 * The conversion itself is what a classifier already does: {@see load()} a payload into a
 * resident {@see TBayesianVocabulary}, point it at a {@see IBayesianTokenStorage}, and
 * {@see save()} — the classifier writes whichever layout the destination is in.  What this
 * class adds is picking the right classifier variant from the model's stored `kind` marker, so
 * a caller converting a directory of models need not know which of them is Bernoulli and which
 * is Complement, and packaging it as one tested operation with its edge cases handled.
 *
 * Only this direction is supported.  Going back — per-token to payload — would mean enumerating
 * the whole vocabulary, which the per-token layout deliberately does not expose (its point is
 * that the vocabulary need never be fully resident); retrain, or keep the payload copy.
 *
 * ```php
 * $converter = new TBayesianModelConverter();
 * $converter->convert($fileStorage, $sqlTokenStorage, 'comment-spam');
 * // or convert every model a backend holds:
 * $converter->convertAll($fileStorage, $sqlTokenStorage);
 * ```
 *
 * @author Brad Anderson <belisoful@icloud.com>
 * @since 0.1.0
 */
class TBayesianModelConverter
{
	/**
	 * @var array<string, class-string<IBayesianClassifier>> The classifier class for each `kind`
	 * marker.  A custom variant registers its own with {@see registerKind()}.
	 */
	private array $_kinds = [
		'naive-bayes' => TNaiveBayesClassifier::class,
		'multinomial-naive-bayes' => TMultinomialNaiveBayes::class,
		'bernoulli-naive-bayes' => TBernoulliNaiveBayes::class,
		'complement-naive-bayes' => TComplementNaiveBayes::class,
	];

	/**
	 * Registers the classifier class a `kind` marker maps to, for a custom classifier variant.
	 * @param string $kind The kind marker the variant writes into its saved payload.
	 * @param class-string<IBayesianClassifier> $class The classifier class.
	 * @throws TInvalidDataValueException When the class does not implement {@see IBayesianClassifier}.
	 */
	public function registerKind(string $kind, string $class): void
	{
		if (!is_a($class, IBayesianClassifier::class, true)) {
			throw new TInvalidDataValueException('bayesian_classifier_class_invalid', $class);
		}
		$this->_kinds[$kind] = $class;
	}

	/**
	 * Converts one whole-payload model into the per-token layout of the destination storage.
	 *
	 * The destination is written under `$destName`, or the source name when that is null, so a
	 * model can be converted in place (source and destination sharing one database) or copied
	 * under a new name.
	 * @param IBayesianStorage $source The storage holding the whole-payload model.
	 * @param IBayesianTokenStorage $destination The per-token storage to write into.
	 * @param string $name The model name in the source.
	 * @param ?string $destName The name to write under; the source name when null.
	 * @throws TConfigurationException When the source model does not exist.
	 * @throws TInvalidDataValueException When the source model is already per-token, or its kind
	 * marker is one no classifier is registered for.
	 * @throws \Prado\Exceptions\TInvalidOperationException When the destination is not in
	 * per-token mode.
	 */
	public function convert(IBayesianStorage $source, IBayesianTokenStorage $destination, string $name, ?string $destName = null): void
	{
		if (!$destination->getSupportsTokenLookup()) {
			throw new TInvalidDataValueException('bayesian_convert_destination_not_token', $name);
		}
		$payload = $source->load($name);
		if ($payload === null) {
			throw new TConfigurationException('bayesian_classifier_model_missing', $name);
		}
		if (!empty($payload['tokenMode'])) {
			// load() on a per-token backend returns only the metadata, not the token statistics,
			// so a model already in that layout cannot be rebuilt through this path — and would
			// not need converting anyway.
			throw new TInvalidDataValueException('bayesian_convert_source_already_token', $name);
		}
		$classifier = $this->classifierForKind(is_string($payload['kind'] ?? null) ? $payload['kind'] : '', $name);

		// Loading through the classifier (rather than importing the peeked payload) keeps the
		// kind check and the tokenizer restore in one place — the classifier's own load path.
		$classifier->setStorage($source);
		$classifier->load($name);

		$classifier->setStorage($destination);
		$classifier->setName($destName ?? $name);
		$classifier->save();
	}

	/**
	 * Converts every whole-payload model the source holds into the destination's per-token
	 * layout, keeping each model's name.
	 * @param IBayesianStorage $source The storage to read from.
	 * @param IBayesianTokenStorage $destination The per-token storage to write into.
	 * @throws TInvalidDataValueException When the destination is not in per-token mode, or a
	 * source model's kind is unregistered.
	 * @return string[] The names of the models converted.
	 */
	public function convertAll(IBayesianStorage $source, IBayesianTokenStorage $destination): array
	{
		$converted = [];
		foreach ($source->list() as $name) {
			$this->convert($source, $destination, $name);
			$converted[] = $name;
		}
		return $converted;
	}

	/**
	 * Builds a classifier of the variant a `kind` marker names.
	 *
	 * An empty marker — an older payload saved before variants were distinguished — is treated
	 * as the base {@see TNaiveBayesClassifier}, which is the variant that wrote it.
	 * @param string $kind The kind marker.
	 * @param string $name The model name, for the exception message.
	 * @throws TInvalidDataValueException When no classifier is registered for the kind.
	 * @return IBayesianClassifier The classifier.
	 */
	private function classifierForKind(string $kind, string $name): IBayesianClassifier
	{
		if ($kind === '') {
			$kind = 'naive-bayes';
		}
		if (!isset($this->_kinds[$kind])) {
			throw new TInvalidDataValueException('bayesian_convert_kind_unknown', $name, $kind);
		}
		$class = $this->_kinds[$kind];
		return new $class();
	}
}
