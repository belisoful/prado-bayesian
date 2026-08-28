<?php

use Belisoful\Prado\Util\Bayesian\Classifier\TBernoulliNaiveBayes;
use Belisoful\Prado\Util\Bayesian\Classifier\TComplementNaiveBayes;
use Belisoful\Prado\Util\Bayesian\Classifier\TNaiveBayesClassifier;
use Belisoful\Prado\Util\Bayesian\Storage\TFileBayesianStorage;
use Belisoful\Prado\Util\Bayesian\Storage\TMemoryBayesianStorage;
use Belisoful\Prado\Util\Bayesian\Storage\TSqlBayesianStorage;
use Belisoful\Prado\Util\Bayesian\TBayesianModelConverter;

require_once(__DIR__ . '/../../../test_tools/BayesianBackends.php');

/**
 * Covers the whole-payload to per-token converter.
 *
 * A conversion changes only the layout, never the model, so the assertions compare the
 * converted model's scores against the source's — a difference of anything but zero is a bug in
 * the conversion, not a tolerance to accept.
 */
class TBayesianModelConverterTest extends PHPUnit\Framework\TestCase
{
	/** @var string[] Files and directories to remove after the test. */
	private array $_paths = [];

	protected function setUp(): void
	{
		BayesianBackends::requireBackend($this, extension_loaded('pdo'), 'pdo extension not available');
		BayesianBackends::requireBackend(
			$this,
			in_array('sqlite', \PDO::getAvailableDrivers(), true),
			'pdo_sqlite driver not available'
		);
	}

	protected function tearDown(): void
	{
		foreach ($this->_paths as $path) {
			if (is_dir($path)) {
				array_map('unlink', glob($path . '/*') ?: []);
				@rmdir($path);
			} else {
				@unlink($path);
			}
		}
		$this->_paths = [];
	}

	private function sqlFile(): string
	{
		$file = sys_get_temp_dir() . '/bayesian-convert-' . uniqid('', true) . '.sqlite';
		$this->_paths[] = $file;
		return $file;
	}

	private function tokenStorage(string $file): TSqlBayesianStorage
	{
		$storage = new TSqlBayesianStorage();
		$storage->setConnectionString('sqlite:' . $file);
		$storage->setMode(TSqlBayesianStorage::MODE_TOKEN);
		return $storage;
	}

	private function payloadStorage(string $file): TSqlBayesianStorage
	{
		$storage = new TSqlBayesianStorage();
		$storage->setConnectionString('sqlite:' . $file);
		return $storage;
	}

	private function train(TNaiveBayesClassifier $classifier): TNaiveBayesClassifier
	{
		$corpus = [
			['spam', 'cheap pills buy now cheap watches'],
			['spam', 'lowest prices order now limited offer'],
			['ham', 'project meeting agenda tomorrow morning'],
			['ham', 'please review the quarterly report'],
		];
		foreach ($corpus as [$category, $document]) {
			$classifier->trainOne($category, $document);
		}
		return $classifier;
	}

	public function testConvertPreservesScoresForEveryVariant()
	{
		foreach ([TNaiveBayesClassifier::class, TBernoulliNaiveBayes::class, TComplementNaiveBayes::class] as $class) {
			$srcFile = $this->sqlFile();
			$destFile = $this->sqlFile();

			$source = $this->payloadStorage($srcFile);
			$original = new $class();
			$original->setStorage($source);
			$original->setName('m');
			$original->setAlpha(0.5);
			$this->train($original)->save();

			$destination = $this->tokenStorage($destFile);
			(new TBayesianModelConverter())->convert($source, $destination, 'm');

			$converted = new $class();
			$converted->setStorage($destination);
			$converted->load('m');

			self::assertFalse($converted->getVocabulary()->getSupportsFullScan(), $class);
			foreach (['cheap prices now', 'review the report', 'free'] as $probe) {
				self::assertSame($original->score($probe), $converted->score($probe), $class . ' / ' . $probe);
			}
		}
	}

	public function testConvertPicksTheVariantFromTheStoredKind()
	{
		// The caller does not name the variant; the converter reads it from the payload. A
		// Complement model must come back as Complement, not the base classifier.
		$srcFile = $this->sqlFile();
		$destFile = $this->sqlFile();
		$source = $this->payloadStorage($srcFile);
		$original = new TComplementNaiveBayes();
		$original->setStorage($source);
		$original->setName('m');
		$this->train($original)->save();

		(new TBayesianModelConverter())->convert($source, $this->tokenStorage($destFile), 'm');

		// Loading the converted model as the WRONG variant must be refused by the kind marker,
		// proving the marker survived the conversion.
		$wrong = new TBernoulliNaiveBayes();
		$wrong->setStorage($this->tokenStorage($destFile));
		$this->expectException(\Prado\Exceptions\TInvalidDataValueException::class);
		$wrong->load('m');
	}

	public function testConvertCarriesTheTokenizerAndSettings()
	{
		$srcFile = $this->sqlFile();
		$destFile = $this->sqlFile();
		$tokenizer = new \Belisoful\Prado\Util\Bayesian\Tokenizer\TNGramTokenizer();
		$tokenizer->setN(3);
		$tokenizer->setCharacters(true);

		$source = $this->payloadStorage($srcFile);
		$original = new TNaiveBayesClassifier();
		$original->setStorage($source);
		$original->setName('m');
		$original->setTokenizer($tokenizer);
		$original->setUseTfidf(true);
		$this->train($original)->save();

		(new TBayesianModelConverter())->convert($source, $this->tokenStorage($destFile), 'm');

		$converted = new TNaiveBayesClassifier();
		$converted->setStorage($this->tokenStorage($destFile));
		$converted->load('m');
		self::assertInstanceOf(\Belisoful\Prado\Util\Bayesian\Tokenizer\TNGramTokenizer::class, $converted->getTokenizer());
		self::assertSame(3, $converted->getTokenizer()->getN());
		self::assertTrue($converted->getUseTfidf());
	}

	public function testConvertAcrossBackends()
	{
		// The source need not be SQL; a file or in-process blob converts just as well.
		$memory = new TMemoryBayesianStorage();
		$original = new TNaiveBayesClassifier();
		$original->setStorage($memory);
		$original->setName('m');
		$this->train($original)->save();

		$destFile = $this->sqlFile();
		(new TBayesianModelConverter())->convert($memory, $this->tokenStorage($destFile), 'm');

		$converted = new TNaiveBayesClassifier();
		$converted->setStorage($this->tokenStorage($destFile));
		$converted->load('m');
		self::assertSame($original->score('cheap review'), $converted->score('cheap review'));
	}

	public function testConvertToANewName()
	{
		$srcFile = $this->sqlFile();
		$destFile = $this->sqlFile();
		$source = $this->payloadStorage($srcFile);
		$original = new TNaiveBayesClassifier();
		$original->setStorage($source);
		$original->setName('original');
		$this->train($original)->save();

		$destination = $this->tokenStorage($destFile);
		(new TBayesianModelConverter())->convert($source, $destination, 'original', 'renamed');
		self::assertSame(['renamed'], $destination->list());
	}

	public function testConvertInPlaceIntoTheSameDatabase()
	{
		// The common case: promote a blob model to per-token within one database, by pointing a
		// payload-mode and a token-mode storage at the same connection string.
		$file = $this->sqlFile();
		$source = $this->payloadStorage($file);
		$original = new TComplementNaiveBayes();
		$original->setStorage($source);
		$original->setName('m');
		$this->train($original)->save();

		$destination = $this->tokenStorage($file);
		(new TBayesianModelConverter())->convert($source, $destination, 'm');

		$converted = new TComplementNaiveBayes();
		$converted->setStorage($this->tokenStorage($file));
		$converted->load('m');
		self::assertFalse($converted->getVocabulary()->getSupportsFullScan());
		self::assertSame($original->score('cheap prices'), $converted->score('cheap prices'));
	}

	public function testConvertAllConvertsEveryModel()
	{
		$srcFile = $this->sqlFile();
		$destFile = $this->sqlFile();
		$source = $this->payloadStorage($srcFile);
		foreach (['alpha', 'beta', 'gamma'] as $name) {
			$c = new TNaiveBayesClassifier();
			$c->setStorage($source);
			$c->setName($name);
			$this->train($c)->save();
		}

		$destination = $this->tokenStorage($destFile);
		$converted = (new TBayesianModelConverter())->convertAll($source, $destination);
		sort($converted);
		self::assertSame(['alpha', 'beta', 'gamma'], $converted);
		self::assertSame(['alpha', 'beta', 'gamma'], $destination->list());
	}

	public function testConvertRejectsAPayloadDestination()
	{
		$srcFile = $this->sqlFile();
		$source = $this->payloadStorage($srcFile);
		$original = new TNaiveBayesClassifier();
		$original->setStorage($source);
		$original->setName('m');
		$this->train($original)->save();

		$payloadDest = $this->payloadStorage($this->sqlFile());
		try {
			(new TBayesianModelConverter())->convert($source, $payloadDest, 'm');
			self::fail('expected a payload destination to be rejected');
		} catch (\Prado\Exceptions\TInvalidDataValueException $e) {
			self::assertSame('bayesian_convert_destination_not_token', $e->getErrorCode());
		}
	}

	public function testConvertRejectsAnAlreadyPerTokenSource()
	{
		$file = $this->sqlFile();
		$tokenSource = $this->tokenStorage($file);
		$original = new TNaiveBayesClassifier();
		$original->setStorage($tokenSource);
		$original->setName('m');
		$this->train($original)->save();

		// Reading it back through a payload-mode storage on the same DB yields the meta blob,
		// which carries tokenMode=true; the converter must refuse it.
		$payloadView = $this->payloadStorage($file);
		try {
			(new TBayesianModelConverter())->convert($payloadView, $this->tokenStorage($this->sqlFile()), 'm');
			self::fail('expected an already-per-token source to be rejected');
		} catch (\Prado\Exceptions\TInvalidDataValueException $e) {
			self::assertSame('bayesian_convert_source_already_token', $e->getErrorCode());
		}
	}

	public function testConvertRejectsAMissingModel()
	{
		$source = $this->payloadStorage($this->sqlFile());
		$this->expectException(\Prado\Exceptions\TConfigurationException::class);
		(new TBayesianModelConverter())->convert($source, $this->tokenStorage($this->sqlFile()), 'nope');
	}

	public function testConvertRejectsAnUnknownKind()
	{
		// A payload whose kind names a variant the converter has no class for.
		$source = new TMemoryBayesianStorage();
		$source->save('m', ['kind' => 'quantum-naive-bayes', 'categories' => [], 'documentFrequency' => [], 'totalDocuments' => 0]);
		try {
			(new TBayesianModelConverter())->convert($source, $this->tokenStorage($this->sqlFile()), 'm');
			self::fail('expected an unknown kind to be rejected');
		} catch (\Prado\Exceptions\TInvalidDataValueException $e) {
			self::assertSame('bayesian_convert_kind_unknown', $e->getErrorCode());
		}
	}

	public function testRegisterKindRejectsANonClassifier()
	{
		$converter = new TBayesianModelConverter();
		$this->expectException(\Prado\Exceptions\TInvalidDataValueException::class);
		$converter->registerKind('bogus', \stdClass::class);
	}
}
