<?php

use Prado\Util\Bayesian\Classifier\TBernoulliNaiveBayes;
use Prado\Util\Bayesian\Classifier\TComplementNaiveBayes;
use Prado\Util\Bayesian\Classifier\TNaiveBayesClassifier;
use Prado\Util\Bayesian\Storage\TSqlBayesianStorage;
use Prado\Util\Bayesian\TBayesianModule;
use Prado\Xml\TXmlDocument;

require_once(__DIR__ . '/../../../test_tools/BayesianTestApplication.php');
require_once(__DIR__ . '/../../../test_tools/BayesianBackends.php');

/**
 * Covers several models living in one storage backend, and one module owning several of them.
 *
 * The interesting cases are the ones where models could bleed into each other: overlapping
 * vocabularies, opposite labels, incremental training on one while another is loaded. A model is
 * identified only by its name, so isolation is a property of the storage keying rather than of
 * anything the classifier does, and it is worth pinning down explicitly.
 */
class TBayesianMultiModelTest extends PHPUnit\Framework\TestCase
{
	/** @var string[] Files to remove after the test. */
	private array $_files = [];

	protected function setUp(): void
	{
		BayesianBackends::requireBackend($this, extension_loaded('pdo'), 'pdo extension not available');
		BayesianBackends::requireBackend(
			$this,
			in_array('sqlite', \PDO::getAvailableDrivers(), true),
			'pdo_sqlite driver not available'
		);
		BayesianTestApplication::get();
	}

	protected function tearDown(): void
	{
		foreach ($this->_files as $file) {
			@unlink($file);
		}
		$this->_files = [];
	}

	private function file(): string
	{
		$file = sys_get_temp_dir() . '/bayesian-multi-' . uniqid('', true) . '.sqlite';
		$this->_files[] = $file;
		return $file;
	}

	private function storage(string $file, string $mode): TSqlBayesianStorage
	{
		$storage = new TSqlBayesianStorage();
		$storage->setConnectionString('sqlite:' . $file);
		$storage->setMode($mode);
		return $storage;
	}

	/** @return string[] Both storage layouts; multi-model behaviour must not differ between them. */
	public static function modes(): array
	{
		return [TSqlBayesianStorage::MODE_PAYLOAD, TSqlBayesianStorage::MODE_TOKEN];
	}

	public function testOneBackendHoldsSeveralModelsOfDifferentVariantsAndTokenizers()
	{
		foreach (self::modes() as $mode) {
			$storage = $this->storage($this->file(), $mode);

			$spam = new TNaiveBayesClassifier();
			$spam->setStorage($storage);
			$spam->setName('comment-spam');
			$spam->trainOne('spam', 'cheap pills buy watches');
			$spam->trainOne('ham', 'project meeting agenda');
			$spam->save();

			$language = new TBernoulliNaiveBayes();
			$language->setStorage($storage);
			$language->setName('language-id');
			$language->trainOne('en', 'the quick brown fox');
			$language->trainOne('fr', 'le renard brun rapide');
			$language->save();

			$topic = new TComplementNaiveBayes();
			$topic->setStorage($storage);
			$topic->setName('topic');
			$topic->trainOne('tech', 'database index query optimizer');
			$topic->trainOne('sport', 'match goal referee season');
			$topic->save();

			self::assertSame(['comment-spam', 'language-id', 'topic'], $storage->list(), $mode);

			$loadedSpam = new TNaiveBayesClassifier();
			$loadedSpam->setStorage($storage);
			$loadedSpam->load('comment-spam');
			$loadedLanguage = new TBernoulliNaiveBayes();
			$loadedLanguage->setStorage($storage);
			$loadedLanguage->load('language-id');
			$loadedTopic = new TComplementNaiveBayes();
			$loadedTopic->setStorage($storage);
			$loadedTopic->load('topic');

			self::assertSame('spam', $loadedSpam->classify('cheap watches'), $mode);
			self::assertSame('fr', $loadedLanguage->classify('le renard'), $mode);
			self::assertSame('tech', $loadedTopic->classify('query optimizer'), $mode);
			self::assertSame(['spam', 'ham'], $loadedSpam->getVocabulary()->getCategoryNames(), $mode);
			self::assertSame(['en', 'fr'], $loadedLanguage->getVocabulary()->getCategoryNames(), $mode);
		}
	}

	public function testModelsWithFullyOverlappingVocabulariesStayIsolated()
	{
		// Two models over the same tokens with opposite labels: if the storage keyed anything
		// by token alone, these would contaminate each other and both would be wrong.
		foreach (self::modes() as $mode) {
			$storage = $this->storage($this->file(), $mode);

			$a = new TNaiveBayesClassifier();
			$a->setStorage($storage);
			$a->setName('A');
			$a->trainOne('yes', 'alpha beta gamma');
			$a->trainOne('no', 'delta epsilon');
			$a->save();

			$b = new TNaiveBayesClassifier();
			$b->setStorage($storage);
			$b->setName('B');
			$b->trainOne('no', 'alpha beta gamma');
			$b->trainOne('yes', 'delta epsilon');
			$b->save();

			$soloStorage = $this->storage($this->file(), $mode);
			$solo = new TNaiveBayesClassifier();
			$solo->setStorage($soloStorage);
			$solo->setName('A');
			$solo->trainOne('yes', 'alpha beta gamma');
			$solo->trainOne('no', 'delta epsilon');
			$solo->save();

			$loadedA = new TNaiveBayesClassifier();
			$loadedA->setStorage($storage);
			$loadedA->load('A');
			$loadedB = new TNaiveBayesClassifier();
			$loadedB->setStorage($storage);
			$loadedB->load('B');
			$loadedSolo = new TNaiveBayesClassifier();
			$loadedSolo->setStorage($soloStorage);
			$loadedSolo->load('A');

			self::assertSame('yes', $loadedA->classify('alpha'), $mode);
			self::assertSame('no', $loadedB->classify('alpha'), $mode);
			// Sharing a backend must not change a model's numbers at all.
			self::assertSame($loadedSolo->score('alpha beta'), $loadedA->score('alpha beta'), $mode);
			self::assertSame(2, $loadedA->getVocabulary()->getTotalDocuments(), $mode);
			self::assertSame(
				$loadedSolo->getVocabulary()->getVocabularySize(),
				$loadedA->getVocabulary()->getVocabularySize(),
				$mode
			);
		}
	}

	public function testTrainingOneModelLeavesItsNeighbourUntouched()
	{
		$storage = $this->storage($this->file(), TSqlBayesianStorage::MODE_TOKEN);
		foreach (['A', 'B'] as $name) {
			$c = new TNaiveBayesClassifier();
			$c->setStorage($storage);
			$c->setName($name);
			$c->trainOne('yes', 'alpha beta');
			$c->trainOne('no', 'gamma delta');
			$c->save();
		}
		$writer = new TNaiveBayesClassifier();
		$writer->setStorage($storage);
		$writer->load('B');
		$writer->trainOne('no', 'alpha alpha alpha epsilon');

		$reader = new TNaiveBayesClassifier();
		$reader->setStorage($storage);
		$reader->load('A');
		self::assertSame(2, $reader->getVocabulary()->getTotalDocuments());
		self::assertSame(4, $reader->getVocabulary()->getVocabularySize());

		$readerB = new TNaiveBayesClassifier();
		$readerB->setStorage($storage);
		$readerB->load('B');
		self::assertSame(3, $readerB->getVocabulary()->getTotalDocuments());
		self::assertSame(5, $readerB->getVocabulary()->getVocabularySize());
	}

	public function testTokenizerSettingsTravelPerModel()
	{
		foreach (self::modes() as $mode) {
			$storage = $this->storage($this->file(), $mode);
			$tokenizer = new \Prado\Util\Bayesian\Tokenizer\TWordTokenizer();
			$tokenizer->setMinLength(7);
			$tokenizer->setStopWords(['ignoreme']);

			$c = new TNaiveBayesClassifier();
			$c->setStorage($storage);
			$c->setName('C');
			$c->setTokenizer($tokenizer);
			$c->trainOne('x', 'enormous ignoreme tiny');
			$c->trainOne('y', 'gigantic minuscule');
			$c->save();

			$loaded = new TNaiveBayesClassifier();
			$loaded->setStorage($storage);
			$loaded->load('C');
			self::assertSame(7, $loaded->getTokenizer()->getMinLength(), $mode);
			self::assertSame(['ignoreme'], $loaded->getTokenizer()->getStopWords(), $mode);
		}
	}

	private function multiModelXml(string $file): TXmlDocument
	{
		$xml = new TXmlDocument();
		$xml->loadFromString(
			'<module DefaultClassifierID="spam">'
			. '<storage class="TSqlBayesianStorage" ConnectionString="sqlite:' . $file . '" Mode="token" />'
			. '<classifier id="spam" class="TComplementNaiveBayes" Model="comment-spam" Alpha="0.5">'
			. '<tokenizer class="TWordTokenizer" MinLength="3" /></classifier>'
			. '<classifier id="lang" class="TBernoulliNaiveBayes" Model="language-id">'
			. '<tokenizer class="TNGramTokenizer" N="3" Characters="true" /></classifier>'
			. '</module>'
		);
		return $xml;
	}

	public function testModuleHoldsSeveralClassifiersOverOneStorage()
	{
		$file = $this->file();
		$module = new TBayesianModule();
		$module->init($this->multiModelXml($file));

		self::assertSame(['spam', 'lang'], array_keys($module->getClassifiers()));
		self::assertInstanceOf(TComplementNaiveBayes::class, $module->getClassifier('spam'));
		self::assertInstanceOf(TBernoulliNaiveBayes::class, $module->getClassifier('lang'));
		self::assertSame('comment-spam', $module->getClassifier('spam')->getName());
		self::assertSame('language-id', $module->getClassifier('lang')->getName());
		self::assertSame(0.5, $module->getClassifier('spam')->getAlpha());
		// One backend, shared by every model the module owns.
		self::assertSame($module->getStorage(), $module->getClassifier('spam')->getStorage());
		self::assertSame($module->getStorage(), $module->getClassifier('lang')->getStorage());
	}

	public function testModuleGivesEachClassifierItsConfiguredTokenizer()
	{
		$module = new TBayesianModule();
		$module->init($this->multiModelXml($this->file()));

		$spamTokenizer = $module->getClassifier('spam')->getTokenizer();
		self::assertInstanceOf(\Prado\Util\Bayesian\Tokenizer\TWordTokenizer::class, $spamTokenizer);
		self::assertSame(3, $spamTokenizer->getMinLength());

		$languageTokenizer = $module->getClassifier('lang')->getTokenizer();
		self::assertInstanceOf(\Prado\Util\Bayesian\Tokenizer\TNGramTokenizer::class, $languageTokenizer);
		self::assertSame(3, $languageTokenizer->getN());
		self::assertTrue($languageTokenizer->getCharacters());
	}

	public function testTheFirstConfiguredClassifierIsTheDefault()
	{
		$module = new TBayesianModule();
		$module->init($this->multiModelXml($this->file()));
		self::assertSame($module->getClassifier('spam'), $module->getClassifier());
	}

	public function testDefaultClassifierIdSelectsAnotherAsTheDefault()
	{
		// DefaultClassifierID is a module property, so PRADO applies it before init(); set it
		// the same way here rather than relying on the attribute in the XML body.
		$module = new TBayesianModule();
		$module->setDefaultClassifierID('lang');
		$module->init($this->multiModelXml($this->file()));
		self::assertSame($module->getClassifier('lang'), $module->getClassifier());
		self::assertSame('lang', $module->getDefaultClassifierID());
	}

	public function testModuleEagerlyLoadsEveryConfiguredModel()
	{
		$file = $this->file();
		$module = new TBayesianModule();
		$module->init($this->multiModelXml($file));
		$module->getClassifier('spam')->trainOne('spam', 'cheap pills watches');
		$module->getClassifier('spam')->trainOne('ham', 'project meeting agenda');
		$module->getClassifier('spam')->save();
		$module->getClassifier('lang')->trainOne('en', 'the quick brown fox');
		$module->getClassifier('lang')->trainOne('fr', 'le renard brun rapide');
		$module->getClassifier('lang')->save();

		$booted = new TBayesianModule();
		$booted->init($this->multiModelXml($file));
		self::assertTrue($booted->getClassifier('spam')->getIsTrained());
		self::assertTrue($booted->getClassifier('lang')->getIsTrained());
		self::assertSame('spam', $booted->getClassifier('spam')->classify('cheap watches'));
		self::assertSame('fr', $booted->getClassifier('lang')->classify('le renard'));
	}

	public function testAnUnknownClassifierIdIsAConfigurationError()
	{
		$module = new TBayesianModule();
		$module->init($this->multiModelXml($this->file()));
		try {
			$module->getClassifier('nope');
			self::fail('expected an unknown classifier id to raise');
		} catch (\Prado\Exceptions\TConfigurationException $e) {
			self::assertSame('bayesian_classifier_id_unknown', $e->getErrorCode());
		}
	}

	public function testASingleUnnamedClassifierStillBehavesAsTheDefault()
	{
		// The one-model configuration must not have to know about ids.
		$xml = new TXmlDocument();
		$xml->loadFromString(
			'<module>'
			. '<classifier class="TComplementNaiveBayes" Alpha="0.25" />'
			. '<storage class="TMemoryBayesianStorage" />'
			. '</module>'
		);
		$module = new TBayesianModule();
		$module->setDefaultClassifier('only');
		$module->init($xml);
		self::assertSame([], $module->getClassifiers());
		self::assertInstanceOf(TComplementNaiveBayes::class, $module->getClassifier());
		self::assertSame(0.25, $module->getClassifier()->getAlpha());
		self::assertSame('only', $module->getClassifier()->getName());
	}

	public function testClassifiersCanBeRegisteredFromCode()
	{
		$module = new TBayesianModule();
		$module->setStorage(new \Prado\Util\Bayesian\Storage\TMemoryBayesianStorage());
		$extra = new TBernoulliNaiveBayes();
		$module->addClassifier('extra', $extra);

		self::assertTrue($module->hasClassifier('extra'));
		self::assertFalse($module->hasClassifier('missing'));
		self::assertSame($extra, $module->getClassifier('extra'));
		self::assertSame($module->getStorage(), $extra->getStorage());
	}

	public function testPhpConfigurationAcceptsSeveralClassifiers()
	{
		$file = $this->file();
		$module = new TBayesianModule();
		$module->init([
			'storage' => ['class' => 'TSqlBayesianStorage', 'ConnectionString' => 'sqlite:' . $file, 'Mode' => 'token'],
			'classifier' => [
				'spam' => ['class' => 'TComplementNaiveBayes', 'Model' => 'comment-spam', 'Alpha' => 0.5],
				'lang' => ['class' => 'TBernoulliNaiveBayes', 'Model' => 'language-id'],
			],
		]);
		self::assertSame(['spam', 'lang'], array_keys($module->getClassifiers()));
		self::assertSame('comment-spam', $module->getClassifier('spam')->getName());
		self::assertSame(0.5, $module->getClassifier('spam')->getAlpha());
		self::assertSame($module->getStorage(), $module->getClassifier('lang')->getStorage());
	}
}
