<?php

use Belisoful\Prado\Util\Bayesian\Classifier\TNaiveBayesClassifier;
use Belisoful\Prado\Util\Bayesian\Storage\TFileBayesianStorage;
use Belisoful\Prado\Util\Bayesian\Storage\TMemoryBayesianStorage;
use Belisoful\Prado\Util\Bayesian\TBayesianModule;
use Prado\Exceptions\TConfigurationException;

require_once(__DIR__ . '/../../../test_tools/BayesianTestApplication.php');

/** Exposes the protected factories for direct testing. */
class TestableBayesianModule extends TBayesianModule
{
	public function createClassifierPublic(array $properties): void
	{
		$this->createClassifier($properties);
	}

	public function createStoragePublic(array $properties): void
	{
		$this->createStorage($properties);
	}
}

class TBayesianModuleTest extends PHPUnit\Framework\TestCase
{
	private array $_tempDirs = [];

	protected function tearDown(): void
	{
		foreach ($this->_tempDirs as $dir) {
			foreach (glob($dir . '/*') ?: [] as $file) {
				@unlink($file);
			}
			@rmdir($dir);
		}
	}

	public function testDefaultClassifierIsCreatedLazily()
	{
		$module = new TBayesianModule();
		self::assertInstanceOf(TNaiveBayesClassifier::class, $module->getClassifier());
		self::assertSame($module->getClassifier(), $module->getClassifier(), 'The classifier is created once.');
	}

	public function testSetClassifierReplacesDefault()
	{
		$module = new TBayesianModule();
		$custom = new TNaiveBayesClassifier();
		$module->setClassifier($custom);
		self::assertSame($custom, $module->getClassifier());
	}

	public function testStorageIsNullByDefault()
	{
		$module = new TBayesianModule();
		self::assertNull($module->getStorage());
	}

	public function testSetStorageWiresClassifierStorage()
	{
		$module = new TBayesianModule();
		$classifier = new TNaiveBayesClassifier();
		$module->setClassifier($classifier);
		$storage = new TMemoryBayesianStorage();
		$module->setStorage($storage);
		self::assertSame($storage, $module->getStorage());
		self::assertSame($storage, $classifier->getStorage());
	}

	public function testSetClassifierWiresExistingStorage()
	{
		$module = new TBayesianModule();
		$storage = new TMemoryBayesianStorage();
		$module->setStorage($storage);
		$classifier = new TNaiveBayesClassifier();
		$module->setClassifier($classifier);
		self::assertSame($storage, $classifier->getStorage());
	}

	public function testCreateClassifierInstantiatesAndConfigures()
	{
		$module = new TestableBayesianModule();
		$module->createClassifierPublic([
			'class' => TNaiveBayesClassifier::class,
			'Name' => 'spam',
			'Alpha' => 0.5,
		]);
		$classifier = $module->getClassifier();
		self::assertInstanceOf(TNaiveBayesClassifier::class, $classifier);
		self::assertSame('spam', $classifier->getName());
		self::assertEqualsWithDelta(0.5, $classifier->getAlpha(), 1e-9);
	}

	public function testCreateClassifierRejectsInvalidClass()
	{
		$module = new TestableBayesianModule();
		$this->expectException(TConfigurationException::class);
		$module->createClassifierPublic(['class' => \stdClass::class]);
	}

	public function testCreateClassifierRejectsMissingClass()
	{
		$module = new TestableBayesianModule();
		$this->expectException(TConfigurationException::class);
		$module->createClassifierPublic([]);
	}

	public function testCreateStorageInstantiatesAndConfigures()
	{
		$dir = $this->tempDir();
		$module = new TestableBayesianModule();
		$module->createStoragePublic([
			'class' => TFileBayesianStorage::class,
			'Directory' => $dir,
		]);
		$storage = $module->getStorage();
		self::assertInstanceOf(TFileBayesianStorage::class, $storage);
		self::assertSame($dir, $storage->getDirectory());
	}

	public function testCreateStorageRejectsInvalidClass()
	{
		$module = new TestableBayesianModule();
		$this->expectException(TConfigurationException::class);
		$module->createStoragePublic(['class' => \stdClass::class]);
	}

	public function testCreateStorageRejectsMissingClass()
	{
		$module = new TestableBayesianModule();
		$this->expectException(TConfigurationException::class);
		$module->createStoragePublic([]);
	}

	public function testDefaultClassifierNameSetter()
	{
		$module = new TBayesianModule();
		$module->setDefaultClassifier('my-model');
		self::assertSame('my-model', $module->getDefaultClassifier());
		$module->setDefaultClassifier('');
		self::assertNull($module->getDefaultClassifier());
	}

	public function testInitWithArrayStorageConfigCreatesNamedUntrainedClassifier()
	{
		BayesianTestApplication::get();
		$module = new TBayesianModule();
		$module->setDefaultClassifier('m');
		$module->init(['storage' => ['class' => TMemoryBayesianStorage::class]]);
		self::assertInstanceOf(TMemoryBayesianStorage::class, $module->getStorage());
		$classifier = $module->getClassifier();
		self::assertSame('m', $classifier->getName());
		self::assertFalse($classifier->getIsTrained(), 'no model is stored yet');
		self::assertSame($module->getStorage(), $classifier->getStorage());
	}

	public function testInitLoadsExistingModelFromStorage()
	{
		BayesianTestApplication::get();
		$storage = new TMemoryBayesianStorage();
		$trainer = new TNaiveBayesClassifier();
		$trainer->setStorage($storage);
		$trainer->setName('m');
		$trainer->trainOne('spam', 'cheap offer click');
		$trainer->trainOne('ham', 'meeting report lunch');
		$trainer->save();

		$module = new TBayesianModule();
		$module->setStorage($storage);
		$module->setDefaultClassifier('m');
		$module->init(null);
		$classifier = $module->getClassifier();
		self::assertNotSame($trainer, $classifier);
		self::assertTrue($classifier->getIsTrained());
		self::assertSame('m', $classifier->getName());
		self::assertSame('spam', $classifier->classify('cheap click'));
	}

	public function testInitWithoutDefaultClassifierDoesNotLoad()
	{
		BayesianTestApplication::get();
		$storage = new TMemoryBayesianStorage();
		$trainer = new TNaiveBayesClassifier();
		$trainer->setStorage($storage);
		$trainer->setName('m');
		$trainer->trainOne('spam', 'cheap offer');
		$trainer->save();
		$module = new TBayesianModule();
		$module->setStorage($storage);
		$module->init(null);
		self::assertNull($module->getClassifier()->getName());
		self::assertFalse($module->getClassifier()->getIsTrained());
	}

	public function testInitKeepsExplicitClassifierName()
	{
		BayesianTestApplication::get();
		$module = new TBayesianModule();
		$module->setDefaultClassifier('m');
		$module->init(['classifier' => ['class' => TNaiveBayesClassifier::class, 'Name' => 'explicit']]);
		self::assertSame('explicit', $module->getClassifier()->getName());
	}

	public function testInitRejectsNonStorageClass()
	{
		BayesianTestApplication::get();
		$module = new TBayesianModule();
		try {
			$module->init(['storage' => ['class' => 'Belisoful\\Prado\\Util\\Bayesian\\TBayesianCategory']]);
			self::fail('expected exception');
		} catch (TConfigurationException $e) {
			self::assertSame('bayesian_storage_class_invalid', $e->getErrorCode());
		}
	}

	public function testInitRejectsNonexistentStorageClass()
	{
		BayesianTestApplication::get();
		$module = new TBayesianModule();
		try {
			$module->init(['storage' => ['class' => 'Nope\\Missing']]);
			self::fail('expected exception');
		} catch (TConfigurationException $e) {
			self::assertSame('bayesian_storage_class_invalid', $e->getErrorCode());
		}
	}

	public function testInitRejectsNonClassifierClass()
	{
		BayesianTestApplication::get();
		$module = new TBayesianModule();
		try {
			$module->init(['classifier' => ['class' => TMemoryBayesianStorage::class]]);
			self::fail('expected exception');
		} catch (TConfigurationException $e) {
			self::assertSame('bayesian_classifier_class_invalid', $e->getErrorCode());
		}
	}

	public function testInitAcceptsShortClassMapName()
	{
		BayesianTestApplication::get();
		if (\Prado\Prado::usingClass('TMemoryBayesianStorage') === null) {
			self::markTestSkipped('The Prado3 short-name class map is not registered in this test environment.');
		}
		$module = new TBayesianModule();
		$module->init(['storage' => ['class' => 'TMemoryBayesianStorage']]);
		self::assertInstanceOf(TMemoryBayesianStorage::class, $module->getStorage());
	}

	public function testInitAcceptsDottedClassName()
	{
		BayesianTestApplication::get();
		$module = new TBayesianModule();
		$module->init(['storage' => ['class' => 'Prado.Bayesian.Storage.TMemoryBayesianStorage']]);
		self::assertInstanceOf(TMemoryBayesianStorage::class, $module->getStorage());
	}

	public function testInitWithClassifierElementConfiguresClassifier()
	{
		BayesianTestApplication::get();
		$module = new TBayesianModule();
		$module->init(['classifier' => ['class' => \Belisoful\Prado\Util\Bayesian\Classifier\TBernoulliNaiveBayes::class, 'Alpha' => 0.5]]);
		$classifier = $module->getClassifier();
		self::assertInstanceOf(\Belisoful\Prado\Util\Bayesian\Classifier\TBernoulliNaiveBayes::class, $classifier);
		self::assertEqualsWithDelta(0.5, $classifier->getAlpha(), 1e-12);
	}

	public function testInitWithClassifierAndStorageWiresBoth()
	{
		BayesianTestApplication::get();
		$module = new TBayesianModule();
		$module->init([
			'classifier' => ['class' => TNaiveBayesClassifier::class],
			'storage' => ['class' => TMemoryBayesianStorage::class],
		]);
		self::assertSame($module->getStorage(), $module->getClassifier()->getStorage());
	}

	public function testInitWithXmlElementCreatesClassifierAndStorage()
	{
		BayesianTestApplication::get();
		$doc = new \Prado\Xml\TXmlDocument();
		$doc->loadFromString('<module id="bayesian" class="Belisoful\\Prado\\Util\\Bayesian\\TBayesianModule">'
			. '<classifier class="Belisoful\\Prado\\Util\\Bayesian\\Classifier\\TBernoulliNaiveBayes" Alpha="0.5"/>'
			. '<storage class="Belisoful\\Prado\\Util\\Bayesian\\Storage\\TMemoryBayesianStorage"/>'
			. '</module>');
		$module = new TBayesianModule();
		$module->init($doc);
		$classifier = $module->getClassifier();
		self::assertInstanceOf(\Belisoful\Prado\Util\Bayesian\Classifier\TBernoulliNaiveBayes::class, $classifier);
		self::assertEqualsWithDelta(0.5, $classifier->getAlpha(), 1e-12);
		self::assertInstanceOf(TMemoryBayesianStorage::class, $module->getStorage());
		self::assertSame($module->getStorage(), $classifier->getStorage());
	}

	public function testInitWithXmlElementWithoutChildrenLeavesDefaults()
	{
		BayesianTestApplication::get();
		$doc = new \Prado\Xml\TXmlDocument();
		$doc->loadFromString('<module id="bayesian"/>');
		$module = new TBayesianModule();
		$module->init($doc);
		self::assertNull($module->getStorage());
		self::assertInstanceOf(TNaiveBayesClassifier::class, $module->getClassifier());
	}

	public function testInitPropagatesStorageExistsFailure()
	{
		BayesianTestApplication::get();
		$storage = new class () extends \Prado\TComponent implements \Belisoful\Prado\Util\Bayesian\Storage\IBayesianStorage {
			public function save(string $name, array $payload): void
			{
			}
			public function load(string $name): ?array
			{
				return null;
			}
			public function exists(string $name): bool
			{
				throw new TConfigurationException('bayesian_storage_directory_unwritable', '/nowhere');
			}
			public function delete(string $name): void
			{
			}
			public function list(): array
			{
				return [];
			}
		};
		$module = new TBayesianModule();
		$module->setStorage($storage);
		$module->setDefaultClassifier('m');
		try {
			$module->init(null);
			self::fail('expected exception');
		} catch (TConfigurationException $e) {
			self::assertSame('bayesian_storage_directory_unwritable', $e->getErrorCode());
		}
	}

	private function tempDir(): string
	{
		$dir = sys_get_temp_dir() . '/bymod-' . uniqid('', true);
		$this->_tempDirs[] = $dir;
		return $dir;
	}
}
