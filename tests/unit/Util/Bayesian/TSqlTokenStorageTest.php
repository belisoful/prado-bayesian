<?php

use Prado\Util\Bayesian\Classifier\TBernoulliNaiveBayes;
use Prado\Util\Bayesian\Classifier\TComplementNaiveBayes;
use Prado\Util\Bayesian\Classifier\TNaiveBayesClassifier;
use Prado\Util\Bayesian\Storage\TSqlBayesianStorage;
use Prado\Util\Bayesian\TLazyBayesianVocabulary;

require_once(__DIR__ . '/../../../test_tools/BayesianBackends.php');

/**
 * Covers per-token storage: the layout that lets a model outgrow the process scoring against it.
 *
 * The claim these tests exist to defend is equivalence.  A per-token model is a different
 * layout, read through a different vocabulary, with the O(|V|) aggregates restored from storage
 * instead of recomputed — and none of that may change a single score. Most of what follows
 * therefore trains the same corpus twice and compares, rather than asserting numbers that could
 * both drift together.
 */
class TSqlTokenStorageTest extends PHPUnit\Framework\TestCase
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
	}

	protected function tearDown(): void
	{
		foreach ($this->_files as $file) {
			@unlink($file);
		}
		$this->_files = [];
	}

	private function storage(string $mode = TSqlBayesianStorage::MODE_TOKEN): TSqlBayesianStorage
	{
		$file = sys_get_temp_dir() . '/bayesian-token-' . uniqid('', true) . '.sqlite';
		$this->_files[] = $file;
		$storage = new TSqlBayesianStorage();
		$storage->setConnectionString('sqlite:' . $file);
		$storage->setMode($mode);
		return $storage;
	}

	/** @return array<int, array{0:string,1:string}> A small two-class corpus. */
	private function corpus(): array
	{
		return [
			['spam', 'cheap pills buy now cheap watches'],
			['spam', 'lowest prices order now limited offer'],
			['spam', 'free money fast click here'],
			['ham', 'project meeting agenda tomorrow morning'],
			['ham', 'please review the quarterly report'],
			['ham', 'deployment finished staging looks healthy'],
		];
	}

	private function train(TNaiveBayesClassifier $classifier): TNaiveBayesClassifier
	{
		foreach ($this->corpus() as [$category, $document]) {
			$classifier->trainOne($category, $document);
		}
		return $classifier;
	}

	/** @return string[] The classifier classes that must behave identically in both modes. */
	public static function classifierClasses(): array
	{
		return [TNaiveBayesClassifier::class, TBernoulliNaiveBayes::class, TComplementNaiveBayes::class];
	}

	public function testPerTokenScoresAreIdenticalToPayloadScores()
	{
		// The whole point.  Same corpus, same settings, two layouts — the scores must agree
		// exactly, not approximately: nothing in the per-token path is a different calculation.
		foreach (self::classifierClasses() as $class) {
			foreach ([false, true] as $useTfidf) {
				$payloadStore = $this->storage(TSqlBayesianStorage::MODE_PAYLOAD);
				$payload = new $class();
				$payload->setUseTfidf($useTfidf);
				$payload->setAlpha(0.7);
				$payload->setStorage($payloadStore);
				$payload->setName('m');
				$this->train($payload)->save();
				$payloadLoaded = new $class();
				$payloadLoaded->setStorage($payloadStore);
				$payloadLoaded->load('m');

				$tokenStore = $this->storage();
				$token = new $class();
				$token->setUseTfidf($useTfidf);
				$token->setAlpha(0.7);
				$token->setStorage($tokenStore);
				$token->setName('m');
				$this->train($token)->save();
				$tokenLoaded = new $class();
				$tokenLoaded->setStorage($tokenStore);
				$tokenLoaded->load('m');

				$label = $class . ' tfidf=' . var_export($useTfidf, true);
				self::assertFalse(
					$tokenLoaded->getVocabulary()->getSupportsFullScan(),
					$label . ': the loaded vocabulary should be storage-backed'
				);
				foreach (['cheap prices now', 'review the report', 'entirely unseen wording', 'free'] as $probe) {
					self::assertSame($payloadLoaded->score($probe), $tokenLoaded->score($probe), $label . ' / ' . $probe);
					self::assertSame($payloadLoaded->classify($probe), $tokenLoaded->classify($probe), $label . ' / ' . $probe);
				}
			}
		}
	}

	public function testLoadedModelIsStorageBackedAndReportsItsScalars()
	{
		$storage = $this->storage();
		$source = new TNaiveBayesClassifier();
		$source->setStorage($storage);
		$source->setName('m');
		$this->train($source)->save();

		$loaded = new TNaiveBayesClassifier();
		$loaded->setStorage($storage);
		$loaded->load('m');
		$vocabulary = $loaded->getVocabulary();

		self::assertInstanceOf(TLazyBayesianVocabulary::class, $vocabulary);
		self::assertTrue($loaded->getIsTrained());
		self::assertSame(6, $vocabulary->getTotalDocuments());
		self::assertSame($source->getVocabulary()->getVocabularySize(), $vocabulary->getVocabularySize());
		self::assertSame(['spam', 'ham'], $vocabulary->getCategoryNames());
	}

	public function testEnumeratingAStorageBackedVocabularyThrows()
	{
		// Returning the prefetched slice would answer "the whole vocabulary" with a fraction of
		// it, and no caller could tell.  It has to refuse instead.
		$storage = $this->storage();
		$source = new TNaiveBayesClassifier();
		$source->setStorage($storage);
		$source->setName('m');
		$this->train($source)->save();

		$loaded = new TNaiveBayesClassifier();
		$loaded->setStorage($storage);
		$loaded->load('m');

		$this->expectException(\Prado\Exceptions\TInvalidOperationException::class);
		$loaded->getVocabulary()->getDocumentFrequency();
	}

	public function testCategoryTokenMapsOfAStorageBackedModelThrow()
	{
		$storage = $this->storage();
		$source = new TNaiveBayesClassifier();
		$source->setStorage($storage);
		$source->setName('m');
		$this->train($source)->save();

		$loaded = new TNaiveBayesClassifier();
		$loaded->setStorage($storage);
		$loaded->load('m');

		$this->expectException(\Prado\Exceptions\TInvalidOperationException::class);
		$loaded->getVocabulary()->getCategory('spam')->getTokenCounts();
	}

	public function testPayloadStorageRefusesAPerTokenModel()
	{
		// Both layouts keep a metadata row in the same table.  Read through the payload path a
		// per-token model would import as an untrained classifier and quietly classify
		// everything the same way, which is the failure mode worth an exception.
		$file = sys_get_temp_dir() . '/bayesian-token-' . uniqid('', true) . '.sqlite';
		$this->_files[] = $file;
		$tokenStore = new TSqlBayesianStorage();
		$tokenStore->setConnectionString('sqlite:' . $file);
		$tokenStore->setMode(TSqlBayesianStorage::MODE_TOKEN);
		$source = new TNaiveBayesClassifier();
		$source->setStorage($tokenStore);
		$source->setName('m');
		$this->train($source)->save();

		$payloadStore = new TSqlBayesianStorage();
		$payloadStore->setConnectionString('sqlite:' . $file);
		$reader = new TNaiveBayesClassifier();
		$reader->setStorage($payloadStore);

		try {
			$reader->load('m');
			self::fail('expected a per-token model to be refused by the payload path');
		} catch (\Prado\Exceptions\TInvalidDataValueException $e) {
			self::assertSame('bayesian_classifier_token_mode_payload', $e->getErrorCode());
		}
	}

	public function testIncrementalTrainingMatchesTrainingTheWholeCorpusResident()
	{
		// Training against storage writes only the document's rows.  The resulting model must
		// still be the one a fully resident run would have produced.
		$storage = $this->storage();
		$seed = new TNaiveBayesClassifier();
		$seed->setStorage($storage);
		$seed->setName('m');
		$seed->trainOne('spam', 'cheap pills');
		$seed->trainOne('ham', 'team meeting');
		$seed->save();

		$incremental = new TNaiveBayesClassifier();
		$incremental->setStorage($storage);
		$incremental->load('m');
		$incremental->trainOne('spam', 'discount watches cheap');

		$resident = new TNaiveBayesClassifier();
		$resident->trainOne('spam', 'cheap pills');
		$resident->trainOne('ham', 'team meeting');
		$resident->trainOne('spam', 'discount watches cheap');

		$reloaded = new TNaiveBayesClassifier();
		$reloaded->setStorage($storage);
		$reloaded->load('m');

		self::assertSame(3, $reloaded->getVocabulary()->getTotalDocuments());
		self::assertSame(
			$resident->getVocabulary()->getVocabularySize(),
			$reloaded->getVocabulary()->getVocabularySize()
		);
		foreach (['cheap watches', 'team meeting', 'pills'] as $probe) {
			self::assertSame($resident->score($probe), $reloaded->score($probe), $probe);
		}
	}

	public function testIncrementalTrainingIsVisibleToAnotherReader()
	{
		$storage = $this->storage();
		$seed = new TNaiveBayesClassifier();
		$seed->setStorage($storage);
		$seed->setName('m');
		$seed->trainOne('spam', 'cheap pills');
		$seed->trainOne('ham', 'team meeting');
		$seed->save();

		$writer = new TNaiveBayesClassifier();
		$writer->setStorage($storage);
		$writer->load('m');
		$writer->trainOne('spam', 'lottery winner claim prize');

		$reader = new TNaiveBayesClassifier();
		$reader->setStorage($storage);
		$reader->load('m');
		self::assertSame('spam', $reader->classify('lottery prize'));
		self::assertSame(3, $reader->getVocabulary()->getTotalDocuments());
	}

	public function testDeleteRemovesTheTokenRowsNotJustTheMetadata()
	{
		// A model whose token rows outlived its metadata row would come back from the dead on
		// the next save under the same name, carrying the deleted model's counts.
		$storage = $this->storage();
		$source = new TNaiveBayesClassifier();
		$source->setStorage($storage);
		$source->setName('m');
		$this->train($source)->save();
		self::assertTrue($storage->exists('m'));

		$storage->delete('m');
		self::assertFalse($storage->exists('m'));
		self::assertSame([], $storage->loadTokenCategories('m'));
		self::assertSame([], $storage->loadTokens('m', ['cheap', 'meeting']));
	}

	public function testSavingReplacesRatherThanAccumulates()
	{
		$storage = $this->storage();
		$first = new TNaiveBayesClassifier();
		$first->setStorage($storage);
		$first->setName('m');
		$this->train($first)->save();

		$second = new TNaiveBayesClassifier();
		$second->setStorage($storage);
		$second->setName('m');
		$second->trainOne('spam', 'only this document now');
		$second->trainOne('ham', 'and this one');
		$second->save();

		$loaded = new TNaiveBayesClassifier();
		$loaded->setStorage($storage);
		$loaded->load('m');
		self::assertSame(2, $loaded->getVocabulary()->getTotalDocuments());
		self::assertSame([], $storage->loadTokens('m', ['quarterly']), 'rows of the replaced model should be gone');
	}

	public function testLoadTokensAnswersOnlyForTheTokensAsked()
	{
		$storage = $this->storage();
		$source = new TNaiveBayesClassifier();
		$source->setStorage($storage);
		$source->setName('m');
		$this->train($source)->save();

		$rows = $storage->loadTokens('m', ['cheap', 'not-a-real-token']);
		self::assertArrayHasKey('cheap', $rows);
		self::assertArrayNotHasKey('not-a-real-token', $rows);
		self::assertSame(2, $rows['cheap']['spam']['count'], '"cheap" occurs twice in one spam document');
		self::assertSame(1, $rows['cheap']['spam']['docCount'], 'in a single document');
	}

	public function testLoadTokensChunksALargeTokenList()
	{
		// The IN() list is chunked so a document tokenized into many n-grams cannot exceed a
		// driver's bind-parameter limit.
		$storage = $this->storage();
		$source = new TNaiveBayesClassifier();
		$source->setStorage($storage);
		$source->setName('m');
		$document = [];
		for ($i = 0; $i < 1200; $i++) {
			$document[] = 'tok' . $i;
		}
		$source->trainOne('a', $document);
		$source->trainOne('b', ['other']);
		$source->save();

		$rows = $storage->loadTokens('m', $document);
		self::assertCount(1200, $rows);
	}

	public function testTokenOperationsRequireTokenMode()
	{
		$storage = $this->storage(TSqlBayesianStorage::MODE_PAYLOAD);
		self::assertFalse($storage->getSupportsTokenLookup());
		$this->expectException(\Prado\Exceptions\TInvalidOperationException::class);
		$storage->loadTokens('m', ['a']);
	}

	public function testModeRejectsAnUnknownValue()
	{
		$storage = $this->storage();
		$this->expectException(\Prado\Exceptions\TInvalidDataValueException::class);
		$storage->setMode('sideways');
	}

	public function testBernoulliAndComplementRestoreTheirFullScanAggregates()
	{
		// Both keep a per-category quantity summed over the whole vocabulary.  A storage-backed
		// vocabulary cannot rebuild it, so it has to come back with the model — and if it did
		// not, these scores would be silently shifted rather than raise.
		foreach ([TBernoulliNaiveBayes::class, TComplementNaiveBayes::class] as $class) {
			$storage = $this->storage();
			$source = new $class();
			$source->setStorage($storage);
			$source->setName('m');
			$this->train($source)->save();

			$loaded = new $class();
			$loaded->setStorage($storage);
			$loaded->load('m');
			self::assertSame($source->score('cheap prices now'), $loaded->score('cheap prices now'), $class);
		}
	}

	public function testAStorageBackedModelCannotBeSavedBack()
	{
		// Writing the model out means reading all of it, which is the one thing this vocabulary
		// cannot do.
		$storage = $this->storage();
		$source = new TNaiveBayesClassifier();
		$source->setStorage($storage);
		$source->setName('m');
		$this->train($source)->save();

		$loaded = new TNaiveBayesClassifier();
		$loaded->setStorage($storage);
		$loaded->load('m');

		$this->expectException(\Prado\Exceptions\TInvalidOperationException::class);
		$loaded->save();
	}
}
