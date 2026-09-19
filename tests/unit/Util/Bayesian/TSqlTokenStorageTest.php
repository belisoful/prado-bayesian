<?php

use Belisoful\Prado\Util\Bayesian\Classifier\TBernoulliNaiveBayes;
use Belisoful\Prado\Util\Bayesian\Classifier\TComplementNaiveBayes;
use Belisoful\Prado\Util\Bayesian\Classifier\TNaiveBayesClassifier;
use Belisoful\Prado\Util\Bayesian\Storage\TSqlBayesianStorage;
use Belisoful\Prado\Util\Bayesian\TBayesianTokenHistogram;
use Belisoful\Prado\Util\Bayesian\TLazyBayesianVocabulary;

require_once(__DIR__ . '/../../../test_tools/BayesianBackends.php');
require_once(__DIR__ . '/../../../test_tools/BayesianVariantIncrementalTests.php');

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
	use BayesianVariantIncrementalTests;

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
	public function testTwoLoadedInstancesTrainingTheSameModelLoseNoCounts()
	{
		// Two web requests (or a request and a worker) each load the model, then each train one
		// document.  Whichever writes second must add to what the first wrote, not overwrite it
		// with counts computed from its own stale snapshot.
		$storage = $this->storage();
		$seed = new TNaiveBayesClassifier();
		$seed->setStorage($storage);
		$seed->setName('m');
		$seed->trainOne('spam', 'cheap pills');
		$seed->trainOne('ham', 'team meeting');
		$seed->save();

		$first = new TNaiveBayesClassifier();
		$first->setStorage($this->storageFor($storage));
		$first->load('m');
		$second = new TNaiveBayesClassifier();
		$second->setStorage($this->storageFor($storage));
		$second->load('m');

		$first->trainOne('spam', 'cheap watches');
		$second->trainOne('spam', 'cheap lottery prize');
		$second->trainOne('news', 'brand new category');

		$reader = new TNaiveBayesClassifier();
		$reader->setStorage($this->storageFor($storage));
		$reader->load('m');
		$vocabulary = $reader->getVocabulary();
		self::assertSame(5, $vocabulary->getTotalDocuments(), 'every document counts');
		self::assertSame(10, $vocabulary->getVocabularySize(), 'cheap pills team meeting watches lottery prize brand new category');
		self::assertSame(3, $vocabulary->getCategory('spam')->getDocumentCount());
		self::assertSame(7, $vocabulary->getCategory('spam')->getTotalTokens());
		self::assertSame(1, $vocabulary->getCategory('news')->getDocumentCount());
		self::assertSame(['spam', 'ham', 'news'], $vocabulary->getCategoryNames(), 'a category another writer added is present');
		$rows = $storage->loadTokens('m', ['cheap', 'watches', 'prize']);
		self::assertSame(3, $rows['cheap']['spam']['count']);
		self::assertSame(3, $rows['cheap']['spam']['docCount']);
		self::assertSame(1, $rows['watches']['spam']['count']);
		self::assertSame(1, $rows['prize']['spam']['count']);

		// The writer that trained last also sees the true totals, not its own snapshot plus one.
		self::assertSame(5, $second->getVocabulary()->getTotalDocuments());
		self::assertSame(10, $second->getVocabulary()->getVocabularySize());
		// A model trained through two instances scores exactly as one trained resident.
		$resident = new TNaiveBayesClassifier();
		foreach ([['spam', 'cheap pills'], ['ham', 'team meeting'], ['spam', 'cheap watches'], ['spam', 'cheap lottery prize'], ['news', 'brand new category']] as [$category, $document]) {
			$resident->trainOne($category, $document);
		}
		foreach (['cheap prize', 'team', 'brand new'] as $probe) {
			self::assertSame($resident->score($probe), $reader->score($probe), $probe);
		}
	}

	/** Opens a second, independent storage on the same database file. */
	private function storageFor(TSqlBayesianStorage $storage): TSqlBayesianStorage
	{
		$other = new TSqlBayesianStorage();
		$other->setConnectionString($storage->getConnectionString());
		$other->setMode($storage->getMode());
		return $other;
	}

	public function testApplyDeltasAcceptsNegativeDeltasClampedAtZero()
	{
		// An "untrain" needs to subtract.  A count can never go below zero, and removing a
		// document never shrinks the vocabulary: a token that was seen stays seen.
		$storage = $this->storage();
		$source = new TNaiveBayesClassifier();
		$source->setStorage($storage);
		$source->setName('m');
		$source->trainOne('spam', 'cheap cheap pills');
		$source->trainOne('ham', 'team meeting');
		$source->save();

		$storage->applyDeltas('m', 'spam', ['cheap' => ['count' => -1, 'docCount' => 0]], [], ['documentCount' => 0, 'totalTokens' => -1]);
		$rows = $storage->loadTokens('m', ['cheap']);
		self::assertSame(1, $rows['cheap']['spam']['count']);
		self::assertSame(1, $rows['cheap']['spam']['docCount']);
		self::assertSame(2, $storage->loadTokenCategories('m')['spam']['totalTokens']);

		$storage->applyDeltas('m', 'spam', ['cheap' => ['count' => -10, 'docCount' => -10], 'pills' => ['count' => -10, 'docCount' => -10]], [], ['documentCount' => -10, 'totalTokens' => -10]);
		$rows = $storage->loadTokens('m', ['cheap', 'pills']);
		self::assertSame(0, $rows['cheap']['spam']['count'], 'clamped at zero');
		self::assertSame(0, $rows['cheap']['spam']['docCount']);
		self::assertSame(0, $rows['pills']['spam']['count']);
		$categories = $storage->loadTokenCategories('m');
		self::assertSame(0, $categories['spam']['documentCount']);
		self::assertSame(0, $categories['spam']['totalTokens']);
		$meta = $storage->loadTokenMeta('m');
		self::assertSame(1, $meta['totalDocuments'], 'only the ham document remains: the total is the sum of the category counts');
		self::assertSame(2, $meta['vocabularySize'], 'cheap and pills, which no document contains any more, left the vocabulary');

		// A negative delta for a token the model has never seen leaves it out of the
		// vocabulary: nothing went below zero, and no document contains it.
		$storage->applyDeltas('m', 'spam', ['never' => ['count' => -3, 'docCount' => -1]], [], ['documentCount' => 0, 'totalTokens' => 0]);
		$rows = $storage->loadTokens('m', ['never']);
		self::assertSame(0, $rows['never']['spam']['count'] ?? 0);
		self::assertSame(0, $rows['never']['spam']['docCount'] ?? 0);
		self::assertSame(2, $storage->loadTokenMeta('m')['vocabularySize']);
	}

	public function testCountersAreDerivedFromTheTablesNotFromTheMetaArgument()
	{
		// Whatever a writer believes the totals are is irrelevant; the stored counters come from
		// what was actually written.
		$storage = $this->storage();
		$source = new TNaiveBayesClassifier();
		$source->setStorage($storage);
		$source->setName('m');
		$source->trainOne('spam', 'cheap pills');
		$source->save();
		$storage->applyDeltas('m', 'spam', ['new' => ['count' => 1, 'docCount' => 1]], ['totalDocuments' => 999, 'vocabularySize' => 999, 'kind' => 'naive-bayes'], ['documentCount' => 1, 'totalTokens' => 1]);
		$meta = $storage->loadTokenMeta('m');
		self::assertSame(2, $meta['totalDocuments']);
		self::assertSame(3, $meta['vocabularySize']);
		self::assertSame('naive-bayes', $meta['kind'], 'the rest of the metadata is stored as given');
		self::assertSame(TSqlBayesianStorage::TOKEN_LAYOUT_VERSION, $meta['layoutVersion']);
	}

	public function testALegacyTokenLayoutIsUpgradedOnFirstRead()
	{
		// 0.1.0 wrote no vocabulary or counters tables and kept the totals inside the metadata
		// JSON.  Such a model must keep working, with its counters rebuilt from its rows.
		$storage = $this->storage();
		$storage->save('legacy', ['kind' => 'naive-bayes', 'tokenMode' => true, 'totalDocuments' => 3, 'vocabularySize' => 4, 'categoryOrder' => ['spam', 'ham'], 'alpha' => 1.0]);
		$connection = $storage->getDbConnection();
		$table = $storage->getTable();
		$connection->createCommand("INSERT INTO {$table}_categories (model, category, doc_count, total_tokens) VALUES ('legacy', 'spam', 2, 4), ('legacy', 'ham', 1, 2)")->execute();
		$connection->createCommand("INSERT INTO {$table}_tokens (model, token, category, cnt, doccnt) VALUES ('legacy', 'cheap', 'spam', 3, 2), ('legacy', 'pills', 'spam', 1, 1), ('legacy', 'team', 'ham', 1, 1), ('legacy', 'meeting', 'ham', 1, 1)")->execute();
		foreach (['_vocab', '_counters'] as $suffix) {
			$connection->createCommand("DELETE FROM {$table}{$suffix} WHERE model = 'legacy'")->execute();
		}

		$meta = $storage->loadTokenMeta('legacy');
		self::assertSame(3, $meta['totalDocuments']);
		self::assertSame(4, $meta['vocabularySize']);
		self::assertSame(TSqlBayesianStorage::TOKEN_LAYOUT_VERSION, $meta['layoutVersion'], 'the model is stamped with the current layout');
		self::assertSame(4, (int) $connection->createCommand("SELECT COUNT(*) FROM {$table}_vocab WHERE model = 'legacy'")->queryScalar());

		$loaded = new TNaiveBayesClassifier();
		$loaded->setStorage($storage);
		$loaded->load('legacy');
		self::assertSame('spam', $loaded->classify('cheap'));
		// ... and incremental training on the upgraded model keeps the counters exact.
		$loaded->trainOne('spam', 'cheap offer');
		$meta = $storage->loadTokenMeta('legacy');
		self::assertSame(4, $meta['totalDocuments']);
		self::assertSame(5, $meta['vocabularySize']);
	}

	public function testANewerTokenLayoutIsRefused()
	{
		$storage = $this->storage();
		$storage->save('future', ['kind' => 'naive-bayes', 'tokenMode' => true, 'layoutVersion' => TSqlBayesianStorage::TOKEN_LAYOUT_VERSION + 1]);
		try {
			$storage->loadTokenMeta('future');
			self::fail('expected a newer layout to be refused');
		} catch (\Prado\Exceptions\TInvalidDataValueException $e) {
			self::assertSame('bayesian_storage_layout_unsupported', $e->getErrorCode());
		}
	}

	public function testDeleteRemovesTheVocabularyAndCounterRows()
	{
		$storage = $this->storage();
		$source = new TNaiveBayesClassifier();
		$source->setStorage($storage);
		$source->setName('m');
		$this->train($source)->save();
		$storage->delete('m');
		$connection = $storage->getDbConnection();
		$table = $storage->getTable();
		self::assertSame(0, (int) $connection->createCommand("SELECT COUNT(*) FROM {$table}_vocab WHERE model = 'm'")->queryScalar());
		self::assertSame(0, (int) $connection->createCommand("SELECT COUNT(*) FROM {$table}_counters WHERE model = 'm'")->queryScalar());
	}

	public function testTokensLongerThanTheColumnRoundTrip()
	{
		// MySQL and PostgreSQL store tokens in VARCHAR(191); a URL from a regex tokenizer or a
		// run of characters is easily longer.  The storage must take it, and give it back under
		// the original token, with two distinct long tokens kept apart.
		$storage = $this->storage();
		$long = str_repeat('a', 300);
		$other = str_repeat('a', 299) . 'b';
		$multibyte = str_repeat('é', 250);
		$source = new TNaiveBayesClassifier();
		$source->setStorage($storage);
		$source->setName('m');
		$source->trainOne('x', [$long, $long, $multibyte]);
		$source->trainOne('y', [$other]);
		$source->save();
		$rows = $storage->loadTokens('m', [$long, $other, $multibyte, 'absent']);
		self::assertSame(2, $rows[$long]['x']['count']);
		self::assertSame(1, $rows[$other]['y']['count']);
		self::assertSame(1, $rows[$multibyte]['x']['count']);
		self::assertArrayNotHasKey($other, $rows[$long] ?? [], 'distinct long tokens do not share a row');
		self::assertArrayNotHasKey('absent', $rows);
		$loaded = new TNaiveBayesClassifier();
		$loaded->setStorage($storage);
		$loaded->load('m');
		self::assertSame(3, $loaded->getVocabulary()->getVocabularySize());
		self::assertSame('x', $loaded->classify([$long]));
		$loaded->trainOne('y', [$long]);
		$rows = $storage->loadTokens('m', [$long]);
		self::assertSame(1, $rows[$long]['y']['count'], 'incremental training reaches the same row');
		self::assertSame(3, $storage->loadTokenMeta('m')['vocabularySize'], 'an encoded token is still one vocabulary entry');
	}

	public function testTableNameLengthIsBounded()
	{
		$storage = $this->storage();
		$storage->setTable(str_repeat('t', 48));
		self::assertSame(str_repeat('t', 48), $storage->getTable());
		$this->expectException(\Prado\Exceptions\TInvalidDataValueException::class);
		$storage->setTable(str_repeat('t', 49));
	}
	public function testThePrefetchedBatchIsBounded()
	{
		// A worker that scores documents for hours must not hold every token it ever read.
		$storage = $this->storage();
		$source = new TNaiveBayesClassifier();
		$source->setStorage($storage);
		$source->setName('m');
		$this->train($source)->save();
		$loaded = new TNaiveBayesClassifier();
		$loaded->setStorage($storage);
		$loaded->load('m');
		/** @var TLazyBayesianVocabulary $vocabulary */
		$vocabulary = $loaded->getVocabulary();
		self::assertSame(TLazyBayesianVocabulary::DEFAULT_MAX_BATCH_TOKENS, $vocabulary->getMaxBatchTokens());
		$vocabulary->setMaxBatchTokens(4);
		self::assertSame(4, $vocabulary->getMaxBatchTokens());

		$vocabulary->prefetch(['cheap', 'pills']);
		self::assertTrue($vocabulary->hasToken('cheap'));
		$vocabulary->prefetch(['meeting']);
		self::assertTrue($vocabulary->hasToken('cheap'), 'within the cap the batch accumulates');
		self::assertTrue($vocabulary->hasToken('meeting'));
		// This fetch would make five: the batch restarts from the new document only.
		$vocabulary->prefetch(['report', 'review']);
		self::assertTrue($vocabulary->hasToken('report'));
		self::assertTrue($vocabulary->hasToken('review'));
		self::assertFalse($vocabulary->hasToken('cheap'), 'the earlier tokens were dropped');
		// A document longer than the cap is still fetched whole.
		$vocabulary->prefetch(['cheap', 'pills', 'buy', 'now', 'watches', 'prices']);
		foreach (['cheap', 'pills', 'buy', 'now', 'watches', 'prices'] as $token) {
			self::assertTrue($vocabulary->hasToken($token), $token);
		}
		self::assertSame(0, $vocabulary->getMaxBatchTokens() < 1 ? 1 : 0);
		$vocabulary->setMaxBatchTokens(0);
		self::assertSame(1, $vocabulary->getMaxBatchTokens(), 'the cap is at least one token');
		// Scores are unaffected by any of this.
		self::assertSame($source->score('cheap prices now'), $loaded->score('cheap prices now'));
	}
	public function testUntrainingAgainstStorageMatchesTheResidentModel()
	{
		$storage = $this->storage();
		$seed = new TNaiveBayesClassifier();
		$seed->setStorage($storage);
		$seed->setName('m');
		$seed->trainOne('spam', 'cheap pills');
		$seed->trainOne('ham', 'team meeting');
		$seed->trainOne('spam', 'cheap watches lottery');
		$seed->trainOne('news', 'brand new category');
		$seed->save();

		$writer = new TNaiveBayesClassifier();
		$writer->setStorage($this->storageFor($storage));
		$writer->load('m');
		$writer->untrainOne('spam', 'cheap watches lottery');
		$writer->untrainOne('news', 'brand new category');

		$resident = new TNaiveBayesClassifier();
		$resident->trainOne('spam', 'cheap pills');
		$resident->trainOne('ham', 'team meeting');

		$reader = new TNaiveBayesClassifier();
		$reader->setStorage($this->storageFor($storage));
		$reader->load('m');
		$vocabulary = $reader->getVocabulary();
		self::assertSame(2, $vocabulary->getTotalDocuments());
		self::assertSame($resident->getVocabulary()->getVocabularySize(), $vocabulary->getVocabularySize(), 'tokens no document contains have left the vocabulary');
		self::assertSame(['spam', 'ham'], $vocabulary->getCategoryNames(), 'a category with no documents is not present');
		$vocabulary->prefetch(['watches', 'cheap']);
		self::assertFalse($vocabulary->hasToken('watches'));
		self::assertTrue($vocabulary->hasToken('cheap'));
		foreach (['cheap pills', 'team', 'watches lottery', 'brand new'] as $probe) {
			self::assertSame($resident->score($probe), $reader->score($probe), $probe);
		}
		self::assertSame(2, $writer->getVocabulary()->getTotalDocuments(), 'the writer re-read the totals');

		// Training the token again brings it back into the vocabulary once.
		$writer->trainOne('spam', 'watches');
		$writer->trainOne('spam', 'watches');
		self::assertSame($resident->getVocabulary()->getVocabularySize(), $reader->getVocabulary()->getVocabularySize(), 'the reader snapshot is unchanged until refreshed');
		$reader->getVocabulary()->refresh();
		self::assertSame($resident->getVocabulary()->getVocabularySize() + 1, $reader->getVocabulary()->getVocabularySize());
	}

	/** Removes a model's histograms and their advertisement, as a store written before they existed. */
	private function stripHistograms(TSqlBayesianStorage $storage, string $name): void
	{
		$connection = $storage->getDbConnection();
		$command = $connection->createCommand('DELETE FROM ' . $storage->getTable() . '_hist WHERE model = :model');
		$command->bindValue(':model', $name);
		$command->execute();
		$command = $connection->createCommand('SELECT payload FROM ' . $storage->getTable() . ' WHERE name = :name');
		$command->bindValue(':name', $name);
		$meta = json_decode((string) $command->queryScalar(), true);
		unset($meta['histograms']);
		$command = $connection->createCommand('UPDATE ' . $storage->getTable() . ' SET payload = :payload WHERE name = :name');
		$command->bindValue(':payload', json_encode($meta));
		$command->bindValue(':name', $name);
		$command->execute();
	}

	/** @return array<int, array{0:string}> The modes that take histogram work off the training write. */
	public static function relaxedModes(): array
	{
		return [[TSqlBayesianStorage::HISTOGRAM_DEFERRED], [TSqlBayesianStorage::HISTOGRAM_PERIODIC]];
	}

	/**
	 * Saves the corpus as a Complement model in the given histogram mode and returns a trainer
	 * loaded from a second, default-configured storage.
	 * @return array{0:TSqlBayesianStorage, 1:TComplementNaiveBayes}
	 */
	private function relaxedModel(string $mode): array
	{
		$storage = $this->storage();
		$storage->setHistogramMode($mode);
		$source = new TComplementNaiveBayes();
		$source->setStorage($storage);
		$source->setName('m');
		$this->train($source)->save();
		$trainer = new TComplementNaiveBayes();
		$trainer->setStorage($this->storageFor($storage));
		$trainer->load('m');
		return [$storage, $trainer];
	}

	/**
	 * @dataProvider relaxedModes
	 */
	public function testARelaxedModeLeavesTheHistogramsToMaintenance(string $mode)
	{
		[$storage, $trainer] = $this->relaxedModel($mode);
		$families = [TBayesianTokenHistogram::FAMILY_GLOBAL, TBayesianTokenHistogram::FAMILY_CATEGORY_GLOBAL, TBayesianTokenHistogram::FAMILY_COMPLEMENT];
		$saved = $storage->loadTokenHistograms('m', $families);
		self::assertNotSame([], $saved);
		self::assertSame($mode, $storage->loadTokenMeta('m')['histogramMode']);

		$reference = $this->train(new TComplementNaiveBayes());
		foreach ($this->extraDocuments() as [$category, $document]) {
			$reference->trainOne($category, $document);
			$trainer->trainOne($category, $document);
		}
		// The trainer came from a storage configured for the default mode: the model's own
		// mode decides, and its training writes leave the histograms alone.
		self::assertSame($mode, $storage->loadTokenMeta('m')['histogramMode'], 'training keeps the mode');
		self::assertEquals($saved, $storage->loadTokenHistograms('m', $families), 'untouched by training');
		$pending = $storage->getPendingTokenCount('m');
		if ($mode === TSqlBayesianStorage::HISTOGRAM_DEFERRED) {
			self::assertGreaterThan(0, $pending, 'the trained tokens are journalled');
		} else {
			self::assertSame(0, $pending, 'periodic mode keeps no journal');
		}

		self::assertGreaterThan(0, $storage->maintainTokenHistograms('m'));
		self::assertSame(0, $storage->getPendingTokenCount('m'));
		$reader = new TComplementNaiveBayes();
		$reader->setStorage($this->storageFor($storage));
		$reader->load('m');
		$this->assertSameScores($reference, $reader, $mode . ' after maintenance');
		$maintained = $storage->loadTokenHistograms('m', $families);
		$storage->rebuildTokenHistograms('m', $families);
		self::assertEquals($maintained, $storage->loadTokenHistograms('m', $families));
	}

	public function testDeferredFoldingWorksInBatchesAndFollowsUntraining()
	{
		[$storage, $trainer] = $this->relaxedModel(TSqlBayesianStorage::HISTOGRAM_DEFERRED);
		foreach ($this->extraDocuments() as [$category, $document]) {
			$trainer->trainOne($category, $document);
		}
		$pending = $storage->getPendingTokenCount('m');
		self::assertSame(2, $storage->foldTokenHistograms('m', 2), 'a fold takes at most its limit');
		self::assertSame($pending - 2, $storage->getPendingTokenCount('m'));
		foreach (array_reverse($this->extraDocuments()) as [$category, $document]) {
			$trainer->untrainOne($category, $document);
		}
		while ($storage->foldTokenHistograms('m', 3) > 0) {
			// Drain the journal a few tokens at a time.
		}
		self::assertSame(0, $storage->getPendingTokenCount('m'));
		self::assertSame(0, $storage->foldTokenHistograms('m'), 'nothing left to fold');

		$reference = $this->train(new TComplementNaiveBayes());
		$reader = new TComplementNaiveBayes();
		$reader->setStorage($this->storageFor($storage));
		$reader->load('m');
		$this->assertSameScores($reference, $reader, 'trained, half folded, untrained, folded');
	}

	public function testAModelChangesHistogramModeAndStaysExact()
	{
		[$storage, $trainer] = $this->relaxedModel(TSqlBayesianStorage::HISTOGRAM_PERIODIC);
		$reference = $this->train(new TComplementNaiveBayes());
		$extras = $this->extraDocuments();
		$reference->trainOne(...$extras[0]);
		$trainer->trainOne(...$extras[0]);

		// Periodic to deferred: the stale histograms are rebuilt on the way.
		$storage->setTokenHistogramMode('m', TSqlBayesianStorage::HISTOGRAM_DEFERRED);
		self::assertSame(TSqlBayesianStorage::HISTOGRAM_DEFERRED, $storage->loadTokenMeta('m')['histogramMode']);
		$reference->trainOne(...$extras[1]);
		$trainer->trainOne(...$extras[1]);
		self::assertGreaterThan(0, $storage->getPendingTokenCount('m'));

		// Deferred to immediate: the journal is folded on the way, and writes are exact again.
		$storage->setTokenHistogramMode('m', TSqlBayesianStorage::HISTOGRAM_IMMEDIATE);
		self::assertSame(0, $storage->getPendingTokenCount('m'));
		$reference->trainOne(...$extras[2]);
		$trainer->trainOne(...$extras[2]);
		$this->assertSameScores($reference, $trainer, 'immediate again');
		self::assertSame(0, $storage->maintainTokenHistograms('m'), 'an immediate model needs no maintenance');
	}

	public function testBernoulliIsExactInEveryHistogramMode()
	{
		// Its histogram moves from the rows a write already locks, so there is nothing to defer.
		foreach (self::relaxedModes() as [$mode]) {
			$storage = $this->storage();
			$storage->setHistogramMode($mode);
			$source = new TBernoulliNaiveBayes();
			$source->setStorage($storage);
			$source->setName('m');
			$this->train($source)->save();
			$trainer = new TBernoulliNaiveBayes();
			$trainer->setStorage($this->storageFor($storage));
			$trainer->load('m');
			$reference = $this->train(new TBernoulliNaiveBayes());
			foreach ($this->extraDocuments() as [$category, $document]) {
				$reference->trainOne($category, $document);
				$trainer->trainOne($category, $document);
			}
			$this->assertSameScores($reference, $trainer, 'bernoulli ' . $mode);
			self::assertSame(0, $storage->getPendingTokenCount('m'));
		}
	}

	public function testHistogramModeRejectsAnUnknownValue()
	{
		$storage = $this->storage();
		self::assertSame(TSqlBayesianStorage::HISTOGRAM_IMMEDIATE, $storage->getHistogramMode());
		$storage->setHistogramMode('deferred');
		self::assertSame(TSqlBayesianStorage::HISTOGRAM_DEFERRED, $storage->getHistogramMode());
		$this->expectException(\Prado\Exceptions\TInvalidDataValueException::class);
		$storage->setHistogramMode('eventually');
	}

	public function testMaintenanceOfAnUnknownModelDoesNothing()
	{
		$storage = $this->storage();
		self::assertSame(0, $storage->maintainTokenHistograms('nobody'));
		self::assertSame(0, $storage->foldTokenHistograms('nobody'));
		self::assertSame(0, $storage->getPendingTokenCount('nobody'));
	}
}
