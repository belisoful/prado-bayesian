<?php

use Belisoful\Prado\Util\Bayesian\Classifier\TBernoulliNaiveBayes;
use Belisoful\Prado\Util\Bayesian\Classifier\TComplementNaiveBayes;
use Belisoful\Prado\Util\Bayesian\Classifier\TNaiveBayesClassifier;
use Belisoful\Prado\Util\Bayesian\Storage\TRedisBayesianStorage;
use Belisoful\Prado\Util\Bayesian\TLazyBayesianVocabulary;

require_once(__DIR__ . '/../../../test_tools/BayesianBackends.php');

/**
 * Covers Redis per-token storage.  These need both `ext-redis` and a reachable server, so they
 * skip on a machine without either — the encode/decode helpers this layout depends on are
 * additionally proven in isolation by their own reflection test, so the parsing is not left
 * resting on a live Redis being present.
 *
 * As with the SQL suite, the load-bearing assertion is equivalence: a per-token model must score
 * exactly as the same corpus stored whole.
 */
class TRedisTokenStorageTest extends PHPUnit\Framework\TestCase
{
	/** @var TRedisBayesianStorage[] Storages whose keys must be cleared after the test. */
	private array $_storages = [];

	protected function setUp(): void
	{
		BayesianBackends::requireBackend($this, extension_loaded('redis'), 'redis extension not available');
	}

	protected function tearDown(): void
	{
		foreach ($this->_storages as $storage) {
			try {
				foreach ($storage->list() as $name) {
					$storage->delete($name);
				}
				$storage->getRedis()->del($storage->getIndexKey());
			} catch (\Throwable $e) {
				// Best-effort cleanup; a dead connection here should not mask the test result.
			}
		}
		$this->_storages = [];
	}

	private function storage(string $mode = TRedisBayesianStorage::MODE_TOKEN): TRedisBayesianStorage
	{
		$storage = new TRedisBayesianStorage();
		// A unique prefix per storage isolates concurrent test runs sharing one Redis.
		$storage->setKeyPrefix('test-' . uniqid('', true) . ':model:');
		$storage->setIndexKey('test-' . uniqid('', true) . ':index');
		$storage->setMode($mode);
		try {
			$storage->setHost('127.0.0.1');
			$storage->setPort(6379);
			$storage->setTimeout(0.5);
			$storage->getRedis();   // force the connection attempt
		} catch (\Throwable $e) {
			BayesianBackends::requireBackend($this, false, 'No reachable Redis at 127.0.0.1:6379: ' . $e->getMessage());
		}
		$this->_storages[] = $storage;
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
		foreach (self::classifierClasses() as $class) {
			foreach ([false, true] as $useTfidf) {
				$payloadStore = $this->storage(TRedisBayesianStorage::MODE_PAYLOAD);
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
				self::assertFalse($tokenLoaded->getVocabulary()->getSupportsFullScan(), $label);
				foreach (['cheap prices now', 'review the report', 'entirely unseen wording', 'free'] as $probe) {
					self::assertSame($payloadLoaded->score($probe), $tokenLoaded->score($probe), $label . ' / ' . $probe);
					self::assertSame($payloadLoaded->classify($probe), $tokenLoaded->classify($probe), $label . ' / ' . $probe);
				}
			}
		}
	}

	public function testLoadedModelIsStorageBacked()
	{
		$storage = $this->storage();
		$source = new TNaiveBayesClassifier();
		$source->setStorage($storage);
		$source->setName('m');
		$this->train($source)->save();

		$loaded = new TNaiveBayesClassifier();
		$loaded->setStorage($storage);
		$loaded->load('m');

		self::assertInstanceOf(TLazyBayesianVocabulary::class, $loaded->getVocabulary());
		self::assertTrue($loaded->getIsTrained());
		self::assertSame(6, $loaded->getVocabulary()->getTotalDocuments());
		self::assertSame(['spam', 'ham'], $loaded->getVocabulary()->getCategoryNames());
	}

	public function testCategoriesNamedLikeTheTypeBytesRoundTrip()
	{
		// The token-hash field is a type byte ('c'/'d') plus the category, so a category whose
		// own name starts with 'c' or 'd' is the case most likely to be mis-parsed.
		$build = function (TRedisBayesianStorage $storage) {
			$c = new TNaiveBayesClassifier();
			$c->setStorage($storage);
			$c->setName('m');
			$c->trainOne('cat', 'alpha beta gamma');
			$c->trainOne('dog', 'delta epsilon zeta');
			$c->save();
			return $c;
		};
		$payloadStore = $this->storage(TRedisBayesianStorage::MODE_PAYLOAD);
		$payload = $build($payloadStore);
		$tokenStore = $this->storage();
		$build($tokenStore);

		$loadedPayload = new TNaiveBayesClassifier();
		$loadedPayload->setStorage($payloadStore);
		$loadedPayload->load('m');
		$loadedToken = new TNaiveBayesClassifier();
		$loadedToken->setStorage($tokenStore);
		$loadedToken->load('m');
		self::assertSame($loadedPayload->score('alpha beta'), $loadedToken->score('alpha beta'));
	}

	public function testIncrementalTrainingMatchesResidentTraining()
	{
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

	public function testIncrementalTrainingIsAtomicPerField()
	{
		// Training the same token in the same category twice must accumulate, which is the
		// HINCRBY path: a second document adds to the first rather than replacing it.
		$storage = $this->storage();
		$c = new TNaiveBayesClassifier();
		$c->setStorage($storage);
		$c->setName('m');
		$c->trainOne('a', 'token token other');
		$c->trainOne('b', 'unrelated');
		$c->save();

		$writer = new TNaiveBayesClassifier();
		$writer->setStorage($storage);
		$writer->load('m');
		$writer->trainOne('a', 'token token token');

		$rows = $storage->loadTokens('m', ['token']);
		self::assertSame(5, $rows['token']['a']['count'], 'two + three occurrences accumulate');
		self::assertSame(2, $rows['token']['a']['docCount'], 'across two documents');
	}

	public function testDeleteRemovesEveryPerTokenKey()
	{
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
		// The token set itself must be gone, or a later save would see stale members.
		self::assertSame(0, (int) $storage->getRedis()->exists($storage->getKeyPrefix() . 'm:__toks'));
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
		self::assertSame([], $storage->loadTokens('m', ['quarterly']), 'a token only in the replaced model is gone');
	}

	public function testLoadTokensAnswersOnlyForKnownTokens()
	{
		$storage = $this->storage();
		$source = new TNaiveBayesClassifier();
		$source->setStorage($storage);
		$source->setName('m');
		$this->train($source)->save();

		$rows = $storage->loadTokens('m', ['cheap', 'not-a-real-token']);
		self::assertArrayHasKey('cheap', $rows);
		self::assertArrayNotHasKey('not-a-real-token', $rows);
		self::assertSame(2, $rows['cheap']['spam']['count']);
		self::assertSame(1, $rows['cheap']['spam']['docCount']);
	}

	public function testPayloadStorageRefusesAPerTokenModel()
	{
		$tokenStore = $this->storage();
		$source = new TNaiveBayesClassifier();
		$source->setStorage($tokenStore);
		$source->setName('m');
		$this->train($source)->save();

		$payloadStore = new TRedisBayesianStorage();
		$payloadStore->setKeyPrefix($tokenStore->getKeyPrefix());
		$payloadStore->setIndexKey($tokenStore->getIndexKey());
		$payloadStore->setHost('127.0.0.1');
		$payloadStore->setPort(6379);
		$reader = new TNaiveBayesClassifier();
		$reader->setStorage($payloadStore);
		try {
			$reader->load('m');
			self::fail('expected a per-token model to be refused by the payload path');
		} catch (\Prado\Exceptions\TInvalidDataValueException $e) {
			self::assertSame('bayesian_classifier_token_mode_payload', $e->getErrorCode());
		}
	}

	public function testTokenOperationsRequireTokenMode()
	{
		$storage = $this->storage(TRedisBayesianStorage::MODE_PAYLOAD);
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

	public function testListSpansBothModelsInTheSameKeyspace()
	{
		$storage = $this->storage();
		foreach (['zeta', 'alpha', 'mid'] as $name) {
			$c = new TNaiveBayesClassifier();
			$c->setStorage($storage);
			$c->setName($name);
			$c->trainOne('x', 'one two');
			$c->trainOne('y', 'three four');
			$c->save();
		}
		self::assertSame(['alpha', 'mid', 'zeta'], $storage->list());
	}
	/** Opens a second, independent storage on the same keys. */
	private function storageFor(TRedisBayesianStorage $storage): TRedisBayesianStorage
	{
		$other = new TRedisBayesianStorage();
		$other->setKeyPrefix($storage->getKeyPrefix());
		$other->setIndexKey($storage->getIndexKey());
		$other->setMode($storage->getMode());
		return $other;
	}

	public function testTwoLoadedInstancesTrainingTheSameModelLoseNoCounts()
	{
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
		self::assertSame(5, $vocabulary->getTotalDocuments());
		self::assertSame(10, $vocabulary->getVocabularySize());
		self::assertSame(3, $vocabulary->getCategory('spam')->getDocumentCount());
		self::assertSame(7, $vocabulary->getCategory('spam')->getTotalTokens());
		self::assertSame(1, $vocabulary->getCategory('news')->getDocumentCount());
		self::assertSame(['spam', 'ham', 'news'], $vocabulary->getCategoryNames());
		$rows = $storage->loadTokens('m', ['cheap']);
		self::assertSame(3, $rows['cheap']['spam']['count']);
		self::assertSame(3, $rows['cheap']['spam']['docCount']);
		self::assertSame(5, $second->getVocabulary()->getTotalDocuments(), 'the last writer sees the true totals');

		$resident = new TNaiveBayesClassifier();
		foreach ([['spam', 'cheap pills'], ['ham', 'team meeting'], ['spam', 'cheap watches'], ['spam', 'cheap lottery prize'], ['news', 'brand new category']] as [$category, $document]) {
			$resident->trainOne($category, $document);
		}
		foreach (['cheap prize', 'team', 'brand new'] as $probe) {
			self::assertSame($resident->score($probe), $reader->score($probe), $probe);
		}
	}

	public function testApplyDeltasAcceptsNegativeDeltasClampedAtZero()
	{
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
		self::assertSame(2, $storage->loadTokenCategories('m')['spam']['totalTokens']);

		$storage->applyDeltas('m', 'spam', ['cheap' => ['count' => -10, 'docCount' => -10]], [], ['documentCount' => -10, 'totalTokens' => -10]);
		$rows = $storage->loadTokens('m', ['cheap']);
		self::assertSame(['count' => 0, 'docCount' => 0], $rows['cheap']['spam'] ?? ['count' => 0, 'docCount' => 0], 'clamped at zero');
		$categories = $storage->loadTokenCategories('m');
		self::assertSame(['documentCount' => 0, 'totalTokens' => 0], $categories['spam']);
		$meta = $storage->loadTokenMeta('m');
		self::assertSame(1, $meta['totalDocuments']);
		self::assertSame(3, $meta['vocabularySize'], 'a token no document contains any more leaves the vocabulary');

		$storage->applyDeltas('m', 'spam', ['never' => ['count' => -3, 'docCount' => -1]], [], ['documentCount' => 0, 'totalTokens' => 0]);
		self::assertArrayNotHasKey('never', $storage->loadTokens('m', ['never']), 'a token no document contains is gone');
		self::assertSame(3, $storage->loadTokenMeta('m')['vocabularySize']);
	}

	public function testCountersAreDerivedFromTheKeysNotFromTheMetaArgument()
	{
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
		self::assertSame('naive-bayes', $meta['kind']);
		self::assertSame(TRedisBayesianStorage::TOKEN_LAYOUT_VERSION, $meta['layoutVersion']);
	}

	public function testALegacyTokenLayoutIsUpgradedOnFirstRead()
	{
		// 0.1.0 packed each category's scalars as "docs:tokens" in one hash and kept the totals
		// in the metadata JSON.
		$storage = $this->storage();
		$redis = $storage->getRedis();
		$key = $storage->getKeyPrefix() . 'legacy';
		$redis->set($key, json_encode(['kind' => 'naive-bayes', 'tokenMode' => true, 'totalDocuments' => 3, 'vocabularySize' => 4, 'categoryOrder' => ['spam', 'ham'], 'alpha' => 1.0]));
		$redis->sAdd($storage->getIndexKey(), 'legacy');
		$redis->hSet($key . ':__cat', 'spam', '2:4');
		$redis->hSet($key . ':__cat', 'ham', '1:2');
		foreach ([['cheap', 'spam', 3, 2], ['pills', 'spam', 1, 1], ['team', 'ham', 1, 1], ['meeting', 'ham', 1, 1]] as [$token, $category, $count, $docs]) {
			$redis->hSet($key . ':__t:' . $token, 'c' . $category, (string) $count);
			$redis->hSet($key . ':__t:' . $token, 'd' . $category, (string) $docs);
			$redis->sAdd($key . ':__toks', $token);
		}

		$meta = $storage->loadTokenMeta('legacy');
		self::assertSame(3, $meta['totalDocuments']);
		self::assertSame(4, $meta['vocabularySize']);
		self::assertSame(TRedisBayesianStorage::TOKEN_LAYOUT_VERSION, $meta['layoutVersion']);
		self::assertSame(['ham' => ['documentCount' => 1, 'totalTokens' => 2], 'spam' => ['documentCount' => 2, 'totalTokens' => 4]], $storage->loadTokenCategories('legacy'));
		self::assertSame(0, (int) $redis->exists($key . ':__cat'), 'the packed hash is gone');

		$loaded = new TNaiveBayesClassifier();
		$loaded->setStorage($this->storageFor($storage));
		$loaded->load('legacy');
		self::assertSame(['spam', 'ham'], $loaded->getVocabulary()->getCategoryNames());
		self::assertSame('spam', $loaded->classify('cheap'));
		$loaded->trainOne('spam', 'cheap offer');
		$meta = $storage->loadTokenMeta('legacy');
		self::assertSame(4, $meta['totalDocuments']);
		self::assertSame(5, $meta['vocabularySize']);
	}

	public function testLegacyCategoriesAreReadableBeforeTheMetadataIsTouched()
	{
		// loadTokenCategories() can be the first call on a model; it must not misread the packed
		// form as the current one.
		$storage = $this->storage();
		$redis = $storage->getRedis();
		$key = $storage->getKeyPrefix() . 'legacy';
		$redis->set($key, json_encode(['kind' => 'naive-bayes', 'tokenMode' => true]));
		$redis->hSet($key . ':__cat', 'spam', '2:4');
		self::assertSame(['spam' => ['documentCount' => 2, 'totalTokens' => 4]], $this->storageFor($storage)->loadTokenCategories('legacy'));
	}

	public function testANewerTokenLayoutIsRefused()
	{
		$storage = $this->storage();
		$storage->getRedis()->set($storage->getKeyPrefix() . 'future', json_encode(['kind' => 'naive-bayes', 'tokenMode' => true, 'layoutVersion' => TRedisBayesianStorage::TOKEN_LAYOUT_VERSION + 1]));
		try {
			$storage->loadTokenMeta('future');
			self::fail('expected a newer layout to be refused');
		} catch (\Prado\Exceptions\TInvalidDataValueException $e) {
			self::assertSame('bayesian_storage_layout_unsupported', $e->getErrorCode());
		}
	}

	public function testAModelNameContainingTheReservedSequenceIsRejected()
	{
		// "m:__t:x" is exactly the key that token "x" of model "m" lives under.
		$storage = $this->storage();
		foreach (['m:__t:x', 'm:__cat', ':__toks'] as $name) {
			try {
				$storage->save($name, []);
				self::fail("expected {$name} to be rejected");
			} catch (\Prado\Exceptions\TInvalidDataValueException $e) {
				self::assertSame('bayesian_storage_name_invalid', $e->getErrorCode());
			}
		}
		// A colon on its own is fine.
		$storage->save('site:comments', ['a' => 1]);
		self::assertTrue($storage->exists('site:comments'));
	}

	public function testDeleteDuringTrainingLeavesNoOrphanKeys()
	{
		$storage = $this->storage();
		$source = new TNaiveBayesClassifier();
		$source->setStorage($storage);
		$source->setName('m');
		$this->train($source)->save();
		$storage->delete('m');
		$redis = $storage->getRedis();
		$scan = null;
		$leftovers = [];
		while (($keys = $redis->scan($scan, $storage->getKeyPrefix() . '*', 1000)) !== false) {
			foreach ($keys as $key) {
				$leftovers[] = $key;
			}
			if ($scan === 0) {
				break;
			}
		}
		self::assertSame([], $leftovers);
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
}
