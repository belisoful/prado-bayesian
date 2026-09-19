<?php

use Belisoful\Prado\Util\Bayesian\Classifier\TBernoulliNaiveBayes;
use Belisoful\Prado\Util\Bayesian\Classifier\TComplementNaiveBayes;
use Belisoful\Prado\Util\Bayesian\Classifier\TNaiveBayesClassifier;
use Belisoful\Prado\Util\Bayesian\Storage\IBayesianHistogramStorage;
use Belisoful\Prado\Util\Bayesian\TBayesianTokenHistogram;

/**
 * The tests that prove Bernoulli and Complement Naive Bayes train incrementally through
 * per-token storage, shared by the SQL and Redis test classes.
 *
 * Both variants need a per-category sum over the whole vocabulary.  The storage keeps the
 * integer histograms those sums are computed from, moving them atomically with every training
 * write, so a storage-backed model must score exactly — bit for bit — as a resident model
 * trained on the same documents.  Every test here therefore compares against a resident
 * reference rather than asserting numbers.
 *
 * The host class supplies `storage()`, `storageFor()`, `corpus()`, `train()` and
 * `stripHistograms()`.
 */
trait BayesianVariantIncrementalTests
{
	/** @return string[] The variants whose aggregates span the vocabulary. */
	public static function variantClasses(): array
	{
		return [TBernoulliNaiveBayes::class, TComplementNaiveBayes::class];
	}

	/** @return array<int, array{0:string,1:string}> Documents trained after the model was saved. */
	private function extraDocuments(): array
	{
		return [
			['spam', 'cheap cheap lottery prize now'],
			['ham', 'meeting notes review tomorrow'],
			['news', 'election results tomorrow morning'],
			['news', 'cheap flights results'],
		];
	}

	/** @return string[] Documents to compare scores on. */
	private function probes(): array
	{
		return ['cheap prices now', 'review the election results', 'meeting tomorrow morning lottery', 'words nobody trained'];
	}

	private function assertSameScores(TNaiveBayesClassifier $expected, TNaiveBayesClassifier $actual, string $message): void
	{
		foreach ($this->probes() as $probe) {
			self::assertSame($expected->score($probe), $actual->score($probe), $message . ': ' . $probe);
			self::assertSame($expected->logScores($probe), $actual->logScores($probe), $message . ' (log): ' . $probe);
		}
	}

	public function testVariantsTrainIncrementallyThroughStorage()
	{
		foreach (self::variantClasses() as $class) {
			$storage = $this->storage();
			$source = new $class();
			$source->setStorage($storage);
			$source->setName('m');
			$this->train($source)->save();

			$reference = $this->train(new $class());
			$trainer = new $class();
			$trainer->setStorage($this->storageFor($storage));
			$trainer->load('m');
			foreach ($this->extraDocuments() as [$category, $document]) {
				$reference->trainOne($category, $document);
				$trainer->trainOne($category, $document);
				// The trainer itself keeps scoring, exactly, after each of its own writes.
				$this->assertSameScores($reference, $trainer, $class . ' trainer after ' . $document);
			}

			$reader = new $class();
			$reader->setStorage($this->storageFor($storage));
			$reader->load('m');
			$this->assertSameScores($reference, $reader, $class . ' fresh reader');
		}
	}

	public function testVariantsUntrainIncrementallyThroughStorage()
	{
		foreach (self::variantClasses() as $class) {
			$storage = $this->storage();
			$source = new $class();
			$source->setStorage($storage);
			$source->setName('m');
			$this->train($source)->save();

			$trainer = new $class();
			$trainer->setStorage($this->storageFor($storage));
			$trainer->load('m');
			foreach ($this->extraDocuments() as [$category, $document]) {
				$trainer->trainOne($category, $document);
			}
			foreach (array_reverse($this->extraDocuments()) as [$category, $document]) {
				$trainer->untrainOne($category, $document);
			}

			// Back where it started: the model scores as one that never saw the documents.
			$reference = $this->train(new $class());
			$reader = new $class();
			$reader->setStorage($this->storageFor($storage));
			$reader->load('m');
			$this->assertSameScores($reference, $reader, $class . ' after untraining');
			$this->assertSameScores($reference, $trainer, $class . ' trainer after untraining');
		}
	}

	public function testVariantsTrainedFromTwoInstancesStayExact()
	{
		foreach (self::variantClasses() as $class) {
			$storage = $this->storage();
			$source = new $class();
			$source->setStorage($storage);
			$source->setName('m');
			$this->train($source)->save();

			$first = new $class();
			$first->setStorage($this->storageFor($storage));
			$first->load('m');
			$second = new $class();
			$second->setStorage($this->storageFor($storage));
			$second->load('m');

			$reference = $this->train(new $class());
			foreach ($this->extraDocuments() as $index => [$category, $document]) {
				// Each instance trains from its own stale snapshot of the model.
				($index % 2 === 0 ? $first : $second)->trainOne($category, $document);
				$reference->trainOne($category, $document);
			}

			$reader = new $class();
			$reader->setStorage($this->storageFor($storage));
			$reader->load('m');
			$this->assertSameScores($reference, $reader, $class . ' two writers');
		}
	}

	public function testHistogramsAreRebuiltForAModelStoredWithoutThem()
	{
		// A per-token model written before the storage kept histograms has none, and its
		// metadata does not advertise any.  Loading it rebuilds them in the store, once.
		foreach (self::variantClasses() as $class) {
			$storage = $this->storage();
			$source = new $class();
			$source->setStorage($storage);
			$source->setName('m');
			$this->train($source)->save();
			$this->stripHistograms($storage, 'm');
			self::assertSame([], TBayesianTokenHistogram::flatten($storage->loadTokenHistograms('m', TBayesianTokenHistogram::FAMILIES)), 'stripped');

			$reference = $this->train(new $class());
			$trainer = new $class();
			$trainer->setStorage($this->storageFor($storage));
			$trainer->load('m');
			$this->assertSameScores($reference, $trainer, $class . ' after the rebuild');
			foreach ($this->extraDocuments() as [$category, $document]) {
				$reference->trainOne($category, $document);
				$trainer->trainOne($category, $document);
			}
			$reader = new $class();
			$reader->setStorage($this->storageFor($storage));
			$reader->load('m');
			$this->assertSameScores($reference, $reader, $class . ' trained after the rebuild');
		}
	}

	public function testTrainingAStrippedModelWithoutLoadingRebuildsItsHistograms()
	{
		// applyDeltas() is handed metadata that asks for histograms the store does not hold:
		// it must build them rather than move rows that are not there.
		$storage = $this->storage();
		$source = new TBernoulliNaiveBayes();
		$source->setStorage($storage);
		$source->setName('m');
		$this->train($source)->save();
		$trainer = new TBernoulliNaiveBayes();
		$trainer->setStorage($this->storageFor($storage));
		$trainer->load('m');
		$this->stripHistograms($storage, 'm');

		$reference = $this->train(new TBernoulliNaiveBayes());
		foreach ($this->extraDocuments() as [$category, $document]) {
			$reference->trainOne($category, $document);
			$trainer->trainOne($category, $document);
		}
		$reader = new TBernoulliNaiveBayes();
		$reader->setStorage($this->storageFor($storage));
		$reader->load('m');
		$this->assertSameScores($reference, $reader, 'self-healed');
	}

	public function testMaintainedHistogramsEqualARebuildFromTheTokenRows()
	{
		foreach (self::variantClasses() as $class) {
			$storage = $this->storage();
			self::assertInstanceOf(IBayesianHistogramStorage::class, $storage);
			$source = new $class();
			$source->setStorage($storage);
			$source->setName('m');
			$this->train($source)->save();
			$trainer = new $class();
			$trainer->setStorage($this->storageFor($storage));
			$trainer->load('m');
			foreach ($this->extraDocuments() as [$category, $document]) {
				$trainer->trainOne($category, $document);
			}
			$trainer->untrainOne('news', 'cheap flights results');

			$families = TBayesianTokenHistogram::families($storage->loadTokenMeta('m')['histograms'] ?? null);
			self::assertNotSame([], $families, $class . ' advertises its histograms');
			$maintained = $storage->loadTokenHistograms('m', $families);
			self::assertNotSame([], TBayesianTokenHistogram::flatten($maintained));
			$storage->rebuildTokenHistograms('m', $families);
			self::assertEquals($maintained, $storage->loadTokenHistograms('m', $families), $class);
		}
	}

	public function testTheMultinomialClassifierKeepsNoHistograms()
	{
		// Its likelihood needs none, so its training writes pay for none.
		$storage = $this->storage();
		$source = new TNaiveBayesClassifier();
		$source->setStorage($storage);
		$source->setName('m');
		$this->train($source)->save();
		$trainer = new TNaiveBayesClassifier();
		$trainer->setStorage($this->storageFor($storage));
		$trainer->load('m');
		$trainer->trainOne('spam', 'cheap lottery');
		self::assertSame([], TBayesianTokenHistogram::flatten($storage->loadTokenHistograms('m', TBayesianTokenHistogram::FAMILIES)));
		self::assertSame([], TBayesianTokenHistogram::families($storage->loadTokenMeta('m')['histograms'] ?? null));
	}

	public function testDeletingAModelDeletesItsHistograms()
	{
		$storage = $this->storage();
		$source = new TComplementNaiveBayes();
		$source->setStorage($storage);
		$source->setName('m');
		$this->train($source)->save();
		self::assertNotSame([], TBayesianTokenHistogram::flatten($storage->loadTokenHistograms('m', TBayesianTokenHistogram::FAMILIES)));
		$storage->delete('m');
		self::assertSame([], TBayesianTokenHistogram::flatten($storage->loadTokenHistograms('m', TBayesianTokenHistogram::FAMILIES)));
	}

	public function testUntrainingADocumentThatWasNeverTrainedKeepsTheHistogramsConsistent()
	{
		// Every count clamps at zero, so the histograms must follow the clamped counts, not
		// the deltas that were asked for.
		foreach (self::variantClasses() as $class) {
			$storage = $this->storage();
			$source = new $class();
			$source->setStorage($storage);
			$source->setName('m');
			$this->train($source)->save();
			$trainer = new $class();
			$trainer->setStorage($this->storageFor($storage));
			$trainer->load('m');
			$trainer->untrainOne('ham', 'cheap cheap cheap never seen words');
			$trainer->untrainOne('spam', 'cheap cheap cheap');

			$families = TBayesianTokenHistogram::families($storage->loadTokenMeta('m')['histograms'] ?? null);
			$maintained = $storage->loadTokenHistograms('m', $families);
			$storage->rebuildTokenHistograms('m', $families);
			self::assertEquals($maintained, $storage->loadTokenHistograms('m', $families), $class);
		}
	}
}
