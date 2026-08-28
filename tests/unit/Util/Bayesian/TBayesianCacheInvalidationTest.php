<?php

use Prado\Util\Bayesian\Classifier\TBernoulliNaiveBayes;
use Prado\Util\Bayesian\Classifier\TComplementNaiveBayes;
use Prado\Util\Bayesian\TBayesianCategory;

/**
 * Covers the derived caches that Bernoulli and Complement keep.
 *
 * Both hoist an O(|V|) quantity out of the per-classification path — Bernoulli's absent-token
 * constant, Complement's corpus counts and per-category weight norms.  A stale one of those
 * does not raise; it silently shifts every score.  So each test here mutates the model in a way
 * that must invalidate, and asserts the score matches a classifier built fresh from the same
 * training — the only comparison that would catch a cache that failed to notice.
 */
class TBayesianCacheInvalidationTest extends PHPUnit\Framework\TestCase
{
	/** @return array{0:TBernoulliNaiveBayes,1:TBernoulliNaiveBayes} A warmed classifier and a pristine twin. */
	private function bernoulliPair(): array
	{
		$make = function () {
			$c = new TBernoulliNaiveBayes();
			$c->trainOne('spam', 'cheap pills buy now');
			$c->trainOne('ham', 'project meeting agenda');
			return $c;
		};
		return [$make(), $make()];
	}

	public function testBernoulliMatchesTheLiteralFullVocabularySum()
	{
		// The identity under test: sum over V of [present ? log p : log(1-p)] equals
		// sum over V of log(1-p) plus a correction over the document's tokens alone.
		$c = new TBernoulliNaiveBayes();
		$c->setAlpha(0.7);
		$c->trainOne('a', 'alpha beta gamma delta epsilon');
		$c->trainOne('a', 'alpha beta zeta');
		$c->trainOne('b', 'eta theta iota kappa');
		$c->trainOne('b', 'eta theta lambda mu');

		$document = ['alpha', 'beta', 'eta', 'unseen-token'];
		$alpha = $c->getAlpha();
		$vocabulary = $c->getVocabulary();
		$expected = [];
		foreach ($vocabulary->getCategories() as $category) {
			$denominator = $category->getDocumentCount() + 2.0 * $alpha;
			$present = array_flip($document);
			$sum = 0.0;
			foreach ($vocabulary->getDocumentFrequency() as $token => $_) {
				$p = ($category->getTokenDocumentCount((string) $token) + $alpha) / $denominator;
				if ($p <= 0.0 || $p >= 1.0) {
					continue;
				}
				$sum += isset($present[$token]) ? log($p) : log(1.0 - $p);
			}
			$prior = log($category->getDocumentCount() / $vocabulary->getTotalDocuments());
			$expected[$category->getName()] = $prior + $sum;
		}
		$expected = \Prado\Util\Bayesian\Math\TBayesMath::normalize($expected);

		foreach ($c->score($document) as $name => $score) {
			self::assertEqualsWithDelta($expected[$name], $score, 1e-12, "category {$name}");
		}
	}

	public function testBernoulliCacheIsDroppedByFurtherTraining()
	{
		[$warm, $fresh] = $this->bernoulliPair();
		$warm->score('cheap pills');            // builds the cached constant

		$warm->trainOne('spam', 'discount offer cheap');
		$fresh->trainOne('spam', 'discount offer cheap');
		self::assertSame($fresh->score('cheap pills'), $warm->score('cheap pills'));
	}

	public function testBernoulliCacheIsDroppedByAnAlphaChange()
	{
		[$warm, $fresh] = $this->bernoulliPair();
		$warm->score('cheap pills');

		$warm->setAlpha(0.25);
		$fresh->setAlpha(0.25);
		self::assertSame($fresh->score('cheap pills'), $warm->score('cheap pills'));
	}

	public function testBernoulliCacheIsDroppedByDirectVocabularyMutation()
	{
		// getVocabulary() hands out live objects.  A cache keyed only on the classifier's own
		// training calls would not see this, and every later score would be quietly wrong.
		[$warm, $fresh] = $this->bernoulliPair();
		$warm->score('cheap pills');

		foreach ([$warm, $fresh] as $c) {
			$c->getVocabulary()->addDocument('spam', ['cheap', 'cheap', 'watches']);
		}
		self::assertSame($fresh->score('cheap pills'), $warm->score('cheap pills'));
	}

	public function testBernoulliCacheIsDroppedByDirectCategoryMutation()
	{
		// The subtlest case: the mutation happens on a TBayesianCategory, so the vocabulary's
		// own totals do not move at all.
		[$warm, $fresh] = $this->bernoulliPair();
		$warm->score('cheap pills');

		foreach ([$warm, $fresh] as $c) {
			$c->getVocabulary()->getCategory('spam')->addTokenDocument('meeting');
		}
		self::assertSame($fresh->score('cheap pills'), $warm->score('cheap pills'));
	}

	public function testBernoulliIsUnaffectedByRepeatedTokensInTheDocument()
	{
		// Bernoulli's feature is presence, not count.
		$c = new TBernoulliNaiveBayes();
		$c->trainOne('spam', 'cheap pills');
		$c->trainOne('ham', 'project meeting');
		self::assertSame($c->score('cheap'), $c->score('cheap cheap cheap'));
	}

	public function testComplementScoresACategoryNamedHashLikeAnyOther()
	{
		// The per-category norms live in a map keyed by category name; a reserved key in that
		// map would be a name a caller could legitimately use.
		$build = function (string $first) {
			$c = new TComplementNaiveBayes();
			$c->trainOne($first, 'alpha beta gamma delta');
			$c->trainOne($first, 'alpha beta epsilon');
			$c->trainOne('other', 'zeta eta theta iota');
			$c->trainOne('other', 'zeta eta kappa');
			return $c->score('alpha beta gamma');
		};
		$hashNamed = $build('#');
		$plainNamed = $build('renamed');
		self::assertEqualsWithDelta($plainNamed['renamed'], $hashNamed['#'], 1e-12);
		self::assertEqualsWithDelta($plainNamed['other'], $hashNamed['other'], 1e-12);
	}

	public function testComplementCacheIsDroppedByDirectCategoryMutation()
	{
		$make = function () {
			$c = new TComplementNaiveBayes();
			$c->trainOne('spam', 'cheap pills buy now');
			$c->trainOne('ham', 'project meeting agenda');
			return $c;
		};
		$warm = $make();
		$fresh = $make();
		$warm->score('cheap pills');

		foreach ([$warm, $fresh] as $c) {
			$c->getVocabulary()->getCategory('spam')->addToken('agenda', 3);
		}
		self::assertSame($fresh->score('cheap pills'), $warm->score('cheap pills'));
	}

	public function testVocabularyStateSignatureChangesOnEveryKindOfMutation()
	{
		$c = new TBernoulliNaiveBayes();
		$c->trainOne('a', 'one two');
		$vocabulary = $c->getVocabulary();

		$seen = [$vocabulary->getStateSignature()];
		$vocabulary->addDocument('b', ['three']);
		$seen[] = $vocabulary->getStateSignature();
		$vocabulary->getCategory('a')->addToken('four');
		$seen[] = $vocabulary->getStateSignature();
		$vocabulary->getCategory('a')->addTokenDocument('four');
		$seen[] = $vocabulary->getStateSignature();
		$vocabulary->getCategory('a')->addDocument();
		$seen[] = $vocabulary->getStateSignature();
		$vocabulary->getOrCreateCategory('c');
		$seen[] = $vocabulary->getStateSignature();
		$vocabulary->setStats([new TBayesianCategory('z')], ['tok' => 1], 1);
		$seen[] = $vocabulary->getStateSignature();

		self::assertCount(count($seen), array_unique($seen), 'every mutation must change the signature');
	}

	public function testGenerationCountersOnlyEverIncrease()
	{
		$category = new TBayesianCategory('a');
		$start = $category->getGeneration();
		$category->addDocument();
		$afterDocument = $category->getGeneration();
		$category->addToken('t');
		$afterToken = $category->getGeneration();

		self::assertGreaterThan($start, $afterDocument);
		self::assertGreaterThan($afterDocument, $afterToken);
	}
}
