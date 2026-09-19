<?php

/**
 * IBayesianHistogramStorage interface file.
 *
 * @author Brad Anderson <belisoful@icloud.com>
 * @link https://github.com/belisoful/prado-bayesian
 * @license https://github.com/pradosoft/prado/blob/master/LICENSE
 */

namespace Belisoful\Prado\Util\Bayesian\Storage;

/**
 * IBayesianHistogramStorage interface.
 *
 * A per-token storage that also keeps the {@see \Belisoful\Prado\Util\Bayesian\TBayesianTokenHistogram}
 * histograms of a model, which is what lets Bernoulli and Complement Naive Bayes train
 * incrementally against it: their per-category aggregates are sums over the whole vocabulary,
 * and the histograms are those sums in a form the store can move by exact increments.
 *
 * The contract has three parts:
 *
 * - **Which families a model keeps is part of its metadata**: the `histograms` key, a list of
 *   family letters, written by the classifier that owns the model.  A model whose metadata
 *   names no family keeps none, and its training writes pay for none.
 * - **{@see IBayesianTokenStorage::saveTokenModel()} writes the named families** from the
 *   statistics it is given, and **{@see IBayesianTokenStorage::applyDeltas()} moves them** in
 *   the same atomic unit as the token counts: the cells the touched tokens counted towards
 *   before the write lose one each, the cells they count towards afterwards gain one.  When
 *   the metadata handed to `applyDeltas()` names a family the stored metadata does not — a
 *   model written before histograms existed — the write rebuilds instead of moving.
 * - **{@see rebuildTokenHistograms()} recomputes them inside the store**, the only operation
 *   here proportional to the model, for migrating such a model or repairing one.
 *
 * @author Brad Anderson <belisoful@icloud.com>
 * @since 0.2.0
 */
interface IBayesianHistogramStorage extends IBayesianTokenStorage
{
	/**
	 * Returns the stored histograms of the given families.
	 * @param string $name The model name.
	 * @param string[] $families The family letters wanted.
	 * @return array<string, array<string, array<int, int>>> The histograms, as family =>
	 * category => value => token count; empty when the model keeps none.
	 */
	public function loadTokenHistograms(string $name, array $families): array;

	/**
	 * Recomputes the given families from the model's token statistics, atomically with respect
	 * to training writes, and records them (together with any family already kept) in the
	 * model's `histograms` metadata so that later writes keep them current.
	 * @param string $name The model name.
	 * @param string[] $families The family letters to build.
	 */
	public function rebuildTokenHistograms(string $name, array $families): void;
}
