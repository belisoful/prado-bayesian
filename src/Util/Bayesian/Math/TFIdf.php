<?php

/**
 * TFIdf class file.
 *
 * @author Brad Anderson <belisoful@icloud.com>
 * @link https://github.com/belisoful/prado-bayesian
 * @license https://github.com/pradosoft/prado/blob/master/LICENSE
 */

namespace Belisoful\Prado\Util\Bayesian\Math;

/**
 * TFIdf class.
 *
 * Term-frequency × inverse-document-frequency weighting.  The raw Naive Bayes product treats
 * every token as equally informative; a token that appears in nearly every document is a poor
 * discriminator, while a token that appears in only a few should count more.  TF-IDF
 * re-weights the term counts of the document being classified by this intuition (using the
 * document frequencies learned during training) before the classifier sums the per-token
 * log-probabilities.
 *
 * The class is a static utility — build the IDF once from the document-frequency map of a
 * corpus, then ask for the weight of any (term, frequency) pair.  Document frequency is the
 * number of documents the term appeared in (not the total occurrence count).
 *
 * @author Brad Anderson <belisoful@icloud.com>
 * @since 0.1.0
 */
final class TFIdf
{
	/**
	 * Computes the inverse-document-frequency weight for a term from a document-frequency map.
	 *
	 * Uses the smoothed form: `log((N + 1) / (df + 1)) + 1` so terms that appear in every
	 * document still get a non-zero, non-explosive weight.
	 *
	 * The document frequency is passed as a count rather than looked up in a map: the caller
	 * knows which term it is weighting, and a classifier scoring against storage-backed
	 * statistics has the one count without holding the whole map.
	 * @param int $documentFrequency The number of documents containing the term (>= 0).
	 * @param int $totalDocuments The total number of documents in the corpus; must be positive.
	 * @return float The IDF weight (>= 1.0 by the smoothed formulation).
	 */
	public static function idf(int $documentFrequency, int $totalDocuments): float
	{
		if ($totalDocuments <= 0) {
			return 1.0;
		}
		if ($documentFrequency < 0) {
			$documentFrequency = 0;
		}
		return log(($totalDocuments + 1) / ($documentFrequency + 1)) + 1.0;
	}

	/**
	 * Computes the TF-IDF weight of a (term, frequency) pair under a given IDF map.
	 *
	 * The term-frequency factor is `1 + log(frequency)` when the frequency is positive,
	 * which compresses long documents' contributions so a 100× repetition is not 100× the
	 * influence of a single occurrence.
	 * @param int $frequency The occurrence count of the term in the document (>= 0).
	 * @param int $documentFrequency The number of corpus documents containing the term (>= 0).
	 * @param int $totalDocuments The total number of documents in the corpus; must be positive.
	 * @return float The TF-IDF weight; 0.0 when the term does not occur in the document.
	 */
	public static function weight(int $frequency, int $documentFrequency, int $totalDocuments): float
	{
		if ($frequency <= 0) {
			return 0.0;
		}
		return self::termFrequency($frequency) * self::idf($documentFrequency, $totalDocuments);
	}

	/**
	 * The smoothed term-frequency factor `1 + log(frequency)`.
	 * @param int $frequency The occurrence count (>= 1).
	 * @return float The TF factor.
	 */
	public static function termFrequency(int $frequency): float
	{
		if ($frequency <= 0) {
			return 0.0;
		}
		return 1.0 + log($frequency);
	}
}
