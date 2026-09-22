<?php

use Belisoful\Prado\Util\Bayesian\Classifier\TNaiveBayesClassifier;
use Belisoful\Prado\Util\Bayesian\TBayesianModule;
use Belisoful\Prado\Util\Bayesian\TBayesianRecommender;
use Belisoful\Prado\Web\Services\TBayesianService;
use Prado\Exceptions\TConfigurationException;

/** Exposes a stub application so the module-resolution path can be tested without a full app. */
class StubAppBayesianService extends TBayesianService
{
	/** @var array<string, TBayesianModule> */
	public array $modules = [];

	// Disable global-event listening so construction never reaches into the stub application.
	public function getAutoGlobalListen()
	{
		return false;
	}

	public function getApplication()
	{
		$modules = $this->modules;
		return new class ($modules) {
			/** @param array<string, mixed> $modules */
			public function __construct(private array $modules)
			{
			}
			public function getModulesByType($type, $strict = false): array
			{
				return $this->modules;
			}
			public function getModule($id)
			{
				return $this->modules[$id] ?? null;
			}
		};
	}
}


/** Reports no application at all so the null-application guard of the resolver can be tested. */
class NoAppBayesianService extends TBayesianService
{
	public function getAutoGlobalListen()
	{
		return false;
	}

	public function getApplication()
	{
		return null;
	}
}

/** Records the headers instead of sending them (header() cannot be used once PHPUnit has written to stdout). */
class RecordingHttpResponse extends \Prado\Web\THttpResponse
{
	/** @var string[] */
	public array $headers = [];

	public function appendHeader($header, bool $replace = true, int $response_code = 0): void
	{
		$this->headers[] = $header;
	}
}

/** Denies both service permissions the way TPermissionsBehavior does for a user without them. */
class DenyingBayesianBehavior extends \Prado\Util\TBehavior
{
	/** @var string[] The dynamic events seen. */
	public array $seen = [];

	public function dyClassify($denied, $callchain)
	{
		$this->seen[] = 'dyClassify';
		return true;
	}

	public function dyRecommend($denied, $callchain)
	{
		$this->seen[] = 'dyRecommend';
		return true;
	}

	public function dyTag($denied, $callchain)
	{
		$this->seen[] = 'dyTag';
		return true;
	}
}

/** Exposes the protected JSON encoder. */
class EncodingBayesianService extends TBayesianService
{
	public function encodePublic(array $result): string
	{
		return $this->encode($result);
	}
}

class TBayesianServiceTest extends PHPUnit\Framework\TestCase
{
	public function testClassifyAction()
	{
		$service = new TBayesianService();
		$classifier = new TNaiveBayesClassifier();
		$classifier->trainOne('spam', 'cheap offer click free');
		$classifier->trainOne('ham', 'meeting report tomorrow lunch');
		$service->setClassifier($classifier);
		$response = $service->runService(['action' => 'classify', 'text' => 'cheap click free']);
		self::assertSame('spam', $response['category']);
		self::assertArrayHasKey('spam', $response['scores']);
		self::assertArrayHasKey('ham', $response['scores']);
		$sum = $response['scores']['spam'] + $response['scores']['ham'];
		self::assertEqualsWithDelta(1.0, $sum, 1e-9);
	}

	public function testClassifyActionDefaultWhenNoAction()
	{
		$service = new TBayesianService();
		$classifier = new TNaiveBayesClassifier();
		$classifier->trainOne('spam', 'cheap offer');
		$classifier->trainOne('ham', 'meeting tomorrow');
		$service->setClassifier($classifier);
		$response = $service->runService(['text' => 'cheap offer']);
		self::assertSame('spam', $response['category']);
	}

	public function testClassifyActionWithCategoryParam()
	{
		$service = new TBayesianService();
		$classifier = new TNaiveBayesClassifier();
		$classifier->trainOne('spam', 'cheap offer');
		$classifier->trainOne('ham', 'meeting tomorrow');
		$service->setClassifier($classifier);
		$response = $service->runService(['text' => 'cheap offer', 'category' => 'spam']);
		self::assertTrue($response['isSpam']);
	}

	public function testClassifyActionRequiresText()
	{
		$service = new TBayesianService();
		$classifier = new TNaiveBayesClassifier();
		$service->setClassifier($classifier);
		try {
			$service->runService(['action' => 'classify']);
			self::fail('expected exception');
		} catch (\Prado\Exceptions\TInvalidDataValueException $e) {
			self::assertSame('bayesian_service_text_required', $e->getErrorCode());
		}
	}

	public function testUnknownActionThrows()
	{
		$service = new TBayesianService();
		$service->setClassifier(new TNaiveBayesClassifier());
		try {
			$service->runService(['action' => 'wat', 'text' => 'foo']);
			self::fail('expected exception');
		} catch (\Prado\Exceptions\TInvalidDataValueException $e) {
			self::assertSame('bayesian_service_action_unknown', $e->getErrorCode());
		}
	}

	public function testGetClassifierThrowsWhenUnset()
	{
		$service = new TBayesianService();
		$this->expectException(TConfigurationException::class);
		$service->getClassifier();
	}

	public function testRecommenderIsLazilyCreatedFromClassifier()
	{
		$service = new TBayesianService();
		$classifier = new TNaiveBayesClassifier();
		$classifier->trainOne('liked', 'red shoes sneakers');
		$classifier->trainOne('ignored', 'red hat scarf');
		$service->setClassifier($classifier);
		$recommender = $service->getRecommender();
		self::assertInstanceOf(TBayesianRecommender::class, $recommender);
		// Same instance on second call.
		self::assertSame($recommender, $service->getRecommender());
	}

	public function testSetRecommenderReplacesDefault()
	{
		$service = new TBayesianService();
		$service->setClassifier(new TNaiveBayesClassifier());
		$custom = new TBayesianRecommender();
		$service->setRecommender($custom);
		self::assertSame($custom, $service->getRecommender());
	}

	public function testRecommendAction()
	{
		$service = new TBayesianService();
		$classifier = new TNaiveBayesClassifier();
		$classifier->trainOne('liked', 'red shoes blue sneakers leather boots');
		$classifier->trainOne('ignored', 'red hat blue scarf leather belt');
		$service->setClassifier($classifier);
		$response = $service->runService([
			'action' => 'recommend',
			'context' => ['red shoes'],
			'candidates' => ['red sneakers', 'blue sneakers', 'red hat', 'leather wallet'],
		]);
		self::assertArrayHasKey('scores', $response);
		self::assertCount(4, $response['scores']);
	}

	public function testRecommendActionWithEmptyCandidatesThrows()
	{
		$service = new TBayesianService();
		$classifier = new TNaiveBayesClassifier();
		$classifier->trainOne('liked', 'red shoes');
		$service->setClassifier($classifier);
		$this->expectException(\Prado\Exceptions\TInvalidDataValueException::class);
		$service->runService([
			'action' => 'recommend',
			'context' => ['red shoes'],
			'candidates' => [],
		]);
	}

	public function testResolvesClassifierFromModuleAutoDetect()
	{
		$module = new TBayesianModule();
		$classifier = new TNaiveBayesClassifier();
		$classifier->trainOne('spam', 'cheap offer');
		$module->setClassifier($classifier);

		$service = new StubAppBayesianService();
		$service->modules = ['bayesian' => $module];
		// No classifier set explicitly; it is resolved from the registered module.
		self::assertSame($classifier, $service->getClassifier());
	}

	public function testResolvesClassifierFromModuleByID()
	{
		$wrong = new TBayesianModule();
		$wrong->setClassifier(new TNaiveBayesClassifier());
		$right = new TBayesianModule();
		$target = new TNaiveBayesClassifier();
		$right->setClassifier($target);

		$service = new StubAppBayesianService();
		$service->setModuleID('right');
		$service->modules = ['wrong' => $wrong, 'right' => $right];
		self::assertSame($target, $service->getClassifier());
	}

	public function testExplicitClassifierTakesPrecedenceOverModule()
	{
		$module = new TBayesianModule();
		$module->setClassifier(new TNaiveBayesClassifier());
		$explicit = new TNaiveBayesClassifier();

		$service = new StubAppBayesianService();
		$service->modules = ['bayesian' => $module];
		$service->setClassifier($explicit);
		self::assertSame($explicit, $service->getClassifier());
	}

	private function trainedService(): TBayesianService
	{
		$service = new TBayesianService();
		$classifier = new TNaiveBayesClassifier();
		$classifier->trainOne('spam', 'cheap offer click free');
		$classifier->trainOne('ham', 'meeting report tomorrow lunch');
		$service->setClassifier($classifier);
		return $service;
	}

	public function testArrayTextParameterIsRejected()
	{
		$service = $this->trainedService();
		try {
			$service->runService(['text' => ['x']]);
			self::fail('expected exception');
		} catch (\Prado\Exceptions\TInvalidDataValueException $e) {
			self::assertSame('bayesian_service_parameter_invalid', $e->getErrorCode());
			self::assertStringContainsString("'text'", $e->getErrorMessage());
		}
	}

	public function testArrayActionParameterIsRejected()
	{
		$service = $this->trainedService();
		try {
			$service->runService(['action' => ['classify'], 'text' => 'cheap']);
			self::fail('expected exception');
		} catch (\Prado\Exceptions\TInvalidDataValueException $e) {
			self::assertSame('bayesian_service_parameter_invalid', $e->getErrorCode());
			self::assertStringContainsString("'action'", $e->getErrorMessage());
		}
	}

	public function testArrayCategoryParameterIsRejected()
	{
		$service = $this->trainedService();
		try {
			$service->runService(['text' => 'cheap', 'category' => ['spam']]);
			self::fail('expected exception');
		} catch (\Prado\Exceptions\TInvalidDataValueException $e) {
			self::assertSame('bayesian_service_parameter_invalid', $e->getErrorCode());
		}
	}

	public function testNestedCandidatesAreRejected()
	{
		$service = $this->trainedService();
		try {
			$service->runService(['action' => 'recommend', 'context' => ['cheap'], 'candidates' => [['a']]]);
			self::fail('expected exception');
		} catch (\Prado\Exceptions\TInvalidDataValueException $e) {
			self::assertSame('bayesian_service_parameter_invalid', $e->getErrorCode());
			self::assertStringContainsString("'candidates'", $e->getErrorMessage());
		}
	}

	public function testNestedContextIsRejected()
	{
		$service = $this->trainedService();
		try {
			$service->runService(['action' => 'recommend', 'context' => [['cheap']], 'candidates' => ['a']]);
			self::fail('expected exception');
		} catch (\Prado\Exceptions\TInvalidDataValueException $e) {
			self::assertSame('bayesian_service_parameter_invalid', $e->getErrorCode());
			self::assertStringContainsString("'context'", $e->getErrorMessage());
		}
	}

	public function testScalarContextAndCandidatesAreAccepted()
	{
		$service = $this->trainedService();
		$service->getRecommender()->setPositiveCategory('spam');
		$response = $service->runService(['action' => 'recommend', 'context' => 'cheap', 'candidates' => 'offer']);
		self::assertSame(['offer'], array_keys($response['scores']));
	}

	public function testTextLongerThanMaxLengthIsRejected()
	{
		$service = $this->trainedService();
		self::assertSame(65536, $service->getMaxTextLength());
		$service->setMaxTextLength(10);
		try {
			$service->runService(['text' => str_repeat('a', 11)]);
			self::fail('expected exception');
		} catch (\Prado\Exceptions\TInvalidDataValueException $e) {
			self::assertSame('bayesian_service_text_too_long', $e->getErrorCode());
			self::assertStringContainsString('10', $e->getErrorMessage());
		}
		// Exactly the limit is still accepted.
		$response = $service->runService(['text' => str_repeat('a', 10)]);
		self::assertArrayHasKey('category', $response);
	}

	public function testMaxTextLengthZeroDisablesLimit()
	{
		$service = $this->trainedService();
		$service->setMaxTextLength(0);
		self::assertSame(0, $service->getMaxTextLength());
		$response = $service->runService(['text' => str_repeat('cheap ', 20000)]);
		self::assertSame('spam', $response['category']);
	}

	public function testNegativeMaxTextLengthIsClampedToZero()
	{
		$service = new TBayesianService();
		$service->setMaxTextLength(-5);
		self::assertSame(0, $service->getMaxTextLength());
	}

	public function testUntrainedClassifierThrowsNotTrained()
	{
		$service = new TBayesianService();
		$service->setClassifier(new TNaiveBayesClassifier());
		try {
			$service->runService(['text' => 'anything']);
			self::fail('expected exception');
		} catch (\Prado\Exceptions\TInvalidOperationException $e) {
			self::assertSame('bayesian_classifier_not_trained', $e->getErrorCode());
		}
	}

	public function testEncodeEmitsScoresAsJsonObjectWithNumericKeys()
	{
		$service = new EncodingBayesianService();
		$classifier = new TNaiveBayesClassifier();
		$classifier->trainOne('liked', ['2024', 'shoes']);
		$classifier->trainOne('ignored', ['1999', 'hat']);
		$service->setClassifier($classifier);
		$result = $service->runService(['action' => 'recommend', 'context' => ['shoes'], 'candidates' => ['2024', '1999']]);
		// PHP has turned the numeric candidate keys into ints...
		self::assertSame([2024, 1999], array_keys($result['scores']));
		$json = $service->encodePublic($result);
		$decoded = json_decode($json, true);
		self::assertIsArray($decoded);
		self::assertStringStartsWith('{"scores":{', $json);
		// ...but the JSON must be an object keyed by the string form, not a list.
		self::assertStringContainsString('"2024":', $json);
		self::assertStringContainsString('"1999":', $json);
		$scores = json_decode($json)->scores;
		self::assertInstanceOf(\stdClass::class, $scores, 'scores decode as an object, not a list');
		self::assertTrue(property_exists($scores, '2024'));
		self::assertTrue(property_exists($scores, '1999'));
	}

	public function testEncodeEmitsEmptyScoresAsEmptyObject()
	{
		$service = new EncodingBayesianService();
		self::assertSame('{"scores":{}}', $service->encodePublic(['scores' => []]));
	}

	public function testEncodeLeavesNonArrayScoresAlone()
	{
		$service = new EncodingBayesianService();
		self::assertSame('{"category":"spam","scores":"x"}', $service->encodePublic(['category' => 'spam', 'scores' => 'x']));
	}

	public function testEncodeSurvivesInvalidUtf8InCategoryName()
	{
		$service = new EncodingBayesianService();
		$json = $service->encodePublic(['category' => "caf\xE9", 'scores' => ["caf\xE9" => 1.0]]);
		self::assertNotSame('', $json);
		self::assertNotSame('{"error":"encode_failed"}', $json);
		$decoded = json_decode($json, true);
		self::assertIsArray($decoded);
		self::assertSame(JSON_ERROR_NONE, json_last_error());
		self::assertStringStartsWith('caf', $decoded['category']);
		self::assertCount(1, $decoded['scores']);
	}

	public function testEncodePreservesZeroFraction()
	{
		$service = new EncodingBayesianService();
		self::assertSame('{"scores":{"a":1.0,"b":0.0}}', $service->encodePublic(['scores' => ['a' => 1.0, 'b' => 0.0]]));
	}

	public function testModuleIDRoundTripsAndEmptyBecomesNull()
	{
		$service = new TBayesianService();
		self::assertNull($service->getModuleID());
		$service->setModuleID('bayesian');
		self::assertSame('bayesian', $service->getModuleID());
		$service->setModuleID('');
		self::assertNull($service->getModuleID());
		$service->setModuleID(null);
		self::assertNull($service->getModuleID());
	}

	public function testRecommendWithoutContextUsesCandidatesAlone()
	{
		$service = $this->trainedService();
		$service->getRecommender()->setPositiveCategory('spam');
		$response = $service->runService(['action' => 'recommend', 'candidates' => ['cheap offer', 'meeting']]);
		self::assertSame(['cheap offer', 'meeting'], array_keys($response['scores']));
		self::assertGreaterThan($response['scores']['meeting'], $response['scores']['cheap offer']);
	}

	public function testRecommendWithUnknownPositiveCategoryThrows()
	{
		$service = $this->trainedService();
		// The default positive category 'liked' is not one of the classifier's spam/ham labels.
		try {
			$service->runService(['action' => 'recommend', 'candidates' => ['cheap offer']]);
			self::fail('expected exception');
		} catch (\Prado\Exceptions\TInvalidOperationException $e) {
			self::assertSame('bayesian_recommender_category_unknown', $e->getErrorCode());
		}
	}

	public function testResolverReturnsNullWithoutApplication()
	{
		$service = new NoAppBayesianService();
		try {
			$service->getClassifier();
			self::fail('expected exception');
		} catch (TConfigurationException $e) {
			self::assertSame('bayesian_service_classifier_missing', $e->getErrorCode());
		}
	}

	// ---- live-application tests -------------------------------------------------------------

	/**
	 * Installs an unbuffered, header-recording response on the shared test application and
	 * fills the request with the given parameters.
	 * @param array<string, mixed> $params
	 */
	private function prepareLiveRequest(array $params): RecordingHttpResponse
	{
		$app = BayesianTestApplication::get();
		$response = new RecordingHttpResponse();
		$response->setBufferOutput(false);
		$app->setResponse($response);
		$request = $app->getRequest();
		$request->clear();
		foreach ($params as $key => $value) {
			$request->add($key, $value);
		}
		return $response;
	}

	/** Runs the service against the live request and returns [decoded JSON body, response]. */
	private function runLive(TBayesianService $service, array $params): array
	{
		$response = $this->prepareLiveRequest($params);
		ob_start();
		try {
			$service->run();
		} finally {
			$body = ob_get_clean();
		}
		self::assertIsString($body);
		$decoded = json_decode($body, true);
		self::assertIsArray($decoded, 'body is JSON: ' . $body);
		return [$decoded, $response];
	}

	public function testRunWritesJsonClassificationWithHeaders()
	{
		[$body, $response] = $this->runLive($this->trainedService(), ['text' => 'cheap click free', 'category' => 'spam']);
		self::assertSame(200, $response->getStatusCode());
		self::assertSame('application/json', $response->getContentType());
		self::assertSame('UTF-8', $response->getCharset());
		self::assertContains('X-Content-Type-Options: nosniff', $response->headers);
		self::assertContains('Content-Type: application/json;charset=UTF-8', $response->headers);
		self::assertSame('spam', $body['category']);
		self::assertTrue($body['isSpam']);
		self::assertEqualsWithDelta(1.0, array_sum($body['scores']), 1e-9);
	}

	public function testRunIgnoresUnrecognizedRequestParameters()
	{
		[$body, $response] = $this->runLive($this->trainedService(), ['text' => 'meeting report', 'other' => 'x', 'action' => 'classify']);
		self::assertSame(200, $response->getStatusCode());
		self::assertSame('ham', $body['category']);
		self::assertArrayNotHasKey('isSpam', $body);
	}

	public function testRunMissingTextIs400()
	{
		[$body, $response] = $this->runLive($this->trainedService(), []);
		self::assertSame(400, $response->getStatusCode());
		self::assertSame('bayesian_service_text_required', $body['error']);
		self::assertArrayHasKey('message', $body);
	}

	public function testRunArrayParameterIs400()
	{
		[$body, $response] = $this->runLive($this->trainedService(), ['text' => ['x']]);
		self::assertSame(400, $response->getStatusCode());
		self::assertSame('bayesian_service_parameter_invalid', $body['error']);
	}

	public function testRunUnknownActionIs400()
	{
		[$body, $response] = $this->runLive($this->trainedService(), ['action' => 'wat', 'text' => 'x']);
		self::assertSame(400, $response->getStatusCode());
		self::assertSame('bayesian_service_action_unknown', $body['error']);
	}

	public function testRunTextTooLongIs413()
	{
		$service = $this->trainedService();
		$service->setMaxTextLength(5);
		[$body, $response] = $this->runLive($service, ['text' => 'cheap offer']);
		self::assertSame(413, $response->getStatusCode());
		self::assertSame('bayesian_service_text_too_long', $body['error']);
	}

	public function testRunUntrainedClassifierIs503()
	{
		$service = new TBayesianService();
		$service->setClassifier(new TNaiveBayesClassifier());
		[$body, $response] = $this->runLive($service, ['text' => 'anything']);
		self::assertSame(503, $response->getStatusCode());
		self::assertSame('bayesian_classifier_not_trained', $body['error']);
	}

	public function testRunRecommendWritesScoresObject()
	{
		$response = $this->prepareLiveRequest(['action' => 'recommend', 'context' => ['cheap'], 'candidates' => ['offer', 'meeting']]);
		$service = $this->trainedService();
		$service->getRecommender()->setPositiveCategory('spam');
		ob_start();
		try {
			$service->run();
		} finally {
			$json = ob_get_clean();
		}
		self::assertSame(200, $response->getStatusCode());
		self::assertStringStartsWith('{"scores":{', $json);
		$body = json_decode($json, true);
		self::assertSame(['offer', 'meeting'], array_keys($body['scores']));
	}

	public function testResolvesClassifierFromLiveApplicationModuleAutoDetect()
	{
		$app = BayesianTestApplication::get();
		$module = new TBayesianModule();
		$classifier = new TNaiveBayesClassifier();
		$module->setClassifier($classifier);
		$id = 'bayesianAuto' . uniqid();
		$app->setModule($id, $module);
		$service = new TBayesianService();
		self::assertSame($classifier, $service->getClassifier());
	}

	public function testResolvesClassifierFromLiveApplicationModuleByID()
	{
		$app = BayesianTestApplication::get();
		$module = new TBayesianModule();
		$classifier = new TNaiveBayesClassifier();
		$module->setClassifier($classifier);
		$id = 'bayesianById' . uniqid();
		$app->setModule($id, $module);
		$service = new TBayesianService();
		$service->setModuleID($id);
		self::assertSame($classifier, $service->getClassifier());
	}

	public function testModuleIDPointingAtNonBayesianModuleThrows()
	{
		$app = BayesianTestApplication::get();
		$id = 'notBayesian' . uniqid();
		$app->setModule($id, new \Prado\Data\TDataSourceConfig());
		$service = new TBayesianService();
		$service->setModuleID($id);
		try {
			$service->getClassifier();
			self::fail('expected exception');
		} catch (TConfigurationException $e) {
			self::assertSame('bayesian_service_classifier_missing', $e->getErrorCode());
		}
	}

	public function testModuleIDPointingAtMissingModuleThrows()
	{
		BayesianTestApplication::get();
		$service = new TBayesianService();
		$service->setModuleID('definitelyMissing' . uniqid());
		try {
			$service->getClassifier();
			self::fail('expected exception');
		} catch (TConfigurationException $e) {
			self::assertSame('bayesian_service_classifier_missing', $e->getErrorCode());
		}
	}
	// ---- input limits -----------------------------------------------------------------------

	public function testRecommendCandidateCountIsCapped()
	{
		$service = $this->trainedService();
		$service->getRecommender()->setPositiveCategory('spam');
		self::assertSame(100, $service->getMaxCandidates());
		$service->setMaxCandidates(2);
		try {
			$service->runService(['action' => 'recommend', 'candidates' => ['a', 'b', 'c']]);
			self::fail('expected exception');
		} catch (\Prado\Exceptions\TInvalidDataValueException $e) {
			self::assertSame('bayesian_service_candidates_too_many', $e->getErrorCode());
			self::assertStringContainsString('2', $e->getErrorMessage());
		}
		// Exactly the cap is accepted; repeats count as sent, since they arrive that way.
		$response = $service->runService(['action' => 'recommend', 'candidates' => ['cheap', 'offer']]);
		self::assertCount(2, $response['scores']);
		$service->setMaxCandidates(0);
		self::assertSame(0, $service->getMaxCandidates());
		$response = $service->runService(['action' => 'recommend', 'candidates' => ['a', 'b', 'c', 'd']]);
		self::assertCount(4, $response['scores']);
		$service->setMaxCandidates(-3);
		self::assertSame(0, $service->getMaxCandidates());
	}

	public function testRecommendContextAndCandidatesHonourMaxTextLength()
	{
		$service = $this->trainedService();
		$service->getRecommender()->setPositiveCategory('spam');
		$service->setMaxTextLength(10);
		try {
			$service->runService(['action' => 'recommend', 'context' => ['cheap', 'offers'], 'candidates' => ['a']]);
			self::fail('the joined context is 12 bytes');
		} catch (\Prado\Exceptions\TInvalidDataValueException $e) {
			self::assertSame('bayesian_service_text_too_long', $e->getErrorCode());
			self::assertStringContainsString("'context'", $e->getErrorMessage());
		}
		try {
			$service->runService(['action' => 'recommend', 'context' => ['cheap'], 'candidates' => ['a', str_repeat('b', 11)]]);
			self::fail('one candidate is 11 bytes');
		} catch (\Prado\Exceptions\TInvalidDataValueException $e) {
			self::assertSame('bayesian_service_text_too_long', $e->getErrorCode());
			self::assertStringContainsString("'candidates'", $e->getErrorMessage());
		}
		$response = $service->runService(['action' => 'recommend', 'context' => ['cheap'], 'candidates' => [str_repeat('b', 10)]]);
		self::assertCount(1, $response['scores']);
	}

	public function testRunTooManyCandidatesIs413()
	{
		$service = $this->trainedService();
		$service->setMaxCandidates(1);
		[$body, $response] = $this->runLive($service, ['action' => 'recommend', 'candidates' => ['a', 'b']]);
		self::assertSame(413, $response->getStatusCode());
		self::assertSame('bayesian_service_candidates_too_many', $body['error']);
	}

	// ---- permissions --------------------------------------------------------------------------

	public function testServiceDeclaresItsPermissions()
	{
		$service = new TBayesianService();
		self::assertInstanceOf(\Prado\Security\Permissions\IPermissions::class, $service);
		$events = $service->getPermissions(null);
		self::assertCount(3, $events);
		$byName = [];
		foreach ($events as $event) {
			self::assertInstanceOf(\Prado\Security\Permissions\TPermissionEvent::class, $event);
			$byName[$event->getName()] = $event->getEvents();
		}
		self::assertSame(['dyclassify'], array_map('strtolower', $byName[TBayesianService::PERM_CLASSIFY]));
		self::assertSame(['dyrecommend'], array_map('strtolower', $byName[TBayesianService::PERM_RECOMMEND]));
		self::assertSame(['dytag'], array_map('strtolower', $byName[TBayesianService::PERM_TAG]));
	}

	public function testADeniedPermissionRefusesTheActionBeforeAnyWork()
	{
		$service = $this->trainedService();
		$behavior = new DenyingBayesianBehavior();
		$service->attachBehavior('deny', $behavior);
		try {
			$service->runService(['text' => 'cheap']);
			self::fail('expected the classify permission to refuse');
		} catch (\Prado\Exceptions\THttpException $e) {
			self::assertSame('bayesian_service_permission_denied', $e->getErrorCode());
			self::assertSame(401, $e->getStatusCode(), 'no application user: the caller is anonymous');
			self::assertStringContainsString(TBayesianService::PERM_CLASSIFY, $e->getErrorMessage());
		}
		try {
			$service->runService(['action' => 'recommend', 'candidates' => ['x']]);
			self::fail('expected the recommend permission to refuse');
		} catch (\Prado\Exceptions\THttpException $e) {
			self::assertSame('bayesian_service_permission_denied', $e->getErrorCode());
			self::assertStringContainsString(TBayesianService::PERM_RECOMMEND, $e->getErrorMessage());
		}
		self::assertSame(['dyClassify', 'dyRecommend'], $behavior->seen);
		// Without the behavior the same requests go through: the check is opt-in.
		$service->detachBehavior('deny');
		self::assertSame('spam', $service->runService(['text' => 'cheap'])['category']);
	}

	public function testRunReportsADeniedPermissionAsJson()
	{
		$service = $this->trainedService();
		$service->attachBehavior('deny', new DenyingBayesianBehavior());
		[$body, $response] = $this->runLive($service, ['text' => 'cheap']);
		self::assertSame(401, $response->getStatusCode());
		self::assertSame('bayesian_service_permission_denied', $body['error']);
	}

	// ---- authorization rules ------------------------------------------------------------------

	public function testAuthorizationRulesAreEmptyByDefault()
	{
		$service = new TBayesianService();
		$rules = $service->getAuthorizationRules();
		self::assertInstanceOf(\Prado\Security\TAuthorizationRuleCollection::class, $rules);
		self::assertCount(0, $rules);
		self::assertSame($rules, $service->getAuthorizationRules());
	}

	public function testInitReadsRulesFromXmlConfiguration()
	{
		$xml = new \Prado\Xml\TXmlDocument();
		$xml->loadFromString('<service id="bayesian" class="TBayesianService"><authorization><allow roles="editor" verb="post" /><deny users="*" /></authorization></service>');
		$service = new TBayesianService();
		$service->init($xml);
		$rules = $service->getAuthorizationRules();
		self::assertCount(2, $rules);
		self::assertSame('allow', $rules->itemAt(0)->getAction());
		self::assertSame(['editor'], $rules->itemAt(0)->getRoles());
		self::assertSame('post', $rules->itemAt(0)->getVerb());
		self::assertSame('deny', $rules->itemAt(1)->getAction());
		self::assertTrue($rules->itemAt(1)->getEveryoneApplied(), 'users="*" applies to everyone');
	}

	public function testInitReadsRulesFromPhpConfiguration()
	{
		$service = new TBayesianService();
		$service->init([
			'class' => 'TBayesianService',
			'authorization' => [
				['action' => 'allow', 'users' => 'alice, bob', 'ips' => '10.0.0.*'],
				['action' => 'deny', 'users' => '*'],
			],
		]);
		$rules = $service->getAuthorizationRules();
		self::assertCount(2, $rules);
		self::assertSame(['alice', 'bob'], $rules->itemAt(0)->getUsers());
		self::assertSame(['10.0.0.*'], $rules->itemAt(0)->getIPRules());
		self::assertSame('deny', $rules->itemAt(1)->getAction());
	}

	public function testInitWithoutRulesLeavesTheServiceOpen()
	{
		$service = new TBayesianService();
		$service->init(null);
		self::assertCount(0, $service->getAuthorizationRules());
		$service->init(['class' => 'TBayesianService']);
		self::assertCount(0, $service->getAuthorizationRules());
	}

	public function testInitRejectsAnUnknownRuleTag()
	{
		$xml = new \Prado\Xml\TXmlDocument();
		$xml->loadFromString('<service id="bayesian"><authorization><permit roles="editor" /></authorization></service>');
		$service = new TBayesianService();
		try {
			$service->init($xml);
			self::fail('expected exception');
		} catch (TConfigurationException $e) {
			self::assertSame('bayesian_service_rule_invalid', $e->getErrorCode());
			self::assertStringContainsString('permit', $e->getErrorMessage());
		}
		$service = new TBayesianService();
		try {
			$service->init(['authorization' => [['action' => 'permit', 'roles' => 'editor']]]);
			self::fail('expected exception');
		} catch (TConfigurationException $e) {
			self::assertSame('bayesian_service_rule_invalid', $e->getErrorCode());
		}
	}

	/** Puts a user on the shared test application, or removes it when null. */
	private function setLiveUser(?\Prado\Security\IUser $user): void
	{
		$app = BayesianTestApplication::get();
		if ($user !== null) {
			$app->setUser($user);
			return;
		}
		$property = new \ReflectionProperty(\Prado\TApplication::class, '_user');
		$property->setValue($app, null);
	}

	private function liveUser(string $name, bool $guest, array $roles = []): \Prado\Security\TUser
	{
		$user = new \Prado\Security\TUser(new \Prado\Security\TUserManager());
		$user->setName($name);
		$user->setIsGuest($guest);
		$user->setRoles($roles);
		return $user;
	}

	/** A service that allows editors and denies everyone else. */
	private function restrictedService(): TBayesianService
	{
		$service = $this->trainedService();
		$service->getAuthorizationRules()->add(new \Prado\Security\TAuthorizationRule('allow', '', 'editor'));
		$service->getAuthorizationRules()->add(new \Prado\Security\TAuthorizationRule('deny', '*', ''));
		return $service;
	}

	public function testRulesRefuseAGuestWith401()
	{
		try {
			$this->setLiveUser($this->liveUser('Guest', true));
			[$body, $response] = $this->runLive($this->restrictedService(), ['text' => 'cheap']);
			self::assertSame(401, $response->getStatusCode());
			self::assertSame('bayesian_service_unauthorized', $body['error']);
			self::assertArrayNotHasKey('category', $body);
		} finally {
			$this->setLiveUser(null);
		}
	}

	public function testRulesRefuseAnAuthenticatedUserWithoutTheRoleWith403()
	{
		try {
			$this->setLiveUser($this->liveUser('carol', false, ['reader']));
			[$body, $response] = $this->runLive($this->restrictedService(), ['text' => 'cheap']);
			self::assertSame(403, $response->getStatusCode());
			self::assertSame('bayesian_service_unauthorized', $body['error']);
			self::assertStringContainsString('carol', $body['message']);
		} finally {
			$this->setLiveUser(null);
		}
	}

	public function testRulesAdmitAUserWithTheRole()
	{
		try {
			$this->setLiveUser($this->liveUser('dave', false, ['editor']));
			[$body, $response] = $this->runLive($this->restrictedService(), ['text' => 'cheap']);
			self::assertSame(200, $response->getStatusCode());
			self::assertSame('spam', $body['category']);
		} finally {
			$this->setLiveUser(null);
		}
	}

	public function testRulesWithoutAnApplicationUserFailClosed()
	{
		// Rules were configured but no authentication module put a user on the application:
		// refusing is the only safe answer, and the message says what is missing.
		$this->setLiveUser(null);
		[$body, $response] = $this->runLive($this->restrictedService(), ['text' => 'cheap']);
		self::assertSame(401, $response->getStatusCode());
		self::assertSame('bayesian_service_user_required', $body['error']);
	}

	public function testNoRulesMeansNoUserIsNeeded()
	{
		$this->setLiveUser(null);
		[$body, $response] = $this->runLive($this->trainedService(), ['text' => 'cheap']);
		self::assertSame(200, $response->getStatusCode());
		self::assertSame('spam', $body['category']);
	}

	public function testRulesApplyToRecommendToo()
	{
		try {
			$this->setLiveUser($this->liveUser('Guest', true));
			$service = $this->restrictedService();
			$service->getRecommender()->setPositiveCategory('spam');
			[$body, $response] = $this->runLive($service, ['action' => 'recommend', 'candidates' => ['cheap']]);
			self::assertSame(401, $response->getStatusCode());
			self::assertSame('bayesian_service_unauthorized', $body['error']);
		} finally {
			$this->setLiveUser(null);
		}
	}
	public function testClassifyReportsWhetherScoresAreCalibrated()
	{
		$service = $this->trainedService();
		$response = $service->runService(['text' => 'cheap click']);
		self::assertFalse($response['calibrated']);
		$classifier = $service->getClassifier();
		self::assertInstanceOf(TNaiveBayesClassifier::class, $classifier);
		$classifier->setCalibration(new \Belisoful\Prado\Util\Bayesian\Calibration\TTemperatureScaling(2.0));
		$response = $service->runService(['text' => 'cheap click']);
		self::assertTrue($response['calibrated']);
		self::assertEqualsWithDelta(1.0, array_sum($response['scores']), 1e-9);
	}

	// ---- tagging ------------------------------------------------------------------------------

	private function taggedService(): TBayesianService
	{
		$tagger = new \Belisoful\Prado\Util\Bayesian\TBayesianTagger();
		$tagger->train(['php', 'security'], 'validate every request parameter before the sql query');
		$tagger->train(['php'], 'composer autoloads the classes');
		$tagger->train(['cooking'], 'simmer the sauce for twenty minutes');
		$tagger->train([], 'the meeting moved to tuesday');
		$service = new TBayesianService();
		$service->setClassifier($tagger->getClassifier());
		return $service;
	}

	public function testTagActionReturnsTagsAboveTheThreshold()
	{
		$service = $this->taggedService();
		self::assertSame(0.5, $service->getTagThreshold());
		self::assertSame(0, $service->getMaxTags());
		$response = $service->runService(['action' => 'tag', 'text' => 'escape the sql query parameters']);
		self::assertSame(['php', 'security'], array_keys($response['tags']));
		self::assertFalse($response['calibrated']);
		self::assertInstanceOf(\Belisoful\Prado\Util\Bayesian\TBayesianTagger::class, $service->getTagger());
		self::assertSame($service->getTagger(), $service->getTagger());
	}

	public function testTagThresholdAndMaxTagsShapeTheBuiltTagger()
	{
		$service = $this->taggedService();
		$service->setTagThreshold(0.0);
		$service->setMaxTags(1);
		self::assertSame(0.0, $service->getTagThreshold());
		self::assertSame(1, $service->getMaxTags());
		$response = $service->runService(['action' => 'tag', 'text' => 'escape the sql query parameters']);
		self::assertSame(['php'], array_keys($response['tags']));
		self::assertSame(0.0, $service->getTagger()->getThreshold());
		self::assertSame(1, $service->getTagger()->getMaxTags());
		$service->setTagThreshold(NAN);
		self::assertSame(0.5, $service->getTagThreshold());
		$service->setTagThreshold(7.0);
		self::assertSame(1.0, $service->getTagThreshold());
		$service->setMaxTags(-1);
		self::assertSame(0, $service->getMaxTags());
	}

	public function testTagActionValidatesText()
	{
		$service = $this->taggedService();
		try {
			$service->runService(['action' => 'tag']);
			self::fail('expected exception');
		} catch (\Prado\Exceptions\TInvalidDataValueException $e) {
			self::assertSame('bayesian_service_text_required', $e->getErrorCode());
		}
		$service->setMaxTextLength(5);
		try {
			$service->runService(['action' => 'tag', 'text' => 'far too long']);
			self::fail('expected exception');
		} catch (\Prado\Exceptions\TInvalidDataValueException $e) {
			self::assertSame('bayesian_service_text_too_long', $e->getErrorCode());
		}
	}

	public function testTagActionNeedsATrainedTaggerAndAMultinomialClassifier()
	{
		$service = new TBayesianService();
		$service->setClassifier(new TNaiveBayesClassifier());
		try {
			$service->runService(['action' => 'tag', 'text' => 'anything']);
			self::fail('expected exception');
		} catch (\Prado\Exceptions\TInvalidOperationException $e) {
			self::assertSame('bayesian_classifier_not_trained', $e->getErrorCode());
		}
		$service = new TBayesianService();
		$service->setClassifier($this->createStub(\Belisoful\Prado\Util\Bayesian\Classifier\IBayesianClassifier::class));
		try {
			$service->getTagger();
			self::fail('expected exception');
		} catch (TConfigurationException $e) {
			self::assertSame('bayesian_tagger_classifier_invalid', $e->getErrorCode());
		}
	}

	public function testAnInjectedTaggerIsUsedAsIs()
	{
		$tagger = new \Belisoful\Prado\Util\Bayesian\TBayesianTagger();
		$tagger->train(['a'], 'alpha beta');
		$tagger->train([], 'gamma delta');
		$tagger->setThreshold(0.1);
		$service = new TBayesianService();
		$service->setTagger($tagger);
		self::assertSame($tagger, $service->getTagger());
		$response = $service->runService(['action' => 'tag', 'text' => 'alpha']);
		self::assertSame(['a'], array_keys($response['tags']));
	}

	public function testTagPermissionIsChecked()
	{
		$service = $this->taggedService();
		$service->attachBehavior('deny', new DenyingBayesianBehavior());
		try {
			$service->runService(['action' => 'tag', 'text' => 'sql']);
			self::fail('expected exception');
		} catch (\Prado\Exceptions\THttpException $e) {
			self::assertSame('bayesian_service_permission_denied', $e->getErrorCode());
			self::assertStringContainsString(TBayesianService::PERM_TAG, $e->getErrorMessage());
		}
	}

	public function testRunWritesTagsAsAJsonObject()
	{
		$service = $this->taggedService();
		[$body, $response] = $this->runLive($service, ['action' => 'tag', 'text' => 'escape the sql query parameters']);
		self::assertSame(200, $response->getStatusCode());
		self::assertSame(['php', 'security'], array_keys($body['tags']));
		self::assertFalse($body['calibrated']);
		$service->setTagThreshold(1.0);
		$service = $this->taggedService();
		$service->setTagThreshold(1.0);
		$response = $this->prepareLiveRequest(['action' => 'tag', 'text' => 'escape the sql query parameters']);
		ob_start();
		try {
			$service->run();
		} finally {
			$json = ob_get_clean();
		}
		self::assertStringStartsWith('{"tags":{}', $json, 'no tags is still an object');
	}
}
