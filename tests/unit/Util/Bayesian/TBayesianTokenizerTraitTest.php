<?php

use Prado\TComponent;
use Prado\Util\Bayesian\Tokenizer\IBayesianTokenizer;
use Prado\Util\Bayesian\Tokenizer\TBayesianTokenizerTrait;
use Prado\Util\Bayesian\Tokenizer\TWordTokenizer;

/** A tokenizer that takes the trait's defaults: no configuration properties at all. */
class BareTraitTokenizer extends TComponent implements IBayesianTokenizer
{
	use TBayesianTokenizerTrait;

	public function tokenize(string $text): array
	{
		return $text === '' ? [] : [$this->normalizeText($text)];
	}
}

/** A tokenizer whose setters cover every branch of the trait's type coercion. */
class TypedTraitTokenizer extends TComponent implements IBayesianTokenizer
{
	use TBayesianTokenizerTrait;
	private float $_ratio = 1.0;
	private $_label = 'none';
	private ?TWordTokenizer $_inner = null;

	public function tokenize(string $text): array
	{
		return $this->matchAll('/\w+/u', $text)[0] ?? [];
	}

	protected function getConfigProperties(): array
	{
		return ['Ratio', 'Label', 'Inner'];
	}

	public function getRatio(): float
	{
		return $this->_ratio;
	}

	public function setRatio(float $value): void
	{
		$this->_ratio = $value;
	}

	public function getLabel()
	{
		return $this->_label;
	}

	// Deliberately untyped, to exercise the "no declared type" coercion branch.
	public function setLabel($value): void
	{
		$this->_label = $value;
	}

	public function getInner(): ?TWordTokenizer
	{
		return $this->_inner;
	}

	public function setInner(?TWordTokenizer $value): void
	{
		$this->_inner = $value;
	}
}

class TBayesianTokenizerTraitTest extends PHPUnit\Framework\TestCase
{
	public function testDefaultConfigPropertiesAreEmpty()
	{
		$tokenizer = new BareTraitTokenizer();
		self::assertSame([], $tokenizer->exportConfig());
		// Importing anything is a no-op when no properties are declared.
		$tokenizer->importConfig(['anything' => 'ignored']);
		self::assertSame([], $tokenizer->exportConfig());
		self::assertSame(['abc'], $tokenizer->tokenize('ABC'));
	}

	public function testFloatSetterCoercesNumericStrings()
	{
		$tokenizer = new TypedTraitTokenizer();
		$tokenizer->importConfig(['ratio' => '2.5']);
		self::assertSame(2.5, $tokenizer->getRatio());
		$tokenizer->importConfig(['ratio' => 3]);
		self::assertSame(3.0, $tokenizer->getRatio());
	}

	public function testFloatSetterFallsBackToZeroForNonScalars()
	{
		$tokenizer = new TypedTraitTokenizer();
		$tokenizer->importConfig(['ratio' => ['nope']]);
		self::assertSame(0.0, $tokenizer->getRatio());
	}

	public function testUntypedSetterReceivesValueUnchanged()
	{
		$tokenizer = new TypedTraitTokenizer();
		$tokenizer->importConfig(['label' => ['a', 'b']]);
		self::assertSame(['a', 'b'], $tokenizer->getLabel());
	}

	public function testObjectTypedSetterReceivesValueUnchanged()
	{
		$tokenizer = new TypedTraitTokenizer();
		$inner = new TWordTokenizer();
		$tokenizer->importConfig(['inner' => $inner]);
		self::assertSame($inner, $tokenizer->getInner());
	}

	public function testNullableSetterAcceptsNull()
	{
		$tokenizer = new TypedTraitTokenizer();
		$tokenizer->setInner(new TWordTokenizer());
		$tokenizer->importConfig(['inner' => null]);
		self::assertNull($tokenizer->getInner());
	}

	public function testExportConfigReadsEveryDeclaredProperty()
	{
		$tokenizer = new TypedTraitTokenizer();
		$tokenizer->setRatio(0.25);
		$tokenizer->setLabel('x');
		$config = $tokenizer->exportConfig();
		self::assertSame(['ratio', 'label', 'inner'], array_keys($config));
		self::assertSame(0.25, $config['ratio']);
		self::assertSame('x', $config['label']);
		self::assertNull($config['inner']);
	}

	public function testMatchAllReturnsEmptyWhenNothingMatches()
	{
		$tokenizer = new TypedTraitTokenizer();
		self::assertSame([], $tokenizer->tokenize('...'));
		self::assertSame(['ab', 'cd'], $tokenizer->tokenize('ab cd'));
	}
}
