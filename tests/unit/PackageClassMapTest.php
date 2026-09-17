<?php

/**
 * Guards the two files composer.json registers system-wide through `extra.prado`:
 * `config/prado-bayesian-classes.json` (the Prado3 short-name class map) and
 * `config/errorMessages.txt` (the error codes the extension throws).
 *
 * The phpunit bootstrap registers both the way TApplicationConfiguration does for an installed
 * extension, so a short name that does not resolve or a code without a message shows up here
 * rather than in an application's error page.
 */
class PackageClassMapTest extends PHPUnit\Framework\TestCase
{
	/**
	 * @return array<string, string> The short name => FQN map.
	 */
	private function classMap(): array
	{
		$json = (string) file_get_contents(__DIR__ . '/../../config/prado-bayesian-classes.json');
		$map = json_decode($json, true);
		self::assertIsArray($map);
		return $map;
	}

	/**
	 * The map holds interfaces and a trait as well as classes; class_exists() ignores those.
	 * @param string $name The type name.
	 * @return bool Whether it resolves.
	 */
	private function typeExists(string $name): bool
	{
		return class_exists($name) || interface_exists($name) || trait_exists($name);
	}

	/**
	 * @return array<string, string> Every declared type under src/, FQN => file.
	 */
	private function sourceTypes(): array
	{
		$root = dirname(__DIR__, 2) . '/src';
		$types = [];
		foreach (new RecursiveIteratorIterator(new RecursiveDirectoryIterator($root)) as $file) {
			if ($file->getExtension() !== 'php') {
				continue;
			}
			$relative = substr($file->getPathname(), strlen($root) + 1, -4);
			$types['Belisoful\\Prado\\' . str_replace('/', '\\', $relative)] = $file->getPathname();
		}
		return $types;
	}

	public function testClassMapIsAValidNonEmptyObject(): void
	{
		$map = $this->classMap();
		self::assertNotEmpty($map);
		foreach ($map as $short => $fqn) {
			self::assertIsString($short);
			self::assertIsString($fqn);
			self::assertMatchesRegularExpression('/^[TI][A-Z][A-Za-z]+$/', $short, "{$short} is not a T*/I* short name");
		}
	}

	public function testEveryMappedFqnAutoloads(): void
	{
		foreach ($this->classMap() as $short => $fqn) {
			self::assertTrue($this->typeExists($fqn), "Mapped FQN does not autoload: {$fqn}");
			self::assertSame($short, (new ReflectionClass($fqn))->getShortName(), "{$short} maps to a differently named type");
		}
	}

	public function testEveryShortNameResolvesThroughTheRegisteredClassMap(): void
	{
		foreach ($this->classMap() as $short => $fqn) {
			$resolved = \Prado\Prado::usingClass($short);
			self::assertSame($fqn, $resolved, "Short name {$short} does not resolve to {$fqn}");
		}
	}

	public function testEverySourceTypeIsInTheMap(): void
	{
		$mapped = array_values($this->classMap());
		foreach ($this->sourceTypes() as $fqn => $file) {
			self::assertContains($fqn, $mapped, "{$fqn} (" . basename($file) . ') is missing from config/prado-bayesian-classes.json');
		}
	}

	public function testEveryMappedTypeHasASourceFile(): void
	{
		$types = $this->sourceTypes();
		foreach ($this->classMap() as $short => $fqn) {
			self::assertArrayHasKey($fqn, $types, "{$fqn} is mapped but has no file under src/");
		}
	}

	/**
	 * @return array<string, true> The codes defined in config/errorMessages.txt.
	 */
	private function definedCodes(): array
	{
		$defined = [];
		foreach (file(__DIR__ . '/../../config/errorMessages.txt') ?: [] as $line) {
			if (preg_match('/^\s*(bayesian_[a-z0-9_]+)\s*=/', $line, $match)) {
				self::assertArrayNotHasKey($match[1], $defined, "Error code {$match[1]} is defined twice");
				$defined[$match[1]] = true;
			}
		}
		return $defined;
	}

	/**
	 * @return array<string, string> Every `bayesian_*` code named in the source, code => file.
	 */
	private function usedCodes(): array
	{
		$used = [];
		foreach ($this->sourceTypes() as $file) {
			preg_match_all("/'(bayesian_[a-z0-9]+_[a-z0-9_]+)'/", (string) file_get_contents($file), $matches);
			foreach ($matches[1] as $code) {
				$used[$code] = $file;
			}
		}
		return $used;
	}

	public function testEveryErrorCodeUsedInSourceIsDefined(): void
	{
		$defined = $this->definedCodes();
		$used = $this->usedCodes();
		self::assertNotEmpty($used, 'No error codes were found in the source.');
		foreach ($used as $code => $file) {
			self::assertArrayHasKey($code, $defined, "Error code '{$code}' is raised in " . basename($file) . ' but not defined in config/errorMessages.txt');
		}
	}

	public function testEveryDefinedErrorCodeIsUsedInSource(): void
	{
		$used = $this->usedCodes();
		foreach (array_keys($this->definedCodes()) as $code) {
			self::assertArrayHasKey($code, $used, "Error code '{$code}' is defined in config/errorMessages.txt but never raised");
		}
	}

	public function testEveryDefinedErrorCodeResolvesToAMessage(): void
	{
		foreach (array_keys($this->definedCodes()) as $code) {
			$exception = new \Prado\Exceptions\TInvalidDataValueException($code, 'a', 'b', 'c');
			self::assertNotSame($code, $exception->getMessage(), "Error code {$code} did not resolve to its message text");
			self::assertStringNotContainsString('{0}', $exception->getMessage(), "Error code {$code} left a placeholder unfilled");
		}
	}

	public function testEveryErrorCodeIsDocumented(): void
	{
		$documentation = (string) file_get_contents(__DIR__ . '/../../docs/configuration.md');
		foreach (array_keys($this->definedCodes()) as $code) {
			self::assertStringContainsString('`' . $code . '`', $documentation, "Error code {$code} is not listed in docs/configuration.md");
		}
	}
}
