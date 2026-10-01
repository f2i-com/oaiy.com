<?php
/**
 * Write or check the recorded fixtures of the protocol package (platform/protocol/relay/v1/fixtures/).
 *
 *   php tests/fixtures.php --check     verify the committed fixtures without changing them (the default)
 *   php tests/fixtures.php --write     record them again from the real relay code (new random values: commit the result)
 *   php tests/fixtures.php --write aokie     only the Aokie set (fixtures/aokie/); `pairing` only the pairing ceremony and sealed tokens
 *
 * The fixtures are recorded by driving the relay in temporary directories on loopback, like the test suite does: nothing
 * outside the temporary tree is touched. Sealed boxes and tokens are random, so a rewrite changes them; a check never
 * does.
 */

if (PHP_SAPI !== 'cli') {
    exit;
}
$needSodium = !extension_loaded('sodium');
if ($needSodium && getenv('OAIY_TEST_REEXEC') === false) {
    $env = getenv();
    $env['OAIY_TEST_REEXEC'] = '1';
    $env['OAIY_TEST_PHP_FLAGS'] = '-d extension=sodium';
    $proc = proc_open(array_merge([PHP_BINARY, '-d', 'extension=sodium'], $_SERVER['argv']), [0 => STDIN, 1 => STDOUT, 2 => STDERR], $pipes, null, $env);
    exit(is_resource($proc) ? proc_close($proc) : 2);
}
error_reporting(E_ALL);
ini_set('display_errors', '1');
set_error_handler(static function (int $no, string $msg, string $file, int $line): bool {
    if (!(error_reporting() & $no)) {
        return false;
    }
    throw new ErrorException($msg, 0, $no, $file, $line);
});

$testsDir = __DIR__;
require $testsDir . '/lib/Assert.php';
require $testsDir . '/lib/Tmp.php';
\OaiyTest\Tmp::init();
putenv('OAIY_TEST_CLOCK=' . \OaiyTest\Tmp::clockFile());
require $testsDir . '/prepend.php';
define('OAIY_RELAY', true);
require_once dirname($testsDir) . '/src/autoload.php';
foreach (glob($testsDir . '/lib/*.php') as $f) {
    require_once $f;
}

$mode = $argv[1] ?? '--check';
$which = $argv[2] ?? 'all';
if (!in_array($mode, ['--check', '--write'], true) || !in_array($which, ['all', 'pairing', 'aokie'], true)) {
    fwrite(STDERR, "usage: php tests/fixtures.php [--check|--write [all|pairing|aokie]]\n");
    exit(2);
}
$dir = \OaiyTest\Fixtures::dir();
$files = [];
$exit = 0;
try {
    if ($mode === '--write') {
        if ($which !== 'aokie') {
            $made = \OaiyTest\Fixtures::pairing();
            $files['pairing-ceremony.json'] = $made['ceremony'];
            $files['sealed-token.json'] = $made['sealed'];
        }
        foreach ($files as $name => $doc) {
            $bad = $name === 'sealed-token.json' ? \OaiyTest\Fixtures::checkSealed($doc) : \OaiyTest\Fixtures::checkCeremony($doc);
            if ($bad) {
                throw new RuntimeException("$name would not pass its own check: " . implode('; ', $bad));
            }
        }
        if (!is_dir($dir)) {
            mkdir($dir, 0755, true);
        }
        foreach ($files as $name => $doc) {
            file_put_contents($dir . '/' . $name, \OaiyTest\Fixtures::encode($doc));
            echo "wrote $name\n";
        }
        if ($which !== 'pairing') {
            $aokie = \OaiyTest\AokieFixtures::record();
            $bad = \OaiyTest\AokieFixtures::check(\OaiyTest\AokieFixtures::roundTrip($aokie));
            if ($bad) {
                throw new RuntimeException('the Aokie recording would not pass its own check: ' . implode('; ', $bad));
            }
            if (!is_dir(\OaiyTest\AokieFixtures::dir())) {
                mkdir(\OaiyTest\AokieFixtures::dir(), 0755, true);
            }
            foreach ($aokie as $name => $doc) {
                file_put_contents(\OaiyTest\AokieFixtures::dir() . '/' . $name, \OaiyTest\AokieFixtures::encode($doc));
                echo "wrote aokie/$name\n";
            }
        }
    } else {
        foreach (['pairing-ceremony.json' => 'checkCeremony', 'sealed-token.json' => 'checkSealed'] as $name => $check) {
            $raw = @file_get_contents($dir . '/' . $name);
            $doc = is_string($raw) ? json_decode($raw, true) : null;
            if (!is_array($doc)) {
                echo "MISSING $name\n";
                $exit = 1;
                continue;
            }
            $bad = \OaiyTest\Fixtures::$check($doc);
            if ($raw !== \OaiyTest\Fixtures::encode($doc)) {
                $bad[] = 'the file is not written the way the package writes it (indent, order, line endings)';
            }
            echo $bad ? "FAIL $name: " . implode('; ', $bad) . "\n" : "ok   $name\n";
            $exit = $exit || $bad ? 1 : 0;
        }
        $aokie = \OaiyTest\AokieFixtures::load();
        $bad = [];
        foreach (\OaiyTest\AokieFixtures::FILES as $name) {
            $raw = @file_get_contents(\OaiyTest\AokieFixtures::dir() . '/' . $name);
            if (!is_string($raw) || !is_array($aokie[$name])) {
                $bad[] = "aokie/$name is missing";
            } elseif ($raw !== \OaiyTest\AokieFixtures::encode(json_decode($raw, false))) {
                $bad[] = "aokie/$name is not written the way the package writes it";
            }
        }
        $bad = array_merge($bad, $bad ? [] : \OaiyTest\AokieFixtures::check($aokie));
        echo $bad ? 'FAIL aokie/: ' . implode('; ', $bad) . "\n" : "ok   aokie/ (" . count(\OaiyTest\AokieFixtures::FILES) . " files)\n";
        $exit = $exit || $bad ? 1 : 0;
    }
} finally {
    \OaiyTest\Tmp::cleanup();
}
exit($exit);
