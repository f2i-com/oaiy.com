<?php
/**
 * Loaded with auto_prepend_file by the test harness only (and required by tests/run.php). It defines the constants
 * that let the relay's clock read a file and let a test point a server at its own data directory. The release zip
 * does not contain tests/, and a request header cannot define a PHP constant, so no configuration or header can
 * leave a hook live on a real host.
 */
$oaiyClockFile = getenv('OAIY_TEST_CLOCK');
if (is_string($oaiyClockFile) && $oaiyClockFile !== '' && !defined('OAIY_TEST_CLOCK_FILE')) {
    define('OAIY_TEST_CLOCK_FILE', $oaiyClockFile);
}
$oaiyDataDir = getenv('OAIY_TEST_DATA');
if (is_string($oaiyDataDir) && $oaiyDataDir !== '' && !defined('OAIY_TEST_DATA_DIR')) {
    define('OAIY_TEST_DATA_DIR', $oaiyDataDir);
}
unset($oaiyClockFile, $oaiyDataDir);
