<?php
/**
 * Loaded with auto_prepend_file by the test harness only (and required by tests/run.php). It defines the one
 * constant that lets the relay's clock read a file. The release zip does not contain tests/, and a request header
 * cannot define a PHP constant, so no configuration or header can leave a clock hook live on a real host.
 */
$oaiyClockFile = getenv('OAIY_TEST_CLOCK');
if (is_string($oaiyClockFile) && $oaiyClockFile !== '' && !defined('OAIY_TEST_CLOCK_FILE')) {
    define('OAIY_TEST_CLOCK_FILE', $oaiyClockFile);
}
unset($oaiyClockFile);
