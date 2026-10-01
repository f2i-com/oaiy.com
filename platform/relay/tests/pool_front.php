<?php
declare(strict_types=1);

/**
 * A pool of W workers in front of W single-threaded `php -S` backends that share one relay data directory: the one property of a
 * PHP-FPM host that decides what a hold costs. A connection is served by exactly one free backend for its whole life, and when every
 * backend is busy the next connection WAITS its turn (the accept queue), as it does with pm.max_children = W.
 *
 *   php pool_front.php <listen port> <backend port>[,<backend port>...]
 *
 * It listens on 127.0.0.1 only and connects to 127.0.0.1 only. The tests start it (Pool::start) and stop it with their other servers.
 */

if (PHP_SAPI !== 'cli') {
    exit(1);
}
$listen = (int)($argv[1] ?? 0);
$free = array_map('intval', explode(',', (string)($argv[2] ?? '')));
if ($listen <= 0 || $free === [] || in_array(0, $free, true)) {
    fwrite(STDERR, "usage: php pool_front.php <listen port> <backend port>[,<backend port>...]\n");
    exit(2);
}
// A backlog of 4096, as a real front has: with PHP's default of 32 a burst of fifty connections overflows the queue, the operating
// system drops the extra connection attempts and a client retries a second later, which would show as a slow answer that is the
// harness's and not the relay's.
$server = @stream_socket_server('tcp://127.0.0.1:' . $listen, $errno, $errstr, STREAM_SERVER_BIND | STREAM_SERVER_LISTEN, stream_context_create(['socket' => ['backlog' => 4096]]));
if ($server === false) {
    fwrite(STDERR, "cannot listen on 127.0.0.1:$listen: $errstr\n");
    exit(1);
}
stream_set_blocking($server, false);

/** @var list<array{c:resource,buf:string}> $waiting clients that have connected and have no backend yet, oldest first */
$waiting = [];
/** @var array<int,array{c:resource,b:resource,port:int,toB:string,toC:string,bDone:bool}> $pairs */
$pairs = [];
$nextId = 0;

$close = static function (int $id) use (&$pairs, &$free): void {
    $p = $pairs[$id];
    @fclose($p['c']);
    @fclose($p['b']);
    $free[] = $p['port'];
    unset($pairs[$id]);
};

while (true) {
    // Read what the waiting clients have sent (a client is served once its request headers are in, as a front does; a connection that
    // has said nothing yet does not take a worker), and forget the ones that have gone.
    foreach ($waiting as $i => $w) {
        $d = @fread($w['c'], 65536);
        if ($d !== '' && $d !== false) {
            $waiting[$i]['buf'] .= $d;
        } elseif (feof($w['c'])) {
            @fclose($w['c']);
            unset($waiting[$i]);
        }
    }
    // Give the oldest waiting client with a whole request the oldest free backend.
    foreach ($waiting as $i => $w) {
        if ($free === []) {
            break;
        }
        if (strpos($w['buf'], "\r\n\r\n") === false) {
            continue;
        }
        unset($waiting[$i]);
        $port = array_shift($free);
        $b = @stream_socket_client('tcp://127.0.0.1:' . $port, $e1, $e2, 2.0);
        if ($b === false) {
            $free[] = $port;
            @fclose($w['c']);
            continue;
        }
        stream_set_blocking($b, false);
        $pairs[$nextId++] = ['c' => $w['c'], 'b' => $b, 'port' => $port, 'toB' => $w['buf'], 'toC' => '', 'bDone' => false];
    }
    // Only the listening socket and the sockets of the pairs are selected on: a select over more than a few dozen sockets fails on some
    // builds of PHP for Windows (FD_SETSIZE), so the waiting clients are read without one, at the top of every turn.
    $read = [$server];
    $write = [];
    foreach ($pairs as $p) {
        $read[] = $p['c'];
        $read[] = $p['b'];
        if ($p['toB'] !== '') {
            $write[] = $p['b'];
        }
        if ($p['toC'] !== '') {
            $write[] = $p['c'];
        }
    }
    $except = null;
    if (@stream_select($read, $write, $except, 0, $waiting === [] ? 20000 : 3000) === false) {
        usleep(10000);
        continue;
    }
    foreach ($read as $s) {
        if ($s === $server) {
            while (($c = @stream_socket_accept($server, 0)) !== false) {
                stream_set_blocking($c, false);
                $waiting[] = ['c' => $c, 'buf' => ''];
            }
            continue;
        }
        foreach ($pairs as $id => $p) {
            if ($p['c'] === $s) {
                $d = @fread($s, 65536);
                if ($d === '' || $d === false) {
                    if (feof($s)) {
                        $close($id); // the client hung up: the backend's request is over for the pool, as a proxy would close upstream
                    }
                } else {
                    $pairs[$id]['toB'] .= $d;
                }
                continue 2;
            }
            if ($p['b'] === $s) {
                $d = @fread($s, 65536);
                if ($d === '' || $d === false) {
                    if (feof($s)) {
                        $pairs[$id]['bDone'] = true;
                    }
                } else {
                    $pairs[$id]['toC'] .= $d;
                }
                continue 2;
            }
        }
    }
    foreach ($write as $s) {
        foreach ($pairs as $id => $p) {
            if ($p['b'] === $s && $p['toB'] !== '') {
                $n = @fwrite($s, $p['toB']);
                if ($n === false) {
                    $close($id);
                } else {
                    $pairs[$id]['toB'] = (string)substr($p['toB'], $n);
                }
                continue 2;
            }
            if ($p['c'] === $s && $p['toC'] !== '') {
                $n = @fwrite($s, $p['toC']);
                if ($n === false) {
                    $close($id);
                } else {
                    $pairs[$id]['toC'] = (string)substr($p['toC'], $n);
                }
                continue 2;
            }
        }
    }
    foreach ($pairs as $id => $p) {
        if ($p['bDone'] && $p['toC'] === '') {
            $close($id); // the backend answered and closed: everything it said has reached the client
        }
    }
}
