"""The eGPU plugin's looks (plugin/plugin.py): Docker's engine asked on its own socket, with short waits, and the card's
and Docker's looks kept, answered at once, looked at again on a thread of their own, kept for the plugin's next start,
and looked at now on `setup.status {refresh: true}`. No Docker, card or system_profiler needed: fake sockets and looks.

    python3 tools/tinygpu/plugin_test.py
"""
from __future__ import annotations

import importlib.util, io, json, os, pathlib, socket, sys, tempfile, threading, time, unittest
from unittest import mock

HERE = pathlib.Path(__file__).resolve().parent
DATA = tempfile.mkdtemp(prefix="egpu-plugin-test-")
os.environ["OAIY_PLUGIN_DATA_DIR"] = DATA
spec = importlib.util.spec_from_file_location("egpu_plugin", HERE / "plugin" / "plugin.py")
plugin = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plugin)


class FakeEngine:
    """A unix socket that answers each path with a status, or takes the call and says nothing (`silent`)."""

    def __init__(self, answers: dict | None = None, silent: bool = False):
        self.dir = tempfile.mkdtemp(prefix="egpu-sock-", dir="/tmp")
        self.path = os.path.join(self.dir, "docker.sock")
        self.answers, self.silent, self.asked = answers or {}, silent, []
        self.server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server.bind(self.path)
        self.server.listen(8)
        threading.Thread(target=self.serve, daemon=True).start()

    def serve(self):
        while True:
            try:
                conn, _ = self.server.accept()
            except OSError:
                return
            with conn:
                said = b""
                while b"\r\n\r\n" not in said:
                    chunk = conn.recv(4096)
                    if not chunk:
                        break
                    said += chunk
                path = said.split(b" ")[1].decode() if said.count(b" ") >= 2 else ""
                self.asked.append(path)
                if self.silent:
                    time.sleep(3)
                    continue
                status = self.answers.get(path, 404)
                conn.sendall(f"HTTP/1.1 {status} X\r\nContent-Length: 2\r\n\r\nOK".encode())

    def close(self):
        self.server.close()


def kept(name: str, fresh: float, look):
    k = plugin.Kept(name, fresh, look)
    setattr(plugin, f"_{name}", k)
    return k


class Counted:
    """A look that counts its calls, takes `wait` seconds, and says how many it has been."""

    def __init__(self, wait: float = 0.0):
        self.calls, self.wait = 0, wait

    def __call__(self):
        self.calls += 1
        time.sleep(self.wait)
        return {"found": True, "n": self.calls}


class DockerEngine(unittest.TestCase):
    def test_an_answering_engine_is_running_with_its_image(self):
        e = FakeEngine({"/_ping": 200, f"/images/{plugin.IMAGE}/json": 200})
        with mock.patch.object(plugin, "docker_socket", return_value=e.path), \
             mock.patch.object(plugin, "quiet", side_effect=AssertionError("Docker's program asked")):
            d = plugin.look_docker()
        e.close()
        self.assertEqual((d["running"], d["image"], d["stalled"]), (True, True, False))
        self.assertEqual(e.asked, ["/_ping", f"/images/{plugin.IMAGE}/json"])

    def test_an_engine_without_the_image(self):
        e = FakeEngine({"/_ping": 200})
        with mock.patch.object(plugin, "docker_socket", return_value=e.path):
            d = plugin.look_docker()
        e.close()
        self.assertEqual((d["running"], d["image"]), (True, False))

    def test_an_engine_that_takes_the_call_and_says_nothing_is_said_so_in_a_second_and_a_half(self):
        e = FakeEngine(silent=True)
        began = time.monotonic()
        with mock.patch.object(plugin, "docker_socket", return_value=e.path):
            d = plugin.look_docker()
        took = time.monotonic() - began
        e.close()
        self.assertEqual((d["running"], d["image"], d["stalled"]), (False, False, True))
        self.assertLess(took, 2.0)

    def test_no_engine_is_said_at_once(self):
        began = time.monotonic()
        with mock.patch.object(plugin, "docker_socket", return_value=os.path.join(DATA, "no-such.sock")):
            d = plugin.look_docker()
        self.assertEqual((d["running"], d["stalled"]), (False, False))
        self.assertLess(time.monotonic() - began, 0.2)

    def test_an_engine_elsewhere_is_asked_by_dockers_own_program(self):
        ran = []
        def quiet(argv, timeout, env=None):
            ran.append(argv[1])
            return plugin.subprocess.CompletedProcess(argv, 0, "27.0" if argv[1] == "info" else "", "")
        with mock.patch.object(plugin, "docker_socket", return_value=None), \
             mock.patch.object(plugin, "docker_cli", return_value="/usr/local/bin/docker"), \
             mock.patch.object(plugin, "quiet", side_effect=quiet):
            d = plugin.look_docker()
        self.assertEqual((d["running"], d["image"]), (True, True))
        self.assertEqual(ran, ["info", "image"])


class DockerSocket(unittest.TestCase):
    def setUp(self):
        self.home = pathlib.Path(tempfile.mkdtemp(prefix="egpu-home-"))
        self.env = {k: v for k, v in os.environ.items() if k not in ("DOCKER_HOST", "DOCKER_CONTEXT")}

    def socket_with(self, env: dict) -> str | None:
        with mock.patch.dict(os.environ, {**self.env, **env}, clear=True), mock.patch.object(plugin, "HOME", self.home):
            return plugin.docker_socket()

    def test_docker_host(self):
        self.assertEqual(self.socket_with({"DOCKER_HOST": "unix:///run/user/501/docker.sock"}), "/run/user/501/docker.sock")
        self.assertIsNone(self.socket_with({"DOCKER_HOST": "tcp://10.0.0.5:2376"}))

    def test_the_context_in_use(self):
        import hashlib
        meta = self.home / ".docker" / "contexts" / "meta" / hashlib.sha256(b"colima").hexdigest()
        meta.mkdir(parents=True)
        (meta / "meta.json").write_text(json.dumps({"Name": "colima", "Endpoints": {"docker": {"Host": "unix:///Users/sam/.colima/docker.sock"}}}))
        (self.home / ".docker" / "config.json").write_text(json.dumps({"currentContext": "colima"}))
        self.assertEqual(self.socket_with({}), "/Users/sam/.colima/docker.sock")

    def test_docker_desktops_own_socket_else_dockers_place(self):
        self.assertEqual(self.socket_with({}), "/var/run/docker.sock")
        (self.home / ".docker" / "run").mkdir(parents=True)
        (self.home / ".docker" / "run" / "docker.sock").write_text("")
        self.assertEqual(self.socket_with({}), str(self.home / ".docker" / "run" / "docker.sock"))


class KeptLooks(unittest.TestCase):
    def setUp(self):
        self.file = pathlib.Path(tempfile.mkdtemp(prefix="egpu-kept-")) / "checks.json"
        self.patch = mock.patch.object(plugin, "KEPT_FILE", self.file)
        self.patch.start()
        kept("card", 10, Counted())
        kept("docker", 15, Counted())

    def tearDown(self):
        self.patch.stop()

    def test_a_fresh_look_is_answered_as_it_is(self):
        look = Counted()
        k = kept("card", 10, look)
        self.assertEqual(k.get()["n"], 1)
        self.assertEqual(k.get()["n"], 1)
        self.assertEqual(look.calls, 1)

    def test_an_old_look_is_answered_at_once_and_looked_at_again_on_its_own(self):
        look = Counted(wait=0.4)
        k = kept("card", 10, look)
        k.get()
        k.stale()
        began = time.monotonic()
        self.assertEqual(k.get()["n"], 1)          # (the old one, at once)
        self.assertLess(time.monotonic() - began, 0.1)
        self.assertTrue(k.busy())
        k.get()                                    # (asked again while it looks: no second look)
        k.looking.join()
        self.assertEqual(look.calls, 2)
        self.assertEqual(k.get()["n"], 2)

    def test_now_waits_for_a_new_look(self):
        look = Counted()
        k = kept("card", 10, look)
        k.get()
        self.assertEqual(k.get(now=True)["n"], 2)

    def test_looks_are_kept_for_the_next_start_and_answered_as_old(self):
        plugin._card.get()
        plugin._docker.get()
        on_disk = json.loads(self.file.read_text())
        self.assertEqual(sorted(on_disk), ["card", "docker"])
        look = Counted(wait=0.2)
        kept("card", 10, look)                       # (a new start)
        plugin.kept_looks()
        began = time.monotonic()
        self.assertEqual(plugin.card()["n"], 1)      # (last run's, at once)
        self.assertLess(time.monotonic() - began, 0.1)
        self.assertEqual(plugin._card.seen, on_disk["card"]["seen"])
        plugin._card.looking.join()
        self.assertEqual(look.calls, 1)

    def test_a_kept_file_that_is_not_ours_is_left_unanswered(self):
        self.file.write_text("[1, 2]")
        plugin.kept_looks()
        self.file.write_text(json.dumps({"card": {"value": "x", "seen": 1}}))
        plugin.kept_looks()
        self.assertIsNone(plugin._card.value)


class SetupStatus(unittest.TestCase):
    def setUp(self):
        self.patches = [mock.patch.object(plugin, "KEPT_FILE", pathlib.Path(tempfile.mkdtemp()) / "checks.json"),
                        mock.patch.object(plugin, "driver", return_value={"installed": True, "enabled": True, "app": True, "state": "allowed", "said": ""}),
                        mock.patch.object(plugin, "compiler", return_value={"ready": False, "built": False})]
        for p in self.patches:
            p.start()
        self.card, self.docker = Counted(), Counted()
        self.card_look = lambda: {**self.card(), "up": True, "arch": "sm_89"}
        kept("card", 10, self.card_look)
        kept("docker", 15, lambda: {**self.docker(), "running": False, "stalled": False})

    def tearDown(self):
        for p in self.patches:
            p.stop()

    def test_answered_as_kept_and_refreshed_on_asking(self):
        first = plugin.setup_status()
        plugin.setup_status()
        self.assertEqual((self.card.calls, self.docker.calls), (1, 1))
        self.assertIsNotNone(first["checked"]["at"])
        plugin.setup_status({"refresh": True})
        self.assertEqual((self.card.calls, self.docker.calls), (2, 2))
        plugin.setup_status({"refresh": "no"})     # (any truthy value refreshes; a payload that is no object never)
        plugin.setup_status(["refresh"])
        self.assertEqual(self.card.calls, 3)

    def test_the_two_looks_are_waited_for_side_by_side(self):
        slow = Counted(wait=0.5)
        kept("card", 10, lambda: {**slow(), "up": True})
        kept("docker", 15, lambda: (time.sleep(0.5), {"running": False})[1])
        began = time.monotonic()
        plugin.setup_status()
        self.assertLess(time.monotonic() - began, 0.9)

    def test_the_kept_card_is_not_written_into(self):
        plugin.setup_status()
        self.assertNotIn("covered", plugin._card.value)

    def test_a_finished_job_has_its_looks_taken_again(self):
        plugin.setup_status()
        plugin.start("driver", lambda job: None)["started"]
        for _ in range(50):
            if plugin._job[0].state != "running":
                break
            time.sleep(0.01)
        time.sleep(0.05)
        plugin.setup_status()
        for k in (plugin._card, plugin._docker):
            if k.looking:
                k.looking.join()
        self.assertEqual((self.card.calls, self.docker.calls), (2, 2))


class Watcher(unittest.TestCase):
    def test_a_card_the_pack_carries_kernels_for_never_asks_docker(self):
        kept("card", 10, lambda: {"found": True, "up": True, "arch": "sm_89"})
        kept("docker", 15, mock.Mock(side_effect=AssertionError("Docker asked")))
        with mock.patch.object(plugin, "runtime", return_value={"pack": {"archs": ["sm_86", "sm_89", "sm_120"], "built": "x"}}):
            self.assertFalse(plugin.watch_once(set()))


class Commands(unittest.TestCase):
    def test_a_commands_payload_reaches_it(self):
        seen = []
        lines = [{"jsonrpc": "2.0", "id": 1, "method": "connector.request", "params": {"command": "setup.status", "payload": {"refresh": True}}},
                 {"jsonrpc": "2.0", "id": 2, "method": "connector.request", "params": {"command": "setup.status"}},
                 {"jsonrpc": "2.0", "id": 3, "method": "connector.request", "params": {"command": "setup.status", "payload": "refresh"}},
                 {"jsonrpc": "2.0", "method": "plugin.shutdown"}]
        out = io.StringIO()
        with mock.patch.dict(plugin.COMMANDS, {"setup.status": lambda p: seen.append(p) or {"ok": 1}}), \
             mock.patch.object(plugin, "kept_looks"), mock.patch.object(plugin, "watch", lambda: None), \
             mock.patch.object(sys, "stdin", io.StringIO("".join(json.dumps(l) + "\n" for l in lines))), \
             mock.patch.object(sys, "stdout", out):
            plugin.main()
        self.assertEqual(seen, [{"refresh": True}, None, None])
        self.assertEqual([json.loads(l)["result"]["ok"] for l in out.getvalue().splitlines()], [True, True, True])


if __name__ == "__main__":
    unittest.main(verbosity=1)
