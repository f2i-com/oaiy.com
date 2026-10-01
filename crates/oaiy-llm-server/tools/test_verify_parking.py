#!/usr/bin/env python3
"""
Tests of verify_parking.py's own logic, without a model: it is run against tools/mock_server.py, a stand-in
for the gateway and the Studio's log on 127.0.0.1 (a port the OS picks) that applies the engine's rules
(continue the live state, else bring a stashed one back, else read; set aside what a prompt displaces) to
words instead of tokens.  Nothing here touches the real gateway (8080) or Studio (7860).

    cd crates/oaiy-llm-server && python -m unittest tools/test_verify_parking.py
"""
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "verify_parking.py")
MOCK = os.path.join(HERE, "mock_server.py")
NOTES = ["--notes-a", "400", "--notes-b", "100"]


class Mock:
    """The mock server on a loopback port the OS picks; `flags` are its MOCK_* switches."""

    def __init__(self, **flags):
        env = {**os.environ, **{f"MOCK_{k.upper()}": "1" for k, v in flags.items() if v}}
        self.proc = subprocess.Popen([sys.executable, MOCK, "0"], stdout=subprocess.PIPE, env=env, text=True)
        self.port = int(self.proc.stdout.readline())
        self.url = f"http://127.0.0.1:{self.port}"

    def stop(self):
        self.proc.kill()
        self.proc.wait()
        self.proc.stdout.close()


def run(url, *args, **env):
    """verify_parking.py against `url` for the gateway and the Studio: (exit code, output)."""
    e = {**os.environ, "OAIY_GATEWAY": url, "OAIY_STUDIO": url, "PYTHONIOENCODING": "utf-8", **env}
    p = subprocess.run([sys.executable, SCRIPT, *args], capture_output=True, text=True, encoding="utf-8", env=e, timeout=300)
    return p.returncode, p.stdout + p.stderr


class VerifyParking(unittest.TestCase):
    def mock(self, **flags):
        m = Mock(**flags)
        self.addCleanup(m.stop)
        return m

    def recorded(self, m, tmp):
        """The two runs the determinism check compares, on a mock that is fresh for each."""
        ref, swp = os.path.join(tmp, "reference.json"), os.path.join(tmp, "swapped.json")
        self.assertEqual(run(m.url, "--mode", "reference", *NOTES, "--out", ref)[0], 0)
        self.assertEqual(run(m.url, "--mode", "swapped", *NOTES, "--restart", "--out", swp)[0], 0)
        return ref, swp

    def compare(self, m):
        with tempfile.TemporaryDirectory() as tmp:
            ref, swp = self.recorded(m, tmp)
            return run(m.url, "--mode", "compare", "--ref", ref, "--swapped", swp)

    def test_a_sequence_that_swaps_in_every_time_passes(self):
        code, out = run(self.mock().url, "--mode", "sequence", *NOTES)
        self.assertEqual(code, 0, out)
        self.assertEqual(out.count("PASS"), 4, out)
        self.assertIn("Qwen restore:", out)

    def test_a_sequence_with_somebody_elses_request_in_it_is_inconclusive(self):
        code, out = run(self.mock(interleave=True).url, "--mode", "sequence", *NOTES)
        self.assertEqual(code, 2, out)
        self.assertIn("INCONCLUSIVE", out)

    def test_identical_answers_after_a_swap_pass(self):
        code, out = self.compare(self.mock())
        self.assertEqual(code, 0, out)
        self.assertIn("PASS", out)

    def test_a_different_answer_after_a_swap_fails(self):
        code, out = self.compare(self.mock(bad=True))
        self.assertEqual(code, 1, out)
        self.assertIn("FAIL", out)

    def test_two_empty_answers_prove_nothing(self):
        code, out = self.compare(self.mock(empty=True))
        self.assertEqual(code, 2, out)
        self.assertIn("INCONCLUSIVE", out)

    def test_a_request_that_slipped_in_makes_the_comparison_inconclusive(self):
        code, out = self.compare(self.mock(interleave=True))
        self.assertEqual(code, 2, out)
        self.assertIn("INCONCLUSIVE", out)

    def test_the_cache_source_falls_back_to_the_servers_own_log_line(self):
        # A gateway that does not pass oaiy_progress through: the `Qwen cache:` line says where A2 came from.
        code, out = self.compare(self.mock(no_progress=True))
        self.assertEqual(code, 0, out)

    def test_two_references_are_not_accepted_as_a_swapped_run(self):
        # Nothing was swapped in either: refused (--swapped must come from --mode swapped).
        m = self.mock()
        with tempfile.TemporaryDirectory() as tmp:
            ref = os.path.join(tmp, "reference.json")
            fake = os.path.join(tmp, "fake.json")
            self.assertEqual(run(m.url, "--mode", "reference", *NOTES, "--out", ref)[0], 0)
            self.assertEqual(run(m.url, "--mode", "reference", *NOTES, "--restart", "--out", fake)[0], 0)
            code, out = run(m.url, "--mode", "compare", "--ref", ref, "--swapped", fake)
            self.assertNotEqual(code, 0, out)

    def test_the_baseline_of_two_references_says_the_engine_agrees_with_itself(self):
        m = self.mock()
        with tempfile.TemporaryDirectory() as tmp:
            one, two = os.path.join(tmp, "one.json"), os.path.join(tmp, "two.json")
            self.assertEqual(run(m.url, "--mode", "reference", *NOTES, "--out", one)[0], 0)
            self.assertEqual(run(m.url, "--mode", "reference", *NOTES, "--restart", "--out", two)[0], 0)
            code, out = run(m.url, "--mode", "compare", "--baseline", "--ref", one, "--swapped", two)
            self.assertEqual(code, 0, out)
            self.assertIn("BASELINE", out)

    def test_a_server_that_already_answered_is_refused_for_the_recorded_runs(self):
        m = self.mock()
        with tempfile.TemporaryDirectory() as tmp:
            out_file = os.path.join(tmp, "x.json")
            self.assertEqual(run(m.url, "--mode", "reference", *NOTES, "--out", out_file)[0], 0)
            code, out = run(m.url, "--mode", "reference", *NOTES, "--out", out_file)
            self.assertNotEqual(code, 0, out)
            self.assertIn("already answered", out)

    def test_it_talks_to_loopback_only(self):
        p = subprocess.run([sys.executable, SCRIPT, "--help"], capture_output=True, text=True,
                           env={**os.environ, "OAIY_GATEWAY": "http://example.com:8080"})
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("loopback only", p.stdout + p.stderr)


if __name__ == "__main__":
    unittest.main()
