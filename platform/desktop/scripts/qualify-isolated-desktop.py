#!/usr/bin/env python3
"""Windows executable smoke checks; no GUI automation or visual qualification.

Run with --exe ABSOLUTE_BUNDLED_EXE --output NEW_ABSOLUTE_DIRECTORY.
Only owned processes are launched/stopped. Reserved normal ports are inspected
through the Windows TCP table, never contacted. Artifacts remain in --output.
"""
import argparse
import ctypes
from ctypes import wintypes
import hashlib
import http.client
import json
import os
from pathlib import Path
import re
import socket
import stat
import subprocess
import sys
import time

FORBIDDEN = {17872, 17972, 17973}
DIAGNOSTIC = "isolated dashboard requested plugin inventory"
REFUSAL = "isolated_capability_unavailable"


class CheckFailure(Exception):
    pass


def require(condition, code):
    if not condition:
        raise CheckFailure(code)


class TcpOwners:
    """Read listener ownership without sending packets or invoking netstat."""
    def __init__(self):
        self.api = ctypes.WinDLL("iphlpapi", use_last_error=True).GetExtendedTcpTable
        self.api.argtypes = [ctypes.c_void_p, ctypes.POINTER(wintypes.DWORD),
                             wintypes.BOOL, wintypes.ULONG, ctypes.c_int, wintypes.ULONG]
        self.api.restype = wintypes.DWORD

    def listeners(self):
        import struct
        rows = []
        for family, width in [(socket.AF_INET, 24), (socket.AF_INET6, 56)]:
            size = wintypes.DWORD(0)
            result = self.api(None, ctypes.byref(size), False, family, 3, 0)
            require(result in (0, 122), "tcp_table_size_failed")
            for _ in range(4):
                buffer = ctypes.create_string_buffer(max(size.value, 4))
                result = self.api(buffer, ctypes.byref(size), False, family, 3, 0)
                if result != 122:
                    break
            require(result == 0, "tcp_table_read_failed")
            count = struct.unpack_from("<I", buffer)[0]
            require(4 + count * width <= len(buffer), "tcp_table_invalid_size")
            for offset in range(4, 4 + count * width, width):
                if family == socket.AF_INET:
                    _, address, port, _, _, pid = struct.unpack_from("<6I", buffer, offset)
                    address = socket.inet_ntop(family, address.to_bytes(4, "little"))
                else:
                    address, _, port, _, _, _, _, pid = struct.unpack_from("<16sII16sIIII", buffer, offset)
                    address = socket.inet_ntop(family, address)
                rows.append((family, address, socket.ntohs(port & 0xffff), pid))
        return rows

    def protected(self):
        rows = self.listeners()
        return {str(port): sorted({pid for _, _, found, pid in rows if found == port})
                for port in sorted(FORBIDDEN)}

    def verify(self, process, port):
        require(port not in FORBIDDEN and 1024 <= port <= 65535, "unsafe_api_port")
        require(process.poll() is None, "owned_process_exited")
        owners = {pid for family, address, found, pid in self.listeners()
                  if family == socket.AF_INET and address == "127.0.0.1" and found == port}
        require(owners == {process.pid}, "api_listener_owner_mismatch")


def request(owners, process, port, method, route, origin=True):
    owners.verify(process, port)
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        connection.connect()
        owners.verify(process, port)  # Recheck before sending even one HTTP byte.
        headers = {"Content-Type": "application/json"}
        if origin:
            headers["Origin"] = "tauri://localhost"
        connection.request(method, route, body=b"{}" if method == "POST" else None, headers=headers)
        response = connection.getresponse()
        data = response.read(1024 * 1024 + 1)
        require(len(data) <= 1024 * 1024, "oversized_api_response")
        return response.status, json.loads(data)
    finally:
        connection.close()


def wait_ready(process, root, owners):
    deadline = time.monotonic() + 30
    log = root / "data" / "logs" / "oaiy-desktop.log"
    while time.monotonic() < deadline:
        require(process.poll() is None, "startup_process_exited")
        text = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
        match = re.search(r"OAIY API listening on http://127\.0\.0\.1:(\d+)\s", text)
        # This happens BEFORE this harness makes its first request to the app.
        # The Windows dashboard alone uses the diagnostic's http://tauri origin.
        if match and DIAGNOSTIC in text:
            port = int(match.group(1))
            owners.verify(process, port)
            return port
        time.sleep(0.1)
    raise CheckFailure("native_dashboard_startup_timeout")


def inspect_instance(owners, process, port, root):
    status, health = request(owners, process, port, "GET", "/api/health")
    require(status == 200 and health.get("product") == "oaiy-desktop", "health_identity_failed")
    status, config = request(owners, process, port, "GET", "/api/config")
    require(status == 200, "config_failed")
    canonical = root.resolve()
    for key in ["activeDir", "defaultDir", "configuredDir", "modelsActiveDir",
                "modelsDefaultDir", "modelsConfiguredDir"]:
        value = config.get(key)
        if value is not None:
            require(isinstance(value, str) and Path(value).is_absolute(), "invalid_config_path")
            require(".." not in Path(value).parts and Path(value).is_relative_to(canonical), "config_path_escaped_root")
    require(Path(config["activeDir"]) == canonical / "data", "data_root_mismatch")
    require(Path(config["modelsActiveDir"]) == canonical / "models", "model_root_mismatch")
    status, plugins = request(owners, process, port, "GET", "/api/plugins")
    require(status == 200 and plugins.get("plugins") == [], "unexpected_plugin_inventory")
    require(Path(plugins["root"]) == canonical / "data" / "plugins", "plugin_root_mismatch")
    marker = json.loads((root / ".oaiy-isolated.json").read_text(encoding="utf-8"))
    require(marker.get("version") == 1 and marker.get("kind") == "oaiy-desktop-isolated", "invalid_root_marker")
    require(re.fullmatch(r"com\.oaiy\.isolated\.[0-9a-f]{48}", marker.get("identifier", "")), "invalid_isolated_identity")
    profile = (root / "webview").lstat()
    require(stat.S_ISDIR(profile.st_mode) and not profile.st_file_attributes & 0x400, "webview_profile_not_owned")
    entries = [entry.lstat() for entry in (root / "webview").iterdir()]
    require(entries and all(not entry.st_file_attributes & 0x400 for entry in entries)
            and any(stat.S_ISDIR(entry.st_mode) or stat.S_ISREG(entry.st_mode) for entry in entries), "webview_profile_not_owned_and_populated")
    appdata = os.environ.get("APPDATA")  # Parent environment only; never passed to the app.
    if appdata:
        require(not os.path.lexists(Path(appdata) / marker["identifier"]), "unexpected_isolated_appdata_directory")
    return {"root": root.name, "pid": process.pid, "port": port,
            "identifier": marker["identifier"], "dashboardReachedApiBeforeHarness": True,
            "ownedConfigAndPluginRoots": True, "autoloadedPlugins": 0,
            "ownedWebViewProfilePopulated": True,
            "isolatedIdentifierAppDataAbsent": True if appdata else None}


def stop(process):
    if process.poll() is None:
        process.terminate()  # Popen retains this process's handle; no PID/name targeting.
        process.wait(timeout=15)
    return {"pid": process.pid, "stopped": process.poll() is not None,
            "exitCode": process.returncode}


def run(exe, output):
    with exe.open("rb") as file:
        digest = hashlib.file_digest(file, "sha256").hexdigest()
    report = {"passed": False, "nativeVisualQualification": False, "checks": [],
              "instances": [], "cleanup": [], "executableSha256": digest}
    processes = []
    owned_ports = set()
    phase = "initialization"
    owners = None
    try:
        owners = TcpOwners()
        report["protectedPortOwnersBefore"] = owners.protected()
        temp = output / "temp"
        temp.mkdir()
        system = os.environ["SystemRoot"]
        environment = {"SystemRoot": system, "WINDIR": system,
                       "PATH": str(Path(system) / "System32"), "TEMP": str(temp), "TMP": str(temp)}
        startup = subprocess.STARTUPINFO()
        startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startup.wShowWindow = 0

        def launch(root):
            child = subprocess.Popen([str(exe), "--isolated-root", str(root), "--hidden"],
                cwd=output, env=environment, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, startupinfo=startup, creationflags=subprocess.CREATE_NO_WINDOW)
            processes.append(child)
            return child

        phase = "instance_a"
        a_root = output / "instance-a"
        a = launch(a_root)
        a_port = wait_ready(a, a_root, owners)
        owned_ports.add(a_port)
        report["instances"].append(inspect_instance(owners, a, a_port, a_root))
        phase = "duplicate_root"
        duplicate = launch(a_root)
        require(duplicate.wait(timeout=30) == 2, "duplicate_root_not_refused")
        owners.verify(a, a_port)
        report["checks"].append("same_root_refused_original_alive")
        phase = "instance_b"
        b_root = output / "instance-b"
        b = launch(b_root)
        b_port = wait_ready(b, b_root, owners)
        owned_ports.add(b_port)
        report["instances"].append(inspect_instance(owners, b, b_port, b_root))
        require(a_port != b_port and report["instances"][0]["identifier"] != report["instances"][1]["identifier"], "instances_not_independent")
        stop(b)
        owners.verify(a, a_port)
        report["checks"].append("different_roots_have_independent_identity_port_and_data")
        phase = "capability_refusals"
        for method, route in [("GET", "/api/ai/codex/status"), ("GET", "/api/engines/catalog"),
                ("POST", "/api/setup/plugins/example/check/foo"), ("POST", "/api/services/example/start"),
                ("POST", "/api/mcp"), ("POST", "/api/bridge/runs"), ("POST", "/api/update/check")]:
            status, body = request(owners, a, a_port, method, route)
            require(status == 403 and isinstance(body.get("error"), dict)
                    and body["error"].get("code") == REFUSAL, "capability_not_refused:" + route)
            report["checks"].append({"route": route, "status": status, "code": REFUSAL})
        status, body = request(owners, a, a_port, "POST", "/api/plugins/install", origin=False)
        require(status == 403 and isinstance(body.get("error"), str), "origin_auth_not_preserved")
        report["checks"].append("plugin_install_without_origin_refused_by_auth")
        phase = "invalid_roots"
        foreign = output / "foreign"
        foreign.mkdir()
        sentinel = foreign / "keep.txt"
        sentinel.write_text("unchanged", encoding="utf-8")
        for name, root in [("relative", "relative-root"), ("network", r"\\invalid.example\share\qualification"), ("foreign", foreign)]:
            refused = launch(root)
            require(refused.wait(timeout=30) == 2, "invalid_root_not_refused")
            report["checks"].append(name + "_root_refused_before_gui")
        require(sentinel.read_text(encoding="utf-8") == "unchanged" and len(list(foreign.iterdir())) == 1, "foreign_root_changed")
        require(not (output / "relative-root").exists(), "relative_root_created")
        owners.verify(a, a_port)
        report["passed"] = True
    except Exception as error:
        report["failure"] = {"phase": phase, "type": type(error).__name__,
                             "code": str(error) if isinstance(error, CheckFailure) else "native_smoke_exception"}
    finally:
        for process in reversed(processes):
            try:
                report["cleanup"].append(stop(process))
            except Exception as error:
                report["passed"] = False
                report["cleanup"].append({"pid": process.pid, "stopped": False, "type": type(error).__name__})
        if owners:
            try:
                deadline = time.monotonic() + 5
                while True:
                    remaining = sorted({port for _, _, port, _ in owners.listeners()} & owned_ports)
                    if not remaining or time.monotonic() >= deadline:
                        break
                    time.sleep(0.1)
                report["ownedApiPortsReleased"] = not remaining
                report["remainingOwnedPorts"] = remaining
                report["passed"] &= not remaining
                report["protectedPortOwnersAfter"] = owners.protected()
                unchanged = report.get("protectedPortOwnersBefore") == report["protectedPortOwnersAfter"]
                report["protectedPortOwnersUnchanged"] = unchanged
                report["passed"] &= unchanged
            except Exception as error:
                report["passed"] = False
                report["ownershipCheckFailure"] = type(error).__name__
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report["passed"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(os.name == "nt", "windows_required")
    require(args.exe.is_absolute() and args.exe.is_file(), "absolute_executable_required")
    require(args.output.is_absolute() and not args.output.exists(), "new_absolute_output_required")
    for ancestor in args.output.parents:
        meta = ancestor.lstat()
        require(not meta.st_file_attributes & 0x400, "output_reparse_ancestor_refused")
    args.output.mkdir()
    passed = run(args.exe.resolve(), args.output.resolve())
    print(json.dumps({"passed": passed, "report": "report.json", "nativeVisualQualification": False}))
    return 0 if passed else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(json.dumps({"passed": False, "type": type(error).__name__,
                          "code": str(error) if isinstance(error, CheckFailure) else "invalid_harness_input"}))
        sys.exit(1)
