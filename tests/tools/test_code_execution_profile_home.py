"""Real child-process profile isolation regression for #110303.

No provider calls or live credentials: each profile contains a fixture marker.
"""
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from tools.code_execution_env import _build_child_env


def _child_env(tmp_path):
    return _build_child_env(
        rpc_endpoint="fixture-socket", rpc_token="fixture-rpc-token",
        tmpdir=str(tmp_path), child_python=sys.executable,
    )


def test_concurrent_profile_children_never_read_default_or_sibling_state(tmp_path, monkeypatch):
    default = tmp_path / "default"
    default.mkdir()
    (default / ".env").write_text("default-marker", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(default))
    monkeypatch.setenv("PAPERCLIP_API_KEY", "fixture-default-secret")
    homes = [tmp_path / name for name in ("alpha", "beta", "missing")]
    for home in homes:
        home.mkdir()
    for home in homes[:2]:
        (home / ".env").write_text(home.name, encoding="utf-8")
    barrier = Barrier(len(homes))

    def run_child(home):
        token = set_hermes_home_override(home)
        try:
            barrier.wait(timeout=10)
            env = _child_env(tmp_path)
            child = subprocess.run([
                sys.executable, "-c",
                "import json,os; from pathlib import Path; "
                "p=Path(os.environ['HERMES_HOME'])/'.env'; "
                "print(json.dumps({'home':os.environ['HERMES_HOME'], "
                "'marker':p.read_text(encoding='utf-8') if p.exists() else None, "
                "'credential_present':'PAPERCLIP_API_KEY' in os.environ}))",
            ], env=env, capture_output=True, text=True, check=True, timeout=15)
            return json.loads(child.stdout)
        finally:
            reset_hermes_home_override(token)

    with ThreadPoolExecutor(max_workers=len(homes)) as pool:
        results = list(pool.map(run_child, homes))
    assert [item["home"] for item in results] == [str(home) for home in homes]
    assert [item["marker"] for item in results] == [home.name for home in homes[:2]] + [None]
    assert all(not item["credential_present"] for item in results)
    assert os.environ["HERMES_HOME"] == str(default)


def test_unscoped_child_preserves_process_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    token = set_hermes_home_override(None)
    try:
        assert _child_env(tmp_path)["HERMES_HOME"] == str(tmp_path)
    finally:
        reset_hermes_home_override(token)
