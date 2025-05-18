# tests/test_sandbox.py
import pytest
from pysand.manager import SandboxManager

@pytest.fixture(params=["virtualenv", "docker"])
def mgr(request):
    return SandboxManager(backend=request.param, packages=["pytest"])

def test_create_and_teardown(mgr, tmp_path):
    sandbox = mgr.create()
    assert sandbox.is_ready()
    sandbox.teardown()
    assert not sandbox.is_ready()

def test_run_simple_code(mgr):
    sandbox = mgr.create()
    out, err, code = sandbox.exec("print('hello pysand')")
    assert code == 0
    assert out.strip() == "hello pysand"
    sandbox.teardown()

def test_install_additional_package(mgr):
    sandbox = mgr.create()
    sandbox.install = getattr(sandbox, "install", None)
    # virtualenv has install; docker installs at create time
    if sandbox.install:
        sandbox.install(["requests"])
    out, err, code = sandbox.exec("import requests; print(True)")
    assert code == 0
    sandbox.teardown()

def test_timeout(mgr):
    sandbox = mgr.create()
    out, err, code = sandbox.exec("import time; time.sleep(5)", timeout=1)
    assert code != 0
    sandbox.teardown()

def test_run_file(tmp_path, mgr):
    sandbox = mgr.create()
    script = tmp_path / "script.py"
    script.write_text("print(2 + 3)")
    out, err, code = sandbox.exec_file(str(script))
    assert code == 0
    assert out.strip() == "5"
    sandbox.teardown()
