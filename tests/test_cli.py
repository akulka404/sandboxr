# add a new file tests/test_cli.py:
import subprocess, sys, pytest
@pytest.mark.parametrize("args,expected", [
    (["--backend","virtualenv","--code","print(1+1)"], "2"),
    (["--backend","virtualenv","--packages","requests","--code","import requests;print(True)"], "True"),
])
def test_sandboxr_cli(tmp_path, args, expected):
    cmd = [sys.executable, "-m", "sandboxr.cli", *args]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.stdout.strip() == expected
    assert result.returncode == 0
