# pysand/backends/virtualenv.py

import venv
import subprocess
import shutil
import tempfile
import os
from typing import List

class VirtualenvSandbox:
    def __init__(self, packages: List[str], base_dir=None):
        self.packages = packages or []
        self.base_dir = base_dir or tempfile.mkdtemp(prefix="pysand_")
        self.env_dir = os.path.join(self.base_dir, "env")

    def create(self):
        venv.create(self.env_dir, with_pip=True)
        self.python = os.path.join(self.env_dir, "bin", "python")
        if self.packages:
            subprocess.check_call(
                [self.python, "-m", "pip", "install", *self.packages],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        self.ready = True
        return self

    def is_ready(self):
        return getattr(self, "ready", False)

    def exec(self, code: str, timeout=30):
        proc = subprocess.Popen(
            [self.python, "-c", code],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        try:
            out, err = proc.communicate(timeout=timeout)
            return out.decode(), err.decode(), proc.returncode
        except subprocess.TimeoutExpired:
            proc.kill()
            return "", "Timed out", -1

    def exec_file(self, filepath: str, timeout=30):
        proc = subprocess.Popen(
            [self.python, filepath],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        try:
            out, err = proc.communicate(timeout=timeout)
            return out.decode(), err.decode(), proc.returncode
        except subprocess.TimeoutExpired:
            proc.kill()
            return "", "Timed out", -1

    def install(self, packages: List[str]):
        """
        Install additional packages into an existing virtualenv sandbox.
        """
        subprocess.check_call(
            [self.python, "-m", "pip", "install", *packages],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

    def teardown(self):
        shutil.rmtree(self.base_dir, ignore_errors=True)
        self.ready = False
