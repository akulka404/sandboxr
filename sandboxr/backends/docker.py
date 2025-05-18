# sandboxr/backends/docker.py

import subprocess
import tempfile
import shutil
import os
import uuid
from typing import List

class DockerSandbox:
    def __init__(self, packages: List[str], base_dir=None):
        self.packages = packages or []
        self.id = uuid.uuid4().hex[:8]
        self.image_tag = f"sandboxr_{self.id}"
        self.build_dir = base_dir or tempfile.mkdtemp(prefix="sandboxr_docker_")
        self.dockerfile = os.path.join(self.build_dir, "Dockerfile")
        # Create output directory
        self.output_dir = os.path.join(self.build_dir, "output")
        os.makedirs(self.output_dir, exist_ok=True)

    def create(self):
        df = ["FROM python:3.9-slim", "WORKDIR /sandbox"]
        if self.packages:
            df.append("RUN pip install " + " ".join(self.packages))
        with open(self.dockerfile, "w") as f:
            f.write("\n".join(df))
        subprocess.check_call(
            ["docker", "build", "--quiet", "-t", self.image_tag, self.build_dir],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        self.ready = True
        return self

    def is_ready(self):
        return getattr(self, "ready", False)

    def exec(self, code: str, timeout=30):
        cmd = ["docker", "run", "--rm", self.image_tag, "python", "-c", code]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            out, err = proc.communicate(timeout=timeout)
            return out.decode(), err.decode(), proc.returncode
        except subprocess.TimeoutExpired:
            proc.kill()
            return "", "Timed out", -1

    def exec_file(self, filepath: str, timeout=30):
        abs_path = os.path.abspath(filepath)
        cmd = [
            "docker", "run", "--rm",
            "-v", f"{abs_path}:/sandbox/script.py",
            "-v", f"{self.output_dir}:/sandbox/output",
            self.image_tag, "python", "/sandbox/script.py"
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            out, err = proc.communicate(timeout=timeout)
            return out.decode(), err.decode(), proc.returncode
        except subprocess.TimeoutExpired:
            proc.kill()
            return "", "Timed out", -1

    def install(self, packages: List[str]):
        """
        Append a RUN layer to the Dockerfile for the new packages,
        then remove the old image and rebuild quietly.
        """
        with open(self.dockerfile, "a") as f:
            f.write("\nRUN pip install " + " ".join(packages))

        subprocess.check_call(
            ["docker", "rmi", "-f", self.image_tag],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        subprocess.check_call(
            ["docker", "build", "--quiet", "-t", self.image_tag, self.build_dir],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

    def teardown(self):
        subprocess.call(["docker", "rmi", "-f", self.image_tag],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL)
        shutil.rmtree(self.build_dir, ignore_errors=True)
        self.ready = False
