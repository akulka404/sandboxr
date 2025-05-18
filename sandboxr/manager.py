# pysand/manager.py
from .backends.virtualenv import VirtualenvSandbox
from .backends.docker import DockerSandbox

class SandboxManager:
    def __init__(self, backend: str, packages=None, base_dir=None):
        self.backend = backend
        self.packages = packages or []
        self.base_dir = base_dir

    def create(self):
        if self.backend == "virtualenv":
            return VirtualenvSandbox(self.packages, base_dir=self.base_dir).create()
        elif self.backend == "docker":
            return DockerSandbox(self.packages, base_dir=self.base_dir).create()
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
