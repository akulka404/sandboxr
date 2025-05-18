from setuptools import setup, find_packages

# Read in the README for the long description on PyPI
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="sandboxr",
    version="0.1.2",
    author="Ani Kulkarni",
    author_email="aniruddha.k1911@gmail.com",
    description="Quickly spin up isolated Python sandboxes (virtualenv or Docker), install dependencies, execute code, and tear down.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/akulka404/sandboxr",
    packages=find_packages(),
    install_requires=[],
    entry_points={
        "console_scripts": [
            "sandboxr=sandboxr.cli:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
    ],
    python_requires=">=3.6",
    keywords=["sandbox", "virtualenv", "docker", "automation", "cli"],
    include_package_data=True,
    zip_safe=False,
)
