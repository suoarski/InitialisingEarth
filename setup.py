##############################
# BUILDING CONDA ENVIRONMENT
##############################
#
# conda env export --from-history | grep -v "prefix" > conda-env.yml
# conda env create -f conda-env.yml
# conda activate gospl-package
#
##############################
# BUILDING PYTHON PACKAGE PYPI
##############################
#
# python3 -m pip install --user --upgrade setuptools wheel
# python3 setup.py sdist
# python3 -m pip install --user --upgrade twine
# /usr/local/bin/twine check dist/*
# /usr/local/bin/twine upload dist/*
#
##############################

import os
import io
import sys
import subprocess
from setuptools import find_packages

try:
    subprocess.call([sys.executable, "-m", "pip", "install", "numpy"])
except ImportError:
    pass

from numpy.distutils.core import setup, Extension
from distutils.command.sdist import sdist

try:
    from distutils.command import bdist_conda
except ImportError:
    pass

# in development set version to none and ...
PYPI_VERSION = "0.0.0"

# Place install_requires into the text file "requirements.txt"
with open("requirements.txt") as f2:
    requirements = f2.read().strip().splitlines()


packs = find_packages(include=["earthinit", "earthinit.*"])


def git_version():
    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ["SYSTEMROOT", "PATH"]:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env["LANGUAGE"] = "C"
        env["LANG"] = "C"
        env["LC_ALL"] = "C"
        out = subprocess.Popen(cmd, stdout=subprocess.PIPE, env=env).communicate()[0]
        return out

    try:
        out = _minimal_ext_cmd(["git", "rev-parse", "--short", "HEAD"])
        GIT_REVISION = out.strip().decode("ascii")
    except OSError:
        GIT_REVISION = "Unknown"

    return GIT_REVISION


if PYPI_VERSION is None:
    PYPI_VERSION = git_version()


this_directory = os.path.abspath(os.path.dirname(__file__))
with io.open(os.path.join(this_directory, "README.rst"), encoding="utf-8") as f:
    long_description = f.read()


if __name__ == "__main__":
    setup(
        name="earthinit",
        author="Tristan Salles",
        author_email="tristan.salles@sydney.edu.au",
        url="https://github.com/Geodels/gospl",
        version=PYPI_VERSION,
        license="GPLv3",
        description="A Python interface to evaluate paleogeography through time",
        keywords=["python", "paleogeography", "science",],
        long_description=long_description,
        long_description_content_type="text/markdown",
        packages=packs,
        install_requires=requirements,
        setup_requires=[[p for p in requirements if p.startswith("numpy")][0]],
        python_requires=">=3",
        classifiers=[
            "Intended Audience :: Science/Research",
            "Operating System :: Unix",
            "Operating System :: MacOS",
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
        ],
        cmdclass={"sdist": sdist},
    )
