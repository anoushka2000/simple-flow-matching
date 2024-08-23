from os import path

from setuptools import find_packages, setup

here = path.abspath(path.dirname(__file__))

# base requirements
install_requires = open(path.join(here, "requirements.txt")).read().strip().split("\n")

setup(
    name="simple_flow_matching",
    version="0.0.1",
    packages=find_packages(),
    install_requires=install_requires,
    description="Minimal demo of flow matching with optimal transport.",
)