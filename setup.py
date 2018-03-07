import os
from setuptools import setup, find_packages


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


with open('requirements.txt') as f:
    required = f.read().splitlines()


setup(
    install_requires=required,
    name="holtwintersts",
    version="0.01",
    description="",
    author="Eilif Mikkelsen",
    packages=find_packages(),
)
