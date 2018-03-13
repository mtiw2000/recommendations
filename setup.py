# -*- coding: utf-8 -*-


"""setup.py: setuptools control."""


import re
from setuptools import setup


version = re.search(
    '^__version__\s*=\s*"(.*)"',
    open('bootstrap/bootstrap.py').read(),
    re.M
    ).group(1)


with open("README.rst", "rb") as f:
    long_descr = f.read().decode("utf-8")


setup(
    name = "cmdline-bootstrap",
    packages = ["bootstrap"],
    entry_points = {
        "console_scripts": ['bootstrap = bootstrap.bootstrap:main']
        },
    version = version,
    description = "Python command line application bare bones template.",
    long_description = long_descr,
    author = "Manish Tiwari",
    author_email = "mtiw2000@hotmail.com",
    url = "http://tiwariji.com",
    )