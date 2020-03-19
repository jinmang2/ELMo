#!/usr/bin/env python
import setuptools


setuptools.setup(
  name="elmo",
  version="0.0.0",
  packages=setuptools.find_packages(),
  install_requires=[
    "torch",
    "h5py",
    "numpy",
    "overrides",
  ],
  classifiers=[
    "Development Status :: 2 - Pre-Alpha",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3.6",
  ],
)
