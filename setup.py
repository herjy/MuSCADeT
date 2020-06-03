
from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='MuSCADeT',
      version='0.1',
      description='Code for colour separation of multi-band astronomical images',
      author='Remy Joseph, Frederic Courbin, Jean-Luc Starck',
      author_email='remy.joseph@epfl.ch',
      packages=['MuSCADeT'],
      zip_safe=False,
      classifiers=[
                   "Programming Language :: Python :: 2.7",
                   "License :: OSI Approved :: MIT License",
                   "Operating System :: OS Independent",
                   ])
