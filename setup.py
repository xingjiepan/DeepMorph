#!/usr/bin/env python3

from setuptools import setup, find_packages

setup(
    name='deepmorph',
    version='0.0.0',
    author='Xingjie Pan',
    author_email='xingjiepan@gmail.com',
    url='https://github.com/xingjiepan/DeepMorph',
    packages=setuptools.find_packages(include=['deepmorph']),
    install_requires=[
        'numpy',
        'matplotlib',
    ],
    description='DeepMorph is a deep learning method to learn representations of multi-dimensional cell morphology.',
    long_description=open('README.md').read(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'Intended Audience :: Science/Research',
    ],
)
