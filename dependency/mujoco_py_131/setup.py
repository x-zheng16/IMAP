#!/usr/bin/env python

from setuptools import setup

setup(
    name='mujoco-py-131',
    version='0.5.7',
    description='Python wrapper for Mojoco',
    author='OpenAI',
    packages=['mujoco_py_131'],
    install_requires=[
        'PyOpenGL>=3.1.0',
        'numpy>=1.10.4',
        'six',
    ],
    tests_requires=[
        'nose2'
    ]
)
