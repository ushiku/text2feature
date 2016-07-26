#!/usr/bin/env python
# coding: utf-8
from setuptools import setup, find_packages
from text2feature import __author__, __version__, __license__
 
setup(
        name             = 'text2feature',
        version          = __version__,
        description      = 'textからfeatureにする',
        license          = __license__,
        author           = __author__,
        author_email     = '',
        url              = '',
        keywords         = '',
        packages         = find_packages(),
        install_requires = ['numpy', 'scikit-learn', 'scipy']
        )
