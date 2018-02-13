# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 15:33:19 2016

@author: hfwittmann
"""

from setuptools import setup


def readme():
    with open('README.rst') as f:
        return f.read()


setup(name='timeseries_utils',
      version='0.1',
      description='convenience functions for prediction with artificial data',
      long_description=readme(),
      classifiers=[],
      keywords='quant artificial',
      url='',
      author='H. Felix Wittmann',
      author_email="hfwittmann@gmail.com",
      license='MIT',
      packages=['timeseries_utils'],
      install_requires=['markdown'],
      include_package_data=True,
      zip_safe=False,
      test_suite='nose.collector',
      tests_require=['nose', 'nose-cover3'],
      entry_points={
        'console_scripts': ['allocator=timeseries_utils.command_line:main']
      }
      
      )
