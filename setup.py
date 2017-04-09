# Configuration of the experiments package.
# Copyright (C) 2017 Cesar Perez
# This file is part of pmf.
#
# pmf is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# pmf is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with pmf.  If not, see <http://www.gnu.org/licenses/>.

from setuptools import find_packages, setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='pmf',
      version='0.0.1',

      description='Implementation of the Probabilistic Matrix Factorization algorithm',
      long_description=readme(),

      url='http://github.com/wnohang/python-pmf',
      license='GPLv3',

      keywords='pmf',
      classifiers=[
          'Development Status :: 1 - Planning',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3.2',
          'Programming Language :: Python :: 3.3',
          'Programming Language :: Python :: 3.4',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Topic :: Scientific/Engineering :: Artificial Intelligence'
      ],

      author='Cesar Perez',
      author_email='cesar@bigtruedata.com',

      packages=find_packages(exclude=['test']),
      install_requires=[
          'numpy'
      ],

      test_suite='tests',

      zip_safe=True
)
