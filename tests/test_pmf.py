# Tests for the PMF algorithm implementation.
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

import unittest

import numpy as np

# Subject Under Test
from pmf import Pmf

class ModelInitializationTest(unittest.TestCase):

    def setUp(self):
        pmf = Pmf(numTopics=4, lamb=0.1, gamma=0.2)
        self.model = pmf.train(ratings=None, numRows=2, numCols=3,
                               maxIterations=0)

    def testRowBiasIsInitializedWithCorrectShape(self):
        self.assertTupleEqual((2,), self.model.rowBias.shape)

    def testColBiasIsInitializedWithCorrectShape(self):
        self.assertTupleEqual((3,), self.model.colBias.shape)

    def testRowFactorsIsInitializedWithCorrectShape(self):
        self.assertTupleEqual((2,4), self.model.rowFactors.shape)

    def testColFactorsIsInitializedWithCorrectShape(self):
        self.assertTupleEqual((3,4), self.model.colFactors.shape)

class BiasUpdateTest(unittest.TestCase):

    def testBiasRemainsUnchangedIfGammaIsZero(self):
        pmf = Pmf(numTopics=1, lamb=0.5, gamma=0)

        updatedBias = pmf._updateBias(12, -5)

        self.assertEqual(12, updatedBias)

    def testBiasIsCorrectlyUpdatedWhenLambdaIsZero(self):
        pmf = Pmf(numTopics=1, lamb=0, gamma=0.5)

        updatedBias = pmf._updateBias(10, 4)

        self.assertEqual(12, updatedBias)

    def testBiasIsCorrectlyUpdated(self):
        pmf = Pmf(numTopics=1, lamb=0.25, gamma=0.5)

        updatedBias = pmf._updateBias(8, 4)

        self.assertEqual(9, updatedBias)

class FactorsUpdateTest(unittest.TestCase):

    def testFactorsRemainsUnchangedIfGammaIsZero(self):
        pmf = Pmf(numTopics=3, lamb=0.5, gamma=0)

        factors1 = np.array([1,2,3])
        factors2 = np.array([4,5,6])
        updatedFactors = pmf._updateFactors(factors1, factors2, -5)

        self.assertListEqual([1,2,3], updatedFactors.tolist())

    def testFactorsAreCorrectlyUpdatedWhenLambdaIsZero(self):
        pmf = Pmf(numTopics=3, lamb=0, gamma=0.5)

        factors1 = np.array([1,2,3])
        factors2 = np.array([2,1,2])
        updatedFactors = pmf._updateFactors(factors1, factors2, 2)

        self.assertListEqual([3,3,5], updatedFactors.tolist())

    def testFactorsAreCorrectlyUpdated(self):
        pmf = Pmf(numTopics=3, lamb=0.5, gamma=0.25)

        factors1 = np.array([4,2,3])
        factors2 = np.array([5,2,1])
        updatedFactors = pmf._updateFactors(factors1, factors2, 3)

        self.assertListEqual([7.25, 3.25, 3.375],
                             updatedFactors.tolist())

if __name__ == '__main__':
    pass
