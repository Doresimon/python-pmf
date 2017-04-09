# Implementation of the Probabilistic Matrix Factorization algorithm.
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

import numpy as np

class Pmf:
    '''Probabilistic Matrix Factorization

    Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix Factorization
    Techniques for Recommender Systems. Computer, 42(8), 30-37.
    '''

    class Model:

        def __init__(self, numRows, numCols, numTopics):
            self.ratingAverage = np.random.rand(1)[0]

            self.rowBias = np.random.rand(numRows)
            self.colBias = np.random.rand(numCols)

            self.rowFactors = np.random.rand(numRows, numTopics)
            self.colFactors = np.random.rand(numCols, numTopics)

        def predict(self, row, col):
            bias = self.rowBias[row] + self.colBias[col]

            score = self._scorePrediction(self.rowFactors[row],
                                          self.colFactors[col])

            return self.ratingAverage + bias + score

        def _scorePrediction(self, rowFactors, colFactors):
            return np.dot(rowFactors, colFactors)

    def __init__(self, numTopics, lamb, gamma):
        self.numTopics = numTopics
        self.lamb = lamb
        self.gamma = gamma

    def train(self, ratings, numRows, numCols, maxIterations):
        model = Pmf.Model(numRows, numCols, self.numTopics)

        iteration = 0
        while iteration < maxIterations:
            self._iterate(model, ratings)
            iteration += 1

        return model

    def _iterate(self, model, ratings):
        for (row, col, score) in ratings:
            error = score - model.predict(row, col)

            rowBias = model.rowBias[row]
            colBias = model.colBias[col]
            rowFactors = model.rowFactors[row]
            colFactors = model.colFactors[col]

            model.rowBias[row] = self._updateBias(rowBias, error)
            model.colBias[row] = self._updateBias(colBias, error)

            f1 = self._updateFactors(rowFactors, colFactors, error)
            model.rowFactors[row] = f1

            f2 = self._updateFactors(colFactors, rowFactors, error)
            model.colFactors[row] = f2

    def _updateBias(self, bias, error):
        return bias + self.gamma * (error - self.lamb*bias)

    def _updateFactors(self, factors1, factors2, error):
        return factors1 + self.gamma * (error*factors2 -
                                        self.lamb*factors1)

    def _initializeFactors(self, numFactors, numTopics):
        return np.random.rand(numFactors, numTopics)

if __name__ == '__main__':
    pass
