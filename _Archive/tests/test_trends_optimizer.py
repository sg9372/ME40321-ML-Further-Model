import unittest
import numpy as np
import cvxpy as cp
from _Archive.src.trends_optimizer import trend_optimizer

class test_total_weight_change_optimize(unittest.TestCase):

    # Expected not to adjust weights
    def test_UnchangedWeights1(self):
        values = np.array([[1,1,0], [2, 2, 0]])
        emissions = np.array([[0,0,1], [0,0,2]])
        old_weights = np.array([0.5,0.5, 0])
        end = 3
        weights = trend_optimizer(values, emissions, old_weights, end)
        assert np.all(np.abs(weights - old_weights) <= 0.001)
    
    # Should change
    def test_ChangingWeights(self):
        values = np.array([[1,1,5], [2, 3, 4], [1,1,3]])
        emissions = np.array([[0,0,0], [0,0,0], [0,0,0]])
        old_weights = np.array([0.3, 0.4, 0.3])
        end = 5
        weights = trend_optimizer(values, emissions, old_weights, end)
        result = np.array([0.3, 0.45, 0.25])
        assert np.all(np.abs(result - weights) <= 0.01)