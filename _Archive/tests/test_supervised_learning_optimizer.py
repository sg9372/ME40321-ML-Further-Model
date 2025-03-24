import unittest
import numpy as np
from _Archive.src.supervised_learning_optimizer import supervised_learning_optimizer

class test_supervised_unit_test(unittest.TestCase):

    def test_optimized_weights_sum_to_one(self):
        values = np.array([
            [100, 200, 300],
            [110, 210, 310],
            [120, 220, 320]
        ])
        emissions = np.array([10, 20, 30])
        old_weights = np.array([0.3, 0.4, 0.3])

        optimized_weights = supervised_learning_optimizer(values, emissions, old_weights)
        self.assertAlmostEqual(np.sum(optimized_weights), 1.0, places=5)

    def test_optimized_weights_non_negative(self):
        values = np.array([
            [100, 200, 300],
            [110, 210, 310],
            [120, 220, 320]
        ])
        emissions = np.array([10, 20, 30])
        old_weights = np.array([0.3, 0.4, 0.3])

        optimized_weights = supervised_learning_optimizer(values, emissions, old_weights)
        self.assertTrue(np.all(optimized_weights >= 0))

    def test_optimized_weights_shape(self):
        values = np.array([
            [100, 200, 300],
            [110, 210, 310],
            [120, 220, 320]
        ])
        emissions = np.array([10, 20, 30])
        old_weights = np.array([0.3, 0.4, 0.3])

        optimized_weights = supervised_learning_optimizer(values, emissions, old_weights)
        self.assertEqual(optimized_weights.shape, old_weights.shape)
    
    def test_optimized_weights_with_different_data(self):
        values = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ])
        emissions = np.array([1, 2, 3])
        old_weights = np.array([0.2, 0.3, 0.5])

        optimized_weights = supervised_learning_optimizer(values, emissions, old_weights)
        self.assertAlmostEqual(np.sum(optimized_weights), 1.0, places=5)
        self.assertTrue(np.all(optimized_weights >= 0))
        self.assertEqual(optimized_weights.shape, old_weights.shape)

        self.assertIsInstance(optimized_weights, np.ndarray)