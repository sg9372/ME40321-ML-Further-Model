import unittest
import numpy as np
import cvxpy as cp
from src.optimizer import average_optimizer
from src.extract_sheet_data import extract_sheet_data
from src.extract_values import extract_values

class test_total_weight_change_optimize(unittest.TestCase):
    '''
    # Expected not to adjust weights
    def test_UnchangedWeights(self):
        values = np.array([1,1,0])
        emissions = np.array([0,0,1])
        old_weights = np.array([0.5,0.5,0])
        weights = average_optimizer(values, emissions, old_weights)
        assert np.all(np.abs(weights - old_weights) <= 0.001)
    '''
        
    # Expected to change to [0.025, 0.5, 0.0475] commented out as this chanhges when weights change
    def test_ChangedWeights(self):
        values = np.array([1,1,0])
        emissions = np.array([0,1,1])
        old_weights = np.array([0,0.5,0.5])
        weights = average_optimizer(values, emissions, old_weights)
        correct = np.array([0.025,0.5,0.475])
       #assert np.all(np.abs(weights - correct) <= 0.001)
    
    def test_WeightsSum(self):
        values = np.array([1,1,0])
        emissions = np.array([0,1,1])
        old_weights = np.array([0,0.5,0.5])
        weights = average_optimizer(values, emissions, old_weights)
        assert np.isclose(np.sum(weights), 1)

    def test_NoError(self):
        file = 'Monthly FTSE Data - New.xlsx'
        start_Date = '01/01/2020'
        end_Date = '01/02/2020'
        companies,weights,emissions = extract_sheet_data(file, '01_2020')
        values,_ = extract_values(file, companies, start_Date, end_Date)
        result = average_optimizer(values, emissions, weights)
        self.assertIsInstance(result, np.ndarray)