import sys
import unittest
import os
import tensorflow as tf
from SparseFactorization.sparse_factorization import SparseFactorizationBase

class SparseFactorizationExample(SparseFactorizationBase):

    def __init__(self, hyperparameters={}):
        super().__init__(hyperparameters)

    def get_hyperparameter_defaults(self):
        return {
            "a" : 1,
            "b" : 2,
        }

class SparseFactorizationBaseTest(unittest.TestCase):

    def test_hyperparameter_merging(self):
        target_hyperparams = {"a" : 2, "b": 3}
        fact_obj = SparseFactorizationExample(target_hyperparams)
        fact_obj_parameters = fact_obj.get_hyperparameters()
        assert(fact_obj_parameters == target_hyperparams)

    def test_hyperparameter_extra_args_error(self):
        hyperparams = {"a": 1, "c": 2}
        succeeded = True
        try:
            fact_obj = SparseFactorizationExample(hyperparams)
        except:
            succeeded = False
        assert(not succeeded)

    def test_hyperparameter_default_args(self):
        fact_obj = SparseFactorizationExample()
        assert(fact_obj.get_hyperparameters() == fact_obj.get_hyperparameter_defaults())
        
if __name__ == "__main__":
    unittest.main()
    
