import sys
import unittest
import os
import numpy as np
import random
import tensorflow as tf
from SparseFactorization.sparse_factorization import SparseFactorizationWithL1AndPruningTF
from SparseFactorization.utils import utils

class SparseFactorizationWithL1AndPruningTFTest(unittest.TestCase):

    def test_creation(self):
        factorization = SparseFactorizationWithL1AndPruningTF()

    def test_factorize_small_matrix(self):
        utils.set_all_seeds(0)
        
        hyperparams = {
            "l1_parameter" : 0,
            "learning_rate" : 1e-2,
            "intermediate_dimension" : 20,
            "training_iters" : 3000,
            "seed" : 0
        }
        factorization = SparseFactorizationWithL1AndPruningTF(hyperparameters=hyperparams)            
        small_matrix = np.random.randn(10, 10)
        factors, result_details = factorization.factorize(small_matrix)

        assert(result_details["calculated_frobenius_error"] < .1)
        assert(abs(result_details["calculated_frobenius_error"] - 0.09581156730879141) <= 1e-6)

    def test_factorize_small_matrix_twice(self):

        # Run factorize twice with same seed -- should be the same
        errors = []
        for i in range(2):
            utils.set_all_seeds(0)

            hyperparams = {
                "l1_parameter" : 0,
                "learning_rate" : 1e-2,
                "intermediate_dimension" : 20,
                "training_iters" : 3000,
                "seed" : 0
            }
            factorization = SparseFactorizationWithL1AndPruningTF(hyperparameters=hyperparams)            
            small_matrix = np.random.randn(10, 10)
            factors, result_details = factorization.factorize(small_matrix)

            errors.append(result_details["calculated_frobenius_error"])

        assert(abs(errors[0]-errors[1]) < 1e-6)        

if __name__ == "__main__":
    print("Running test_SparseFactorizationWithL1AndPruningTF.py")
    unittest.main()
