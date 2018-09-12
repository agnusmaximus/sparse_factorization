import tensorflow as tf
import sys
import time
import numpy as np
from sparse_factorize_tensorflow import *

if __name__=="__main__":
    np.random.seed(0)
    tf.set_random_seed(0)
    random.seed(0)
    
    factorize_dimension = int(sys.argv[1])
    product_of_n_matrices_generation = int(sys.argv[2])
    hyperparameter_space = eval(open(sys.argv[3], "r").read())
    outdir = sys.argv[4]
    if len(sys.argv) > 5:
        load_from = sys.argv[5]
    else:
        load_from = None

    matrix_to_factorize = debug_generate_factorizable_matrix(factorize_dimension, product_of_n_matrices_generation)

    n_matrices_ranges = [2, 10, 15, 20, 30]
    #n_matrices_ranges = [30]
    sparsity_ranges = [.1, .2, .4, .6, .8, .9]
    
    binned_results = {}
    for a in n_matrices_ranges:
        for b in sparsity_ranges:
            binned_results[(a,b)] = []

    # Load from pikle
    if load_from is not None:
        binned_results = pkl.load(open(load_from, "rb"))
        for k,v in binned_results.items():
            print(k, len(v))

    n_to_collect_per_bin = 15

    while True:
        for n_matrices in n_matrices_ranges:
            hyperparameters = sample_from_space(hyperparameter_space)
            hyperparameters["n_matrices_to_factorize_into"] = n_matrices
            hyperparameters["intermediate_dimension"] = factorize_dimension
            results = sparse_factorize(matrix_to_factorize, **hyperparameters)

            n_nnz_elements = results["n_nnz_elements"]
            orig_nnz_elements = np.prod(results["target_matrix"].shape)
            result_sparsity = n_nnz_elements / orig_nnz_elements

            for i in range(len(sparsity_ranges)):
                if i == 0:
                    lower = 0
                else:
                    lower = sparsity_ranges[i-1]
                upper = sparsity_ranges[i]
                if lower <= result_sparsity and result_sparsity <= upper:
                    binned_results[(n_matrices, upper)].append(results)

        lengths = [len(v) for k,v in binned_results.items()]
        print("Progress: ", lengths)
        if min(lengths) >= n_to_collect_per_bin:
            break

        with open("%s/factorization_experiment_dim=%d_nprod=%d" % (outdir, factorize_dimension, product_of_n_matrices_generation), "wb") as f:
              pkl.dump(binned_results, f)
    
        
        
        
