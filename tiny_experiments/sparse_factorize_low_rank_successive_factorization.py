"""
1. Generate target matrix A = I + u v^T for some vector u, v of size n, so A has size n.
2. We know A = [I_n I_n] [I_n 0_{n x 1} ; 0_{n x n}  u] [I_n ; v^T], which is a factorization of dimension n x 2n, 2n x (n+1), (n+1) x n. The total sparsity here is 6n (counting the I_n).
3. Run gradient descent to find this factorization A = B C D where B is n x 2n, C is 2n x (n+1), D is (n+1) x n, with total sparsity around 6n.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import sys
import math
import random
import pickle

def factorize(A, hyperparameters):
    #l1_regularizer_parameter = hyperparameters["l1_regularizer_parameter"]
    #dimension = hyperparameters["dimension"]
    #zero_out_threshold = hyperparameters["zero_out_threshold"]
    #lr = hyperparameters["lr"]    
    #niters = hyperparameters["niters"]

    l1_regularizer_parameter = .01
    zero_out_thresholds = [1e-3, 1e-3]
    lr = 5e-2
    niters = 10000
    intermediate_dimension = 200
    
    # Factorize A in to matrices of shape shapes
    tf.reset_default_graph()

    shapes = [(A.shape[0], intermediate_dimension), (intermediate_dimension, A.shape[1])]
    variables = []
    for shape in shapes:
        variables.append(tf.Variable(tf.random_normal(shape, stddev=.1 + tf.eye(*shape), dtype=tf.float32)))
        
    # Multiply the variables together
    to_optimize = variables[0] + tf.eye(*tuple(variables[0].get_shape().as_list()))
    for variable in variables[1:]:
        to_optimize = tf.matmul(to_optimize, tf.eye(*tuple(variable.get_shape().as_list())) + variable)
        #to_optimize = tf.matmul(to_optimize, variable)

    assert(tuple(to_optimize.get_shape().as_list()) == tuple(A.shape))
        
    # Construct the optimization
    target_placeholder = tf.placeholder(tf.float32, shape=A.shape)
    loss_frobenius_error = tf.norm(target_placeholder - to_optimize)

    # Add l1 loss
    l1_parameter_placeholder = tf.placeholder(dtype=tf.float32)
    l1_regularizer = tf.contrib.layers.l1_regularizer(
        scale=1.0, scope=None
    )
    regularization_penalty = tf.contrib.layers.apply_regularization(l1_regularizer, variables)
    loss = loss_frobenius_error + l1_parameter_placeholder * regularization_penalty
    #loss = loss_frobenius_error

    # Create opt
    lr_placeholder = tf.placeholder(dtype=tf.float32)
    opt = tf.train.GradientDescentOptimizer(learning_rate=lr_placeholder)
    minimize = opt.minimize(loss)

    # Zero out values below absolute threshold
    zero_ops = []
    with tf.control_dependencies([minimize]):
        for matrix, thresh in zip(variables, zero_out_thresholds):
            mask = tf.cast(tf.greater(tf.abs(matrix), thresh *
                                      tf.ones_like(matrix, dtype=tf.float32)), tf.float32)
            zero_ops.append(tf.assign(matrix, tf.multiply(mask, matrix)))    
        minimize_and_zero_out = tf.group(zero_ops)    

    # Do optimization
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    results = []
    for i in range(niters):
        _, loss_materialized, loss_frobenius_error_materialized = sess.run([minimize_and_zero_out, loss, loss_frobenius_error],
                                                              feed_dict={
                                                                  target_placeholder : A,
                                                                  l1_parameter_placeholder : l1_regularizer_parameter,
                                                                  lr_placeholder : lr
                                                              })
        
        
        if i % 100 == 0:
            # Also calculate sparsity
            variables_materialized = sess.run(variables)
            total_number_of_nnzs = sum([np.count_nonzero(x) for x in variables_materialized])
            nnzs = [np.count_nonzero(x) for x in variables_materialized]
            
            print("Loss: %g, Loss_frob_error: %g, # nnzs in factored matrices: %d, frob_error / (||u||||v||): %g, nnzs: %s" % (
                loss_materialized, loss_frobenius_error_materialized, total_number_of_nnzs,
                loss_frobenius_error_materialized / (np.linalg.norm(u) * np.linalg.norm(v)),
                str(nnzs)))
            results.append((total_number_of_nnzs, loss_frobenius_error_materialized))

    all_results = {
        "hyperparameters" : hyperparameters,
        "target_matrix" : A,
        "results" : results
    }
    vs = sess.run(variables)
    print([np.count_nonzero(v) for v in vs])
    return [(v + np.eye(*v.shape), np.count_nonzero(v)) for v in vs], all_results
    #return [(v, np.count_nonzero(v)) for v in vs], all_results

def successive_factorization(A):
    matrices = [(A, np.count_nonzero(A))]
    #while True:
    for i in range(4):
        matrices_nnzs = [x[1] for x in matrices]
        print("--------------------------------------")
        print("TOTAL NNZS IN SUCCESSIVE FACTORIZATION: %d" % sum(matrices_nnzs))
        print("NNZS IN FACTORIZATION: %s" % str(matrices_nnzs))
        print("NNZS IN ORIGINAL MATRIX: %d" % np.count_nonzero(A))
        print("TOTAL NNZS (including ident): %g" % (sum([np.count_nonzero(x[0]) for x in matrices])))

        cur_matrix = matrices[0][0]
        for matrix in matrices[1:]:
            cur_matrix = cur_matrix.dot(matrix[0])
        frob_error = np.linalg.norm(A-cur_matrix)
        
        print("FROBENIUS ERROR: %g" % frob_error)
        print("NORMALIZED FROBENIUS ERROR: %g" % (frob_error / np.linalg.norm(A)))
        print("--------------------------------------")        
        
        matrix_to_factorize_index = np.argmax(matrices_nnzs)
        matrix_to_factorize = matrices[matrix_to_factorize_index]
        factorized, result_details = factorize(matrix_to_factorize[0], hyperparameters)

        nnzs_of_factorized = [x[1] for x in factorized]
        print(nnzs_of_factorized, matrix_to_factorize[1])
        if sum(nnzs_of_factorized) < matrix_to_factorize[1]:
            matrices = matrices[:matrix_to_factorize_index] + factorized + matrices[matrix_to_factorize_index+1:]

    for ind, factor in enumerate(matrices):
        plt.imshow(factor[0], cmap='hot', interpolation='nearest')            
        plt.savefig("testing_%d.png" % ind)        
        #print(factorized)

def sample(hyperparameter_space):
    hyp = {}
    for k,v in hyperparameter_space.items():
        if type(v) == type({}):
            assert "min" in v
            assert "max" in v
            assert "scale" in v
            assert "type" in v

            mini = v["min"]
            maxi = v["max"]
            scale = v["scale"]
            typ = v["type"]

            if scale == "linear":
                sampled_value = random.uniform(mini, maxi)
            else:
                mini_log = math.log10(mini)
                maxi_log = math.log10(maxi)
                sampled_value = 10**random.uniform(mini_log, maxi_log)

            if typ == "int":
                sampled_value = int(sampled_value)
                
            hyp[k] = sampled_value
        else:
            hyp[k] = v
    return hyp

if __name__=="__main__":
    np.random.seed(0)
    tf.set_random_seed(0)
    
    outdir = sys.argv[1]

    hyperparameter_space = {
        "dimension" : 10,
        "l1_regularizer_parameter" : {
            "min" : 1e-5,
            "max" : 1e-2,
            "type" : "float",
            "scale" : "log"
        },
        "zero_out_threshold" : {
            "min" : 1e-7,
            "max" : 1e-3,
            "type" : "float",
            "scale" : "log"
        },
        "lr" : {
            "min" : 1e-4,
            "max" : 1e-2,
            "type" : "float",
            "scale" : "log",
        },
        "niters" : 100000
    }
    
    hyperparameters = sample(hyperparameter_space)
    dimension = 100

    # Generate target matrix A = I + u v ^T, u,v of size n
    A = np.eye(dimension)
    for i in range(4):
        u = np.random.randn(dimension, 1)/5
        v = np.random.randn(dimension, 1)/5
        A = A.dot(np.eye(dimension) + u.dot(v.T))

        #u_2 = np.random.randn(dimension, 1)/5
        #v_2 = np.random.randn(dimension, 1)/5
        #A = A.dot(np.eye(dimension) + u_2.dot(v_2.T))

        #u_3 = np.random.randn(dimension, 1)/5
        #v_3 = np.random.randn(dimension, 1)/5
        #A = A.dot(np.eye(dimension) + u_3.dot(v_3.T))    

    ##################################################################3
    # Synthetic solution
    a_1 = np.concatenate([np.eye(dimension), np.eye(dimension)], axis=1)
    a_2 = np.concatenate([
        np.concatenate([np.eye(dimension), np.zeros((dimension, 1))], axis=1),
        np.concatenate([np.zeros((dimension,dimension)), u.reshape(dimension, 1)], axis=1)
    ], axis=0)
    a_3 = np.concatenate([
        np.eye(dimension),
        v.T.reshape(1, dimension)
    ], axis=0)
    solution = [a_1, a_2, a_3]
    print(a_1.shape, a_2.shape, a_3.shape)
    #assert(np.linalg.norm(a_1.dot(a_2).dot(a_3) - A) < 1e-7)
    ##################################################################

    successive_factorization(A)
