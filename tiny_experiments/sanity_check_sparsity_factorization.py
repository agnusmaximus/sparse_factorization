"""
1. Generate target matrix A = I + u v^T for some vector u, v of size n, so A has size n.
2. We know A = [I_n I_n] [I_n 0_{n x 1} ; 0_{n x n}  u] [I_n ; v^T], which is a factorization of dimension n x 2n, 2n x (n+1), (n+1) x n. The total sparsity here is 6n (counting the I_n).
3. Run gradient descent to find this factorization A = B C D where B is n x 2n, C is 2n x (n+1), D is (n+1) x n, with total sparsity around 6n.
"""

import tensorflow as tf
import numpy as np
import sys
import math
import random
import pickle

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
    l1_regularizer_parameter = hyperparameters["l1_regularizer_parameter"]
    dimension = hyperparameters["dimension"]
    zero_out_threshold = hyperparameters["zero_out_threshold"]
    lr = hyperparameters["lr"]    
    niters = hyperparameters["niters"]
    print(hyperparameters)

    l1_regularizer_parameter = .01
    dimension = 10
    zero_out_threshold = 3e-3
    lr = 1e-2
    niters = 100000

    # Generate target matrix A = I + u v ^T, u,v of size n
    u = np.random.randn(dimension, 1)/5
    v = np.random.randn(dimension, 1)/5
    A = np.eye(dimension) + u.dot(v.T)
    
    u_2 = np.random.randn(dimension, 1)/5
    v_2 = np.random.randn(dimension, 1)/5
    A = A.dot(np.eye(dimension) + u_2.dot(v_2.T))

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

    # Set up factorization:
    # A = B C D where B is n x 2n, C is 2n x (n+1), D is (n+1) x n, with total sparsity around 6n
    #shapes = [(dimension, 2*dimension),
    #          (2*dimension, dimension+1),
    #          (dimension+1, dimension)]
    shapes = [(dimension, dimension+1), (dimension+1, dimension)]
    #shapes = [(dimension, dimension), (dimension, dimension)]
    variables = []
    for shape in shapes:
        variables.append(tf.Variable(tf.random_normal(shape, stddev=.001 + tf.eye(*shape), dtype=tf.float32)))
        
    # Multiply the variables together
    to_optimize = variables[0]
    for variable in variables[1:]:
        #to_optimize = tf.matmul(to_optimize, tf.eye(*tuple(variable.get_shape().as_list())) + variable)
        to_optimize = tf.matmul(to_optimize, variable)

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

    # Create opt
    lr_placeholder = tf.placeholder(dtype=tf.float32)
    opt = tf.train.GradientDescentOptimizer(learning_rate=lr_placeholder)
    minimize = opt.minimize(loss)

    # Zero out values below absolute threshold
    zero_ops = []
    with tf.control_dependencies([minimize]):
        for matrix in variables:
            mask = tf.cast(tf.greater(tf.abs(matrix), zero_out_threshold *
                                      tf.ones_like(matrix, dtype=tf.float32)), tf.float32)
            zero_ops.append(tf.assign(matrix, tf.multiply(mask, matrix)))    
        minimize_and_zero_out = tf.group(zero_ops)    

    # Do optimization
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # For debug, initialize the variables to the solution
    """
    for variable, sol in zip(variables, solution):
        sess.run(variable.assign(sol + tf.random_normal(sol.shape, stddev=.1)))
        #sess.run(variable.assign(sol))
    """

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
            total_number_of_nnzs = sum([np.count_nonzero(x.flatten()) for x in variables_materialized])
            
            print("Loss: %g, Loss_frob_error: %g, # nnzs in factored matrices: %d, frob_error / (||u||||v||): %g" % (
                loss_materialized, loss_frobenius_error_materialized, total_number_of_nnzs,
                loss_frobenius_error_materialized / (np.linalg.norm(u) * np.linalg.norm(v))))
            results.append((total_number_of_nnzs, loss_frobenius_error_materialized))

    all_results = {
        "hyperparameters" : hyperparameters,
        "target_matrix" : A,
        "results" : results
    }
    save_name = ",".join([str(k)+"="+str(v) for k,v in hyperparameters.items()])
    with open("%s/%s_save" % (outdir, save_name), "wb") as f:
        pickle.dump(all_results, f)

    
