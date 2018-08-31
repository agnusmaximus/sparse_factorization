import sys
import random
import math
import tensorflow as tf
import os
import numpy as np
import uuid
from mnist_app import *
import pickle as pkl

def merge_dict(dst, src):
    result = dict(dst)
    for k,v in src.items():
        if k not in result:
            result[k] = v
    return result

def debug_generate_factorizable_matrix(dimension, product_of_n_matrices=3):
    A = None
    for i in range(product_of_n_matrices):
        v = np.random.randn(dimension, 1) / 2
        v = v.dot(v.T)
        if A is None:
            A = np.eye(dimension) + v
        else:
            A = A.dot(np.eye(dimension) + v)
    return A

def sparse_factorize(target_matrix, **hyperparameters):
    hyperparameter_defaults = {
        "l1_parameter" : 0.002, # Tune
        "l1_parameter_growth" : 1,
        "grow_l1_every_n_iter" : 1000,
        "intermediate_dimension" : 100, # Tune
        "ntrain_iters" : 10000,
        "n_matrices_to_factorize_into" : 16, # Tune 
        "lr" : 3e-2,
        "lr_decay" : .995,
        "decay_lr_every_n_iter" : 1000,
        "init_normal_stdev" : .01,
        "zero_out_threshold" : 1e-2, # Tune
        "print_every" : 100
    }

    hyperparameters = merge_dict(hyperparameters,
                                 hyperparameter_defaults)

    #hyperparameters["intermediate_dimension"] = 100

    print("sparse_factorize: Using hyperparameters")
    print(hyperparameters)

    # Construct graph
    M,K = target_matrix.shape
    Z = hyperparameters["intermediate_dimension"]

    # Construct variables to optimize
    to_optimize = [
        tf.Variable(tf.random_normal((M, Z), stddev=hyperparameters["init_normal_stdev"]) + tf.eye(M, Z), dtype=tf.float32)
    ] + [
        tf.Variable(tf.random_normal((Z, Z), stddev=hyperparameters["init_normal_stdev"]) +
                    tf.eye(Z), dtype=tf.float32) for i in range(hyperparameters["n_matrices_to_factorize_into"]-2)
    ] + [
        tf.Variable(tf.random_normal((Z, K), stddev=hyperparameters["init_normal_stdev"]) + tf.eye(Z, K), dtype=tf.float32)
    ]

    # Create matmul chain
    cur_matrix = to_optimize[0]
    for matrix in to_optimize[1:]:
        cur_matrix = tf.matmul(cur_matrix, matrix)

    # Matrix placeholders
    target_placeholder = tf.placeholder(tf.float32, shape=target_matrix.shape)

    # Create loss
    raw_mse_loss = tf.norm(cur_matrix - target_placeholder)

    # Add regularization
    loss = raw_mse_loss
    l1_parameter_placeholder = tf.placeholder(dtype=tf.float32)
    l1_regularizer = tf.contrib.layers.l1_regularizer(
        #scale=hyperparameters["l1_parameter"], scope=None
        scale=1.0, scope=None
    )
    regularization_penalty = tf.contrib.layers.apply_regularization(l1_regularizer, to_optimize)
    loss += l1_parameter_placeholder * regularization_penalty

    # Create optimizer
    lr_placeholder = tf.placeholder(dtype=tf.float32)
    opt = tf.train.GradientDescentOptimizer(learning_rate=lr_placeholder)
    #opt = tf.train.AdamOptimizer()
    minimize = opt.minimize(loss)

    # 0 out values that are small
    zero_ops = []
    with tf.control_dependencies([minimize]):
        for matrix in to_optimize:
            mask = tf.cast(tf.greater(tf.abs(matrix), hyperparameters["zero_out_threshold"] *
                                      tf.ones_like(matrix, dtype=tf.float32)), tf.float32)
            zero_ops.append(tf.assign(matrix, tf.multiply(mask, matrix)))
    
        minimize = tf.group(zero_ops)

    # Train
    cur_learning_rate = hyperparameters["lr"]
    log_data = []
    cur_l1 = hyperparameters["l1_parameter"]

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for train_iter in range(hyperparameters["ntrain_iters"]):
        
        if train_iter % hyperparameters["decay_lr_every_n_iter"] == 0:
            cur_learning_rate *= hyperparameters["lr_decay"]

        if train_iter % hyperparameters["grow_l1_every_n_iter"] == 0:
            cur_l1 *= hyperparameters["l1_parameter_growth"]

        sampled_x = np.random.randn(target_matrix.shape[1], 1)
        total_loss_materialized, loss_mse_materialized, _ = (
            sess.run([loss, raw_mse_loss, minimize], feed_dict={
                target_placeholder: target_matrix,
                lr_placeholder: cur_learning_rate,
                l1_parameter_placeholder : cur_l1
            })
        )

        if train_iter % hyperparameters["print_every"] == 0:

            # Compute sum of nnz elements
            n_nnz_elements = 0
            for matrix in to_optimize:
                matrix_materialized = sess.run(matrix)
                n_nnz_elements += np.count_nonzero(matrix_materialized)            
            
            print("Iteration %d of %d, Total Loss (+l1 penalty): %g; MSE loss: %g, nnz: %d" % (
                train_iter,
                hyperparameters["ntrain_iters"],
                total_loss_materialized,
                loss_mse_materialized,
                n_nnz_elements
            ))

            log_data.append({
                "total_loss" : total_loss_materialized,
                "loss_mse" : loss_mse_materialized,
                "nnz" : n_nnz_elements
            })

    # Compute number of nonzero elements
    n_nnz_elements = 0
    for matrix in to_optimize:
        matrix_materialized = sess.run(matrix)
        n_nnz_elements += np.count_nonzero(matrix_materialized)
    original_nnz_elements = np.count_nonzero(target_matrix)

    # Compute frobenius error
    product_of_matrices = sess.run(cur_matrix)
    frobenius_error = np.linalg.norm(product_of_matrices - target_matrix)

    # Materialize actual matrices
    materialized_factorizations = []
    for matrix in to_optimize:
        matrix_materialized = sess.run(matrix)
        materialized_factorizations.append(matrix_materialized)
    
    # Return results
    results = {
        "factorized_matrices" : materialized_factorizations,
        "product_of_matrices" : product_of_matrices,
        "frobenius_error" : frobenius_error,
        "original_nnz_elements" : original_nnz_elements,
        "target_matrix" : target_matrix,
        "n_nnz_elements" : n_nnz_elements,
        "hyperparameter_setting" : hyperparameters,
    }

    # Print summary
    print("sparse_factorize: Summary")
    print("-------------------------")
    print("original matrix nnz elements: %d" % original_nnz_elements)
    print("nnz elements: %d" % n_nnz_elements)
    print("frobenius error: %g" % frobenius_error)
    print("nnzs of factorized matrices: %s" % str([np.count_nonzero(x) for x in materialized_factorizations]))
    return results

def sparse_factorize_tf(app, use_hyperparameters):
    tf.reset_default_graph()

    # Final results
    final_results = {}

    # Train the app
    app.train(niters=1000)

    # Evaluate the app
    test_score = app.eval()
    print("sparse_factorize_tensorflow.py: test score - %g" % test_score)
    
    # Factorize linear layers
    layers_to_factorize = app.get_layers_to_factorize()
    sess = app.get_sess()
    final_results["factorize_results"] = []
    for layer in layers_to_factorize:

        print("sparse_factorize_tensorflow.py: factorizing: %s" % str(layer))

        # Extract raw weights of layer
        layer_materialized = sess.run(layer)

        # Be aware this function adds ops to the graph
        sparse_factorization_results = sparse_factorize(layer_materialized, **use_hyperparameters)

        # Override layer weights with factorized result
        prod = sparse_factorization_results["product_of_matrices"]
        sess.run(tf.assign(layer, prod))

        final_results["factorize_results"].append({
            "factorization_result_details" : sparse_factorization_results,
            "layer_name" : str(layer.name),
            "layer_weights" : layer_materialized
        })

    # Evaluate the app again
    new_test_score_after_factorization = app.eval()
    print("sparse_factorize_tensorflow.py: old test score - %g" % test_score)
    print("sparse_factorize_tensorflow.py: test score - %g" % new_test_score_after_factorization)

    final_results["original_score"] = test_score
    final_results["new_score_after_factorized"] = new_test_score_after_factorization

    return final_results

def sample_from_space(hyperparameter_space):
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
    
if __name__ == "__main__":

    # Create the results directory
    results_directory = sys.argv[1]
    assert os.path.exists(results_directory)

    # Configuration for random sampling
    hyperparameters = {}
    if len(sys.argv) >= 3:
        hyperparameter_space = eval(open(sys.argv[2], "r").read())
        hyperparameters = sample_from_space(hyperparameter_space)
        print("Random sampling hyperparameters")
        print("-------------------------------")
        print(hyperparameters)

    # Create app
    app = MnistApp()

    # Factorize
    results = sparse_factorize_tf(app, hyperparameters)

    # Write to results dir
    results_path = "%s/%s" % (results_directory, uuid.uuid4())
    while os.path.exists(results_path):
        results_path = "%s/%s" % (results_directory, uuid.uuid4())

    with open(results_path, "wb") as f:
        pkl.dump(results, f)
