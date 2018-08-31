import sys
import random
import numpy as np
import tensorflow as tf

np.random.seed(0)

def evaluate_sparse_factorization(A,
                                  n_sparse_matrices=0,
                                  factoring_product_of_n_matrices=0,
                                  dimension=-1,
                                  l1_regularization_factor=10.0,
                                  gauss_noise=False,
                                  lr=1e-5,
                                  lr_decay=.95,
                                  ntrain_iters=40000,
                                  verbose=True):


    # Make sure A is square
    assert(A.shape[0] == A.shape[1])
    assert(n_sparse_matrices > 0)
    assert(A.shape[0] == dimension)

    # Random seed
    tf.reset_default_graph()
    tf.set_random_seed(0)

    # Create sparse matrices
    #matrices = [tf.Variable(tf.random_normal(A.shape, stddev=.01) + tf.eye(A.shape[0]), dtype=tf.float32) for i in range(n_sparse_matrices)]
    matrices = [tf.Variable(tf.random_normal(A.shape, stddev=1e-2) + tf.eye(A.shape[0]), dtype=tf.float32) for i in range(n_sparse_matrices)]
    #matrices = [tf.Variable(tf.random_normal(A.shape, stddev=1e-5), dtype=tf.float32) for i in range(n_sparse_matrices)]
    #matrices = [tf.Variable(tf.random_normal(A.shape, stddev=1e-5), dtype=tf.float32) for i in range(n_sparse_matrices)]

    # Create matmul chain
    cur_matrix = matrices[0]
    for matrix in matrices[1:]:
        cur_matrix = tf.matmul(cur_matrix, matrix)
        #print(matrix.shape)
        #cur_matrix = tf.matmul(cur_matrix, (matrix + tf.eye(50)))
        #z = tf.add(matrix, tf.eye(int(matrix.shape[0])))
        #cur_matrix = tf.matmul(cur_matrix, tf.add(matrix, tf.eye(matrix.shape[0], num_columns=matrix.shape[0])))
        #cur_matrix = tf.matmul(cur_matrix, z)
        
    # Placeholders for A and x
    A_placeholder = tf.placeholder(tf.float32, shape=A.shape)
    x_placeholder = tf.placeholder(tf.float32, shape=(A.shape[0], 1))

    # Create loss
    predictions = tf.matmul(cur_matrix, x_placeholder)
    expected = tf.matmul(A_placeholder, x_placeholder)
    loss_pred_err_mse = tf.losses.mean_squared_error(expected, predictions) 

    # Regularize for sparsity
    loss = loss_pred_err_mse
    l1_regularizer = tf.contrib.layers.l1_regularizer(
        scale=l1_regularization_factor, scope=None
    )
    regularization_penalty = tf.contrib.layers.apply_regularization(l1_regularizer, tf.trainable_variables())
    #regularization_penalty = tf.contrib.layers.apply_regularization(l1_regularizer, [matrices[0]])
    loss += regularization_penalty

    # Create optimizer
    lr_pl = tf.placeholder(dtype=tf.float32)
    opt = tf.train.GradientDescentOptimizer(learning_rate=lr_pl)
    minimize = opt.minimize(loss)
    minimize_alternating = [opt.minimize(loss, var_list=[x]) for x in matrices]    

    # Set to 0 values that are small
    zero_ops = []
    with tf.control_dependencies([minimize]):
        for matrix in matrices:
            mask = tf.cast(tf.greater(tf.abs(matrix), 1e-5 * tf.ones_like(matrix, dtype=tf.float32)), tf.float32)
            zero_ops.append(tf.assign(matrix, tf.multiply(mask, matrix)))
    
        minimize = tf.group(zero_ops)

    #zero_ops_alternating = []
    #for x, c_minimize_op in zip(matrices, minimize_alternating):
    #    zero_ops_alternating.append(tf.group(zero_ops + [c_minimize_op]))
    #minimize_alternating = zero_ops_alternating

    # Gaussian noise op
    gaussian_noise_ops = []
    for matrix in matrices:
        noise = tf.random_normal(shape=matrix.get_shape(), mean=0.0, stddev=1e-4, dtype=tf.float32)
        gaussian_noise_ops.append(matrix.assign_add(noise))
    add_gaussian_noise_to_matrices = tf.group(gaussian_noise_ops)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    log_data = []
    for i in range(ntrain_iters):

        cur_x = np.random.randn(A.shape[0], 1)
        
        loss_value, loss_pred_err_mse_value, _ = sess.run([loss, loss_pred_err_mse, minimize], feed_dict={A_placeholder: A,
                                                                                                          x_placeholder: cur_x,
                                                                                                          lr_pl: lr})

        #loss_value, loss_pred_err_mse_value, _ = sess.run([loss, loss_pred_err_mse, minimize_alternating[0]], feed_dict={A_placeholder: A,
        #                                                                                                                 x_placeholder: cur_x,
        #                                                                                                                 lr_pl: lr})        
        
        if gauss_noise:
            if i % 1000 == 0:
                sess.run(add_gaussian_noise_to_matrices)
        
        if i % 1000 == 0:
            if verbose:
                print("%d of %d" % (i, ntrain_iters))        
                print(loss_value, loss_pred_err_mse_value)
            log_data.append({"total_loss_value" : loss_value,
                             "prediction_error_loss" : loss_pred_err_mse_value})
            lr *= lr_decay

    # Compute sparsity of matrices
    n_total_elements, n_nnz_elements = 0, 0
    for matrix in matrices:
        matrix_materialized = sess.run(matrix)
        n_total_elements += np.prod(A.shape)
        n_nnz_elements += np.count_nonzero(matrix_materialized)

    prod_matrix = sess.run(cur_matrix)
    #print(prod_matrix)
    #print(A)
    #print(np.linalg.norm(prod_matrix-A))
    #print(n_total_elements, n_nnz_elements)
    return n_nnz_elements, np.linalg.norm(prod_matrix-A), log_data

product_of_n_matrices = int(sys.argv[1])
dimension = int(sys.argv[2])
l1 = float(sys.argv[3])
lr = float(sys.argv[4])
print(lr)
lr_decay = float(sys.argv[5])
ntrain_iters = int(sys.argv[6])
factor_into_n_matrices = int(sys.argv[7])

"""product_of_n_matrices = 2
dimension = 50
l1 = 0.0
lr = 1e-3
lr_decay = .95
ntrain_iters = 40000"""

passed_args = {
    "n_sparse_matrices" : factor_into_n_matrices,
    "dimension" : dimension,
    "l1_regularization_factor" : l1,
    "lr" : lr,
    "lr_decay" : lr_decay,
    "ntrain_iters" : ntrain_iters,
    "factoring_product_of_n_matrices" : product_of_n_matrices
}

A = None
for i in range(product_of_n_matrices):
    #v = (np.random.randn(dimension, 1) > 0).astype(np.float) * .5
    #v = (np.random.randn(dimension, 1) > 0)
    v = np.random.randn(dimension, 1) / 2
    v = v.dot(v.T)
    if A is None:
        A = np.eye(dimension) + v
    else:
        A = A.dot(np.eye(dimension) + v)

#A = np.random.rand(dimension, dimension) * 20
        
n_nnz_elements, frob_diff, log_data = evaluate_sparse_factorization(A, **passed_args)
results_dict = dict(passed_args)
results_dict["n_nnz_elements"] = n_nnz_elements
results_dict["frob_diff"] = frob_diff
results_dict["loss_data"] = log_data
print(results_dict)
