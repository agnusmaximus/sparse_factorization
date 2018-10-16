import sys
import torch
import tensorflow as tf
import numpy as np
from SparseFactorization.utils import utils
from SparseFactorization.utils import complex_utils

class SparseFactorizationBase(object):
    """
    Base class for sparse factorization.
    """
    def __init__(self, hyperparameters={}):

        # Merge hyperparameters.
        self.hyperparameters = utils.merge_dict(self.get_hyperparameter_defaults(),
                                                hyperparameters)
        
        # Check hyperparameters are valid
        self.validate_hyperparameters()

    def validate_hyperparameters(self):
        
        # Make sure there are no extra keys that are not in the default dict.
        keys_set_difference = set(self.hyperparameters.keys()) - set(self.get_hyperparameter_defaults().keys())
        if len(keys_set_difference) != 0:
            raise Exception("Unknown keys to class %s: %s" % (self.__class__.__name__, set(keys_set_difference)))        

    def factorize(self, A):
        """
        Factorize A into sparse matrices according to hyperparameters.
        Returns 2 things:
        1) Factorized matrices (in which prod(fact_matrices) ~ A)
        2) Result details (for logging, plotting, etc)
        """
        raise NotImplementedError()

    def get_hyperparameter_defaults(self):
        """
        Returns hyperparameter defaults of sparse factorization.
        """
        return {}

    @staticmethod
    def get_hyperparameter_space():
        raise NotImplementedError()

    def get_hyperparameters(self):
        """
        Returns hyperparameters of the sparse factorization instance.
        """
        return self.hyperparameters

class SparseFactorizationWithL1AndPruningTF(SparseFactorizationBase):

    def __init__(self, hyperparameters={}):
        super().__init__(hyperparameters)
        
    def get_hyperparameter_defaults(self):
        return {
            # The following implies factored into:
            # A = A_1 A_2. If A is MxN, then
            # A_1 is MxIntermediateDimenion,
            # A_2 is IntermediateDimensionxN
            "intermediate_dimension" : 100,
            "number_of_factors" : 2,

            # Optimization parameters
            "learning_rate" : 1e-3,
            "l1_parameter" : 1e-2,
            "pruning_threshold" : 1e-3, # Parameters with abs value smaller than this are zeroed
            "training_iters" : 10000,
            "l1_parameter_decay" : 1,

            # Initialization parameters
            "initialization_stdev" : .01,
            "matrix_initializations" : {},

            # Config parameters
            "log_every" : 100,
            "seed" : 0
        }

    def validate_hyperparameters(self):
        super().validate_hyperparameters()

        if self.get_hyperparameters()["number_of_factors"] < 2:
            raise Exception("Number of factors needs to be >= 2, got: %d" % self.get_hyperparameters()["number_of_factors"])
 
    def factorize(self, A):

        tf.reset_default_graph()
        utils.set_all_seeds(self.get_hyperparameters()["seed"])
        
        # Manage shapes
        first_shape, last_shape = (
            (A.shape[0], self.get_hyperparameters()["intermediate_dimension"]),
            (self.get_hyperparameters()["intermediate_dimension"], A.shape[1])
        )

        intermediate_shapes = [(self.get_hyperparameters()["intermediate_dimension"],
                               self.get_hyperparameters()["intermediate_dimension"]) for i in
                              range(self.get_hyperparameters()["number_of_factors"]-2)]
        all_shapes = [first_shape] + intermediate_shapes + [last_shape]

        # Create tensorflow variables
        variables = []
        for shape in all_shapes:
            stdev = self.get_hyperparameters()["initialization_stdev"]
            variables.append(tf.Variable(tf.random_normal(shape,
                                                          stddev=stdev) +
                                         tf.eye(*shape), dtype=tf.float32))
            #variables.append(tf.Variable(tf.random_normal(shape,
            #                                              stddev=stdev)))
                             
        # Create optimization procedure
        #product_of_factors = variables[0] + tf.eye(*tuple(variables[0].get_shape().as_list()))
        product_of_factors = variables[0]
        for variable in variables[1:]:
            #product_of_factors = tf.matmul(product_of_factors,
            #                               variable + tf.eye(*tuple(variable.get_shape().as_list())))
            product_of_factors = tf.matmul(product_of_factors,
                                           variable)

        A_placeholder = tf.placeholder(dtype=tf.float32, shape=A.shape)
        #frobenius_error = tf.norm(A_placeholder - product_of_factors)
        #frobenius_error = tf.norm(A_placeholder - product_of_factors, 2)
        frobenius_error = tf.nn.l2_loss(A_placeholder - product_of_factors)

        # Add l1 loss
        l1_placeholder = tf.placeholder(dtype=tf.float32)
        l1_regularizer = tf.contrib.layers.l1_regularizer(
            scale=1.0, scope=None
        )
        regularization_penalty = tf.contrib.layers.apply_regularization(l1_regularizer, variables)
        loss = frobenius_error + l1_placeholder * regularization_penalty

        # Create optimization
        lr_placeholder = tf.placeholder(dtype=tf.float32)
        opt = tf.train.GradientDescentOptimizer(learning_rate=lr_placeholder)
        minimize = opt.minimize(loss)

        # Create pruning update
        prune_ops = []
        for variable in variables:
            pruning_threshold = self.get_hyperparameters()["pruning_threshold"]
            mask = tf.cast(tf.greater(tf.abs(variable),
                                      tf.ones_like(variable, dtype=tf.float32) * pruning_threshold), dtype=tf.float32)
            prune_ops.append(tf.assign(variable, tf.multiply(mask, variable)))
        prune_op = tf.group(prune_ops)
        
        # Create tf session
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        # initialize variables
        matrix_initializations = self.get_hyperparameters()["matrix_initializations"]
        for index, variable in enumerate(variables):
            if index in matrix_initializations:
                print("Initializing: %d" % index)
                sess.run(tf.assign(variable, matrix_initializations[index]))

        training_iters = self.get_hyperparameters()["training_iters"]
        log_every = self.get_hyperparameters()["log_every"]
        lr = self.get_hyperparameters()["learning_rate"]
        l1 = self.get_hyperparameters()["l1_parameter"]
        log_result_data = []
        for i in range(training_iters):

            # Minimize
            _ = sess.run([minimize], feed_dict={
                A_placeholder : A,
                lr_placeholder : lr,
                l1_placeholder : l1
            })

            # Prune
            _ = sess.run([prune_op])
            
            if i % log_every == 0 or i == training_iters-1:
                l1 *= self.get_hyperparameters()["l1_parameter_decay"]
                frob_error_materialized, loss_materialized = sess.run([frobenius_error, loss], feed_dict={
                    A_placeholder : A,
                    lr_placeholder : lr,
                    l1_placeholder : l1
                })
                variables_materialized = sess.run(variables)
                nonzeros = [np.count_nonzero(x) for x in variables_materialized]
                sum_nonzeros = sum(nonzeros)

                log_result_data.append({
                    "frobenius_error" : frob_error_materialized,
                    "loss" : loss_materialized,
                    "variables_nonzero_count" : nonzeros,
                    "sum_nonzeros" : sum_nonzeros
                })

                print("SparseFactorizationWithL1AndPruningTF: Frob error: %g, Loss: %g, Variables NNzs: %s, Sum NNzs: %g" % (
                    frob_error_materialized,
                    loss_materialized,
                    str(nonzeros),
                    sum_nonzeros
                ))

        # Extract final logged data
        variables_materialized = [x + np.eye(*x.shape) for x in sess.run(variables)]
        product_of_factors_materialized = sess.run(product_of_factors)        
        
        final_sum_nonzeros = sum([np.count_nonzero(x) for x in variables_materialized])        
        product_of_factors = variables_materialized[0] 
        for variable_materialized in variables_materialized[1:]:
            product_of_factors = product_of_factors.dot(variable_materialized)
        calculated_frobenius_error = np.linalg.norm(product_of_factors - A)

        # Some sanity checks
        #print(np.linalg.norm(product_of_factors_materialized - product_of_factors))
        #assert(np.linalg.norm(product_of_factors_materialized - product_of_factors) < 1e-6)
                
        result_data = {
            "log_result_data" : log_result_data,
            "final_sum_nonzeros" : final_sum_nonzeros,            
            "product_of_factors" : product_of_factors,
            "result_factors" : variables_materialized,
            "calculated_frobenius_error" : calculated_frobenius_error,
            "A" : A
        }

        sess.close()
        
        return  variables_materialized, result_data

class SparseFactorizationWithL1AndPruningPytorch(SparseFactorizationWithL1AndPruningTF):

    def factorize(self, A):

        # Seed
        seed = self.get_hyperparameters()["seed"]
        torch.cuda.manual_seed_all(seed)
        torch.manual_seed(seed)

        # Use cuda device if possible
        device = torch.device('cuda')

        # Create shapes
        first_shape, last_shape = (
            (A.shape[0], self.get_hyperparameters()["intermediate_dimension"]),
            (self.get_hyperparameters()["intermediate_dimension"], A.shape[1])
        )
        
        intermediate_shapes = [(self.get_hyperparameters()["intermediate_dimension"],
                                self.get_hyperparameters()["intermediate_dimension"]) for i in
                               range(self.get_hyperparameters()["number_of_factors"]-2)]
        all_shapes = [first_shape] + intermediate_shapes + [last_shape]

        # Create variables
        variables = []
        for shape in all_shapes:
            stdev = self.get_hyperparameters()["initialization_stdev"]
            weights = np.random.normal(0, scale=stdev, size=shape)
            variables.append(torch.tensor(weights, device=device, requires_grad=True))
        
        # Extract params
        training_iters = self.get_hyperparameters()["training_iters"]
        log_every = self.get_hyperparameters()["log_every"]
        lr = self.get_hyperparameters()["learning_rate"]
        l1 = self.get_hyperparameters()["l1_parameter"]
        log_result_data = []

        # optimizer
        optimizer = torch.optim.SGD(variables, lr=lr)
        for i in range(training_iters):

            # Create prediction
            prediction = variables[0] + torch.eye(*tuple(variables[0].size()), device=device, dtype=torch.double)
            for variable in variables[1:]:
                prediction = prediction.mm(variable + torch.eye(*tuple(variable.size()), device=device, dtype=torch.double))

            # Add l1 regularization
            l1_reg = None
            for variable in variables:
                if l1_reg is None:
                    l1_reg = variable.norm(1)
                else:
                    l1_reg = l1_reg + variable.norm(1)

            # Create the loss
            loss = (prediction - torch.tensor(A, device=device)).norm(2) + l1_reg * l1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Prune weights
            for variable in variables:
                threshold = self.get_hyperparameters()["pruning_threshold"]
                variable.data[np.abs(variable.data) < threshold] = 0                
                            
            if i % log_every == 0 or i == training_iters-1:
                loss_materialized = loss.item()
                frob_error_materialized = np.linalg.norm(prediction.cpu().data.numpy() - A)
                variables_materialized = [x.cpu().data.numpy() for x in variables]
                nonzeros = [np.count_nonzero(x) for x in variables_materialized]
                sum_nonzeros = sum(nonzeros)

                log_result_data.append({
                    "frobenius_error" : frob_error_materialized,
                    "loss" : loss_materialized,
                    "variables_nonzero_count" : nonzeros,
                    "sum_nonzeros" : sum_nonzeros
                })

                print("SparseFactorizationWithL1AndPruningPytorch: Frob error: %g, Loss: %g, Variables NNzs: %s, Sum NNzs: %g" % (
                    frob_error_materialized,
                    loss_materialized,
                    str(nonzeros),
                    sum_nonzeros
                ))                
                
        variables_materialized = [x.cpu().data.numpy() for x in variables]
        variables_materialized = [x + np.eye(*x.shape) for x in variables_materialized]

        final_sum_nonzeros = sum([np.count_nonzero(x) for x in variables_materialized])        
        product_of_factors = variables_materialized[0] 
        for variable_materialized in variables_materialized[1:]:
            product_of_factors = product_of_factors.dot(variable_materialized)
        calculated_frobenius_error = np.linalg.norm(product_of_factors - A)

        result_data = {
            "log_result_data" : log_result_data,
            "final_sum_nonzeros" : final_sum_nonzeros,            
            "product_of_factors" : product_of_factors,
            "result_factors" : variables_materialized,
            "calculated_frobenius_error" : calculated_frobenius_error,
            "A" : A
        }

        return  variables_materialized, result_data
        
class Successive2Factorization(SparseFactorizationBase):

    def __init__(self, hyperparameters, factorization_method, factorization_method_hyperparameters):
        """
        hyperparameters - hyperparameters for successive2factorization
        factorization_method - class of type SparseFactorizationBase to sparse factor matrix
        factorization_method_hyperparameters - hyperparameters for factorization_method
        """
        super().__init__(hyperparameters)
        self.factorization_method = factorization_method
        self.factorization_method_hyperparameters = factorization_method_hyperparameters

    def get_hyperparameter_defaults(self):
        return {
            "n_factorization_rounds" : 4
        }

    def factorize(self, A):
        num_rounds = self.get_hyperparameter_defaults()["n_factorization_rounds"]
        matrices = [A]
        result_details = []
        for i in range(num_rounds):

            # Create factorizer
            factorizer = self.factorization_method(self.factorization_method_hyperparameters)

            # Get densest matrix
            nonzeros = [np.count_nonzero(x) for x in matrices]
            densest_matrix_index = np.argmax(nonzeros)
            densest_matrix = matrices[densest_matrix_index]
            densest_nnzs = nonzeros[densest_matrix_index]

            factorized, _ = factorizer.factorize(densest_matrix)
            factorized_nnzs = [np.count_nonzero(x-np.eye(*x.shape)) for x in factorized]
            factorized_nnzs_sum = sum(factorized_nnzs)

            if factorized_nnzs_sum <= densest_nnzs:
                matrices = matrices[:densest_matrix_index] + factorized + matrices[densest_matrix_index+1:]
                
                print("Successive2Factorization - Factorized matrix from %d => [%d,%d]" %
                      (densest_nnzs, factorized_nnzs[0], factorized_nnzs[1]))
                
                matrices_nnzs = [np.count_nonzero(x) for x in matrices]
                print("Successive2Factorization - Matrix nnzs: %s" % str(matrices_nnzs))
                result_details.append({
                    "matrices_nnzs" : matrices_nnzs
                })
        return matrices, result_details

class SparseFactorizationWithEnforcedStructurePytorch(SparseFactorizationBase):

    def __init__(self, hyperparameters={}):
        super().__init__(hyperparameters)

    def get_hyperparameter_defaults(self):
        return {

            # Optimization parameters
            "learning_rate" : 1e-3,
            "l1_parameter" : 1e-2,
            "pruning_threshold" : 1e-3, # Parameters with abs value smaller than this are zeroed
            "training_iters" : 10000,
            "l1_parameter_decay" : 1,

            # Structure parameters
            # ------------------------------------------------------------
            # What to initialize each matrix to. This also determines number of factors in the
            # optimization.
            # E.g:
            # matrix_initializations = {
            #    0 : np.array([1,1],[1,1])
            #    1 : np.array([1,1],[1,1])
            # }
            # Would make the optimizer do a 2-factorization with the initial matrices specified.
            "matrix_initializations" : {},
            
            # Force every element in the corresponding optimized matrix to 0 if the element
            # of the matrix in the mapping in the same position is also 0.
            # E.g:
            # sparsity_enforcements = {
            #     0 : np.array([0,0,1,1],[1,1,0,0])
            # }
            # would make sure that the optimized matrix 0 would have 0s in the top left and bottom right.
            "sparsity_constraints" : {},

            # Like sparsity enforcements, but actually overrides the value of the optimized matrix
            # to have exactly the same values as the referenced matrix.
            # E.g:
            # value_enforcements = {
            #    0 : np.array([1,2,3,4], [5,6,7,8])
            # }
            # would make sure that the optimized matrix 0 would be overridden to [1,2,3,4][5,6,7,8] after each iteration
            "value_constraints" : {},

            # Complex?
            "is_complex" : True,

            # Config parameters
            "log_every" : 100,
            "seed" : 0
        }
        

    def factorize(self, A):

        # Seed
        seed = self.get_hyperparameters()["seed"]
        torch.cuda.manual_seed_all(seed)
        torch.manual_seed(seed)

        # Use cuda device if possible
        device = torch.device('cuda')

        # Create variables
        nvars = len(self.get_hyperparameters()["matrix_initializations"])
        variables = [None for i in range(nvars)]
        for index, matrix in self.get_hyperparameters()["matrix_initializations"].items():
            variables[index] = torch.tensor(matrix, device=device, requires_grad=True)
        
        # Extract params
        training_iters = self.get_hyperparameters()["training_iters"]
        log_every = self.get_hyperparameters()["log_every"]
        lr = self.get_hyperparameters()["learning_rate"]
        l1 = self.get_hyperparameters()["l1_parameter"]
        log_result_data = []
        is_complex = self.get_hyperparameters()["is_complex"]

        # optimizer
        optimizer = torch.optim.SGD(variables, lr=lr)
        for i in range(training_iters):

            # Create prediction (probably can refactor this code a bit)
            if not is_complex:                

                # The following is a formulation with the + I
                # prediction = variables[0] + torch.eye(*tuple(variables[0].size()), device=device, dtype=torch.double)
                # for variable in variables[1:]:
                #     prediction = prediction.mm(variable + torch.eye(*tuple(variable.size()), device=device, dtype=torch.double))

                prediction = variables[0]
                for variable in variables[1:]:
                    prediction = prediction.mm(variable, device=device, dtype=torch.double)
                
            else:                

                # Complex multiplication
                prediction = variables[0]
                for variable in variables[1:]:
                    stacked_eye = np.stack([np.eye(variable.shape[0]), np.eye(variable.shape[0])], axis=2)
                    prediction = complex_utils.complex_mm(prediction, variable + torch.tensor(stacked_eye, device=device, requires_grad=True))
                    #prediction = complex_utils.complex_mm(prediction, variable)

            # Add l1 regularization
            l1_reg = None
            for variable in variables:
                if l1_reg is None:
                    l1_reg = variable.norm(1)
                else:
                    l1_reg = l1_reg + variable.norm(1)

            # Create the loss
            loss = (prediction - torch.tensor(A, device=device)).norm(2) + l1_reg * l1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Prune weights
            for variable in variables:
                threshold = self.get_hyperparameters()["pruning_threshold"]
                variable.data[np.abs(variable.data) < threshold] = 0

            # Enforce sparsity constraints
            for index, sparsity_constraint in self.get_hyperparameters()["sparsity_constraints"].items():
                #variables[index] *= torch.tensor(sparsity_constraint)
                variables[index].data *= torch.tensor(sparsity_constraint, device=device, dtype=torch.double)
                #print("YOOOO", variables[index].data.cpu().numpy())

            # Enforce value constraints
            for index, value_constraint in self.get_hyperparameters()["value_constraints"].items():
                variables[index].data = torch.tensor(value_constraint, device=device, dtype=torch.double)
                            
            if i % log_every == 0 or i == training_iters-1:
                loss_materialized = loss.item()
                frob_error_materialized = np.linalg.norm(prediction.cpu().data.numpy() - A)
                variables_materialized = [x.cpu().data.numpy() for x in variables]
                nonzeros = [np.count_nonzero(x) for x in variables_materialized]
                sum_nonzeros = sum(nonzeros)

                log_result_data.append({
                    "frobenius_error" : frob_error_materialized,
                    "loss" : loss_materialized,
                    "variables_nonzero_count" : nonzeros,
                    "sum_nonzeros" : sum_nonzeros
                })

                print("SparseFactorizationWithL1AndPruningPytorch: Frob error: %g, Loss: %g, Variables NNzs: %s, Sum NNzs: %g" % (
                    frob_error_materialized,
                    loss_materialized,
                    str(nonzeros),
                    sum_nonzeros
                ))                
                
        variables_materialized = [x.cpu().data.numpy() for x in variables]
        #variables_materialized = [x + np.eye(*x.shape) for x in variables_materialized]

        final_sum_nonzeros = sum([np.count_nonzero(x) for x in variables_materialized])

        product_of_factors = prediction.cpu().data.numpy()
        calculated_frobenius_error = np.linalg.norm(product_of_factors - A)

        result_data = {
            "log_result_data" : log_result_data,
            "final_sum_nonzeros" : final_sum_nonzeros,            
            "product_of_factors" : product_of_factors,
            "result_factors" : variables_materialized,
            "calculated_frobenius_error" : calculated_frobenius_error,
            "A" : A
        }

        return  variables_materialized, result_data
