#!/bin/bash

source /dfs/scratch0/maxlam/env3/bin/activate

for i in `seq 1 1000000`;
do
    python sparse_factorize_tensorflow.py search_results/ mnist_factorize_1layer_hyperparam_space
done    
