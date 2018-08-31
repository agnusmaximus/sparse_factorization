#!/bin/bash

# Variables for all runs
ntrain_iters=100000
dimension=50
lr_decay=.95

# product_of_n_matrices=2, lr=1e-5
product_of_n_matrices=2
lr=1e-3
l1s=( "0" ".001" ".002" ".004" ".008" ".01" ".02" ".04" ".08" ".1" ".2" ".4" ".8" "1" "10" )
factor_into_n_matrices=${product_of_n_matrices}
for l1 in ${l1s[@]}; do
    fname=results_n=${product_of_n_matrices}_l1=${l1}
    #echo "python sparse_factorize.py ${product_of_n_matrices} ${dimension} ${l1} ${lr} ${lr_decay} ${ntrain_iters} > ${fname}"
    #python sparse_factorize.py ${product_of_n_matrices} ${dimension} ${l1} ${lr} ${lr_decay} ${ntrain_iters} ${factor_into_n_matrices}  > ${fname}
done

# product_of_n_matrices=3, lr=1e-5
product_of_n_matrices=3
lr=1e-5
factor_into_n_matrices=${product_of_n_matrices}
#l1s=( "0" ".001" ".002" ".004" ".008" ".01" ".02" ".04" ".08" ".1" ".2" ".4" ".8" "1" "10" )
l1s=( "2" "3" "4" "5" "6" "7" "8" "9" )
for l1 in ${l1s[@]}; do
    fname=results_n=${product_of_n_matrices}_l1=${l1}
    #echo "python sparse_factorize.py ${product_of_n_matrices} ${dimension} ${l1} ${lr} ${lr_decay} ${ntrain_iters} > ${fname}"
    #python sparse_factorize.py ${product_of_n_matrices} ${dimension} ${l1} ${lr} ${lr_decay} ${ntrain_iters} ${factor_into_n_matrices} > ${fname} &
done

# product_of_n_matrices=5, lr=1e-7
product_of_n_matrices=5
lr=1e-7
factor_into_n_matrices=${product_of_n_matrices}
#l1s=( "0" ".001" ".002" ".004" ".008" ".01" ".02" ".04" ".08" ".1" ".2" ".4" ".8" "1" "10" )
#l1s=( "10" "20" "40" "80" )
#l1s=( "50" "60" "70" )
#l1s=( "71" "72" "74" "78" )
#l1s=( "79" "79.1" "79.2" "79.4" "79.8" )
#l1s=( "50" "55" "60" "65" "70" )
#l1s=( "41" "42" "44" "48" )
#l1s=( "45" "45.5" "46" "46.5" "47" )
l1s=( "47.1" "47.2" "47.3" "47.4" "47.5" "47.6" "47.7" "47.8" "47.9" )
for l1 in ${l1s[@]}; do
    fname=results_n=${product_of_n_matrices}_l1=${l1}
    #echo "python sparse_factorize.py ${product_of_n_matrices} ${dimension} ${l1} ${lr} ${lr_decay} ${ntrain_iters} > ${fname}"
    #python sparse_factorize.py ${product_of_n_matrices} ${dimension} ${l1} ${lr} ${lr_decay} ${ntrain_iters} ${factor_into_n_matrices}  > ${fname} &
done

# product_of_n_matrices=5, lr=1e-7
product_of_n_matrices=5
lr=1e-8
factor_into_n_matrices=10
#l1s=( "0" ".001" ".002" ".004" ".008" ".01" ".02" ".04" ".08" ".1" ".2" ".4" ".8" "1" "10" )
#l1s=( "10" "20" "40" "80" )
#l1s=( "50" "60" "70" )
#l1s=( "71" "72" "74" "78" )
#l1s=( "79" "79.1" "79.2" "79.4" "79.8" )
#l1s=( "50" "55" "60" "65" "70" )
#l1s=( "41" "42" "44" "48" )
#l1s=( "45" "45.5" "46" "46.5" "47" )
l1s=( "47.1" "47.2" "47.3" "47.4" "47.5" "47.6" "47.7" "47.8" "47.9" )
for l1 in ${l1s[@]}; do
    fname=results_n=${product_of_n_matrices}_l1=${l1}
    echo "python sparse_factorize.py ${product_of_n_matrices} ${dimension} ${l1} ${lr} ${lr_decay} ${ntrain_iters} ${factor_into_n_matrices} > ${fname}"
    #python sparse_factorize.py ${product_of_n_matrices} ${dimension} ${l1} ${lr} ${lr_decay} ${ntrain_iters} ${factor_into_n_matrices}  > ${fname} &
done
