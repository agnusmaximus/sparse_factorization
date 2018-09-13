export SPARSEHOME="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH="$PYTHONPATH:$SPARSEHOME"
echo "Added Sparse Factorization repository ($SPARSEHOME) to \$PYTHONPATH."
