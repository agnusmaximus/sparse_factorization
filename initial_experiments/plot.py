import matplotlib
matplotlib.use("Agg")
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt

def plot_data(n_matrices, data):
    plt.cla()

    lr = data[0]["lr"]
    lr_decay = data[0]["lr_decay"]
    ntrain_iters = data[0]["ntrain_iters"]
    dimension = data[0]["dimension"]
    title = "Factoring %d Matrices, lr=%f, lr_decay=%f, ntrain_iters=%d, dimension=%d" % (n_matrices, lr, lr_decay, ntrain_iters, dimension)
    name = "plot_n=%d" % n_matrices

    data = sorted(data, key=lambda x:x["n_nnz_elements"])
    data_filtered = [(x["n_nnz_elements"], x["frob_diff"], x["l1_regularization_factor"]) for x in data]
    x_data = [x[0] for x in data_filtered]
    y_data = [x[1] for x in data_filtered]
    print(data_filtered)

    plt.plot(x_data, y_data, marker="o")
    
    plt.xlabel("Number of non-zero elements")
    plt.ylabel("Frobenius of Difference (A-A_1*A_2*...*A_n)")
    plt.title(title)
    #plt.yscale("log")
    plt.savefig(name)

nan = "nan"
nan = -10

plot_dir = sys.argv[1]
files = glob.glob("%s/*" % plot_dir)
data = []
for f in files:
    with open(f, "r") as f_read:
        str_val = f_read.read()
        data.append(eval(str_val))

group_by_n_matrices = {}
for d in data:
    if d["n_sparse_matrices"] not in group_by_n_matrices.keys():
        group_by_n_matrices[d["n_sparse_matrices"]] = []
    group_by_n_matrices[d["n_sparse_matrices"]].append(d)

for n_matrices, cur_data in group_by_n_matrices.items():
    plot_data(n_matrices, cur_data)
    

    
