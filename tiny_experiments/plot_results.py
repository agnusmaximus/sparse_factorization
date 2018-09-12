import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sys
import pickle as pkl
import glob

indir = sys.argv[1]
files = glob.glob("%s/*" % indir)
data = [pkl.load(open(x, "rb")) for x in files]

xs, ys = [], []
for d in data:
    nnzs, loss_frobenius_error_materialized = d["results"][-1]
    xs.append(nnzs)
    ys.append(loss_frobenius_error_materialized)
    
plt.scatter(xs, ys, marker="o")

norm_of_matrix = np.linalg.norm(data[-1]["target_matrix"])
xs = [0, max(xs)]
ys = [norm_of_matrix, norm_of_matrix]
plt.plot(xs, ys, label="Norm of target matrix")

plt.title("Synthetic Toy Experiment Results (dimension=10)")
plt.xlabel("# of nnzs")
plt.ylabel("frobenius error")
plt.legend(loc="best")
plt.savefig("synthetic.png")
