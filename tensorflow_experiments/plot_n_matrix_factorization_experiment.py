import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import sys
import numpy  as np
import pickle as pkl

def plot_data(data):
    keys = data.keys()
    n_matrices = sorted(list(set([x[0] for x in keys])))
    sparsities = sorted(list(set([x[1] for x in keys])))
    colors=iter(cm.rainbow(np.linspace(0,1,len(n_matrices))))

    frobenius_norm_of_target_matrix = np.linalg.norm(data[(2, .1)][0]["target_matrix"])
    
    for n in n_matrices:
        xs, ys = [], []
        for s in sparsities:
            results = data[(n,s)]
            frobenius = [x["frobenius_error"] for x in results]
            if len(frobenius) != 0:
                best_frobenius = min(frobenius)
                ys.append(best_frobenius)
                xs.append(s)
        plt.plot(xs, ys, color=next(colors), label="N factored matrices = %d" % n, marker="o")


    xs = sparsities
    ys = [frobenius_norm_of_target_matrix] * len(xs)
    plt.plot(xs, ys, label="Frobenius of target matrix being factorized")

    #plt.yscale("log")
    plt.ylabel("Frobenius Error (lower is better)")
    plt.xlabel("% of 10,000 nnz")
    plt.legend(loc="best")
    plt.grid()
    plt.savefig("n_matrix_factorization_plot.png")
            
if __name__=="__main__":
    path_to_file = sys.argv[1]
    data = pkl.load(open(path_to_file, "rb"))
    for k,v in data.items():
        print(k, len(v))

    plot_data(data)
