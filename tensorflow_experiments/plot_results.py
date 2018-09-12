import matplotlib
matplotlib.use("Agg")
import sys
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import glob
import pickle
import numpy as np

def extract_data(f):
    with open(f, "rb") as f_read:
        return pickle.load(f_read)

if __name__=="__main__":
    results_dir = sys.argv[1]
    files = glob.glob("%s/*" % results_dir)
    data = []
    for f in files:
        data.append(extract_data(f))

    data = [x for x in data if "fc2" in x["factorize_results"][0]["layer_name"]]
        
    points = []
    baseline_scores = []
    baseline_nnz = []
    for d in data:
        print("-------------------")
        print(d["original_score"])
        print(d["new_score_after_factorized"])
        print(d["factorize_results"][0]["layer_name"])
        print(d["factorize_results"][0]["factorization_result_details"]["n_nnz_elements"])
        print(d["factorize_results"][0]["factorization_result_details"]["original_nnz_elements"])
        print(d["factorize_results"][0]["factorization_result_details"]["hyperparameter_setting"]["n_matrices_to_factorize_into"])
        print("Matrix nnzs:", [np.count_nonzero(x) for x in d["factorize_results"][0]["factorization_result_details"]["factorized_matrices"]])
        print("Intermediate dimension: ", (d["factorize_results"][0]["factorization_result_details"]["hyperparameter_setting"]["intermediate_dimension"]))
        baseline_scores.append(d["original_score"])
        baseline_nnz = d["factorize_results"][0]["factorization_result_details"]["original_nnz_elements"]
        n_matrices = len(d["factorize_results"][0]["factorization_result_details"]["factorized_matrices"])
        points.append((d["factorize_results"][0]["factorization_result_details"]["n_nnz_elements"], d["new_score_after_factorized"], n_matrices))

    # Plot data
    points = [x for x in points if x[0] <= np.mean(baseline_nnz)]
    xvals = [x[0] for x in points]
    yvals = [x[1] for x in points]
    plt.scatter(xvals, yvals, marker="o")

    # Plot baseline score
    baseline_score = np.max(baseline_scores)
    xbaseline_score = np.arange(0, np.mean(baseline_nnz))
    ybaseline_score = [baseline_score for x in xbaseline_score]
    plt.plot(xbaseline_score, ybaseline_score, label="Baseline accuracy")

    # Plot nnz
    baseline_nnz = np.mean(baseline_nnz)
    plt.axvline(x=baseline_nnz, color="k", label="Baseline nnz")
    
    plt.title("MNIST Sparsely Factorize FC2")
    plt.xlabel("n_nnz_elements")
    plt.ylabel("accuracy")
    plt.legend(loc="best")
    plt.savefig("mnist_plot.png")

    ################################################

    plt.cla()
    
    # Plot by number of factorized matrices
    to_plot = {}
    groups = [(2,3), (4,6), (5,7), (8, 10), (11, 30)]
    for point in points:
        x, y, n_matrices = point
        for g in groups:
            if str(g) not in to_plot:
                to_plot[str(g)] = []
            lower, upper = g
            if lower <= n_matrices and n_matrices <= upper:
                to_plot[str(g)].append(point)

    colors=iter(cm.rainbow(np.linspace(0,1,len(groups))))       
    plt.plot(xbaseline_score, ybaseline_score, label="Baseline accuracy")
    plt.axvline(x=baseline_nnz, color="k", label="Baseline nnz")    
    for k,v in to_plot.items():
        xvals = [x[0] for x in v]
        yvals = [x[1] for x in v]
        plt.scatter(xvals, yvals, marker="o", label="n matrices factorized between " + str(k), color=next(colors))

    plt.title("MNIST Sparsely Factorize FC2")
    plt.xlabel("n_nnz_elements")
    plt.ylabel("accuracy")
    plt.legend(loc="best")
    plt.savefig("mnist_plot_colorized.png")
    
