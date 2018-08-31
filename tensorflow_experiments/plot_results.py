import matplotlib
matplotlib.use("Agg")
import sys
import matplotlib.pyplot as plt
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

    points = []
    baseline_scores = []
    baseline_nnz = []
    for d in data:
        print("-------------------")
        print(d["original_score"])
        print(d["new_score_after_factorized"])
        print(d["factorize_results"][0]["factorization_result_details"]["n_nnz_elements"])
        print(d["factorize_results"][0]["factorization_result_details"]["original_nnz_elements"])
        print(d["factorize_results"][0]["factorization_result_details"]["hyperparameter_setting"]["n_matrices_to_factorize_into"])
        baseline_scores.append(d["original_score"])
        baseline_nnz = d["factorize_results"][0]["factorization_result_details"]["original_nnz_elements"]
        points.append((d["factorize_results"][0]["factorization_result_details"]["n_nnz_elements"], d["new_score_after_factorized"]))

    # Plot data
    points = [x for x in points if x[0] <= np.mean(baseline_nnz)]
    xvals = [x[0] for x in points]
    yvals = [x[1] for x in points]
    plt.scatter(xvals, yvals, marker="o")

    # Plot baseline score
    baseline_score = np.mean(baseline_scores)
    xbaseline_score = np.arange(0, max(xvals))
    ybaseline_score = [baseline_score for x in xbaseline_score]
    plt.plot(xbaseline_score, ybaseline_score, label="Baseline accuracy")

    # Plot nnz
    baseline_nnz = np.mean(baseline_nnz)
    plt.axvline(x=baseline_nnz, color="k", label="Baseline nnz")
    
    plt.title("MNIST Sparsely Factorize FC1")
    plt.xlabel("n_nnz_elements")
    plt.ylabel("accuracy")
    plt.legend(loc="best")
    plt.savefig("mnist_plot.png")
