import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

np.random.seed(123)
LOC = 0
N = 100000

# Generate data
n = 10
sigma2 = 1

X = np.random.normal(LOC, sigma2, size=(n, N))


def evaluateVarEstimator(X, ddof=1):
    est = X.var(axis=0, ddof=ddof)
    bias = est.mean() - sigma2
    var = est.var()

    return bias, var, bias ** 2 + var


to_print = ("{}: Bias = {:.4f}, Variance = {:.4f}"
            ", MSE = {:.4f}.")

print(to_print.format("Sample variance estimator", *evaluateVarEstimator(X)))

print(to_print.format("MLE estimator", *evaluateVarEstimator(X, ddof=0)))

ns = [10, 100, 1000]
sigma2s = [1, 10, 100]

gaps = np.zeros(shape=(3, len(ns), len(sigma2s)))

for i in range(len(ns)):
    for j in range(len(sigma2s)):
        X = np.random.normal(LOC, sigma2s[j], size=(ns[i], N))
        gap = np.array(evaluateVarEstimator(X)) - \
              np.array(evaluateVarEstimator(X, ddof=0))

        for l, metric in enumerate(gap):
            gaps[l, i, j] = metric

metrics = ["Bias", "Variance", "MSE"]

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(21, 6))
for i in range(len(metrics)):
    sns.heatmap(gaps[i], ax=axes[i], cmap="coolwarm",
                annot=True, annot_kws={'size': 16}, fmt="0.2f",
                xticklabels=sigma2s, yticklabels=ns)
    axes[i].set_xlabel(r"Population Variance ($\sigma^2$)", size=15)
    axes[i].set_ylabel(r"Sample Size ($n$)", size=15)
    axes[i].set_title(metrics[i], size=20)

fig.tight_layout()

if __name__ == '__main__':
    pass
