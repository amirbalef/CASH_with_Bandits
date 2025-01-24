import numpy as np
from scipy.stats import truncnorm

means = [0, 0.5, 1.0, 0, 0.5, 1.0]
variance = [1, 1, 1, 0.2, 0.2, 0.2]
a_trunc, b_trunc = 0, 1


def F(x, samples):
    return np.sum(samples <= x) / len(samples)


def G(x, samples):
    return 1 - F(x, samples)


for loc, scale in zip(means, variance):
    L = []
    U = []

    for i in range(100):
        a, b = (a_trunc - loc) / scale, (b_trunc - loc) / scale
        r = truncnorm.rvs(a, b, loc=loc, scale=scale, size=10000, random_state=i)
        X = np.linspace(np.min(r), np.max(r), 100)
        A1 = [G(X[-1] - x, r) / x for x in X]
        L.append(min(A1))
        U.append(max(A1))

    # print(
    #     "mean=",
    #     loc,
    #     ",\t std=",
    #     scale,
    #     f",\t L:{np.mean(L):.3f}±{np.std(L):.2f} ,\t  U:{np.mean(U):.2f}±{np.std(U):.2f}",
    # )

    print(
        "mean=",
        loc,
        ",\t std=",
        scale,
        f",\t L:{np.percentile(L, 5):.2f}, {np.median(L):.2f}, {np.percentile(L, 95):.2f} ,\t  U:{np.percentile(U, 5):.2f}, {np.median(U):.2f}, {np.percentile(U, 95):.2f}",
    )