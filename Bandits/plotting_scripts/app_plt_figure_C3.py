import matplotlib.pyplot as plt
import numpy as np
import os
import plotting_utils
import numpy as np
from scipy.stats import truncnorm
import matplotlib.pyplot as plt

plt.rcParams["text.usetex"] = True
plt.rcParams.update({"font.size": 22})


path = "../results/plots_for_paper/app/"
if not os.path.exists(path):
    os.makedirs(path)

means = [0.25, 0.5, 0.75, 0.25, 0.5, 0.75]
variance = [0.5, 0.5, 0.5, 0.2, 0.2, 0.2]
a_trunc, b_trunc = 0, 1


def F(x, samples):
    return np.sum(samples <= x) / len(samples)


def G(x, samples):
    return 1 - F(x, samples)


labels = []
data = []
golbal_X = np.linspace(a_trunc, b_trunc, 100)


for loc, scale in zip(means, variance):
    L = []
    U = []

    for i in range(1000):
        a, b = (a_trunc - loc) / scale, (b_trunc - loc) / scale
        r = truncnorm.rvs(a, b, loc=loc, scale=scale, size=1000, random_state=i)
        X = np.linspace(np.min(r), np.max(r), 100)
        A1 = [G(X[-1] - x, r) / x for x in X]
        L.append(min(A1))
        U.append(max(A1))

    a, b = (a_trunc - loc) / scale, (b_trunc - loc) / scale
    data.append(truncnorm.rvs(a, b, loc=loc, scale=scale, size=10000, random_state=0))
    labels.append(rf"$\mu={loc:.2f}$, $\sigma^2={scale:.1f}$")

    print(
        "mean=",
        loc,
        ",\t std=",
        scale,
        f",\t L:{np.mean(L):.2f}±{np.std(L):.2f} ,\t  U:{np.mean(U):.2f}±{np.std(U):.2f}",
    )


plt.figure(figsize=(7,4.5))
for i in range(len(means)):
    plt.plot(
        golbal_X,
        [G(x, data[i]) for x in golbal_X],
        label=labels[i],
        alpha=1.0,
        color=plotting_utils.CB_color_cycle[i],
        linewidth = 3,
    )

plt.ylabel("G(x)")
plt.xlabel("x")
# plt.title("Scatter plot of L vs. U for different (mean, variance) pairs")
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig(path + "assumption_guassian_example.pdf", bbox_inches="tight")
