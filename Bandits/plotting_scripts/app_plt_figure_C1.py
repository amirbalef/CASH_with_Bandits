import numpy as np
import matplotlib.pyplot as plt

path = "../results/plots_for_paper/app/"


def G_1(x):
    return 1 - x**2


def G_2(x):
    return (1 - x) ** 2


# Generate x values
x = np.linspace(0, 1, 500)  # 500 points between 0 and 1

# Calculate y values
y1 = G_1(x)
y2 = G_2(x)

# Plot the functions
plt.figure(figsize=(6, 6))

# for epsilon in np.linspace(0.1, 0.9, 5):
#     plt.scatter((1 - epsilon), G_1(1 - epsilon), color="black", alpha=0.5)
#     plt.axline(
#         (1, 0),
#         (1 - epsilon, G_1(1 - epsilon)),
#         color="black",
#         linestyle="dotted",
#         linewidth=1,
#         alpha=0.5,
#     )

#     plt.scatter((1 - epsilon), G_2(1 - epsilon), color="black", alpha=0.5)
#     plt.axline(
#         (1, 0),
#         (1 - epsilon, G_2(1 - epsilon)),
#         color="black",
#         linestyle="dotted",
#         linewidth=1,
#         alpha=0.5,
#     )


plt.plot(x, y1, label=r"$G_1(x) = 1-x^2$", color="blue", linewidth=3)

# x1 = 0.90
# L1 = 1.1
# U1 = 1.90

# # Define x data range for tangent line
# xrange = np.linspace(1, 0.6, 100)
# yrange = -(xrange - 1) * U1  # + G_1(x1)
# plt.plot(xrange, yrange, color="darkblue", linewidth=3, linestyle="--")
# plt.annotate(
#     "$ y = U_1 (1-x)$",
#     xy=(0.43, 0.775),
#     xycoords="data",
#     fontsize=18,
#     color="darkblue",
#     rotation=10 - (180 / np.pi) * np.arctan(U1),
# )

# xrange = np.linspace(1, 0.5, 100)
# yrange = -(xrange - 1) * L1  # + G_1(x1)
# plt.plot(xrange, yrange, color="steelblue", linewidth=3, linestyle="--")
# plt.annotate(
#     "$y = L_1 (1-x)$",
#     xy=(0.30, 0.55),
#     xycoords="data",
#     fontsize=18,
#     color="steelblue",
#     rotation=12 - (180 / np.pi) * np.arctan(L1),
# )


plt.plot(x, y2, label=r"$G_2(x) =(1 - x)^2$", color="red", linewidth=2)

# x2 = 0.90
# L2 = 0.1
# U2 = 0.9

# # Define x data range for tangent line
# xrange = np.linspace(1, 0.5, 100)
# yrange = -(xrange - 1) * U2  # + G_2(x2)
# plt.plot(xrange, yrange, color="darkred", linewidth=3, linestyle="--")
# plt.annotate(
#     "$y = U_2(1-x)$",
#     xy=(0.29, 0.44),
#     xycoords="data",
#     fontsize=18,
#     color="darkred",
#     rotation=10 - (180 / np.pi) * np.arctan(U2),
# )

# yrange = -(xrange - 1) * L2  # + G_2(x2)
# plt.plot(xrange, yrange, color="orangered", linewidth=3, linestyle="--")
# plt.annotate(
#     "$y = L_2(1-x)$",
#     xy=(0.25, 0.035),
#     xycoords="data",
#     fontsize=18,
#     color="orangered",
#     rotation=-(180 / np.pi) * np.arctan(L2),
# )


# Add labels, legend, and grid
plt.xlabel("$x$", fontsize=18)
plt.ylabel("$G(x)$", fontsize=18)
# plt.title("Comparison of $G_1(x)$ and $G_2(x)$", fontsize=18)
plt.legend(fontsize=18)
plt.grid(True)
plt.xlim(0, 1)  # Ensure x is between 0 and 1
plt.ylim(0, 1.1)  # Set y-axis limits slightly above 1 for clarity

# Show the plot
plt.tight_layout()
plt.savefig(path + "assumption_toy_example.pdf")
plt.show()

# Plot the functions
plt.figure(figsize=(8, 6))

for epsilon in np.linspace(0.1, 0.9, 5):
    plt.scatter((epsilon), G_1(1 - epsilon), color="black", alpha=0.5)
    plt.axline(
        (0, 0),
        (epsilon, G_1(1 - epsilon)),
        color="black",
        linestyle="dotted",
        linewidth=1,
        alpha=0.5,
    )

    plt.scatter((epsilon), G_2(1 - epsilon), color="black", alpha=0.5)
    plt.axline(
        (0, 0),
        (epsilon, G_2(1 - epsilon)),
        color="black",
        linestyle="dotted",
        linewidth=1,
        alpha=0.5,
    )


epsilon = np.linspace(0, 1, 500)  # 500 points between 0 and 1
# Calculate y values
y1 = G_1(1 - epsilon)
y2 = G_2(1 - epsilon)


plt.plot(epsilon, y1, label=r"$G_1$", color="blue", linewidth=3)

x1 = 0.90
L1 = 1.1
U1 = 1.90

# Define x data range for tangent line
xrange = np.linspace(0, 0.4, 100)
yrange = (xrange) * U1  # + G_1(x1)
plt.plot(xrange, yrange, color="darkblue", linewidth=3, linestyle="--")
plt.annotate(
    "$ y = U_1\\epsilon$",
    xy=(0.4, 0.775),
    xycoords="data",
    fontsize=18,
    color="darkblue",
    rotation=(180 / np.pi) * np.arctan(U1) - 10,
)

xrange = np.linspace(0, 0.5, 100)
yrange = (xrange) * L1  # + G_1(x1)
plt.plot(xrange, yrange, color="steelblue", linewidth=3, linestyle="--")
plt.annotate(
    "$y = L_1\\epsilon$",
    xy=(0.5, 0.55),
    xycoords="data",
    fontsize=18,
    color="steelblue",
    rotation=(180 / np.pi) * np.arctan(L1) - 10,
)


plt.plot(epsilon, y2, label=r"$G_2$", color="red", linewidth=2)

x2 = 0.90
L2 = 0.1
U2 = 0.9

# Define x data range for tangent line
xrange = np.linspace(0, 0.5, 100)
yrange = (xrange) * U2  # + G_2(x2)
plt.plot(xrange, yrange, color="darkred", linewidth=3, linestyle="--")
plt.annotate(
    "$y = U_2\\epsilon$",
    xy=(0.5, 0.44),
    xycoords="data",
    fontsize=18,
    color="darkred",
    rotation=(180 / np.pi) * np.arctan(U2) - 10,
)

yrange = (xrange) * L2  # + G_2(x2)
plt.plot(xrange, yrange, color="orangered", linewidth=3, linestyle="--")
plt.annotate(
    "$y = L_2\\epsilon$",
    xy=(0.5, 0.035),
    xycoords="data",
    fontsize=18,
    color="orangered",
    rotation=(180 / np.pi) * np.arctan(L2) - 2,
)


# Add labels, legend, and grid
plt.xlabel("$\\epsilon$", fontsize=18)
plt.ylabel("$G(1-\\epsilon)$", fontsize=18)
# plt.title("Comparison of $_1(x)$ and $G_2(x)$", fontsize=18)
plt.legend(fontsize=18)
# plt.grid(True)
plt.xlim(0, 1)  # Ensure x is between 0 and 1
plt.ylim(0, 1.1)  # Set y-axis limits slightly above 1 for clarity

# Show the plot
plt.tight_layout()
plt.savefig(path + "assumption_toy_example_2.pdf")
plt.show()

for epsilon in np.linspace(0.1, 0.9, 5):
    print(
        f"epsilon:{epsilon:.2f}",
        f",\t L:{G_1(1 - epsilon) / epsilon:.2f}",
        f",\t L:{G_2(1 - epsilon) / epsilon:.2f}",
    )