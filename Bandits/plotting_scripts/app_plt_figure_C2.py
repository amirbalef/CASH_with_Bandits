import matplotlib.pyplot as plt
import numpy as np
import os
import plotting_utils

plt.rcParams["text.usetex"] = True
plt.rcParams.update({"font.size": 22})


path = "../results/plots_for_paper/app/"
if not os.path.exists(path):
    os.makedirs(path)


# Define the functions G_1 and G_2
def G_1(x):
    return 1 - x**2
def G_2(x):
    return (1 - x) ** 2

# Generate x values
x = np.linspace(0, 1, 500)  # 500 points between 0 and 1

# Calculate y values
y1 = G_1(x)
y2 = G_2(x)

# Create a figure
fig, ax1 = plt.subplots(figsize=(9, 6))

# Plot the functions G_1 and G_2
ax1.plot(x, y1, label=r"$G_1(x) = 1-x^2$", color=plotting_utils.CB_color_cycle[0], linewidth=3)
ax1.plot(x, y2, label=r"$G_2(x) =(1 - x)^2$", color=plotting_utils.CB_color_cycle[1], linewidth=3)

# Set the first axis labels and limits
ax1.set_xlabel("$x$")
ax1.set_ylabel("$y$")
ax1.set_xlim(0, 1)  # Ensure x is between 0 and 1
ax1.set_ylim(0, 1.1)  # Set y-axis limits slightly above 1 for clarity
ax1.grid(True)

# Create a second x-axis at the top
ax2 = ax1.twiny()

# Set the second x-axis limits based on epsilon values
ax2.set_xlim(0, 1)  # Same range as the first axis
ax2.set_xlabel(r"$\epsilon$")

# Add epsilon ticks on the second axis corresponding to 1 - epsilon values on the first axis
epsilon_ticks = np.linspace(0.1, 0.9, 5)
ax2.set_xticks(1 - epsilon_ticks)
ax2.set_xticklabels([f"{epsilon:.1f}" for epsilon in epsilon_ticks])


# Plot additional lines and points
for epsilon in np.linspace(0.1, 0.9, 5):
    ax1.scatter((1 - epsilon), G_1(1 - epsilon), color="black", alpha=0.5)
    ax1.axline(
        (1, 0),
        (1 - epsilon, G_1(1 - epsilon)),
        color="black",
        linestyle="dotted",
        linewidth=1,
        alpha=0.5,
    )

    ax1.scatter((1 - epsilon), G_2(1 - epsilon), color="black", alpha=0.5)
    ax1.axline(
        (1, 0),
        (1 - epsilon, G_2(1 - epsilon)),
        color="black",
        linestyle="dotted",
        linewidth=1,
        alpha=0.5,
    )

# Add tangent lines and annotations for G_1
x1 = 0.90
L1 = 1.1
U1 = 1.90

# Define x data range for tangent line
xrange = np.linspace(1, 0.61, 100)
yrange = -(xrange - 1) * U1  # + G_1(x1)
ax1.plot(xrange, yrange, color="darkblue", linewidth=3, linestyle="dotted")
ax1.annotate(
    "$ y = U_1 \\epsilon$",
    xy=(0.5, 0.75),
    xycoords="data",
    color="darkblue",
    rotation=6 - (180 / np.pi) * np.arctan((6 / 9) * U1),
)

xrange = np.linspace(1, 0.5, 100)
yrange = -(xrange - 1) * L1  # + G_1(x1)
ax1.plot(xrange, yrange, color="darkblue", linewidth=3, linestyle="--")
ax1.annotate(
    "$y = L_1\\epsilon$",
    xy=(0.34, 0.58),
    xycoords="data",
    color="darkblue",
    rotation=5 - (180 / np.pi) * np.arctan((6 / 9) * L1),
)

# Add tangent lines and annotations for G_2
x2 = 0.90
L2 = 0.1
U2 = 0.9

# Define x data range for tangent line
xrange = np.linspace(1, 0.45, 100)
yrange = -(xrange - 1) * U2  # + G_2(x2)
ax1.plot(xrange, yrange, color="darkred", linewidth=3, linestyle="dotted")
ax1.annotate(
    "$y = U_2\\epsilon$",
    xy=(0.29, 0.51),
    xycoords="data",
    color="darkred",
    rotation=4 - (180 / np.pi) * np.arctan((6 / 9) * U2),
)

xrange = np.linspace(1, 0.4, 100)
yrange = -(xrange - 1) * L2  # + G_2(x2)
ax1.plot(xrange, yrange, color="darkred", linewidth=3, linestyle="--")
ax1.annotate(
    "$y = L_2\\epsilon$",
    xy=(0.25, 0.045),
    xycoords="data",
    color="darkred",
    rotation= 4- (180 / np.pi) * np.arctan((9 / 6) * L2),
)

# Add legend
ax1.legend(fontsize=18)

# Show the plot
plt.tight_layout()
plt.savefig(path + "assumption_toy_example_2.pdf")


for epsilon in np.linspace(0.1, 0.9, 5):
    print(
        f"epsilon:{epsilon:.2f}",
        f",\t L:{G_1(1 - epsilon) / epsilon:.2f}",
        f",\t L:{G_2(1 - epsilon) / epsilon:.2f}",
    )