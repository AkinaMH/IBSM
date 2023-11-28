import matplotlib.pyplot as plt


def plot2class(pos, neg, title=""):
    plt.scatter(pos[:, 0], pos[:, 1], marker='o', c="#ff0000", label="Majority")
    plt.scatter(neg[:, 0], neg[:, 1], marker='s', c="#000000", label="Minority")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    # plt.title(title, x=0.5, y=-0.2)


def plot3class(pos, neg, new, title=""):
    plt.scatter(pos[:, 0], pos[:, 1], marker='o', c="#ff0000", label="Majority")
    plt.scatter(neg[:, 0], neg[:, 1], marker='s', c="#000000", label="Minority")
    plt.scatter(new[:, 0], new[:, 1], marker='x', c="#0000ff", label="Synthetic")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    # plt.title(title, x=0.5, y=-0.2)

