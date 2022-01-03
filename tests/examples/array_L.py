from josephson_circuit import Circuit
import numpy as np


n1 = [0, 1, 0, 1, 1, 2, 4]
n2 = [1, 2, 3, 3, 5, 5, 5]
x = [0, 1, 2, 0, 1, 2]
y = [0, 0, 0, 1, 1, 1]

idx = np.lexsort((n2, n1))
rinv = np.argsort(idx)
idx2 = np.append(idx, idx + len(n1))
inv = np.argsort(idx2)


if __name__ == "__main__":
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use("TkAgg")

    array = Circuit(x, y, n1, n2)
    array.plot()
    plt.show()
