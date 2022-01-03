from josephson_circuit import Circuit
import numpy as np


n1 = [1, 0, 2, 3, 4]
n2 = [2, 3, 5, 4, 5]
x = [0, 1, 2, 0, 1, 2]
y = [1, 1, 1, 0, 0, 0]


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
