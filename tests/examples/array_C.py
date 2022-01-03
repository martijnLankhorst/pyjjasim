from josephson_circuit import Circuit
import numpy as np

n1 = [0, 1, 0, 1, 3, 4, 2,  3, 6,  8, 8,  9,  11, 10, 12, 13]
n2 = [1, 2, 6, 4, 4, 5, 10, 7, 12, 9, 11, 11, 13, 14, 13, 14]
x = [0, 2, 4, 1, 2, 3, 0, 1, 2, 3, 4, 2, 0, 2, 4]
y = [4, 4, 4, 3, 3, 3, 2, 2, 2, 2, 2, 1, 0, 0, 0]

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