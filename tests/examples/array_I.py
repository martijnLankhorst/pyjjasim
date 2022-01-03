from josephson_circuit import Circuit
import numpy as np


n1 = [0, 0, 1, 2, 3, 1, 3, 5, 7, 8, 9, 6, 7, 8, 9, 11, 12, 11, 12, 12, 16, 16, 17, 18, 19]
n2 = [3, 4, 2, 3, 4, 6, 8, 10, 8, 9, 10, 11, 12, 13, 14, 12, 13, 15, 15, 16, 17, 18, 19, 19, 20]
x = [3, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 0, 1, 2, 1, 2, 3]
y = [5, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 0, 0, 0]

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
