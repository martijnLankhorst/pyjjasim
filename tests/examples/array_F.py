from josephson_circuit import Circuit
import numpy as np


n1 = [0, 0, 0, 1, 1, 2]
n2 = [1, 2, 3, 2, 3, 3]
x = [0, 1, 0, 1]
y = [0, 0, 1, 1]



if __name__ == "__main__":
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use("TkAgg")

    array = Circuit(x, y, n1, n2)
    array.plot()
    plt.show()
