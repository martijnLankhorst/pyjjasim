from josephson_circuit import Circuit
import numpy as np


n1 = [0, 0, 1, 2, 2, 3]
n2 = [1, 5, 5, 3, 4, 4]
x = [0, 1, 2, 2, 1, 0]
y = [0, 0, 0, 1, 1, 1]



if __name__ == "__main__":
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use("TkAgg")

    array = Circuit(x, y, n1, n2)
    array.plot()
    plt.show()
