from josephson_circuit import Circuit

n1 = [0, 0, 1, 2, 3, 3, 4, 5]
n2 = [1, 2, 3, 3, 4, 5, 6, 6]
x = [0, 1, 0, 1, 2, 1, 2]
y = [2, 2, 1, 1, 1, 0, 0]

if __name__ == "__main__":
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use("TkAgg")

    array = Circuit(x, y, n1, n2)
    array.plot()
    plt.show()