from josephson_circuit import Circuit

n1 = [0]
n2 = [1]
x = [0, 1]
y = [0, 0]

if __name__ == "__main__":
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use("TkAgg")

    array = Circuit(x, y, n1, n2)
    array.plot()
    plt.show()