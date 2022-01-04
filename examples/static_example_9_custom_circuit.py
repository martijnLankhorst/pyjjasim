
import matplotlib
matplotlib.use("TkAgg")

from pyJJAsim import *


"""
EXAMPLE 9: Custom circuit
"""

#  o--x1--o--x1--o
#  |      |      |
#  x3     x1     x2
#  |      |      |
#  o--x1--o--x2--o
#  |             |
#  x1            x1
#  |             |
#  o--x2--o--x1--o
#

# x1    Ic=0.2, Is=0
# x2    Ic=0.5, Is=1
# x3    Ic=1, Is=0

if __name__ == "__main__":

    x = [0, 1, 2, 0, 1, 2, 0, 1, 2]
    y = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    n1 = [0, 1, 0, 2, 3, 4, 3, 4, 5, 6, 7]
    n2 = [1, 2, 3, 5, 4, 5, 6, 7, 8, 7, 8]
    nr = [1, 0, 0, 0, 0, 1, 2, 0, 1, 0, 0]
    Is = np.array([0, 1, 0])
    Ic = np.array([0.2, 0.5, 1])

    G = EmbeddedGraph(x, y, n1, n2)
    array = Circuit(G, critical_current_factors=Ic[nr])
    array.plot()

    prob = StaticProblem(array, current_sources=Is[nr])
    factor, _, conf, _ = prob.compute_maximal_current()
    conf.plot(title=f"current factor={factor}")
    plt.show()
