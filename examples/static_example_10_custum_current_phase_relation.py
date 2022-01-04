from pyJJAsim import *

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("TkAgg")

"""
EXAMPLE 10: Custom current-phase relation
"""

if __name__ == "__main__":

    N = 10

    array = SquareArray(N, N)
    func = lambda Ic, th: Ic * np.sin(th) ** 3
    d_func = lambda Ic, th: 3 * Ic * (np.sin(th) ** 2) * np.cos(th)
    i_func = lambda Ic, th: Ic * (8 + np.cos(3 * th) - 9 * np.cos(th))/12
    cp = CurrentPhaseRelation(func, d_func, i_func)

    f = 0.01
    n = np.zeros(array.face_count())
    n[array.locate_faces(4.5, 4.5)] = 1
    prob = StaticProblem(array, current_phase_relation=cp, vortex_configuration=n, frustration=f)
    config, status, info = prob.compute()
    print(f"status: {status}")
    print(info)
    print(f"total energy: {np.sum(config.get_Etot())}")
    config.plot()
    plt.show()



