from pyJJAsim import *

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")

def make_hole_array(N, L):
    full_array = SquareArray(N, N)
    x, y = full_array.get_node_coordinates()
    hole_face_ids = (x > (N-1-L)/2) & (x < (N-1+L)/2) & \
                    (y > (N-1-L)/2) & (y < (N-1+L)/2)
    hole_array = full_array.remove_nodes(hole_face_ids.flatten())
    return hole_array

def get_vortex_in_hole_configuration(hole_vortex_count):
    hole_face_idx = hole_array.locate_faces((N - 1) / 2, (N - 1) / 2)
    vortex_configuration = np.zeros(hole_array.face_count())
    vortex_configuration[hole_face_idx] = hole_vortex_count
    return vortex_configuration


if __name__ == "__main__":
    # define N by N square array with L by L hole (N and L even)
    N = 14
    L = 4
    beta_L=0

    hole_array = make_hole_array(N, L)
    hole_array.set_inductance_factors(beta_L)
    hole_array.plot()

    x, y = hole_array.get_node_coordinates()
    Is_node = (x==0).astype(int) - (x==(N-1)).astype(int)
    Is = node_to_junction_current(hole_array, Is_node)

    plt.subplots()
    for i in range(4):
        print(f"computing result with {i} vortices in hole")
        n = get_vortex_in_hole_configuration(i)
        problem = StaticProblem(hole_array, vortex_configuration=n, current_sources=Is)
        f, I, _, info = problem.compute_stable_region(angles=np.linspace(0, np.pi, 61))
        plt.plot(f, I, marker="o", label=f"{i} vortices in hole")

    plt.xlabel("frustration")
    plt.ylabel("maximal current")
    plt.title(f"hole array with N={N} and N_gap={L}, beta_L={beta_L}")
    plt.legend()

    plt.show()

