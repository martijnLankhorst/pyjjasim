import matplotlib
matplotlib.use("TkAgg")
from pyjjasim import *
Nx, Ny = 4, 3                                  # unit cell count
ax, ay = 1.0, 2.0                              # stretch factors
sq_array = SquareArray(Nx, Ny, ax, ay)         # Circuit object with R=1, Ic=1, L=0, C=0
sq_graph = EmbeddedSquareGraph(Nx, Ny, ax, ay) # EmbeddedGraph object
hc_array = HoneycombArray(Nx, Ny, ax, ay)
tr_array = TriangularArray(Nx, Ny, ax, ay)
fig, ax1 = sq_array.plot(show_junction_ids=True, ax_position=[0.05, 0.55, 0.4, 0.4], figsize=[10, 8], title="a")
sq_graph.plot(fig=fig, ax_position=[0.55, 0.55, 0.4, 0.4], title="b")
hc_array.plot(fig=fig, ax_position=[0.05, 0.05, 0.4, 0.4], title="c")
tr_array.plot(fig=fig, ax_position=[0.55, 0.05, 0.4, 0.4], title="d")
plt.show()
# ax3.set_position(0.1, 0.1, 0.3, 0.3)
# ax4.set_position(0.5, 0.1, 0.3, 0.3)