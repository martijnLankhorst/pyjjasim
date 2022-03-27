
from pyjjasim import *
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 17}

matplotlib.rc('font', **font)

a = TriangularArray(15, 6)
remove = 31

np.random.seed(1)
o = np.sort(np.random.permutation(a.node_count())[:remove])
print(o)
a = a.remove_nodes(o)
a.plot()
n = np.zeros(180, dtype=int)
n[179] = 2
n[178] = 1
# n[174] = 1
n[79] = 1
f = 0.045 * a.get_face_areas()
p = StaticProblem(a, frustration=f, vortex_configuration=n)
config, _, info = p.compute()
print(info)
fig, ax = config.plot(face_quantity="J", manual_face_label="face current vortices", legend_width_fraction=0.1,
            figsize=[14, 8], show_axes=False, node_face_color=[0.8, 0.8, 0.8], face_quantity_clim=[-3, 3],
            face_quantity_cmap="bwr", manual_vortex_label="n", manual_arrow_label="", arrow_headlength=3,
                      arrow_headaxislength=2.5)
fig.savefig('disordered.png', bbox_inches=fig.get_tightbbox(fig.canvas.get_renderer()))
plt.show()