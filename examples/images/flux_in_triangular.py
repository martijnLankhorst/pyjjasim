
from pyjjasim import *
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")

a = TriangularArray(15, 6)
a.set_inductance_factors(1)
p = AnnealingProblem(a, frustration=0.05)
_, cs, _ = p.compute()
a.set_inductance_factors(4)
q, status, info = StaticProblem(a, vortex_configuration=cs[0].get_n(), frustration=0).compute()
print(info)
q.plot(figsize=[11, 5], show_axes=False,face_quantity=np.abs(q.get_flux()), show_nodes=False, arrow_color=[0.4, 0.3, 1],
           face_quantity_clim=[0.002, 1], face_quantity_logarithmic_colors=True)
#
# c.plot(time_point=Nt-1, figsize=[11, 5], node_diameter=0.5,
#        show_axes=False, node_quantity="potential", vortex_diameter=0.6, arrow_headwidth=2,
#        arrow_headlength=3, arrow_headaxislength=2.5)
plt.show()
