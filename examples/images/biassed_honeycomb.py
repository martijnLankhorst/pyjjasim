
from pyjjasim import *
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")

a = HoneycombArray(10, 10)
Is = 0.4 * a.current_base(angle=0)[:, None, None]
Nt = 5000
f = 0.15
p = TimeEvolutionProblem(a, 0.1, Nt, external_flux=f, current_sources=Is)
c = p.compute()
c.plot(time_point=Nt-1, figsize=[10, 5], node_diameter=0.5,
       show_axes=False, node_quantity="potential", vortex_diameter=0.6, arrow_headwidth=2,
       arrow_headlength=3, arrow_headaxislength=2.5, legend_width_fraction=0.1,
       axis_position=[0.01, 0.01, 0.98, 0.98])
plt.show()
