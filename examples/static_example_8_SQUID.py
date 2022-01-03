import numpy as np

from josephson_circuit import SQUID
import matplotlib
import matplotlib.pyplot as plt

from static_problem import StaticProblem

matplotlib.use("TkAgg")
array = SQUID()
array.plot()

Is = array.horizontal_junctions()
angles = np.linspace(0, np.pi, 31)
prob = StaticProblem(array, current_sources=Is)

plt.subplots()
f, I, _, _ = prob.compute_stable_region(angles=angles)
plt.plot(f, I, color=[0, 0, 1], label="n=0, beta_L=0")

prob = StaticProblem(array, current_sources=Is, vortex_configuration=1)
f, I, _, _ = prob.compute_stable_region(angles=angles)
plt.plot(f, I, color=[0, 0, 1], linestyle="--", label="n=1, beta_L=0")

array.set_inductance_factors(1)
prob = StaticProblem(array, current_sources=Is)
f, I, _, _ = prob.compute_stable_region(angles=angles)
plt.plot(f, I, color=[0, 0, 0], label="n=0, beta_L=1")

prob = StaticProblem(array, current_sources=Is, vortex_configuration=1, frustration=1)
f, I, _, _ = prob.compute_stable_region(angles=angles)
plt.plot(f, I, color=[0, 0, 0], linestyle="--", label="n=1, beta_L=1")

plt.xlabel("frustration")
plt.ylabel("maximal current")
plt.legend()
plt.title("SQUID maximal current")
plt.show()
