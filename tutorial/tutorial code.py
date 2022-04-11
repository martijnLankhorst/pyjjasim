

from pyjjasim import *


# EXAMPLE 1
# %matplotlib inline
a = SquareArray(4, 4)
f = 0.1
n = [0, 0, 0, 0, 1, 0, 0, 0, 0]
problem = StaticProblem(a, frustration=f, vortex_configuration=n)
result, status, info = problem.compute()
print("status: ", status)
print("result: ", result)
print(info)
result.plot()

# EXAMPLE 2
# %matplotlib notebook
sq_array = SquareArray(10, 10)
dt = 0.05
Nt = 1000
f = 0.2
Is = 0.5 * sq_array.current_base(angle=0)
T = 0.02
problem = TimeEvolutionProblem(sq_array, time_step=dt,
                               time_step_count=Nt, current_sources=Is,
                               frustration=f, temperature=T)
config = problem.compute()
config.animate(junction_quantity="supercurrent")
plt.show()

# Example embedded graph
# %matplotlib inline
x = [0, 1, 1, 0.5, 0]
y = [0, 0, 1, 1.5, 1]
node1 = [0, 0, 1, 2, 2, 3]
node2 = [1, 4, 2, 3, 4, 4]
graph = EmbeddedGraph(x, y, node1, node2)
graph.plot(show_node_ids=True, show_edge_ids=True)

# Example regular lattices
Nx, Ny = 4, 3                                  # unit cell count
ax, ay = 1.0, 2.0                              # stretch factors
sq_array = SquareArray(Nx, Ny, ax, ay)         # Circuit object with R=1, Ic=1, L=0, C=0
sq_graph = EmbeddedSquareGraph(Nx, Ny, ax, ay) # EmbeddedGraph object
hc_array = HoneycombArray(Nx, Ny, ax, ay)
tr_array = TriangularArray(Nx, Ny, ax, ay)
fig, ax1 = sq_array.plot(show_junction_ids=True, ax_position=[0.05, 0.55, 0.4, 0.4],
                         figsize=[9, 7], title="a")
sq_graph.plot(fig=fig, ax_position=[0.55, 0.55, 0.4, 0.4], title="b")
hc_array.plot(fig=fig, ax_position=[0.05, 0.05, 0.4, 0.4], title="c")
tr_array.plot(fig=fig, ax_position=[0.55, 0.05, 0.4, 0.4], title="d")


# Example set circuit properties
print(sq_array.get_resistance_factors())
sq_array.set_resistance_factors(2)
R = sq_array.get_resistance_factors()
print(R)
R[4] = 3
sq_array.set_resistance_factors(R)
print(sq_array.get_resistance_factors())

# example faces
print("face count: ", sq_array.face_count())
print("face nodes: ", sq_array.get_faces())
print("face areas: ", sq_array.get_face_areas())
print("face centroids: ", sq_array.get_face_centroids())
xs = [0.2, 1.2]
ys = [5.4, 2.2]
print("nearest faces: ", sq_array.locate_faces(xs, ys))

# example custom circuit
circuit = Circuit(graph, resistance_factors=[1, 2, 3, 4, 5, 6])
circuit.plot()

# example static problem
theta = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
problem = StaticProblem(circuit)
config = StaticConfiguration(problem, theta)
print(config)

# example with non-uniform frustration
array = SquareArray(3, 3)
f1 = [0, 0.1, 0.2, 0.3] # external magnetic flux through each face
problem1 = StaticProblem(array, frustration=f1)
config1, status1, info1 = problem1.compute()
fig, _ = config1.plot(figsize=[10, 4.5], title="Example non-uniform frustration",
                      face_quantity=np.array(f1), axis_position=[0.1, 0.1, 0.4, 0.85],
                      manual_face_label="f")

# example with uniform frustration
f2 = 0.15 # external magnetic flux through each face
problem2 = StaticProblem(array, frustration=f2)
config2, status2, info2 = problem2.compute()
config2.plot(fig=fig, title="Example uniform frustration (f=0.15 in all faces)",
             axis_position=[0.55, 0.1, 0.4, 0.8], show_legend=False)


# example josephson vortices
# %matplotlib inline
array = SquareArray(4, 4)
n1 = [0, 0, 0, 0, 1, 0, 0, 0, 0] # vorticity of each face
n2 = [0, 0, 0, 0, -1, 0, 0, 0, 0] # vorticity of each face
n3 = [0, 0, 0, 0, 1, 0, 0, 0, 1] # vorticity of each face
n4 = [1, 0, 1, 0, 1, 0, 1, 0, 1] # vorticity of each face
n5 = [1, 1, 1, 1, 0, 1, 1, 1, 1] # vorticity of each face
problem1 = StaticProblem(array, vortex_configuration=n1)
config1, status, info = problem1.compute()
print(info)
fig, ax11 = config1.plot(figsize=[10, 7], axis_position=[0.05, 0.55, 0.27, 0.4],
                         show_legend=False, title="f=0")
problem2 = StaticProblem(array, vortex_configuration=n2)
config2, status, info = problem2.compute()
print(info)
fig, ax12 = config2.plot(fig=fig, axis_position=[0.38, 0.55, 0.27, 0.4],
                         show_legend=False, title="f=0")
problem3 = StaticProblem(array, vortex_configuration=n1, frustration=0.2)
config3, status, info = problem3.compute()
print(info)
fig, ax13 = config3.plot(fig=fig, axis_position=[0.71, 0.55, 0.27, 0.4],
                         show_legend=False, title="f=0.2")
problem4 = StaticProblem(array, vortex_configuration=n1, frustration=0.4)
config4, status, info = problem4.compute()
print(info)
fig, ax21 = config4.plot(fig=fig, axis_position=[0.05, 0.05, 0.27, 0.4],
                         show_legend=False, title="f=0.4")
problem5 = StaticProblem(array, vortex_configuration=n1, frustration=0.6)
config5, status, info = problem5.compute()
print(info) # does not converge, no solution exists!
problem6 = StaticProblem(array, vortex_configuration=n3)
config6, status, info = problem6.compute()
print(info) # does not converge, no solution exists!
problem = StaticProblem(array, vortex_configuration=n4, frustration=0.5)
config, status, info = problem.compute()
print(info)
config.plot(fig=fig, axis_position=[0.38, 0.05, 0.27, 0.4], show_legend=False, title="f=0.5")
problem = StaticProblem(array, vortex_configuration=n5, frustration=1.0)
config, status, info = problem.compute()
print(info)
config.plot(fig=fig, axis_position=[0.71, 0.05, 0.27, 0.4], show_legend=False, title="f=1")

array = SquareArray(5, 5)
n = np.zeros(array.face_count(), dtype=int)
n[array.locate_faces(2.5, 1.5)] = 1
problem = StaticProblem(array, vortex_configuration=n)
config, _, _ = problem.compute()
config.plot(figsize=[5, 5])

# example current sources
array = SquareArray(3, 3)
I_source_junc = 0.5 * np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
prob1 = StaticProblem(array, current_sources=I_source_junc)
config1, _, _ = prob1.compute()
config1.plot()

I_source_node = 0.5 * np.array([1, 0, -1, 1, 0, -1, 1, 0, -1])
converted_I_source = node_to_junction_current(array, I_source_node)
prob2 = StaticProblem(array, current_sources=converted_I_source)
config2, _, _ = prob2.compute()
config2.plot()

array = HoneycombArray(3, 3)
I_source = 0.5 * array.current_base(angle=np.pi/2)
prob = StaticProblem(array, current_sources=I_source)
config, _, _ = prob.compute()
config.plot()

# example for frustration bounds
array = SquareArray(6, 6)
n = np.zeros(array.face_count(), dtype=int)
n[array.locate_faces(2.5, 2.5)] = 1
problem = StaticProblem(array, vortex_configuration=n, frustration=1)
bounds, configs, infos = problem.compute_frustration_bounds()
fig, _ = configs[0].plot(figsize=[10, 4.5], title=f"lower frustration bound (f={bounds[0]:.3f})",
                        axis_position=[0.1, 0.1, 0.38, 0.85], show_legend=False)
configs[1].plot(fig=fig, title=f"upper frustration bound (f={bounds[1]:.3f})",
               axis_position=[0.55, 0.1, 0.42, 0.85])

# example for maximum current sources
array = SquareArray(6, 6)
n = np.zeros(array.face_count(), dtype=int)
n[array.locate_faces(2.5, 2.5)] = 1
Is = array.current_base(angle=0)
problem = StaticProblem(array, vortex_configuration=n, current_sources=Is)
I_factor, config, info = problem.compute_maximal_current()
config.plot(figsize=[5, 5], title=f"maximum current (Is={I_factor:.3f})")

# example for whole stable region in f-Is space
array = SquareArray(6, 6)
n = np.zeros(array.face_count(), dtype=int)
n[array.locate_faces(2.5, 2.5)] = 1
Is = array.current_base(angle=0)
problem = StaticProblem(array, vortex_configuration=n, current_sources=Is, frustration=1)
angles = np.linspace(0, np.pi, 61)
f, I, configs, infos = problem.compute_stable_region(angles=angles)
plt.plot(f, I)
plt.xlabel("frustration")
plt.ylabel("current source")
plt.title("stable region")

# example approximations and initial guess
array = TriangularArray(5, 3)
f = 0.02
n1 = np.zeros(array.face_count(), dtype=int)
n2 = np.zeros(array.face_count(), dtype=int)
n2[array.locate_faces(2, 2)] = 1
prob1 = StaticProblem(array, frustration=f, vortex_configuration=n1)
prob2 = StaticProblem(array, frustration=f, vortex_configuration=n2)

# the instruction:
exact, _, _ = prob2.compute()

# does under the hood:
approx = prob2.approximate(algorithm=1)
exact, _, _ = prob2.compute(approx)

fig, _ = exact.plot(title="exact solution \n(with initial guess=london approximation)",
                    axis_position=[0.1, 0.1, 0.35, 0.35], figsize=[10, 8])
print("exact satisfies target n?: ", exact.satisfies_target_vortices()) # -> True

arctan_approx = prob2.approximate(algorithm=0)
london_approx = prob2.approximate(algorithm=1)
arctan_approx.plot(title="arctan approximation", fig=fig, axis_position=[0.1, 0.6, 0.35, 0.35])
london_approx.plot(title="london approximation", fig=fig, axis_position=[0.5, 0.6, 0.35, 0.35])
print(f"arctan_approx is solution?: {arctan_approx.is_solution()}, error: {arctan_approx.get_error()}")
print(f"london_approx is solution?: {london_approx.is_solution()}, error: {london_approx.get_error()}")
print(f"exact is solution?: {exact.is_solution()}, error: {exact.get_error()}")

# manual initial guess
theta_init_guess = np.zeros(array.junction_count())
exact2, _, _ = prob2.compute(theta_init_guess)
exact2.plot(title="exact solution with initial guess theta=0.\n" \
            "coverges but not to target vortex\nconfiguration!", fig=fig,
            axis_position=[0.5, 0.1, 0.35, 0.35])
print("exact2 satisfies target n?: ", exact2.satisfies_target_vortices()) # -> False

# Example dynamic stability
array = SquareArray(7, 6)
prob = StaticProblem(array)
approx1 = prob.approximate_placed_vortices([1], [2.5], [2.5])
approx2 = prob.approximate_placed_vortices([1], [3], [2.5])
config1, _, _ = prob.compute(approx1)
config2, _, _ = prob.compute(approx2)
fig, _ = config1.plot(node_quantity="phi", title="stable configuration", figsize=[12, 5],
                      axis_position=[0.05, 0.1, 0.4, 0.85], show_legend=False)
config2.plot(node_quantity="phi", title="unstable configuration with vortex placed at junction",
             fig=fig, axis_position=[0.5, 0.1, 0.45, 0.85],)
print("config 1 solution properties: ")
config1.report()
print("config 2 solution properties: ")
config2.report()
stable_status = config1.is_stable() # status 0 -> stable, 1-> unstable, 2-> indeterminate
print("config2 stable status: ", stable_status)


# Example vortex in screened array
array = SquareArray(6, 6)
n = np.zeros(array.face_count())
n[array.locate_faces(2.5, 2.5)] = 1
config_no_L, _, _ = StaticProblem(array, vortex_configuration=n).compute()
fig, _ = config_no_L.plot(face_quantity="Phi", title="inductance=0", figsize=[10, 8],
                          axis_position=[0.1, 0.55, 0.4, 0.4], face_quantity_clim=[-1, 1])
array.set_inductance_factors(1) # all junctions get inductor with L=1
config_L1, _, _ = StaticProblem(array, vortex_configuration=n).compute()
config_L1.plot(face_quantity="Phi", title="inductance=1", fig=fig,
               axis_position=[0.55, 0.55, 0.4, 0.4])
array.set_inductance_factors(10)
config_L10, _, _ = StaticProblem(array, vortex_configuration=n).compute()
config_L10.plot(face_quantity="Phi", title="inductance=10", fig=fig,
               axis_position=[0.1, 0.05, 0.4, 0.4])

# It is easier to find static configurations in screened arrays
array = SquareArray(3, 3)
config_no_L, _, info_no_L = StaticProblem(array, vortex_configuration=[1, 0, 0, 1]).compute()
array.set_inductance_factors(1)
config_with_L, _, info_with_L = StaticProblem(array, vortex_configuration=[1, 0, 0, 1]).compute()

print(info_no_L)
print(info_with_L)
config_with_L.plot(fig=fig, axis_position=[0.55, 0.05, 0.4, 0.4],
                   title="inductance=1\nwould not be stable if inductance=0")

# Example time evolution
# %matplotlib notebook
# Example of sweeping the frustration factor
sq_array = SquareArray(10, 10)
Nt = 3000
f = np.linspace(0, 2, Nt)[None, None, :]
problem = TimeEvolutionProblem(sq_array, time_step=0.05, time_step_count=Nt,
                               frustration=f, temperature=0.02)
config = problem.compute()
config.animate(junction_quantity="supercurrent", title="frustration sweep from 0 to 2")
plt.show()

# Example of sweeping temperature
sq_array = SquareArray(10, 10)
Nt = 3000
T = np.linspace(0, 2, Nt)[None, None, :]
problem = TimeEvolutionProblem(sq_array, time_step=0.05, time_step_count=Nt,
                               temperature=T)
config = problem.compute()
config.animate(junction_quantity="supercurrent", title="temperature sweep from 0 to 2")
plt.show()

# Example of sweeping current
sq_array = SquareArray(10, 10)
Nt = 3000
I_base = sq_array.current_base(angle=0)
I = I_base[:, None, None] * np.linspace(0, 2, Nt)
problem = TimeEvolutionProblem(sq_array, time_step=0.05, time_step_count=Nt,
                               temperature=0.02, current_sources=I)
config = problem.compute()
config.animate(junction_quantity="supercurrent", title="current sweep from 0 to 2")
plt.show()

# example of current pulse
# %matplotlib notebook
from scipy.stats import norm
sq_array = SquareArray(20, 20)
dt = 0.05
Nt = 2000
ts = np.arange(0, Nt, 3)
Ih = sq_array.current_base(angle=0)
amp = 2
Is = amp * Ih[:, None, None] * norm(loc=30, scale=1.0).pdf(np.arange(Nt) * dt)
n = np.zeros(sq_array.face_count(), dtype=int)
n[sq_array.locate_faces(19/ 2, 19/2)] = 1
init, _, _ = StaticProblem(sq_array, vortex_configuration=n).compute()
init_th = init.get_theta()[:, None]

problem = TimeEvolutionProblem(sq_array, time_step=dt, time_step_count=Nt, current_sources=Is,
                               store_time_steps=ts, config_at_minus_1=init_th)
out = problem.compute()
out.animate(figsize=(8, 8), title="Move vortex with current pulse ")


# Example IV curve with sweeping current
sq_array = SquareArray(10, 10)
Nt = 10000
I_base = sq_array.current_base(angle=0)[:, None, None]
I_amp = np.append(np.linspace(0, 2, Nt//2), np.linspace(2, 0, Nt//2))
I = I_base * I_amp
problem = TimeEvolutionProblem(sq_array, time_step=0.05, time_step_count=Nt,
                               temperature=0.02, current_sources=I)
config = problem.compute()
V = config.get_voltage()
Vmean = 2 * np.mean(V * I_base, axis=0).ravel()
print(Vmean.shape)
plt.subplots()
plt.plot(I_amp[:Nt//2], Vmean[:Nt//2], color="blue", label="trace")
plt.plot(I_amp[Nt//2:], Vmean[Nt//2:], color="red", label="retrace")
plt.xlabel("I")
plt.ylabel("V")
plt.title("IV sweep")
plt.legend()
# Now average over 100 consecutive steps
plt.subplots()
I_amp_av = np.mean(np.reshape(I_amp, (-1, 100)), axis=1)
Vmean_av = np.mean(np.reshape(Vmean, (-1, 100)), axis=1)
plt.plot(I_amp_av[:Nt//200], Vmean_av[:Nt//200], color="blue", label="trace")
plt.plot(I_amp_av[Nt//200:], Vmean_av[Nt//200:], color="red", label="retrace")
plt.xlabel("I")
plt.ylabel("V")
plt.title("IV sweep averaged over 100 steps")
plt.legend()

# Example of IV curve
sq_array = SquareArray(10, 10)
Nt = 3000
T = 0.02
f = 0
W = 101 # problem count
I_base = sq_array.current_base(angle=0)[:, None, None]
I_amp = np.linspace(0, 2, W)[:, None]
I = I_base * I_amp
problem = TimeEvolutionProblem(sq_array, time_step=0.05, time_step_count=Nt,
                               temperature=T, current_sources=I, frustration=f)
config = problem.compute()
V = config.get_voltage()
Vmean = 2 * np.mean(V * I_base, axis=(0, 2)).ravel()
plt.subplots()
plt.plot(I_amp[:, 0], Vmean)
plt.xlabel("I")
plt.ylabel("V")
plt.title("IV curve")

# Example of IV curve with out-of-equilibrium start conditions
sq_array = SquareArray(30, 30)
Nt = 2000
dt = 0.1
T = 0.02
f = 0
W = 11 # problem count
I_base = sq_array.current_base(angle=0)[:, None, None]
I_amp = np.linspace(0.8, 1.1, W)[:, None]
I = I_base * I_amp

# equilibrium start
problem_eq = TimeEvolutionProblem(sq_array, time_step=dt, time_step_count=Nt,
                               temperature=T, current_sources=I, frustration=f)
config_eq = problem_eq.compute()
V_eq = config_eq.get_voltage()
# take mean over second half of time evolution
Vmean_eq = 2 * np.mean(V_eq[:, :, Nt//2:] * I_base, axis=(0, 2)).ravel()

# non-equilibrium start
th0 = 100 * np.random.rand(sq_array.junction_count(), W)
problem_neq = TimeEvolutionProblem(sq_array, time_step=dt, time_step_count=Nt,
                               temperature=T, current_sources=I, frustration=f,
                               config_at_minus_1=th0)
config_neq = problem_neq.compute()
V_neq = config_neq.get_voltage()
# take mean over second half of time evolution
Vmean_neq = 2 * np.mean(V_neq[:, :, Nt//2:] * I_base, axis=(0, 2)).ravel()

plt.subplots()
plt.plot(I_amp[:, 0], Vmean_eq, label="equilibrium start")
plt.plot(I_amp[:, 0], Vmean_neq, label="non-equilibrium start")
plt.xlabel("I")
plt.ylabel("V")
plt.legend()
plt.title("IV curve")

# TLDR: start conditions can matter (generally only if T is small and/or V is small)

# Example shapiro steps
N = 20
sq_array = SquareArray(N, N)
T = 0.01
dt = 0.1
Nt = 3000
ts = [Nt//3, Nt - 1]
IAmp = 1
Ifreq = 0.25
IDC = np.linspace(0, 2, 61)
Is = lambda i: sq_array.current_base(angle=0)[:, None] * (IDC + np.sin(Ifreq * i * dt))
prob = TimeEvolutionProblem(sq_array, time_step=dt, time_step_count=Nt,
                            current_sources=Is, temperature=T, store_time_steps=ts,
                            store_theta=True, store_current=False, store_voltage=False)
out = prob.compute()
th = out.get_theta()[sq_array.current_base(angle=0)==1, :, :]
th_at_one_third = th[:, :, 0]
last_th = th[:, :, 1]
V = np.mean((last_th - th_at_one_third) / (dt * (ts[1] - ts[0])), axis=0)
plt.subplots()
plt.plot(IDC, V, label='f=0')
plt.xlabel("DC current")
plt.ylabel("mean array voltage")
plt.title("giant shapiro steps in square array")
plt.show()

# Example of annealing
sq_array = SquareArray(20, 20)
f = 0.1
time_step = 2.0
interval_steps = 20
interval_count = 200
vortex_mobility = 0.001
problem_count = 40
start_T = 1.0
T_factor = 1.03
problem = AnnealingProblem(sq_array, time_step=time_step, interval_steps=interval_steps,
                           interval_count=interval_count, vortex_mobility=vortex_mobility,
                           frustration=f, problem_count=problem_count,
                           start_T=start_T, T_factor=T_factor)
# do time simulation
out = problem.compute()
status, vortex_configurations, temperature_profiles = out

energies = [np.mean(v.get_energy()) for v in vortex_configurations]
lowest_state = np.argmin(energies)

fig = plt.figure(figsize=[9, 5])
a1 = fig.add_axes([0.07, 0.6, 0.3, 0.4])
a1.hist(energies)
a1.set_xlabel("mean energy per junction")
a1.set_ylabel("histogram count")

# plot temperature evolution
a2 = fig.add_axes([0.07, 0.1, 0.3, 0.4])
a2.plot(time_step * interval_steps * np.arange(interval_count), temperature_profiles)
a2.set_xlabel("time")
a2.set_ylabel("annealing temperature")

lowest_E = energies[lowest_state]
vortex_configurations[lowest_state].plot(fig=fig, axis_position=[0.45, 0.05, 0.52, 0.9],
                                         title=f"lowest found state at E={lowest_E:.5f}")
plt.show()
