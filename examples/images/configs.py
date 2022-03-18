import matplotlib
matplotlib.use("TkAgg")
from pyjjasim import *

array = SquareArray(4, 4)
n1 = [0, 0, 0, 0, 1, 0, 0, 0, 0] # vorticity of each face
n2 = [0, 0, 0, 0, -1, 0, 0, 0, 0] # vorticity of each face
n3 = [0, 0, 0, 0, 1, 0, 0, 0, 1] # vorticity of each face
n4 = [1, 0, 1, 0, 1, 0, 1, 0, 1] # vorticity of each face

problem = StaticProblem(array, vortex_configuration=n1)
config, status, info = problem.compute()
print(info)
fig, _ = config.plot(figsize=[10, 7], axis_position=[0.05, 0.55, 0.27, 0.4], show_legend=False)

problem = StaticProblem(array, vortex_configuration=n2)
config, status, info = problem.compute()
print(info)
config.plot(fig=fig, axis_position=[0.35, 0.55, 0.27, 0.4], show_legend=False)

problem = StaticProblem(array, vortex_configuration=n1, frustration=0.2)
config, status, info = problem.compute()
print(info)
config.plot(fig=fig, axis_position=[0.68, 0.55, 0.27, 0.4], show_legend=False)

problem = StaticProblem(array, vortex_configuration=n1, frustration=0.4)
config, status, info = problem.compute()
print(info)
config.plot(fig=fig, axis_position=[0.05, 0.05, 0.27, 0.4], show_legend=False)

problem = StaticProblem(array, vortex_configuration=n1, frustration=0.6)
config, status, info = problem.compute()
print(info) # does not converge, no solution exists!

problem = StaticProblem(array, vortex_configuration=n3)
config, status, info = problem.compute()
print(info) # does not converge, no solution exists!

problem = StaticProblem(array, vortex_configuration=n4, frustration=0.5)
config, status, info = problem.compute()
print(info)
fig, ax, handle = config.plot(fig=fig, axis_position=[0.35, 0.05, 0.27, 0.4], show_legend=False,
                              return_plot_handle=True)
print(handle)
plt.show()