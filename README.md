
# PyJJASim
Circuit Simulator including Josephson Junctions in Python

# Installation
PyJJASim requires numpy, scipy and matplotlib

```
pip install pyjjasim
```

# Introduction
PyJJASim is a circuit simulator including Josephson Junctions as components, 
intended to be used on large Josephson Junction Arrays (JJAs). 

PyJJASim is specialized in keeping track of Josephson vortices in the circuit. 
It can also compute static configurations that have vortices at desired 
locations in the circuit.

This requires that the circuit is a planar embedding (in 2D), such that 
one can unambiguously refer to faces of the circuit, and vortices reside
at faces. This imposes that nodes in the circuit must be placed 
at 2D coordinates, and that no junctions can cross. This also means no
hierarchical structure is supported.

# Features
- supports basic components (inductors, resistors, capacitors and current- and voltage sources)
- keep track of (and place) Josephson vortices in the circuit
- compute static configurations 
- determine dynamic stability of static configurations
- maximize parameters that have stable static configurations
- compute time evolutions
- define external magnetic flux through each face
- thermal fluctuations modeling nonzero temperature
- visualization and animation of simulation results

# Documentation
[tutorial](./tutorial/pyjjasim_tutorial.md) \
[API](https://readthedocs.org/projects/pyjjasim/) \
[article](pyJJAsim_A_circuit_simulator_with_Josephson_junctions.pdf) (in preparation for submission)

# Example Usage

````python
from pyjjasim import *

sq_array = SquareArray(10, 10)
problem = StaticProblem(sq_array, frustration=0.1)
config, status, info = problem.compute()
config.plot(node_quantity="phi")
````

Program output:

![alt text](./examples/readme_example_0.png?raw=true)

````python
n = np.zeros(sq_array.face_count())
n[sq_array.locate_faces(x=[2.5,6.5], y=[2.5,6.5])] = 1
config, status, info = problem.new_problem(vortex_configuration=n).compute()
config.plot(node_quantity="phi")
plt.show()
````

Program output:

![alt text](./examples/readme_example_1.png?raw=true)

More examples:


![alt text](./examples/images/biassed_honeycomb.png?raw=true)
*figure 1*: Example of a frustrated honeycomb array with horizontal current bias. 
Snapshot of a time evolution; the vortices are drifting downward. The electric potential 
gradually increases in the direction of the external current
(see examples/images/biassed_honeycomb.py).


![alt text](./examples/images/shapiro_steps.png?raw=true)
*figure 2*: Example of giant Shapiro steps in a square array. The external 
current has a DC and AC component causing resonance in the form of
voltage plateaus (see examples/images/shapiro_steps.py).


![alt text](./examples/images/disordered.png?raw=true)
*figure 3*: Example of a disordered array with different types of Josephson vortices 
(see examples/images/disordered.py).

If you have any questions, comments, complaints, 
bug reports, feature requests, etc.
please contact me at m.lankhorst89@gmail.com!
