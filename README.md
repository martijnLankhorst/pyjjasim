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
[api documentation](https://htmlpreview.github.io/?https://github.com/martijnLankhorst/pyJJAsim/blob/master/doc/_build/html/pyjjasim.html)\
[whitepaper](PyJJASim_Whitepaper.pdf)\
[user manual](PyJJASim_Whitepaper.pdf)

# Example Usage

````python
from pyJJAsim import *
array = SquareArray(3, 3)
problem = StaticProblem(array, frustration=0.1)
config, status, info = problem.compute()
print(config.get_I())
````

Program output:
<pre>
[-0.30917 -0.30917 0 0 0.30917 0.30917 0.30917 0 -0.30917 0.30917 0 -0.30917]
</pre>

If you have any questions, comments, complaints, 
bug reports, feature requests, etc.
please contact me at m.lankhorst89@gmail.com!
