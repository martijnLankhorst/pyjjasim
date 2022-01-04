# PyJJASim

# Installation
PyJJASim requires numpy, scipy and matplotlib

```
pip install pyjjasim
```

# Introduction
PyJJASim is a circuit simulator including Josephson Junctions as components, intended to be used on large Josephson Junction Arrays (JJAs). 

# Documentation
[api documentation](doc\_build\html\pyjjasim.html)
[whitepaper](PyJJASim_Whitepaper.pdf)
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
