from ForcePy import *
import numpy as np
import matplotlib.pyplot as plt

mesh = UniformMesh(0,10,1)
finemesh = UniformMesh(0,11,0.001)
basis = Basis.Quartic(mesh, 1)

xval = [finemesh[x] for x in range(len(finemesh))]
yval = [np.sum(basis.potential(x, mesh)) for x in xval]

plt.plot(xval, yval)
plt.show()
