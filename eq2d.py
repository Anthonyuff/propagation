import numpy as np
import matplotlib.pyplot as plt
dt=0.001
dz=5
nz=500
nt=1000
nx=500
dx=5
c=1000


tempo=np.arange(0,nt*dt,dt)

prof= np.arange(0,nz*dz,dz)

offset=np.arange(0,nx*dx,dx)
