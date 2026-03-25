import numpy as np
import matplotlib.pyplot as plt
dt=0.001
dz=5
nz=500
nt=2000
nx=500
dx=5
c=1500

sx=150
sz=150

tempo=np.arange(0,nt*dt,dt)

prof= np.arange(0,nz*dz,dz)

offset=np.arange(0,nx*dx,dx)

u= np.zeros((nx,nz,nt)) 

def ricker(t,freq):
  
  f_corte = freq

  fc = f_corte / (3 * np.sqrt(np.pi))

  td = t - (0.5 * np.sqrt(np.pi) / fc)

  arg = np.pi * (np.pi * fc * td)**2

  return (1 - 2*arg) * np.exp(-arg)

source =  ricker(tempo,30)
def eq2d(P, dt, dx, dz, nt, nx, nz, c, sx, sz, fonte):

   cte = (c * dt)**2

   for t in range(1, nt-1):

      # fonte
      P[sz, sx, i] += fonte[i]

      for i in range(2, nz-2):
         for j in range(2, nx-2):

            d2x = (
               -P[i, j + 2, t]
                + 16*P[i, j + 1, t]
               - 30*P[i, j, t]
               + 16*P[i, j - 1, t]
               - P[i, j - 2, t]
            ) / (12 * dx**2)

            d2z = (
               -P[i + 2, j, t]
               + 16*P[i + 1, j, t]
               - 30*P[i, j, t]
               + 16*P[i - 1 , j, t]
               - P[i-2, j, t]
            ) / (12 * dz**2)

            laplacian = d2x + d2z

      P[:, :, t+1] = cte * laplacian + 2*P[:, :, t] - P[:, :, t-1]

   return P


U = eq2d(u, dt, dx, dz, nt, nx, nz, c, sx, sz, source)

plt.imshow(U[:, :, 500], cmap="gray", aspect="auto",
           extent=[0, nx*dx, nt*dt, 0])
plt.xlabel("x (m)")
plt.ylabel("z (m)")
plt.title("Snapshot")
plt.show()