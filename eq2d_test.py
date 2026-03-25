import time

import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange

def measure_runtime(func):
  def wrapper(*args, **kwargs):
    start = time.time()
    result = func(*args, **kwargs)
    end = time.time()
    print(f"Runtime: {round(end - start, 4)} seconds")
    return result

  return wrapper

def ricker(t,freq):
  
  f_corte = freq

  fc = f_corte / (3 * np.sqrt(np.pi))

  td = t - (0.5 * np.sqrt(np.pi) / fc)

  arg = np.pi * (np.pi * fc * td)**2

  return (1 - 2*arg) * np.exp(-arg)

@njit(parallel=True)
def laplacian2d(
    upre, d2u_dx2, d2u_dz2, 
    nzz, nxx, dh2,
) -> None:
  inv_dh2 = 1.0 / (5040.0 * dh2)

  for i in prange(4, nzz - 4):
    for j in range(4, nxx - 4):
      d2u_dx2[i, j] = (
          -9   * upre[i-4, j] + 128   * upre[i-3, j] - 1008 * upre[i-2, j] +
          8064 * upre[i-1, j] - 14350 * upre[i,   j] + 8064 * upre[i+1, j] -
          1008 * upre[i+2, j] + 128   * upre[i+3, j] - 9    * upre[i+4, j]
      ) * inv_dh2

      d2u_dz2[i, j] = (
          -9   * upre[i, j-4] + 128   * upre[i, j-3] - 1008 * upre[i, j-2] +
          8064 * upre[i, j-1] - 14350 * upre[i, j]   + 8064 * upre[i, j+1] -
          1008 * upre[i, j+2] + 128   * upre[i, j+3] - 9    * upre[i, j+4]
      ) * inv_dh2

  return d2u_dx2 + d2u_dz2

@measure_runtime
def eq2d(P, dt, dh, nt, nx, nz, c, sx, sz, fonte):

   d2u_dx2 = np.zeros((nz, nx))
   d2u_dz2 = np.zeros((nz, nx))

   #criar matriz de snapshots
   snap=np.zeros((nz,nx,500))
   dh2 = dh * dh
   cte = (c * dt)**2
   s=0
   for t in range(1, nt-1):

      # fonte
      P[sz, sx, t] += fonte[t] / dh2

      laplacian = laplacian2d(
        P[:, :, t], d2u_dx2, d2u_dz2, nz, nx, dh2
      )


      P[:, :, t+1] = cte * laplacian + 2*P[:, :, t] - P[:, :, t-1]

      # salvar snapshots a um certo passo de tempo
      
      if t%4==0 and s<500:
        
        snap[:,:,s] = P[:,:,t]
        s += 1
   return P,snap


def  disp(c,alpha,f,b):
  h=c/(alpha*f)
  dt=h/(b*c)
  return h,dt

nz = 100
nt = 2000
nx = 200
c = 1500
dh,dt= disp(c,3,30,4)

sx = nx // 2
sz = nz // 2

tempo=np.arange(0 ,nt * dt, dt)

prof= np.arange(0, nz * dh, dh)

offset=np.arange(0, nx * dh, dh)

u = np.zeros((nz, nx, nt)) 

source =  ricker(tempo, 30)

U ,snap = eq2d(u, dt, dh, nt, nx, nz, c, sx, sz, source)

from matplotlib.animation import FuncAnimation

fig,ax = plt.subplots()

perc = 99




wave = ax.imshow(snap[:, :, 0], cmap="gray", aspect="auto",extent=[0, nx*dh, nt*dt, 0])


ax.set_xlabel("x (m)")
ax.set_ylabel("z (m)")
ax.set_title("Snapshot")


def atualizar(frame):
    
    vmax= 2 * np.std(snap[:, :, frame])
    vmin=-vmax
    wave.set_data(snap[:,:,frame])
    wave.set_clim(vmin, vmax)
    ax.set_title(f"time = {frame*dt:.3f} s")
    
    return wave,


ani = FuncAnimation(fig, atualizar, frames=500, interval=10)
plt.show()