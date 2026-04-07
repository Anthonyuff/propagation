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
def eq2d(P, dt, dh, nt, nx, nz, c, sx, sz, fonte,fonte2,cerjan):

   d2u_dx2 = np.zeros((nz, nx))
   d2u_dz2 = np.zeros((nz, nx))

   #criar matriz de snapshots
   snap=np.zeros((nz,nx,500))
   dh2 = dh * dh
   cte = (c * dt)**2
   s=0
   offset = int(10 / dh) 
   rx = list(range(20, nx-20, offset))

   
   rz=[20]*len(rx)
   simo=np.zeros((nt,len(rx)))
   
   for t in range(1, nt-1):
      dlay= 150 #delay
      # fonte
      P[20, sx, t] += fonte[t] / dh2
      
      P[20, sx+40, t] += fonte2[t] / dh2
      
      # P[sz, sx+20, t] += fonte[t] / dh2
      # P[sz, sx+30, t] += fonte[t] / dh2
      # P[sz, sx-20, t] += fonte[t] / dh2


      laplacian = laplacian2d(
        P[:, :, t], d2u_dx2, d2u_dz2, nz, nx, dh2
      )


      P[:, :, t+1] = cte * laplacian + 2*P[:, :, t] - P[:, :, t-1]

      # salvar snapshots a um certo passo de tempo
      P[:, :, t] *= cerjan
      P[:, :, t+1] *= cerjan
      
      if t%4==0 and s<500:
        
        snap[:,:,s] = P[:,:,t]
        s += 1

      for j in range(len(rx)):
            
        simo[t,j] = P[rz[j],rx[j], t]

   return P,snap,simo

def cerjang (cerjan,nabc,nx,nz): #função gausiana
  sb=1.5* nabc
  borda = np.zeros(nabc)

  for i in range(nabc):
        dist = nabc - i
        fb = dist / (1.4142 * sb)
        borda[i] = np.exp(-(fb * fb) )

  for ix in range(nx):    
      
    cerjan[:nabc,ix] *= borda
    cerjan[-nabc:,ix] *= borda[::-1]

  for iz in range(nz):    
    
    cerjan[iz,:nabc,] *= borda
    cerjan[iz,-nabc:] *= borda[::-1]
  
  return cerjan


def  disp(c,alpha,f,b):
  cmax = max(c)
  fmax = max(f)
  h=cmax/(alpha*fmax)
  dt=h/(b*cmax)
  return h,dt

def model(inter,c,modelo):
  if len(inter) == 0:
        modelo[:, :] = c[0]
  else:
        modelo[:inter[0], :] = c[0]

        for i in range(1, len(inter)):
            z_ini = inter[i - 1]
            z_fim = inter[i]
            modelo[z_ini:z_fim, :] = c[i]

        modelo[inter[-1]:, :] = c[-1]
  return modelo
  
  
nz = 200
nt = 2000
nx = 200
f=[80]
nabc= 20

modelo=np.zeros((nz,nx))
interfaces = [50, 150]
c = [1000, 1500, 2000]
dh,dt= disp(c,3,f,4)
dh= 4
modelo=model(interfaces,c,modelo)



print(dh,dt)

sx = nx // 2
sz = nz // 2
offset = int(10 / dh) 
rx = list(range(3, nx, offset))
rz=[3]*len(rx)
plt.imshow(modelo,aspect="auto",extent=[0, nx * dh, nz * dh, 0])
plt.scatter(np.array(rx)*dh,np.array(rz)*dh,c= "green", zorder=10,s=12,label="Receptors")
plt.scatter( sx*dh , 3*dh ,c= "red", marker="*", zorder=10,s=120,label="Source")
plt.scatter( sx*dh + 40 , 3*dh ,c= "red", marker="*", zorder=10,s=120)

plt.xlabel("x (m)")
plt.ylabel("z (m)")
plt.legend()
plt.title("Velocity Model")
plt.show()
plt.show()

source2 = np.zeros(nt)
wavelet2 = ricker(np.arange(0, (nt-150)*dt, dt), 30)
source2[150:] = wavelet2

tempo=np.arange(0  ,nt * dt, dt)

prof= np.arange(0, nz * dh, dh)

offset=np.arange(0, nx * dh, dh)

u = np.zeros((nz, nx, nt)) 

source =  ricker(tempo, 30)
#source2 = ricker(tempo2, 30)

cerjan=np.ones((nz,nx))

cerjan= cerjang(cerjan,nabc,nx,nz)
U ,snap,simo = eq2d(u, dt, dh, nt, nx, nz, modelo, sx, sz, source,source2,cerjan)

vmax = np.percentile(np.abs(simo), 99)
vmin = -vmax

plt.imshow(simo, cmap="gray",aspect='auto', extent=[0, 9, nt*dt, 0], vmax=vmax, vmin=vmin)
plt.colorbar()
plt.show()

abs_snap = np.abs(snap)
vmax = np.percentile(abs_snap, 99)
vmin = -vmax


# plt.imshow(U[ 50, 50, :], cmap="gray", aspect="auto",extent=[0, nx*dh, nz*dh, 0], vmax=vmax, vmin=vmin)
# plt.colorbar()
# plt.show()

from matplotlib.animation import FuncAnimation

fig,ax = plt.subplots()

ax.imshow(modelo,cmap='gray',aspect='auto', extent=[0, nx*dh, nz*dh, 0],alpha=0.5)
wave = ax.imshow(snap[:, :, 0], cmap="gray", aspect="auto",extent=[0, nx*dh, nz*dh, 0],alpha=0.7)


ax.set_xlabel("x (m)")
ax.set_ylabel("twt (s)")
ax.set_title("Snapshot")


def atualizar(frame):
    
    abs_snap = np.abs(snap)
    vmax = np.percentile(abs_snap, 99)
    vmin = -vmax
    wave.set_data(snap[:,:,frame])
    wave.set_clim(vmin, vmax)
    ax.set_title(f"time = {frame*dt:.3f} s")
    
    return wave,


ani = FuncAnimation(fig, atualizar, frames=500, interval=10)
#ani.save('monda2d.gif',writer='pilow',fps=30)
plt.show()