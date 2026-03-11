import numpy as np
import matplotlib.pyplot as plt

dt=0.002
dz=5
nz=500
nt=500
c=1000
modelo=np.full(nz,c)
tempo=np.arange(0,nt*dt,dt)
prof= np.arange(0,nz*dz,dz)

P= np.zeros((nz,nt))

def ricker(t):
  f_corte = 30
  fc = f_corte / (3 * np.sqrt(np.pi))
  td = t - (0.5 * np.sqrt(np.pi) / fc)
  arg = np.pi * (np.pi * fc * td)**2
  return (1 - 2*arg) * np.exp(-arg)

ricker2= ricker(tempo)

def eq1d(P,dt,dz,nt,nz):
  #sId = int(234/ dz)
  for i in range(1,nt-1):
    P[250,i] += ricker2[i]
    for n in range(1,nz-1):
        laplacian=	(1*P[n-1,i]-2*P[n+0,i]+1*P[n+1,i])/(1*1.0*dz**2)

        P[n,i+1] = (c*dt)**2 * laplacian + 2*P[n,i] - P[n,i-1]

  return P

P=eq1d(P,dt,dz,nt,nz)

from matplotlib.animation import FuncAnimation

fig,ax = plt.subplots()
linha, = ax.plot(prof, P[:,0])
ax.set_xlim(prof.min(), prof.max())
ax.set_ylim(-1, 1)

def atualizar(frame):
    linha.set_ydata(P[:,frame])
    ax.set_title(f"Tempo = {frame*dt:.3f} s")
    return linha,

ani = FuncAnimation(fig, atualizar, frames=nt, interval=50)
plt.show()