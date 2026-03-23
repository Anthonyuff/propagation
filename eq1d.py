import numpy as np
import matplotlib.pyplot as plt

dt=0.001
dz=5
nz=500
nt=3200
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

# def ricker3(t):
#   f_corte = 40
#   fc = f_corte / (3 * np.sqrt(np.pi))
#   td = t - (0.5 * np.sqrt(np.pi) / fc)
#   arg = np.pi * (np.pi * fc * td)**2
#   return (1 - 2*arg) * np.exp(-arg)

ricker2= ricker(tempo)
# ricker4= ricker3(tempo)
rec_pos = [50,200]
nrec = len(rec_pos)

rec = np.zeros((nt, nrec))
def eq1d(P,dt,dz,nt,nz,rec_pos,rec):
  #sId = int(234/ dz)
  nrec = len(rec_pos)
  for i in range(1,nt-1):
    P[250,i] += ricker2[i]
    # P[300,i] += ricker4[i]
    for n in range(1,nz-1):
        laplacian=	(1*P[n-1,i]-2*P[n+0,i]+1*P[n+1,i])/(1*1.0*dz**2)

        P[n,i+1] = (c*dt)**2 * laplacian + 2*P[n,i] - P[n,i-1]
    for j in range(nrec):
            rec[i,j] = P[rec_pos[j], i]

  return P,rec

P,rec=eq1d(P,dt,dz,nt,nz,rec_pos,rec)

plt.imshow(rec.T, aspect='auto', cmap='gray',
           extent=[0, nt*dt, nz*dz, 0])

plt.xlabel('time (s)')
plt.ylabel('death (m)')
plt.show()

t = np.arange(nt)*dt

for j in range(2):
    plt.plot(t, rec[:,j], label=f"rec {rec_pos[j]}")

plt.xlabel("time (s)")
plt.ylabel("amplitude")
plt.legend()
plt.show()

from matplotlib.animation import FuncAnimation

fig,ax = plt.subplots()
linha, = ax.plot(prof, P[:,0])
ax.set_xlim(prof.min(), prof.max())
ax.set_ylim(-P.max(),P.max() )

def atualizar(frame):
    linha.set_ydata(P[:,frame])
    ax.set_title(f"time = {frame*dt:.3f} s")
    return linha,

ani = FuncAnimation(fig, atualizar, frames=nt, interval=20)
#ani.save('onda.gif',writer='pilow',fps=30)
plt.show()