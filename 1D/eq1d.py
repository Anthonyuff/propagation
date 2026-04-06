import numpy as np
import matplotlib.pyplot as plt

dt=0.001

dz=5

nz=500

nt=1000

c=1500

tempo=np.arange(0,nt*dt,dt)

prof= np.arange(0,nz*dz,dz)

P= np.zeros((nz,nt))

def ricker(t,freq):
  
  f_corte = freq

  fc = f_corte / (3 * np.sqrt(np.pi))

  td = t - (0.5 * np.sqrt(np.pi) / fc)

  arg = np.pi * (np.pi * fc * td)**2

  return (1 - 2*arg) * np.exp(-arg)



ricker1= ricker(tempo,30)

ricker2= ricker(tempo,40)

rec_pos = [150,200]

nrec = len(rec_pos)

rec = np.zeros((nt, nrec))

nabc = 100

#alpha = 0.00000015

cerjan = np.ones(nz)


# for i in range(nabc):
#     dist = nabc - i
#     fator = np.exp(-(alpha * dist)**2)
#     cerjan[i] = fator
#     cerjan[nz-1-i] = fator

sb = 1.5* nabc

borda = np.zeros(nabc)

for i in range(nabc):   #função gausiana 
    dist = nabc - i
    fb = dist / (1.4142 * sb)
    borda[i] = np.exp(-(fb * fb)*0.55)
cerjan[:nabc] = borda
cerjan[-nabc:] = borda[::-1]

def eq1d(P,dt,dz,nt,nz,rec_pos,rec,cerjan):
 
  nrec = len(rec_pos)

  for i in range(1,nt-1):

    P[250,i] += ricker1[i]

    P[300,i] += ricker2[i]

    for n in range(2,nz-2):
        
        laplacian = (-P[n+2, i] + 16*P[n+1, i] - 30*P[n, i] + 16*P[n-1, i] - P[n-2, i])  / (12 * dz**2)

        P[n,i+1] = (c*dt)**2 * laplacian + 2*P[n,i] - P[n,i-1]  
    
    P[:, i] *= cerjan

    P[:, i+1] *= cerjan

    for j in range(nrec):
            
            rec[i,j] = P[rec_pos[j], i]

  return P,rec

P,rec=eq1d(P,dt,dz,nt,nz,rec_pos,rec,cerjan)

plt.plot(cerjan)

plt.show()

plt.imshow(P.T, aspect='auto', cmap='gray', extent=[0, nt*dt, nz*dz, 0])

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

ani = FuncAnimation(fig, atualizar, frames=nt, interval=10)
ani.save('ondac.gif',writer='pilow',fps=30)
plt.show()