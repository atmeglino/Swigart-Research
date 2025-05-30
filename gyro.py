import numpy as np
import pylab as pl
from mpl_toolkits.mplot3d import Axes3D


mu0 = 4*np.pi*1e-7


def Bfield(r,mmo):
    rr = np.sqrt(np.dot(r,r))
    er = r/rr
    return mu0/4/np.pi/rr**3 * (3*np.dot(er,mmo)*er - mmo)

def vec3d(x,y,z): return np.array([x,y,z])

def Bconst():
    return vec3d(0,0,1)


r = vec3d(-1,0,0)
v = vec3d(0,1,1)
q = 1e9
m = 1

trun = 6
nt = 100000
dt = trun/nt

mmo = vec3d(0,0,1)

fig = pl.figure()
ax = fig.add_subplot(projection='3d')

for i in range(nt):
    r = r+0.5*v*dt
    a = q*np.cross(v,Bfield(r,mmo))/m
    v = v+a*dt
    r = r+0.5*v*dt
    if i % 100 == 0:
        print(np.linalg.norm(r))
        # pl.plot(r[0],r[1],'.k')
        ax.scatter(r[0], r[1], r[2], marker=m)
    

ofil = 'tmp.png'
pl.savefig(ofil)
import os
os.system(f'convert {ofil} ~/www/tmp.jpg')


    
