import numpy as np
from scipy.spatial.transform import Rotation as Ro
import scipy as sp

from constants import *

bodyt = np.dtype([('m',float),('r',float),
                  ('x',float),('y',float),('z',float),
                  ('vx',float),('vy',float),('vz',float),
                  ('ax',float),('ay',float),('az',float),
                  ('Q',float),('eta',float),('L',float),('q',float)])

def Bfield(r,mmo):
    rr = np.sqrt(np.dot(r,r))
    er = r/rr
    return mu0/4/np.pi/rr**3 * (3*np.dot(er,mmo)*er - mmo)

def beta_est(rp,rhop,Q):
    return 3*Lsun*Q/(16*np.pi*GNewt*clight*rhop*rp*Msun)

def beta_estx(rp,mp,Q):
    return rp**2*Lsun/4*Q/(GNewt*clight*mp*Msun)

def accGrav(b,soft=1e-99, mthresh=1e10):   # mthresh sets if body is a gravitating mass.
    global Lsun,GNewt,uswind
    b.ax = b.ay = b.az = 0.0
    bm = b[b.m>mthresh]
    if True: # gravational force
        bdx = np.repeat(b.x[:,np.newaxis],len(bm),1)-np.repeat(bm.x[np.newaxis,:],len(b),0)
        bdy = np.repeat(b.y[:,np.newaxis],len(bm),1)-np.repeat(bm.y[np.newaxis,:],len(b),0)
        bdz = np.repeat(b.z[:,np.newaxis],len(bm),1)-np.repeat(bm.z[np.newaxis,:],len(b),0)
        r3 = (bdx**2+bdy**2+bdz**2+soft**2)**(3/2)
        Gm = GNewt*np.repeat(bm.m[np.newaxis,:],len(b),0)
        b.ax = np.sum(-Gm*bdx/r3,axis=1)
        b.ay = np.sum(-Gm*bdy/r3,axis=1)
        b.az = np.sum(-Gm*bdz/r3,axis=1)
    return b.ax, b.ay, b.az

def accMag(b):
    msk = (b.q>0)
    if np.sum(msk)>0:# magnetic field
        xe,ye,ze = b[1].x,b[1].y,b[1].z
        vxe,vye,vze = b[1].vx,b[1].vy,b[1].vz
        mmuEarth = 1e10
        mmo  = mmuEarth = np.array([np.sin(23/180.*np.pi),0,np.cos(23/180.*np.pi)])
        bt = b[msk]
        rx,ry,rz,vx,vy,vz = bt.x-xe,bt.y-ye,bt.z-ze,bt.vx-vxe,bt.vy-vye,bt.vz-vze
        mu0 = 4*np.pi*1e-7
        meter,kg = 100.0,1000.
        B =  Bfield(np.array([rx,ry,rz])/meter,mmo,mu0)
        a = bt.q * np.cross(np.array([vx,vy,vz])/meter,B)/(bt.m/kg) # m/s^2
        a *= meter
        b.ax[msk] += a[0]
        b.ay[msk] += a[1]
        b.az[msk] += a[2]
    return b.ax, b.ay, b.az
        
def accRad(b):
    msk = (b.Q>0)
    if np.sum(msk)>0: #radiation pressure
        xs,ys,zs,vxs,vys,vzs = b[0].x,b[0].y,b[0].z,b[0].vx,b[0].vy,b[0].vz
        bt = b[msk]
        rx,ry,rz,vx,vy,vz = bt.x-xs,bt.y-ys,bt.z-zs,bt.vx-vxs,bt.vy-vys,bt.vz-vzs
        r2 = rx**2+ry**2+rz**2
        rr = np.sqrt(r2)
        Ap = np.pi*bt.r**2  # this is key! r thus defines cross section not physical radius....
        S = Lsun/(4*np.pi*r2)
        etadivQ = bt.eta/bt.Q
        radacc = Ap*S*bt.Q/clight/bt.m*(1+etadivQ*uswind/clight-(1+etadivQ)*(vx*rx+vy*ry+vz*rz)/rr/clight)
        pracc = -Ap*S*bt.Q/clight**2/bt.m*(1+etadivQ)
        b.ax[msk] += radacc*rx/rr + pracc*vx
        b.ay[msk] += radacc*ry/rr + pracc*vy
        b.az[msk] += radacc*rz/rr + pracc*vz
    return b.ax, b.ay, b.az
        
def ode(b):
    n_bodies = len(b)
    
    pos = np.zeros((n_bodies, 3))
    for i in range(n_bodies):
        pos[i] = [b[i].x, b[i].y, b[i].z]
    
    vel = np.zeros((n_bodies, 3))
    for i in range(n_bodies):
        vel[i] = [b[i].vx, b[i].vy, b[i].vz]
    
    acc_grav = np.column_stack([grav_x, grav_y, grav_z])
    acc_mag = np.column_stack([mag_x, mag_y, mag_z]) 
    acc_rad = np.column_stack([rad_x, rad_y, rad_z])
    
    acc = acc_grav + acc_mag + acc_rad
    
    return vel, acc

def state_vector_to_bodies(b):
    n_bodies = len(b)
    bodies = np.zeros(n_bodies, dtype=bodyt)
    
    for i in range(n_bodies):
        bodies[i].x, bodies[i].y, bodies[i].z = b[3*i:3*i+3]
        bodies[i].vx, bodies[i].vy, bodies[i].vz = b[3*n_bodies + 3*i:3*n_bodies + 3*i+3]
        bodies[i].m = b[i]
    
    return bodies
        
def step(b,t,dt):
    b.x,b.y,b.z = b.x+0.5*dt*b.vx,b.y+0.5*dt*b.vy,b.z+0.5*dt*b.vz
    acc(b,t+0.5*dt)
    b.vx,b.vy,b.vz = b.vx+dt*b.ax,b.vy+dt*b.ay,b.vz+dt*b.az
    b.x,b.y,b.z = b.x+0.5*dt*b.vx,b.y+0.5*dt*b.vy,b.z+0.5*dt*b.vz

def step4symp(b,t,dt):
    beta = 2.**(1./3.);
    dt1 = dt/(2.-beta); dt2 = -beta*dt1;
    step(b,t,dt1)
    step(b,t+dt1,dt2)
    step(b,t+dt1+dt2,dt1)
    return
    # why did i do it this way?
    b.x,b.y,b.z = b.x+0.5*dt1*b.vx,b.y+0.5*dt1*b.vy,b.z+0.5*dt1*b.vz
    acc(b,t+0.5*dt1)
    b.vx,b.vy,b.vz = b.vx+dt1*b.ax,b.vy+dt1*b.ay,b.vz+dt1*b.az
    b.x,b.y,b.z = b.x+0.5*(dt1+dt2)*b.vx,b.y+0.5*(dt1+dt2)*b.vy,b.z+0.5*(dt1+dt2)*b.vz
    acc(b,t+dt1+0.5*dt2)
    b.vx,b.vy,b.vz = b.vx+dt2*b.ax,b.vy+dt2*b.ay,b.vz+dt2*b.az
    b.x,b.y,b.z = b.x+0.5*(dt1+dt2)*b.vx,b.y+0.5*(dt1+dt2)*b.vy,b.z+0.5*(dt1+dt2)*b.vz
    acc(b,t+dt1+dt2+0.5*dt1)
    b.vx,b.vy,b.vz = b.vx+dt1*b.ax,b.vy+dt1*b.ay,b.vz+dt1*b.az
    b.x,b.y,b.z = b.x+0.5*dt1*b.vx,b.y+0.5*dt1*b.vy,b.z+0.5*dt1*b.vz
    return

def step6symp(b,t,dt):
    beta = 2.**(1./3.); t1 = dt/(2.-beta); t2 = -beta*t1;
    step4symp(b,t,t1)
    step4symp(b,t+t1,t2)
    step4symp(b,t+t1+t2,t1)

def steps(b,t,trun,ns,order=6):
    dt = trun/ns
    if order==6:
        for i in range(ns): step6symp(b,t,dt)
    else:
        for i in range(ns): step4symp(b,t,dt)

def dist(b): 
    return np.sqrt(b.x**2+b.y**2+b.z**2)

def reldist(b,bref):
    return np.sqrt((b.x-bref.x)**2+(b.y-bref.y)**2+(b.z-bref.z)**2)

def pairsep(b,bref):
    return np.sqrt((b.x-bref.x)**2+(b.y-bref.y)**2+(b.z-bref.z)**2)

def speed(b): 
    return np.sqrt(b.vx**2+b.vy**2+b.vz**2)

def relspeed(b,bref):
    return np.sqrt((b.vx-bref.vx)**2+(b.vy-bref.vy)**2+(b.vz-bref.vz)**2)
    
def energy(b):
    PE = 0
    for i in range(len(b)):
        for j in range(i+1,len(b)):
            dx,dy,dz = b[j].x-b[i].x,b[j].y-b[i].y,b[j].z-b[i].z
            dr = np.sqrt(dx**2+dy**2+dz**2)
            if dr > 1.0e-5: # centimeters here...
                PE += -GNewt*b[i].m*b[j].m/dr
    E = np.sum(0.5*b.m*(b.vx**2 + b.vy**2 + b.vz**2)) + PE
    return E

# other helpers....


def setbody(Q): # returns a single object bodyt
    # Q is listlike with elements of dtype=bodyt. if Q runs out, 
    # tricky, sets the elements in order but you don't need
    # to set all. setbodies((m,r,x,y)) sets just these 4, all else is zero.
    # see above at ...np.dtype([('m',float),('r',float),
    q = np.zeros(1,dtype=bodyt)[0]
    for i in range(len(Q)): q[i] = Q[i]
    return q
    
# conveniences, maybe, for dtype in nbody.py
def pos(p): return np.array([p.x,p.y,p.z])
def vel(p): return np.array([p.vx,p.vy,p.vz])
def posvelrel(ba,bb):
    return np.array([ba.x-bb.x,ba.y-bb.y,ba.z-bb.z]),np.array([ba.vx-bb.vx,ba.vy-bb.vy,ba.vz-bb.vz])
def posrel(ba,bb):
    return np.array([ba.x-bb.x,ba.y-bb.y,ba.z-bb.z])
def velrel(ba,bb):
    return np.array([ba.vx-bb.vx,ba.vy-bb.vy,ba.vz-bb.vz])

def vec3d(x,y,z): return np.array([x,y,z])

def unitvec(a): return a/np.linalg.norm(a)

# for keplerian orbits...

def zenithframe(planet,star): # frame with z out of orbital plane, x is planet-sun dir, y = -v
    ex = unitvec(posrel(star,planet))
    ez = unitvec(np.cross(velrel(planet,star),ex))
    ey = unitvec(np.cross(ez,ex))
    return np.array([ex,ey,ez])

def bodyframe_equinox(obliq,orbinfo=([],0,1,0.0)): # this is over the top.
    # returns simple tilted ref frame z-axis aligned with spin at vernal equinox, i.e., sun on +x axis.
    # unless! orbinfo = (b,staridx,planetidx,dt), then calc ex,ey,ez from orbit info....should work for Mars.
    # b = list of sol sys bodies dtype = bodyt....
    # staridx = index of sun (0 usually!), i.e., sun = b[starindex]
    # planetidx = index of planet...
    # trun = time between epoch of b[] and vernal equinox. may be negative. becareful. 
    if len(orbinfo):
        b = orbinfo[0].copy() # expect a 
        staridx = orbinfo[1]
        planetidx = orbinfo[2]
        trun = orbinfo[3]
        if trun != 0:
            steps(b,0,trun,400)  # integrate
        star,planet = b[staridx],b[planetidx]
        ez = unitvec(np.cross(posrel(planet,star),velrel(planet,star)))
        ex = unitvec(posrel(star,planet))
        ey = unitvec(np.cross(ez,ex))
    else:
        ex,ey,ez = vec3d(1,0,0),vec3d(0,1,0),vec3d(0,0,1)
    soex = np.sin(-obliq/2)*ex # vec to rotate around, ex, times sin(theta/2)...
    robl = Ro.from_quat([soex[0],soex[1],soex[2],np.cos(-obliq/2)]) # set obliquity, x points to sun
    return robl.apply(np.array([ex,ey,ez]))

def bodyframe(tnow,tref,bodyframe_ref,Pspin): # spins ref frame about z axis....
    t,om = tnow-tref,2*np.pi/Pspin
    rotv = unitvec(bodyframe_ref[2])*np.sin(om*t/2)
    rspin = Ro.from_quat([rotv[0],rotv[1],rotv[2],np.cos(om*t/2)]) # rotate to now
    return np.array(rspin.apply(bodyframe_ref))

def bodyframe_earth(b,sunindex,earthindex,epoch,tequinox,obliq,pspin):
    bf = bodyframe_equinox(obliq,orbinfo=(b,sunindex,earthindex,tequinox-epoch))
    dt = np.ceil(tequinox/day)-tequinox/day # no longer aligned with high noon in Greenwich
    #dt += 12*60/day
    bf = bodyframe(-dt*day,0,bf,pspin)
    return bf

def localframelalo(latdeg,londeg,t,tref,bfref,pspin): # latitude, longitude, in degrees
    ebx,eby,ebz = bodyframe(t,tref,bfref,pspin)
    theta,phi = (90-latdeg)*degree,londeg*degree
    ct,st,cp,sp = np.cos(theta),np.sin(theta),np.cos(phi),np.sin(phi)
    e_up = st*cp*ebx + st*sp*eby + ct*ebz # e_r in spherical polar
    e_north = -ct*cp*ebx - ct*sp*eby +st*ebz # -e_theta
    e_east = -sp*ebx + cp*eby            # e_phi
    return np.array([e_up, e_north, e_east])    
    



def orbels(sat,ctrlmass): # a,e,i but i is rel to bg coordinates not anything sensible
    r,v = posvelrel(sat,ctrlmass)
    m = ctrlmass.m
    E = 0.5*np.sum(v**2)-GNewt*m/np.linalg.norm(r)
    L = np.cross(r,v)
    asemi = -GNewt*m/(2*E)
    ecc = np.linalg.norm(np.cross(v,L)/(GNewt*m)-unitvec(r))
    inc = np.arccos(np.dot(unitvec(L),np.array([0.0,0.0,1.0])))
    return asemi,ecc,inc
