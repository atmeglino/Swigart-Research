import numpy as np
from scipy.spatial.transform import Rotation as Ro
import earthsat as es

from constants import *

bodyt = np.dtype([('m',float),('r',float),
                  ('x',float),('y',float),('z',float),
                  ('vx',float),('vy',float),('vz',float),
                  ('ax',float),('ay',float),('az',float),
                  ('Q',float),('eta',float),('L',float),('q',float)])

def Bfield(r,mmo): # r can be a vector or an array of vectors, (n,3) MKS MKS MKS MKS!!!!!
    if r.ndim==1:
        rr = np.linalg.norm(r);
        print(rr/Rearth)
        er = r/rr
        print(er,mu0*mmo/rr**3)
        erdotmmo = np.dot(er,mmo)
    else:
        rr = np.linalg.norm(r,axis=1)[:,np.newaxis]
        er = r/rr
        erdotmmo = np.dot(er,mmo)[:,np.newaxis]
    return mu0/4/np.pi/rr**3 * (3*erdotmmo*er - mmo)

def beta_est(rp,rhop,Q):
    return 3*Lsun*Q/(16*np.pi*GNewt*clight*rhop*rp*Msun)

def beta_estx(rp,mp,Q):
    return rp**2*Lsun/4*Q/(GNewt*clight*mp*Msun)

bfearthref = []
tearthref = 0.0
sunindex = 0
earthindex = 1
magmoearthunitref = []
magmoearthlalo = MagmoEarthlalo # so we can modify this from the calling routine...

def earth_spin_init(b,tnow,si=sunindex,ei=earthindex):
    global bfearthref,tearthref,sunindex,earthindex,magmoearthunitref
    global J2tildeEarth
    sunindex,earthindex=si,ei
    tearthref = TmarchequinoxEarth2025JD * day # this must be a march equinox                                           
    msk = b.m > 1e20 # use only massive bodies here...                                                                  
    J2tildesave,J2tildeEarth = J2tildeEarth,0 # ack! we need spin to do J2 on moon, but don't have it (yet)             
    # so turn J2 off for now..                                                                                          
    bfearthref = bodyframe_earth(b[msk],si,ei,tnow,tearthref,tiltEarth,PspinEarth)
    J2tildeEarth = J2tildesave # now turn it back on and iterate once again...                                          
    bfearthref = bodyframe_earth(b[msk],si,ei,tnow,tearthref,tiltEarth,PspinEarth) # it seems to make very little diff  
    return bfearthref.copy()

def earth_magnetic_moment_init(b,tnow,si=sunindex,ei=earthindex):
    global bfearthref,tearthref,sunindex,earthindex,magmoearthunitref,magmoearthlalo
    sunindex,earthindex=si,ei
    earth_spin_init(b,tnow,si=si,ei=ei)
    magmoearthunitref,_,_ = localframelalo(*magmoearthlalo,tearthref,tearthref,bfearthref,PspinEarth)
    return
    
def earth_magnetic_moment(t):
    global bfearthref,tearthref,sunindex,earthindex,magmoearthunitref
    dt, om = t-tearthref, 2*np.pi/PspinEarth
    # rotate about spin axis
    rotv = unitvec(bfearthref[2])*np.sin(om*t/2)
    rspin = Ro.from_quat([rotv[0],rotv[1],rotv[2],np.cos(om*t/2)]) # rotate to now
    mmu = MagmoEarth *  rspin.apply(magmoearthunitref)
    return mmu

def earth_spin(t): # t just in case, down the road....                                                                  
    global bfearthref
    return bfearthref[2]

def acc(b,t,soft=1e-99, mthresh=1e10):   # mthresh sets if body is a gravitating mass.
    global Lsun,GNewt,uswind
    b.ax = b.ay = b.az = 0.0
    bm = b[b.m>mthresh]
    if True:
        bdx = np.repeat(b.x[:,np.newaxis],len(bm),1)-np.repeat(bm.x[np.newaxis,:],len(b),0)
        bdy = np.repeat(b.y[:,np.newaxis],len(bm),1)-np.repeat(bm.y[np.newaxis,:],len(b),0)
        bdz = np.repeat(b.z[:,np.newaxis],len(bm),1)-np.repeat(bm.z[np.newaxis,:],len(b),0)
        r3 = (bdx**2+bdy**2+bdz**2+soft**2)**(3/2)
        Gm = GNewt*np.repeat(bm.m[np.newaxis,:],len(b),0)
        b.ax = np.sum(-Gm*bdx/r3,axis=1)
        b.ay = np.sum(-Gm*bdy/r3,axis=1)
        b.az = np.sum(-Gm*bdz/r3,axis=1)
        msk = (b.q>0)
        if np.sum(msk)>0:
            # do the magnetic field!
            xe,ye,ze = b[1].x,b[1].y,b[1].z
            vxe,vye,vze = b[1].vx,b[1].vy,b[1].vz
            mmuEarth = 1e22
            mmo  =  earth_magnetic_moment(t)
            bt = b[msk]
            rx,ry,rz,vx,vy,vz = bt.x-xe,bt.y-ye,bt.z-ze,bt.vx-vxe,bt.vy-vye,bt.vz-vze
            mu0 = 4*np.pi*1e-7
            rvec,vvec = np.array([rx,ry,rz]).T,np.array([vx,vy,vz]).T
            B =  Bfield(rvec/meter,mmo)
            a = (bt.q/(bt.m/kg))[:,np.newaxis] * np.cross(vvec,B) # m/s^2 the meters cancel out
            b.ax[msk] += a[:,0]
            b.ay[msk] += a[:,1]
            b.az[msk] += a[:,2]
        msk = (b.Q>0)
        if np.sum(msk)>0:
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
        return 
    for i in range(len(b)):
        dx,dy,dz = (bm.x-b.x[i]),(bm.y-b.y[i]),(bm.z-b.z[i])
        dr3 = (dx**2 + dy**2 + dz**2 + soft**2)**1.5;
        b.ax[i],b.ay[i],b.az[i] = np.sum(GNewt*bm.m*dx/dr3),np.sum(GNewt*bm.m*dy/dr3),np.sum(GNewt*bm.m*dz/dr3)
  
def accGrav(b, t, soft=1e-99, mthresh=1e10):   # mthresh sets if body is a gravitating mass.
    n_bodies = len(b)
    pos = getPosition(b)
    masses = b.m
    msk = masses > mthresh
    
    if not np.any(msk):
        return np.zeros(n_bodies), np.zeros(n_bodies), np.zeros(n_bodies)
    
    grav_masses = masses[msk]
    grav_pos = pos[msk]
    
    dr = grav_pos[np.newaxis, :, :] - pos[:, np.newaxis, :]
    r_sq = np.sum(dr**2, axis=2) + soft**2
    r_cube = r_sq**(3/2)
    
    Gm = GNewt * grav_masses[np.newaxis, :]
    acc_grav = np.sum(-Gm[:, :, np.newaxis] * dr / r_cube[:, :, np.newaxis], axis=1)
    
    '''
    acc_grav = np.zeros((len(b), 3))
    bm = b[b.m>mthresh] # gravity
    bdx = np.repeat(b.x[:,np.newaxis],len(bm),1)-np.repeat(bm.x[np.newaxis,:],len(b),0)
    bdy = np.repeat(b.y[:,np.newaxis],len(bm),1)-np.repeat(bm.y[np.newaxis,:],len(b),0)
    bdz = np.repeat(b.z[:,np.newaxis],len(bm),1)-np.repeat(bm.z[np.newaxis,:],len(b),0)
    r3 = (bdx**2+bdy**2+bdz**2+soft**2)**(3/2)
    Gm = GNewt*np.repeat(bm.m[np.newaxis,:],len(b),0)
    acc_grav[:,0] = np.sum(-Gm * bdx / r3, axis=1) # x-component
    acc_grav[:,1] = np.sum(-Gm * bdy / r3, axis=1) # y-component
    acc_grav[:,2] = np.sum(-Gm * bdz / r3, axis=1) # z-component
    
    return acc_grav[:, 0], acc_grav[:, 1], acc_grav[:, 2]
    '''
    # return acc_grav[:, 0], acc_grav[:, 1], acc_grav[:, 2]
    return r_cube[:, 0], r_cube[:, 1], r_cube[:, 2]

def accJ2(b, t):
    n_bodies = len(b)
    acc_j2 = np.zeros((n_bodies, 3))
    
    pos = getPosition(b)
    earth_pos = pos[1]
    
    msk = b.m < 0.05 * b[1].m
    if not np.any(msk):
        return np.zeros(n_bodies), np.zeros(n_bodies), np.zeros(n_bodies)
    
    rel_pos = pos[msk] - earth_pos
    
    J2 = J2tildeEarth * GNewt * Mearth * Rearth**2
    aJ2 = gravJ2(rel_pos, J2, earth_spin(t))
    acc_j2[msk] = aJ2
    
    '''
    acc_j2 = np.zeros((len(b), 3))
    xe,ye,ze = b[1].x,b[1].y,b[1].z
    msk = b.m < 0.05*b[1].m # just the small stuff, incl. moon. should really limit by distance not mass!       
    bt = b[msk]
    rx,ry,rz = bt.x-xe,bt.y-ye,bt.z-ze
    rvec = np.array([rx,ry,rz]).T
    J2 = J2tildeEarth * GNewt * Mearth * Rearth**2
    aJ2 = gravJ2(rvec,J2,earth_spin(t))
    acc_j2[msk,0] += aJ2[:,0] # x-component
    acc_j2[msk,1] += aJ2[:,1] # y-component
    acc_j2[msk,2] += aJ2[:,2] # z-component
    
    return acc_j2[:, 0], acc_j2[:, 1], acc_j2[:, 2]
    '''
    return acc_j2[:, 0], acc_j2[:, 1], acc_j2[:, 2]

def gravJ2(r,J2,spinaxisplanet):  # returns accel given 3d pos r and physical J2 & spin vector (not J2tilde!)           
    ez = unitvec(spinaxisplanet)
    if r.ndim==1:
        r7 = np.linalg.norm(r)**7
        z = np.dot(r,ez)
        zvec = z*ez
        rp = r - zvec
        rp2 = np.linalg.norm(rp)**2
    else:
        r7 = np.linalg.norm(r,axis=1)[:,np.newaxis]**7
        z = np.dot(r,ez)[:,np.newaxis]
        zvec = z * ez
        rp = r - zvec
        rp2 = np.linalg.norm(r,axis=1)[:,np.newaxis]**2
    z2 = z**2
    aJ2 = (J2/r7) * ((6*z2 - 1.5*rp2) * rp + (3*z2 - 4.5*rp2) * zvec)
    
    '''
    ez = unitvec(spinaxisplanet)
    if r.ndim==1:
        r7 = np.linalg.norm(r)**7
        z = np.dot(r,ez)
        zvec = z*ez
        rp = r - zvec
        rp2 = np.linalg.norm(rp)**2
    else:
        r7 = np.linalg.norm(r,axis=1)[:,np.newaxis]**7
        z = np.dot(r,ez)[:,np.newaxis]
        zvec = z * ez
        rp = r - zvec
        rp2 = np.linalg.norm(r,axis=1)[:,np.newaxis]**2
    z2 = z**2
    aJ2 = (J2/r7) * ((6*z2 - 1.5*rp2) * rp + (3*z2 - 4.5*rp2) * zvec)
    
    return aJ2
    '''
    return aJ2

def accMag(b, t):
    n_bodies = len(b)
    acc_mag = np.zeros((n_bodies, 3))
    
    msk = b.q > 0
    if not np.any(msk):
        return np.zeros(n_bodies), np.zeros(n_bodies), np.zeros(n_bodies)
    
    pos = getPosition(b)
    vel = getVelocity(b)
    
    earth_pos = pos[1]
    earth_vel = vel[1]
    mmo = earth_magnetic_moment(t)
    
    rel_pos = pos[msk] - earth_pos
    rel_vel = vel[msk] - earth_vel
    
    B = Bfield(rel_pos/meter, mmo)
    a = (b.q[msk] / (b.m[msk]/kg))[:, np.newaxis] * np.cross(rel_vel, B)
    acc_mag[msk] = a
    
    '''
    acc_mag = np.zeros((len(b), 3))
    msk = (b['q']>0)
    if np.sum(msk)>0: # magnetic force
        xe,ye,ze = b[1].x,b[1].y,b[1].z
        vxe,vye,vze = b[1].vx,b[1].vy,b[1].vz
        mmuEarth = 1e22
        mmo  =  earth_magnetic_moment(t)
        bt = b[msk]
        rx,ry,rz,vx,vy,vz = bt.x-xe,bt.y-ye,bt.z-ze,bt.vx-vxe,bt.vy-vye,bt.vz-vze
        mu0 = 4*np.pi*1e-7
        rvec,vvec = np.array([rx,ry,rz]).T,np.array([vx,vy,vz]).T
        B =  Bfield(rvec/meter,mmo)
        a = (bt.q/(bt.m/kg))[:,np.newaxis] * np.cross(vvec,B) # m/s^2 the meters cancel out
        acc_mag[msk, 0] = a[:,0] # x-component
        acc_mag[msk, 1] = a[:,1] # y-component
        acc_mag[msk, 2] = a[:,2] # z-component
    return acc_mag[:, 0], acc_mag[:, 1], acc_mag[:, 2]
    '''
    
    return acc_mag[:, 0], acc_mag[:, 1], acc_mag[:, 2]
        
def accRad(b):
    n_bodies = len(b)
    acc_rad = np.zeros((n_bodies, 3))
    
    msk = b.Q > 0
    if not np.any(msk):
        return np.zeros(n_bodies), np.zeros(n_bodies), np.zeros(n_bodies)
    
    pos = getPosition(b)
    vel = getVelocity(b)
    
    sun_pos = pos[0]
    sun_vel = vel[0]
    
    rel_pos = pos[msk] - sun_pos
    rel_vel = vel[msk] - sun_vel
    
    r2 = np.sum(rel_pos**2, axis=1)
    rr = np.sqrt(r2)
    
    Ap = np.pi * b.r[msk]**2
    S = Lsun / (4 * np.pi * r2)
    etadivQ = b.eta[msk] / b.Q[msk]
    
    radial_vel = np.sum(rel_vel * rel_pos, axis=1) / rr
    
    radacc_mag = (Ap * S * b.Q[msk] / clight / b.m[msk] * 
                 (1 + etadivQ * uswind/clight - (1 + etadivQ) * radial_vel/clight))
    
    pracc_mag = -Ap * S * b.Q[msk] / clight**2 / b.m[msk] * (1 + etadivQ)
    
    radial_unit = rel_pos / rr[:, np.newaxis]
    a = radacc_mag[:, np.newaxis] * radial_unit + pracc_mag[:, np.newaxis] * rel_vel
    acc_rad[msk] = a
    
    '''
    acc_rad = np.zeros((len(b), 3))
    msk = (b['Q']>0)
    if np.sum(msk)>0: # radiation pressure
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
        acc_rad[msk, 0] = radacc * rx / rr + pracc * vx # x-component
        acc_rad[msk, 1] = radacc * ry / rr + pracc * vy # y-component
        acc_rad[msk, 2] = radacc * rz / rr + pracc * vz # z-component
    return acc_rad[:, 0], acc_rad[:, 1], acc_rad[:, 2]
    '''
    
    return acc_rad[:, 0], acc_rad[:, 1], acc_rad[:, 2]

def accTotal(b,t,include_grav=True,include_j2=True,include_mag=True,include_rad=True):
    acc_total = np.zeros((len(b), 3))
    if include_grav:
        grav_x, grav_y, grav_z = accGrav(b, t)
        acc_total[:, 0] += grav_x
        acc_total[:, 1] += grav_y
        acc_total[:, 2] += grav_z
    if include_j2:
        j2_x, j2_y, j2_z = accJ2(b, t)
        acc_total[:, 0] += j2_x
        acc_total[:, 1] += j2_y
        acc_total[:, 2] += j2_z
    if include_mag:
        mag_x, mag_y, mag_z = accMag(b, t)
        acc_total[:, 0] += mag_x
        acc_total[:, 1] += mag_y
        acc_total[:, 2] += mag_z
    if include_rad:
        rad_x, rad_y, rad_z = accRad(b)
        acc_total[:, 0] += rad_x
        acc_total[:, 1] += rad_y
        acc_total[:, 2] += rad_z
    return acc_total[:, 0], acc_total[:, 1], acc_total[:, 2]

def getPosition(b):
    '''
    n_bodies = len(b)
    
    pos = np.zeros((n_bodies, 3))
    for i in range(n_bodies):
        pos[i] = [b[i].x, b[i].y, b[i].z]
    
    return pos
    '''
    return np.column_stack([b.x, b.y, b.z])

def getVelocity(b):
    '''
    n_bodies = len(b)
    
    vel = np.zeros((n_bodies, 3))
    for i in range(n_bodies):
        vel[i] = [b[i].vx, b[i].vy, b[i].vz]
    
    return vel
    '''
    return np.column_stack([b.vx, b.vy, b.vz])

def initialState(b):
    '''
    pos, vel = getPosition(b), getVelocity(b)
    init_state = np.concatenate([pos.flatten(), vel.flatten()])
    
    return init_state
    '''
    pos, vel = getPosition(b), getVelocity(b)
    return np.concatenate([pos.flatten(), vel.flatten()])

def ode(t, y, b):
    n_bodies = len(b)
    
    # solve_ivp uses y (flat array of state variables) but our physics functions
    # use b (structured array), so must re-write y into b
    pos = y[:3*n_bodies].reshape(n_bodies, 3)
    vel = y[3*n_bodies:].reshape(n_bodies, 3)
    
    b.x, b.y, b.z = pos.T
    b.vx, b.vy, b.vz = vel.T
    
    acc_x, acc_y, acc_z = accTotal(b, t, include_j2=False, include_mag=False, include_rad=False)
    acc = np.column_stack([acc_x, acc_y, acc_z])
    
    dydt = np.concatenate([vel.flatten(), acc.flatten()])
    
    '''
    n_bodies = len(b)
    
    # solve_ivp uses y (flat array of state variables) but our physics functions
    # use b (structured array), so must re-write y into b
    for i in range(n_bodies):
        b[i].x, b[i].y, b[i].z = y[3*i:3*i+3]
        b[i].vx, b[i].vy, b[i].vz = y[3*n_bodies + 3*i:3*n_bodies + 3*i+3]
    
    vel = getVelocity(b)
    acc = np.column_stack(accTotal(b,t))
    dydt = np.concatenate([vel.flatten(), acc.flatten()])
    
    return dydt
    '''
    
    return dydt
        
def step(b,t,dt,acc):
    b.x,b.y,b.z = b.x+0.5*dt*b.vx,b.y+0.5*dt*b.vy,b.z+0.5*dt*b.vz
    acc(b,t+0.5*dt)
    b.vx,b.vy,b.vz = b.vx+dt*b.ax,b.vy+dt*b.ay,b.vz+dt*b.az
    b.x,b.y,b.z = b.x+0.5*dt*b.vx,b.y+0.5*dt*b.vy,b.z+0.5*dt*b.vz

def step4symp(b,t,dt,acc):
    beta = 2.**(1./3.);
    dt1 = dt/(2.-beta); dt2 = -beta*dt1;
    step(b,t,dt1,acc)
    step(b,t+dt1,dt2,acc)
    step(b,t+dt1+dt2,dt1,acc)
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

def step6symp(b,t,dt,acc):
    beta = 2.**(1./3.); t1 = dt/(2.-beta); t2 = -beta*t1;
    step4symp(b,t,t1,acc)
    step4symp(b,t+t1,t2,acc)
    step4symp(b,t+t1+t2,t1,acc)

def steprk4(b,t,dt,acc):
    def getdrv(x,y,z,vx,vy,vz,t):
        b.x,b.y,b.z,b.vx,b.vy,b.vz = x.copy(),y.copy(),z.copy(),vx.copy(),vy.copy(),vz.copy()
        acc(b,t)
        return b.vx.copy(),b.vy.copy(),b.vz.copy(),b.ax.copy(),b.ay.copy(),b.az.copy()
    x0,y0,z0,vx0,vy0,vz0 = b.x.copy(),b.y.copy(),b.z.copy(),b.vx.copy(),b.vy.copy(),b.vz.copy()
    h = dt # to follow wiki
    # --- k1 ---
    k1x,k1y,k1z,k1vx,k1vy,k1vz = getdrv(x0,y0,z0,vx0,vy0,vz0,t)
    # --- k2 ---
    k2x,k2y,k2z,k2vx,k2vy,k2vz = getdrv(x0+h*k1x/2,y0+h*k1y/2,z0+h*k1z/2,
                                        vx0+h*k1vx/2,vy0+h*k1vy/2,vz0+h*k1vz/2,t+h/2)
    # --- k3 ---
    k3x,k3y,k3z,k3vx,k3vy,k3vz = getdrv(x0+h*k2x/2,y0+h*k2y/2,z0+h*k2z/2,
                                        vx0+h*k2vx/2,vy0+h*k2vy/2,vz0+h*k2vz/2,t+h/2)
    # --- k4 ---
    k4x,k4y,k4z,k4vx,k4vy,k4vz = getdrv(x0+h*k3x,y0+h*k3y,z0+h*k3z,
                                        vx0+h*k3vx,vy0+h*k3vy,vz0+h*k3vz,t+h)
    # final update 
    b.x = x0 + (k1x + 2 * k2x + 2 * k3x + k4x) * dt / 6
    b.y = y0 + (k1y + 2 * k2y + 2 * k3y + k4y) * dt / 6
    b.z = z0 + (k1z + 2 * k2z + 2 * k3z + k4z) * dt / 6
    b.vx = vx0 + (k1vx + 2 * k2vx + 2 * k3vx + k4vx) * dt / 6
    b.vy = vy0 + (k1vy + 2 * k2vy + 2 * k3vy + k4vy) * dt / 6
    b.vz = vz0 + (k1vz + 2 * k2vz + 2 * k3vz + k4vz) * dt / 6
    return


# sets coeffs for the k_i in rk4(5)
def rkk2(y,k1): return y + k1/4
def rkk3(y,k1,k2): return y + 3*k1/32+9*k2/32
def rkk4(y,k1,k2,k3): return y + 1932*k1/2197 - 7200*k2/2197 + 7296*k3/2197
def rkk5(y,k1,k2,k3,k4): return y + 439*k1/216 - 8*k2 + 3680*k3/513 - 845*k4/4104
def rkk6(y,k1,k2,k3,k4,k5): return y - 8*k1/27 + 2*k2 - 3544*k3/2565 + 1859*k4/4104 - 11*k5/40 

def rkd4(y,k1,k3,k4,k5): return y + 25*k1/216 + 1408*k3/2565 + 2197*k4/4104 - k5/5
def rkd5(y,k1,k3,k4,k5,k6): return y + 16*k1/135 + 6656*k3/12825 + 28561*k4/56430 - 9*k5/50 + 2*k6/55

def steprk45(b,t,dt,acc):
    def getdrv(x,y,z,vx,vy,vz,t,h):
        b.x,b.y,b.z,b.vx,b.vy,b.vz = x,y,z,vx,vy,vz
        acc(b,t)
        return h*b.vx,h*b.vy,h*b.vz,h*b.ax,h*b.ay,h*b.az
    x0,y0,z0,vx0,vy0,vz0 = b.x.copy(),b.y.copy(),b.z.copy(),b.vx.copy(),b.vy.copy(),b.vz.copy()
    h = dt # to follow wiki
    k1x,k1y,k1z,k1vx,k1vy,k1vz = getdrv(x0,y0,z0,vx0,vy0,vz0,t,h)
    k2x,k2y,k2z,k2vx,k2vy,k2vz = getdrv(rkk2(x0,k1x),rkk2(y0,k1y),rkk2(z0,k1z),
        rkk2(vx0,k1vx),rkk2(vy0,k1vy),rkk2(vz0,k1vz),t+h/4,h)
    k3x,k3y,k3z,k3vx,k3vy,k3vz = getdrv(rkk3(x0,k1x,k2x),rkk3(y0,k1y,k2y),rkk3(z0,k1z,k2z),
        rkk3(vx0,k1vx,k2vx),rkk3(vy0,k1vy,k2vy),rkk3(vz0,k1vz,k2vz),t+3*h/8,h)
    k4x,k4y,k4z,k4vx,k4vy,k4vz = getdrv(rkk4(x0,k1x,k2x,k3x),rkk4(y0,k1y,k2y,k3y),rkk4(z0,k1z,k2z,k3z),
        rkk4(vx0,k1vx,k2vx,k3vx),rkk4(vy0,k1vy,k2vy,k3vy),rkk4(vz0,k1vz,k2vz,k3vz),t+12*h/13,h)
    k5x,k5y,k5z,k5vx,k5vy,k5vz = getdrv(rkk5(x0,k1x,k2x,k3x,k4x),rkk5(y0,k1y,k2y,k3y,k4y),rkk5(z0,k1z,k2z,k3z,k4z),
        rkk5(vx0,k1vx,k2vx,k3vx,k4vx),rkk5(vy0,k1vy,k2vy,k3vy,k4vy),rkk5(vz0,k1vz,k2vz,k3vz,k4vz),t+h,h)
    k6x,k6y,k6z,k6vx,k6vy,k6vz = getdrv(rkk6(x0,k1x,k2x,k3x,k4x,k5x),rkk6(y0,k1y,k2y,k3y,k4y,k5y),
        rkk6(z0,k1z,k2z,k3z,k4z,k5z),rkk6(vx0,k1vx,k2vx,k3vx,k4vx,k5vx),rkk6(vy0,k1vy,k2vy,k3vy,k4vy,k5vy),
        rkk6(vz0,k1vz,k2vz,k3vz,k4vz,k5vz),t+h/2,h)
    # final update hi res
    b.x,b.y,b.z = rkd5(x0,k1x,k3x,k4x,k5x,k6x),rkd5(y0,k1y,k3y,k4y,k5y,k6y),rkd5(z0,k1z,k3z,k4z,k5z,k6z)
    b.vx,b.vy = rkd5(vx0,k1vx,k3vx,k4vx,k5vx,k6vx),rkd5(vy0,k1vy,k3vy,k4vy,k5vy,k6vy),
    b.vz = rkd5(vz0,k1vz,k3vz,k4vz,k5vz,k6vz)
    return
    # low res 
    b4 = b.copy()
    b4.x,b4.y,b4.z = rkd4(x0,k1x,k3x,k4x,k5x),rkd4(y0,k1y,k3y,k4y,k5y),rkd4(z0,k1z,k3z,k4z,k5z)
    b4.vx,b4.vy,b4.vz = rkd4(vx0,k1vx,k3vx,k4vx,k5vx),rkd4(vy0,k1vy,k3vy,k4vy,k5vy),rkd4(vz0,k1vz,k3vz,k4vz,k5vz)
    err = np.sqrt((b.x-b4.x)**2+(b.y-b4.y)**2+(b.z-b4.z)**2+(b.vx-b4.vx)**2+(b.vy-b4.vy)**2+(b.vz-b4.vz)**2)
    return err

def steps(b,t,trun,ns,acc,method='rk4'):
    dt = trun/ns
    if method in ['rk4','rk45']:
        for i in range(ns): steprk45(b,t,dt,acc)
        return
    elif method=='symp6':
        for i in range(ns): step6symp(b,t,dt,acc)
        return
    elif method=='symp4':
        for i in range(ns): step4symp(b,t,dt,acc)
        return
    else:
        for i in range(ns): step(b,t,dt,acc)
        return
    
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

def vec3dnorm(a): return np.linalg.norm(a)


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
    if len(orbinfo[0]):
        b = orbinfo[0].copy() # expect an array of nbody data 
        staridx = orbinfo[1]
        planetidx = orbinfo[2]
        trun = orbinfo[3]
        if trun != 0:
            steps(b,0,trun,500,acc)  # integrate
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


def eotminutes(jd): # Equaiton of Time, corrects for non circ motion...returns EoT in minutes
    JD_days = jd - 2451545.0
    Cycle = (4 * JD_days) % 1461
    Theta = Cycle * 0.004301
    EoT1 = 7.353 * np.sin(1 * Theta + 6.209)
    EoT2 = 9.927 * np.sin(2 * Theta + 0.37)
    EoT3 = 0.337 * np.sin(3 * Theta + 0.304)
    EoT4 = 0.232 * np.sin(4 * Theta + 0.715)
    EoT = 0.019 + EoT1 + EoT2 + EoT3 + EoT4    
    return EoT

def bodyframe_earth(b,sunindex,earthindex,epoch,tequinox,obliq,pspin):
    bf = bodyframe_equinox(obliq,orbinfo=(b,sunindex,earthindex,tequinox-epoch))
    dt = np.ceil(tequinox/day)-tequinox/day # no longer aligned with high noon in Greenwich
    dt += eotminutes(np.ceil(tequinox/day))*60/day
    bf = bodyframe(-dt*day,0,bf,pspin)
    return bf

def localframelalo(latdeg,londeg,t,tref,bfref,pspin): # latitude, longitude, in degrees
    ebx,eby,ebz = bodyframe(t,tref,bfref,pspin)
    print(t/day,tref/day,pspin,ebx,eby,ebz)
    theta,phi = (90-latdeg)*degree,londeg*degree
    ct,st,cp,sp = np.cos(theta),np.sin(theta),np.cos(phi),np.sin(phi)
    e_up = st*cp*ebx + st*sp*eby + ct*ebz # e_r in spherical polar
    e_north = -ct*cp*ebx - ct*sp*eby +st*ebz # -e_theta
    e_east = -sp*ebx + cp*eby            # e_phi
    return np.array([e_up, e_north, e_east])    
    
def orbels(sat,ctrlmass,ez = vec3d(0,0,1)): # a,e,i but i is rel to bg coordinates not anything sensible
    r,v = posvelrel(sat,ctrlmass)
    m = ctrlmass.m + sat.m
    E = 0.5*np.sum(v**2)-GNewt*m/np.linalg.norm(r)
    L = np.cross(r,v)
    asemi = -GNewt*m/(2*E)
    ecc = np.linalg.norm(np.cross(v,L)/(GNewt*m)-unitvec(r))
    inc = np.arccos(np.dot(unitvec(L),ez))
    return asemi,ecc,inc

def ntimesteps_suggest(trun,tdyn,etol=1e-4):
    '''
    nsteps=   50: rel err Energy = 7.694691377672628e-07:1.5g
    nsteps=  100: rel err Energy = 7.577572034290277e-08:1.5g
    nsteps=  200: rel err Energy = 7.173732970700933e-09:1.5g
    nsteps=  500: rel err Energy = 3.511486626548182e-09:1.5g
    nsteps= 1000: rel err Energy = 2.8272250207867857e-10:1.5g
    nsteps= 2000: rel err Energy = 1.8759125147561206e-11:1.5g
    nsteps= 5000: rel err Energy = 4.945901436348645e-13:1.5g
    '''
    nsteps = 100 * np.ceil(np.abs(trun/tdyn))
    if etol <= 1e-8: nsteps *= 2
    if etol <= 1e-9: nsteps *= 5
    if etol <= 1e-10: nsteps *= 2
    if etol <= 1e-12: nsteps *= 2
    if etol <= 1e-13: nsteps *= 5
    return int(nsteps)