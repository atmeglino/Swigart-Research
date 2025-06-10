# example code.
import numpy as np
import pandas as pd
import pylab as pl
from OLDconstants import *
from OLDnbody import *
from scipy.integrate import solve_ivp

def printposvel(q):
    s = f'{q.x/AU:7.5f},{q.y/AU:7.5f},{q.z/AU:7.5f} AU'
    s += f', {q.vx/km:7.5f}{q.vy/km:7.5f}{q.vz/km:7.5f} km/s'
    print(s)

# ---- Main ----- #
if __name__ == '__main__':

    verbose = True
    plotmode = 'xy'
    
    #parser = argparse.ArgumentParser(description='heating versus cooling circular orbits')
    #args = parser.parse_args()
    #seed = args.seed

    # --- get physical, orbital data from csv file with JPL horizons data --- #
    fil = 'solarsystem.csv'
    dfsolsys = pd.read_csv(fil)

    nbodies = 3 # sun-earth-moon only
    sun = dfsolsys.iloc[0]
    earth = dfsolsys.iloc[3]
    moon = dfsolsys.iloc[4]
    # ugh, best to look at the csv file...sorry
    ndust = 1

    # set up planet info in an array "b" w/elemets of type bodyt
    # bodyt definesmembers m,r,x,y,z,vx,vy,vz and more! *** units are cgs!!!!! ***
    b = np.zeros(nbodies+ndust,dtype=bodyt).view(np.recarray)
    b[0] = setbody((sun.m,sun.r,sun.x,sun.y,sun.z,sun.vx,sun.vy,sun.vz))
    b[1] = setbody((earth.m,earth.r,earth.x,earth.y,earth.z,earth.vx,earth.vy,earth.vz))
    b[2] = setbody((moon.m,moon.r,moon.x,moon.y,moon.z,moon.vx,moon.vy,moon.vz))
    
    '''
    tnowjd = sun.jd
    teqxjd = TmarchequinoxEarth2026JD
    tilt = tiltEarth
    pspin = PspinEarth
    porbit = PorbitEarth

    # bfeq = bodyframe_equinox(tilt,orbinfo=(b,0,1,(teqxjd-tnowjd)*day))
    bfeq = bodyframe_earth(b,0,1,tnowjd*day,teqxjd*day,tilt,pspin)

    # move to a new time, sunset in SLC valley
    tnewjd = 2460834.6236111
    tnewjd = 2461746.4996528 # 2026?
    
    #tnewjd = teqxjd + 6*hour/day
    print('tnew:',pd.to_datetime(tnewjd,unit='D',origin='julian'))
    btmp = b.copy()
    steps(btmp,tnowjd*day,(tnewjd-tnowjd)*day,100)  # integrate!
    esun = unitvec(posrel(btmp[0],btmp[1]))

    lat,lon = 40.7606,-111.8881 # N,W => < 0, for slc
    eup,enorth,eeast = localframelalo(lat,lon,tnewjd*day,teqxjd*day,bfeq,pspin)
    #lat,lon = 0,-45 # N,W => < 0, for slc
    # lat,lon = 0.,0 # N,W => < 0, for slc

    print('degrees off horizon:',90-np.arccos(np.dot(eup,esun))/degree)
    esunperp = esun - np.dot(eup,esun)*eup
    print('degrees off east:',np.arctan2(np.dot(esunperp,enorth),np.dot(esunperp,eeast))/degree)
    '''
    
    '''
    # now set up a tracer particle, orbiting earth...
    tridx = nb # tracer in
    r = 15*earth.r
    v = np.sqrt(GNewt*earth.m/r)
    x,y,z = r,0,0
    vx,vy,vz = 0,v,0
    b[tridx] = setbody((0,0,earth.x+x,earth.y+y,earth.z+z,earth.vx+vx,earth.vy+vy,earth.vz+vz))
    b[tridx].q = 1e3
    '''
    if ndust:
        planet = b[1]
        dustidx = nbodies # tracer in
        rho,rphys = 2.0,1.0*micron
        b[dustidx:].r = rphys
        b[dustidx:].q = 1e-12 # Coulombs
        b[dustidx:].m = 4*np.pi/3*rho*b[dustidx:].r**3
        b[dustidx:].x, b[dustidx:].y, b[dustidx:].z  = planet.x, planet.y, planet.z
        b[dustidx:].vx,b[dustidx:].vy,b[dustidx:].vz = planet.vx,planet.vy,planet.vz
        # ex,ey,ez = bodyframe(tnowjd*day,teqxjd*day,bfeq,pspin)
        r = 15*planet.r
        v = np.sqrt(GNewt*planet.m/r)
        phi = np.random.uniform(0,2*np.pi,ndust)[:,np.newaxis] # new axis to spread around 3d coord variables
        # pos =  r*np.cos(phi)*ex + r*np.sin(phi)*ey
        # vel = -v*np.sin(phi)*ex + v*np.cos(phi)*ey
        # b[dustidx:].x  += pos[:,0]; b[dustidx:].y  += pos[:,1]; b[dustidx:].z  += pos[:,2] 
        # b[dustidx:].vx += vel[:,0]; b[dustidx:].vy += vel[:,1]; b[dustidx:].vz += vel[:,2] 
    
    t_eval = np.linspace(0,0.95*year,500)
    
    res = solve_ivp(ode, (t_eval[0], t_eval[-1]), initialState(b), args=(b,), rtol=1e-4, t_eval=t_eval)
    
    xs = res.y[0,:]
    ys = res.y[1,:]
    zs = res.y[2,:]
    xe = res.y[3,:]
    ye = res.y[4,:]
    ze = res.y[5,:]
    xm = res.y[6,:]
    ym = res.y[7,:]
    zm = res.y[8,:]
    xp = res.y[9,:]
    yp = res.y[10,:]
    zp = res.y[11,:]
    
    pl.clf()
    #pl.plot(xs,ys, '.k', color='green')
    pl.plot(xe,ye, '.k', color='blue')
    pl.plot(xm,ym, ':', color='red', linewidth=4)
    pl.plot(xp,yp, ':', color='pink', linewidth=2)
    # pl.plot(res.t,xe,'.k')
    pl.show()
    
    exit()
    
    # --- all done set up! --- prelim check: orb els of earth...
    a,e,i = orbels(b[1],b[0])
    print('earth orb els:',a/AU,'AU;',e,i*180/np.pi,'deg')

    # set up for plotting
    pl.gca().set_aspect('equal')
    pl.xlabel('x [Earth radii]')
    pl.xlabel('t [Earth radii]')

    # get ready to integrate, define num of timesteps, substeps .....
    P = 2*np.pi*r/v # dynamical time, or use orbital period...
    trun, nt = 27*day, 50
    dt = trun / nt
    ntsub = int(40 * np.ceil(dt / P))

    # --- main time loop --- #
    tnow = tnowjd * day
    for i in range(nt):
        steps(b,tnow,dt,ntsub)  # integrate!
        tnow += dt
        tnow = dt*(i+1)
        # plot pos in x-y relative to earth
        pl.plot((b[1:].x-b[1].x)/earth.r,(b[1:].y-b[1].y)/earth.r,'.k')
        print(i,tnow/year,'yr;',pairsep(b[1],b[-1])/earth.r)

    # dump out a plot of results
    out = 'earthsat.pdf'
    
    pl.savefig(out)
    import os
    if os.path.isfile('/uufs/astro.utah.edu/common/home/u0095165/www/tmp.jpg'):
        os.system('convert '+out+' ~/www/tmp.jpg')