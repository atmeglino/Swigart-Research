# example code.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pylab as pl
import nbody as nb
from constants import *
from scipy.integrate import solve_ivp
import matplotlib as mpl
from matplotlib.collections import LineCollection
from scipy.spatial.transform import Rotation as Ro


def printposvel(q):
    s = f'{q.x/AU:7.5f},{q.y/AU:7.5f},{q.z/AU:7.5f} AU'
    s += f', {q.vx/km:7.5f}{q.vy/km:7.5f}{q.vz/km:7.5f} km/s'
    print(s)

def test_integration_error(b,tnow,trun):
    # integration error check
    methods = ['rk45','leap2','symp4','symp6']  # rk45 is best for mag fields, symp6 is best for pure grav!
    nsteps = [50,100,200,500,1000,2000,5000]
    Estart = nb.energy(b)
    for nstep in nsteps:
        btmp = b.copy()
        nb.steps(btmp,tnow,trun,nstep,nb.acc,method=methods[0]) # [0] is default method
        Efinish = nb.energy(btmp)
        print(f'nsteps={nstep:5d}: rel err Energy = {np.abs((Efinish-Estart)/Estart):1.5e}')

def test_local_frame(tnewjd,latdeg,londeg,tnew,b,tnbody,tref,bfref,pspin):
    # awkward units!  tnewjd is JD, other times are sec
    # pick a loc on Earth and a time (sunset); see where sun is, rel to zenith, horizon
    eup,enorth,eeast = nb.localframelalo(latdeg,londeg,tnew,tref,bfref,pspin) # unit spher'l polar
    btmp = b.copy()
    #print('xy',btmp[1].x/AU,btmp.y[1]/AU)
    #print('tnow',(tnew-tnbody)/year)
    #esun = nb.unitvec(nb.posrel(btmp[0],btmp[1]))
    #print('esun',esun)
    nb.steps(btmp,tnbody,(tnew-tnbody),nb.ntimesteps_suggest(tnew-tnbody,year),nb.acc) 
    esun = nb.unitvec(nb.posrel(btmp[0],btmp[1]))
    #print('esun',esun)
    sunperp = esun - np.dot(eup,esun)*eup # sun's dir projected onto local horizon plane
    zenithang = np.arccos(np.dot(eup,esun))
    horizang = np.arctan2(np.dot(sunperp,enorth),np.dot(sunperp,eeast)) # rel to east
    (horizangrel,ewdir) = (horizang,'East') if (np.abs(horizang)<np.pi/2) else (np.pi-horizang,'West')
    nsdir = 'North' if horizangrel > 0 else 'South'
    print(f'sun is {zenithang/degree} degrees from local zenith.')
    print(f'and {horizangrel/degree} degrees {nsdir} of {ewdir}')
        
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
    b = np.zeros(nbodies+ndust,dtype=nb.bodyt).view(np.recarray)
    b[0] = nb.setbody((sun.m,sun.r,sun.x,sun.y,sun.z,sun.vx,sun.vy,sun.vz))
    b[1] = nb.setbody((earth.m,earth.r,earth.x,earth.y,earth.z,earth.vx,earth.vy,earth.vz))
    b[2] = nb.setbody((moon.m,moon.r,moon.x,moon.y,moon.z,moon.vx,moon.vy,moon.vz))
    
    a = [b[0].x,b[0].y,b[0].z,b[1].x,b[1].y,b[1].z,b[0].vx,b[0].vy,b[0].vz,b[1].vx,b[1].vy,b[1].vz]
    #print(a);
    [print(f'{x:1.13e},',end=' ') for x in a]
    print(' ' )
    
    '''
    Old stuff:
    
    tnowjd = sun.jd
    teqxjd = TmarchequinoxEarth2025JD # this must be a march equinox in julian days
    tilt,pspin,porbit = tiltEarth,PspinEarth,PorbitEarth
    print('this epoch UTC:',pd.to_datetime(tnowjd,unit='D',origin='julian'))
    print(teqxjd)
    print('Earth frame reference date:',pd.to_datetime(teqxjd,unit='D',origin='julian'))
    tstart = tnowjd*day
    '''
    
    tnowjd = sun.jd
    teqxjd = TmarchequinoxEarth2025JD # must be march equinox in julian daya                                     
    print('this epoch UTC:', pd.to_datetime(tnowjd,unit='D',origin='julian'))
    print('Earth frame reference date:', pd.to_datetime(teqxjd,unit='D',origin='julian'))
    bfeq = nb.earth_spin_init(b,tnowjd*day,0,1)
    pspin = PspinEarth
    tstart = tnowjd*day
    
    # here set up the Earth's magnetic field. This is hack…
    nb.earth_magnetic_moment_init(b,tnowjd*day,si=0,ei=1)
    mmo = nb.earth_magnetic_moment(tnowjd*day)
    print(f'mag north rel tilt: {np.arccos(np.dot(bfeq[2],nb.unitvec(mmo)))/degree:1.5} deg')
    
    # bfeq = nb.bodyframe_equinox(tilt,orbinfo=(b,0,1,(teqxjd-tnowjd)*day)) # use this for mars
    # bfeq = nb.bodyframe_earth(b,0,1,tstart,teqxjd*day,tilt,pspin) 

    # here set up the Earth's magnetic field. not sure how to do this in a good way
    # so let's do it. in nbody.py there is a space to calc earth mag mo, just spagetti code it in
    # nb.magmoearthlalo = (90,0)
    nb.earth_magnetic_moment_init(b,tstart,si=0,ei=1)
    mmo = nb.earth_magnetic_moment(tstart)
    print(f'mag north rel tilt: {np.arccos(np.dot(bfeq[2],nb.unitvec(mmo)))/degree:1.5} deg')
    
    '''
    mytest = True
    #mytest = False

    if mytest:
        # test how well we conserve energy....
        trun = 1*year
        print('\n-- energy conservation test --')
        #test_integration_error(b,tstart,trun)

        # resolving a reference frame on Earth
        lat,lon = 40.7606,-111.8881 # N,W => < 0, for slc
        tnewjd = 2460834.6236111 # 2025-06-08 02:57:59.999041536 sunset in SLC
        print('\n-- late spring sunset in SLC! --')
        print('UTC, lat, long:',pd.to_datetime(tnewjd,unit='D',origin='julian'),lat,lon,'degrees')
        test_local_frame(tnewjd,lat,lon, tnewjd*day,b,tstart,teqxjd*day,bfeq,pspin)
    '''

    # now set up a tracer particle, orbiting earth around equatorial plane + random motion
    if ndust:
        planet = b[1]
        dustidx = nbodies # tracer in
        rho,rphys = 2.0,10.0*micron
        b[dustidx:].r = rphys
        # b[dustidx:].q = 1e-12 # Coulombs
        b[dustidx:].q = 0
        b[dustidx:].m = 4*np.pi/3*rho*b[dustidx:].r**3
        b[dustidx:].x, b[dustidx:].y, b[dustidx:].z  = planet.x, planet.y, planet.z
        b[dustidx:].vx,b[dustidx:].vy,b[dustidx:].vz = planet.vx,planet.vy,planet.vz
        ex,ey,ez = nb.bodyframe(tstart,teqxjd*day,bfeq,pspin)
        r = 3*planet.r
        v = np.sqrt(GNewt*planet.m/r)
        # phi = np.random.uniform(0,2*np.pi,ndust)[:,np.newaxis] # new axis to spread around 3d coord variables
        phi = np.pi/2 # for polar orbit starting above north pole
        # pos =  r*np.cos(phi)*ex + r*np.sin(phi)*ey # for equitorial orbit
        # vel = -v*np.sin(phi)*ex + v*np.cos(phi)*ey # for equitorial orbit
        pos =  r*np.cos(phi)*ex + r*np.sin(phi)*ez # for polar orbit
        # vel = -v*np.sin(phi)*ex + v*np.cos(phi)*ez # for polar orbit
        earth_vel = np.array([planet.vx, planet.vy, planet.vz]) # for vel in same dir as earth w/ respect to barycenter
        # earth_vel = np.array([planet.vx - b[0].vx, planet.vy - b[0].vy, planet.vz - b[0].vz]) # for vel in same dir as earth w/ respect to sun
        earth_vel_direction = earth_vel / np.linalg.norm(earth_vel)
        vel = v * earth_vel_direction
        b[dustidx:].x += pos[...,0]
        b[dustidx:].y += pos[...,1]
        b[dustidx:].z += pos[...,2]
        # b[dustidx:].vx += vel[:,0]
        # b[dustidx:].vy += vel[:,1]
        # b[dustidx:].vz += vel[:,2]
        b[dustidx:].vx += vel[...,0] # dust velocity direction matches earth's relative to sun
        b[dustidx:].vy += vel[...,1]
        b[dustidx:].vz += vel[...,2]
    
    # sun-sync orbit:
    
    rc = 1.25*planet.r # starting position for sun-sync orbit
    vc = np.sqrt(GNewt*planet.m/rc) # circular velocity
    esun = nb.unitvec(nb.posrel(b[0],b[1]))  # unit vec form
    # print(esun)
    # exit()
    ptilt = 13*degree
    r = Ro.from_quat([np.sin(ptilt/2)*esun[0], np.sin(ptilt/2)*esun[1], np.sin(ptilt/2)*esun[2], np.cos(ptilt/2)])
    epolar = r.apply(ez)
    print(r.as_matrix())
    evel = -nb.unitvec(np.cross(np.cross(epolar,esun),epolar))  # cross prods to get vel pointed in good direction
    # do a rotation about esun, 13 degrees. 23 degrees would align earth spim w/z axis
    # epolar = qrotate(epolar,quat(np.cos(ptilt/2),np.sin(ptilt/2)*esun)) # ptilt=0 is a polar orbit. 90 deg is equatorial 
    # rotv = esun*np.sin(ptilt/2)
    # epolar = Ro.from_quat([rotv[0],rotv[1],rotv[2],np.cos(ptilt/2)]) # might need to use -ptilt?
    # print(epolar.as_matrix())
    # exit()
    # evel = -nb.unitvec(np.cross(np.cross(epolar,esun),epolar))  # cross prods to get vel pointed in good direction
    '''
    which part of epolar should be used in above cross product?
    '''
    # print(evel)
    # exit()
    '''
    which part of epolar/evel to use in below lines?
    '''
    b[dustidx:].x = planet.x + rc * epolar[0]
    b[dustidx:].y = planet.y + rc * epolar[1]
    b[dustidx:].z = planet.z + rc * epolar[2]
    b[dustidx:].vx = planet.vx + vc * evel[0]
    b[dustidx:].vy = planet.vy + vc * evel[1]
    b[dustidx:].vz = planet.vz + vc * evel[2]

    # --- all done set up! --- prelim check: orb els of earth...
    a,e,i = nb.orbels(b[1],b[0])
    print('earth orb els:',a/AU,'AU;',e,i*180/np.pi,'deg')

    a,e,i = nb.orbels(b[2],b[1],ez=bfeq[2])
    print('moon orb els:',a/Rearth,'AU;',e,i*180/np.pi,'deg')

    '''
    # get ready to integrate, define num of timesteps, substeps .....
    P = 2*np.pi*r/v # dynamical time, or use orbital period...
    trun, nt = 27*day, 250
    dt = trun / nt
    ntsub = nb.ntimesteps_suggest(dt,day) 

    # --- main time loop --- #
    
    tnow = tnowjd * day

    framedat = [] # space to save stuff to plot later...
    for i in range(nt):
        nb.steps(b,tnow,dt,ntsub,nb.acc)  # integrate!
        tnow += dt
        print(f'time: {tnow/day} JD; dist check: {nb.pairsep(b[-1],b[1])/Rearth} R_Earth')
        # below is time, sun_x,sun_y,sun_z,earth_x,earth_y,earth_z,...
        framedat.append([tnow]+[xyz for p in b for xyz in (p.x, p.y, p.z)]) 
        #if i == 2: break
    framedat = np.array(framedat)
    '''
    
    
    # --- done!!! --- #
    t_eval = tstart + np.linspace(0, 7*day, 500)
    
    res = solve_ivp(nb.ode, (t_eval[0], t_eval[-1]), nb.initialState(b), args=(b,), rtol=1e-13, t_eval=t_eval)
    
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

    '''
    print(f'Initial dust position: {xp[0]:.10e} {yp[0]:.10e} {zp[0]:.10e}')
    print(f'Initial earth position: {xe[0]:.10e} {ye[0]:.10e} {ze[0]:.10e}')
    ax_j2, ay_j2, az_j2 = nb.accJ2(b, tstart+0.5*year)
    ax_grav, ay_grav, az_grav = nb.accGrav(b, tstart+0.5*year)
    ax_mag, ay_mag, az_mag = nb.accMag(b, tstart+0.5*year)
    ax_rad, ay_rad, az_rad = nb.accRad(b)
    print(f'J2 acc: {ax_j2[dustidx]:.10e}, {ay_j2[dustidx]:.10e}, {az_j2[dustidx]:.10e}')
    print(f'Grav acc: {ax_grav[dustidx]:.10e}, {ay_grav[dustidx]:.10e}, {az_grav[dustidx]:.10e}')
    print(f'Mag acc: {ax_mag[dustidx]:.10e}, {ay_mag[dustidx]:.10e}, {az_mag[dustidx]:.10e}')
    print(f'Rad acc: {ax_rad[dustidx]:.10e}, {ay_rad[dustidx]:.10e}, {az_rad[dustidx]:.10e}')
    '''
    
    print(f'Final dust position: {xp[-1]:.10e} {yp[-1]:.10e} {zp[-1]:.10e}')
    print(f'Final earth position: {xe[-1]:.10e} {ye[-1]:.10e} {ze[-1]:.10e}')
    
    exit()
    
    
    pl.clf()
    # Plot below to see sun, earth, moon system
    #pl.plot(xs,ys, '.k', color='green')
    #pl.plot(xe,ye, '.k', color='blue')
    #pl.plot(xm,ym, ':', color='red', linewidth=4)
    #pl.plot(xp,yp, ':', color='pink', linewidth=2)
    #pl.plot(res.t,xe,'.k')
    
    # Plot below to see moon and particle relative to earth (?)
    #pl.plot(xm-xe,ym-ye,'.m', color='red')
    pl.plot(xp-xe,yp-ye,'.m', color='pink')
    pl.show()
    pl.savefig('fig.png', dpi=300)
    pl.close()
    
    '''
    for xi, yi, zi in zip(xp[::10], yp[::10], zp[::10]):
        print(f"{xi:.6e} {yi:.6e} {zi:.6e}")
    '''
    
    framedat = []
    # framedat.append([float(len(b)),tnow]+[q for p in b for q in (p.m,p.r,p.x,p.y,p.z,p.vx,p.vy,p.vz,p.L)])
    for i in range(len(xs)):
        # make a list of lists
        # each list is for each body, with time, r, pos, vel, L (set L=0 for everyone)
        framedat = np.append(framedat, [[res.t[i]] + [p.m, p.r, p.x[i], p.y[i], p.z[i], p.vx[i], p.vy[i], p.vz[i], 0] for p in b])
    framedat = np.array(framedat)
    fbin = 'earthsat.bin'
    if fbin: framedat.tofile(fbin)
    
    exit()
    
    # dump out a plot of results
    out = ''
    out = 'earthsat.pdf'

    if out:
        # set up for plotting, just the dust in earth frame
        pl.gca().set_aspect('equal')
        pl.xlabel('x [Earth radii]')
        pl.ylabel('z [Earth radii]')
        x = framedat[:,3*nbodies+1::3]-framedat[:,[4]]
        y = framedat[:,3*nbodies+2::3]-framedat[:,[5]]
        z = framedat[:,3*nbodies+3::3]-framedat[:,[6]]
        #x = framedat[:,1+3:3*nbodies:3]-framedat[:,[4]]
        #y = framedat[:,2+3:3*nbodies:3]-framedat[:,[5]]
        pl.plot(x/Rearth,z/Rearth,'-')
        pl.savefig(out)
        import os
        if os.path.isfile('/uufs/astro.utah.edu/common/home/u0095165/www/tmp.jpg'):
            os.system('cp '+out+' ~/www/tmp.pdf')
            os.system('convert '+out+' ~/www/tmp.jpg')
