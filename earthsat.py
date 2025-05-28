# example code.
import numpy as np
import pandas as pd
import pylab as pl
from constants import *
from nbody import *


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

    nb = 3 # sun-earth-moon only
    sun = dfsolsys.iloc[0]
    earth = dfsolsys.iloc[3]
    moon = dfsolsys.iloc[4]
    # ugh, best to look at the csv file...sorry
    ndust = 1

    # set up planet info in an array "b" w/elemets of type bodyt
    # bodyt definesmembers m,r,x,y,z,vx,vy,vz and more! *** units are cgs!!!!! ***
    b = np.zeros(nb+ndust,dtype=bodyt).view(np.recarray)
    b[0] = setbody((sun.m,sun.r,sun.x,sun.y,sun.z,sun.vx,sun.vy,sun.vz))
    b[1] = setbody((earth.m,earth.r,earth.x,earth.y,earth.z,earth.vx,earth.vy,earth.vz))
    b[2] = setbody((moon.m,moon.r,moon.x,moon.y,moon.z,moon.vx,moon.vy,moon.vz))
    tnowjd = sun.jd

    #exp,eyp,ezp = planetframe(b,planetindex=1,starindex=0
    Pearth, tilt, jdsummer = PorbitEarth, tiltEarth, TjunesolsticeEarth2025JD
    while jdsummer < sun.jd: jdsummer += Pearth/day # check this or use fmod
    daysleft = np.fmod(jdsummer - tnowjd, Pearth/day) # until next summer
    bsummer = b.copy()
    steps(bsummer,tnowjd*day,daysleft*day,1000)
    exp,eyp,ezp = zenithframe(bsummer[1],bsummer[0]) # orbit frame
    exp,eyp,ezp = exp*np.cos(tilt)-ezp*np.sin(tilt),eyp,exp*np.sin(tilt)+ezp*np.cos(tilt) # earth frame, z = North pole
    tequinoxjd, Pearth,Pspinearth = TmarchequinoxEarth2026JD, PorbitEarth, PspinEarth

    print('tnow:',pd.to_datetime(tnowjd,unit='D',origin='julian'))
    print('tnow:',pd.to_datetime(jdsummer,unit='D',origin='julian'))
    print('teqx:',pd.to_datetime(tequinoxjd,unit='D',origin='julian'))

    print(zenithframe(b[0],b[1]),'bf at tnow')
    bfeq = bodyframe_equinox(b[1],b[0],tnowjd*day,tequinoxjd*day,Pearth,0) # units cgs, radians
    print(bfeq,'bf at equinox using tnow data')
    
    (ebx,eby,ebz) = bodyframe(tnowjd*day,tequinoxjd*day,bfeq,Pspinearth)
    lat,lon = 40.7606,111.8881 # N,W, for slc
    theta,phi = (90-lat)*degree,(-lon)*degree
    ct,st,cp,sp = np.cos(theta),np.sin(theta),np.cos(phi),np.sin(phi)
    eslczenith = vec3d(st*cp*ebx,st*sp*eby,cp*ebz)
    #print('sun off zeninth slc',np.arccos(
    #May 28, 2025, at 8:49 PM mountain -> UTC 2025-05-29 01:49
    tnowjd = 2460824.5756944
    print('teqx:',pd.to_datetime(tnowjd,unit='D',origin='julian'))
    if 0:
        print('comapring with summer version....')
        b = bsummer
        tnowjd = jdsummer
        print(zenithframe(b[0],b[1]),'bf at summer')
        bfeq = bodyframe_equinox(b[1],b[0],jdsummer*day,tequinoxjd*day,Pearth,0) # units cgs, radians
        print(bfeq,'bf at equinox using summer info')

    quit()
    bfeq = bodyframe_equinox(b[1],b[0],tnowjd*day,tequinoxjd*day,tperiod,tilt) # units cgs, radians
    print(bfeq)
    quit()
    #hour = 

    # check:
    esun = unitvec(posrel(b[0],b[1]))
    lat,lon = 40.7606,111.8881 # N,W, for slc
    theta,phi = (90-lat)*degree,(-lon)*degree
    # zenith dir in SLC on start day
    eslc = np.sin(theta)*np.cos(phi)*exp + np.sin(theta)*np.sin(phi)*eyp + np.cos(theta)*ezp
    # latitude ot the sun in Earth frame:
    sunlat = np.pi/2-np.arccos(np.dot(esun,ezp))
    print('summer solstice:',pd.to_datetime(jdsummer,unit='D',origin='julian'))
    print('starting time:',pd.to_datetime(tnowjd,unit='D',origin='julian'))
    print('sun latitude this date:',sunlat/degree)

    
    
    '''
    es = unitvec(np.array([np.dot(ex,ezplanet),np.dot(ey,ezplanet),0])) # spin projected to orbital plane
    phisummer = np.arctan2(es[1],es[0]) # y,x where x is dir to sun, so zero here summer
    nextsummer = tnowjd + Pearth * phisummer /2/np.pi / day
    print('next summer (est):',pd.to_datetime(nextsummer,unit='D',origin='julian'))
    trefEarth = jdsummer*day
    '''
    # check: guess sunrise in SLC at start date
    
    # now set up a tracer particle, orbiting earth...
    tridx = nb # tracer in
    r = 15*earth.r
    v = np.sqrt(GNewt*earth.m/r)
    x,y,z = r,0,0
    vx,vy,vz = 0,v,0
    b[tridx] = setbody((0,0,earth.x+x,earth.y+y,earth.z+z,earth.vx+vx,earth.vy+vy,earth.vz+vz))
    # b[tridx].q = 1e3
    
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