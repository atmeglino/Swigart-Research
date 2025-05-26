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

    # now set up a tracer particle, orbiting earth...
    tridx = nb # tracer in
    r = 15*earth.r
    v = np.sqrt(GNewt*earth.m/r)
    x,y,z = r,0,0
    vx,vy,vz = 0,v,0
    b[tridx] = setbody((0,0,earth.x+x,earth.y+y,earth.z+z,earth.vx+vx,earth.vy+vy,earth.vz+vz))

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
    for i in range(nt):
        steps(b,dt,ntsub)  # integrate!
        tnow = dt*(i+1)
        # plot pos in x-y relative to earth
        # pl.plot((b[1:].x-b[1].x)/earth.r,(b[1:].y-b[1].y)/earth.r,'.k')
        colors = ['red', 'blue', 'gray', 'green']
        pl.scatter((b[1:].x-b[1].x)/earth.r, (b[1:].y-b[1].y)/earth.r, 
           c=colors[1:], s=20)
        print(i,tnow/year,'yr;',pairsep(b[1],b[-1])/earth.r)

    # dump out a plot of results
    out = 'earthsat.pdf'
    
    pl.savefig(out)
    import os
    if os.path.isfile('/uufs/astro.utah.edu/common/home/u0095165/www/tmp.jpg'):
        os.system('convert '+out+' ~/www/tmp.jpg')