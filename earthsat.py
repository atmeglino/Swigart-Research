# example code.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pylab as pl
import nbody as nb
from constants import *
import duster as d
from scipy.integrate import solve_ivp
import matplotlib as mpl
from matplotlib.collections import LineCollection
from scipy.spatial.transform import Rotation as Ro
import time


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
    # [print(f'{x:1.13e},',end=' ') for x in a]
    # print(' ' )
    
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
    # print('this epoch UTC:', pd.to_datetime(tnowjd,unit='D',origin='julian'))
    # print('Earth frame reference date:', pd.to_datetime(teqxjd,unit='D',origin='julian'))
    bfeq = nb.earth_spin_init(b,tnowjd*day,0,1)
    pspin = PspinEarth
    tstart = tnowjd*day
    
    # here set up the Earth's magnetic field. This is hack…
    nb.earth_magnetic_moment_init(b,tnowjd*day,si=0,ei=1)
    mmo = nb.earth_magnetic_moment(tnowjd*day)
    # print(f'mag north rel tilt: {np.arccos(np.dot(bfeq[2],nb.unitvec(mmo)))/degree:1.5} deg')
    
    # bfeq = nb.bodyframe_equinox(tilt,orbinfo=(b,0,1,(teqxjd-tnowjd)*day)) # use this for mars
    # bfeq = nb.bodyframe_earth(b,0,1,tstart,teqxjd*day,tilt,pspin) 

    # here set up the Earth's magnetic field. not sure how to do this in a good way
    # so let's do it. in nbody.py there is a space to calc earth mag mo, just spagetti code it in
    # nb.magmoearthlalo = (90,0)
    nb.earth_magnetic_moment_init(b,tstart,si=0,ei=1)
    mmo = nb.earth_magnetic_moment(tstart)
    # print(f'mag north rel tilt: {np.arccos(np.dot(bfeq[2],nb.unitvec(mmo)))/degree:1.5} deg')
    
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

    # now set up a tracer particle
    if ndust:
        dustidx = nbodies # tracer in
        rho,rphys = 2.0,10.0*micron
        b[dustidx:].r = rphys
        b[dustidx:].q = 1e-12 # Coulombs
        # b[dustidx:].q = 0
        b[dustidx:].Q = 0.8
        # b[dustidx:].eta = 1
        b[dustidx:].m = 4*np.pi/3*rho*b[dustidx:].r**3
        ex,ey,ez = nb.bodyframe(tstart,teqxjd*day,bfeq,pspin)
    
    # equitorial orbit:
    # nb.orbitEquatorial(b, 1.25, dustidx, ndust, ex, ey)
    
    # polar orbit:
    # nb.orbitPolar(b, 1.25, dustidx, ex, ey, ez)
    
    # polar orbit - max shadowing
    # nb.orbitPolarMaxShading(b, 1.25, dustidx, ez)

    # sun-sync orbit: 
    nb.orbitSunSync(b, dustidx, ez)
    

    # --- all done set up! --- prelim check: orb els of earth...
    a,e,i = nb.orbels(b[1],b[0])
    # print('earth orb els:',a/AU,'AU;',e,i*180/np.pi,'deg')

    a,e,i = nb.orbels(b[2],b[1],ez=bfeq[2])
    # print('moon orb els:',a/Rearth,'AU;',e,i*180/np.pi,'deg')
    
    
    # --- semi-major axis for dust particle ---
    '''
    inc = (90 - 13)*np.pi/180
    dOmdt = 2*np.pi/year
    ecc = 0
    
    a_semi = (1.5*J2tildeEarth*REarth**2*np.sqrt(GNewt*MEarth)*np.cos(inc)/(1-ecc**2)**2/dOmdt)**(2./7.)
    
    print(a_semi/REarth)
    '''
    
    

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
    
    t_eval = tstart + np.linspace(0, year/4, 500)
    
    res = solve_ivp(nb.ode, (t_eval[0], t_eval[-1]), nb.initialState(b), args=(b,), rtol=1e-7, t_eval=t_eval)
    
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
    currentState = nb.initialState(b)
    
    for i in range(53):
        t_start_chunk = tstart + i * week
        t_end_chunk = tstart + (i + 1) * week
        
        t_eval = tstart + np.linspace(t_start_chunk, t_end_chunk, 300)
        
        res = solve_ivp(nb.ode, (t_eval[0], t_eval[-1]), currentState, args=(b,), rtol=1e-6, t_eval=t_eval)
        
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
        
        b[0].x = xs[-1]
        b[0].y = ys[-1]
        b[0].z = zs[-1]
        b[0].vx = res.y[12,-1]
        b[0].vy = res.y[13,-1]
        b[0].vz = res.y[14,-1]
        
        b[1].x = xe[-1]
        b[1].y = ye[-1]
        b[1].z = ze[-1]
        b[1].vx = res.y[15,-1]
        b[1].vy = res.y[16,-1]
        b[1].vz = res.y[17,-1]
        
        b[2].x = xm[-1]
        b[2].y = ym[-1]
        b[2].z = zm[-1]
        b[2].vx = res.y[18,-1]
        b[2].vy = res.y[19,-1]
        b[2].vz = res.y[20,-1]
        
        b[dustidx].x = xp[-1]
        b[dustidx].y = yp[-1]
        b[dustidx].z = zp[-1]
        b[dustidx].vx = res.y[21,-1]
        b[dustidx].vy = res.y[22,-1]
        b[dustidx].vz = res.y[23,-1]
        
        currentState = nb.initialState(b)
    '''
    
    
    '''
    dust_pos = []
    for j in range(ndust):
        base_idx = 9 + 3*j
        xp = res.y[base_idx, :]
        yp = res.y[base_idx + 1, :]
        zp = res.y[base_idx + 2, :]
        dust_pos.append((xp, yp, zp))
        
    print(dust_pos)
    '''
    
    esun = np.column_stack([xs-xe, ys-ye, zs-ze])
    esun_unit = esun / np.linalg.norm(esun, axis=1)[:, None]
    
    xp_rot = np.zeros_like(xp)
    yp_rot = np.zeros_like(yp)
    zp_rot = np.zeros_like(zp)
    
    for i in range(len(xp)):
        particle_pos = np.array([xp[i]-xe[i], yp[i]-ye[i], zp[i]-ze[i]])
        
        current_sun_dir = esun_unit[i]
        target_dir = np.array([1, 0, 0])  # +x axis
        
        rotation = Ro.align_vectors([target_dir], [current_sun_dir])[0] # creates rotation object
        
        rotated_pos = rotation.apply(particle_pos)
        
        xp_rot[i] = rotated_pos[0]
        yp_rot[i] = rotated_pos[1]
        zp_rot[i] = rotated_pos[2]
    
    pl.figure()
    pl.clf()
    pl.plot(xp_rot, yp_rot, ':', color='pink', linewidth=2)
    pl.savefig('corotating.png', dpi=300)
    pl.close()
    
    # Plot below to see sun, earth, moon system
    pl.figure()
    pl.clf()
    #pl.plot(xs,ys, '.k', color='green')
    pl.plot(xe,ye, '.k', color='blue')
    #pl.plot(xm,ym, ':', color='red', linewidth=4)
    pl.plot(xp,yp, ':', color='pink', linewidth=2)
    #pl.plot(res.t,xe,'.k')
    pl.savefig('system.png', dpi=300)
    pl.close()
    
    # Plot below to see moon and particle relative to earth
    pl.figure()
    pl.clf()
    #pl.plot(xm-xe,ym-ye,'.m', color='red')
    pl.plot(xp-xe,yp-ye,'.m', color='pink')
    pl.savefig('reltoearth.png', dpi=300)
    pl.close()
    
    '''
    print(f'Final dust position: {xp[-1]:.10e} {yp[-1]:.10e} {zp[-1]:.10e}')
    print(f'Final earth position: {xe[-1]:.10e} {ye[-1]:.10e} {ze[-1]:.10e}')
    '''
    
    
    '''
    for xi, yi, zi in zip(xp[::10], yp[::10], zp[::10]):
        print(f"{xi:.6e} {yi:.6e} {zi:.6e}")
    '''
    
    # isotropic scattering:
    
    for i in range(0, len(xp), 10):
        b[dustidx].x = xp[i]
        b[dustidx].y = yp[i]
        b[dustidx].z = zp[i]
        b[1].x = xe[i]
        b[1].y = ye[i]
        b[1].z = ze[i]
        b[0].x = xs[i]
        b[0].y = ys[i]
        b[0].z = zs[i]

        if d.illuminated(b[dustidx], b[1], b[0]):
            E_recv = (Lsun * np.pi * b[dustidx].r**2) / (4 * np.pi * nb.reldist(b[0], b[dustidx])**2)
            E_deliv = E_recv * d.skyfraction(Rearth, nb.reldist(b[dustidx], b[1]))
        else:
            E_recv = 0
            E_deliv = 0

        if d.shade(b[dustidx], b[1], b[0]):
            E_removed = E_recv
        else:
            E_removed = 0

        E_change = E_deliv - E_removed
        

        print(f"Step {i}:")
        print(f"  Energy received: {E_recv}")
        print(f"  Energy delivered: {E_deliv}")
        print(f"  Energy removed: {E_removed}")
        print(f"  Net change in energy: {E_change}")
    
    
    '''
    energy_calc_times = np.linspace(0, 1*year, 10)  # 10 points across the year
    energy_window = 2.5 * hour  # 2.5 hours of data for each period
    current_state = nb.initialState(b)
    last_time = 0

    for period_idx, start_time in enumerate(energy_calc_times):
        print(f"\n=== Energy Period {period_idx + 1}/10 ===")
        print(f"Time: {start_time/day:.1f} days from start")
        
        # Short integration for just this 2.5 hour window
        t_eval = tstart + start_time + np.linspace(0, energy_window, 500)
        
        # Update initial state to this time point
        if period_idx > 0:
            gap_time = start_time - last_time
            if gap_time > 0:
                # Quick integration to get to this time
                temp_res = solve_ivp(nb.ode, (tstart + last_time, tstart + start_time), 
                                current_state, args=(b,), rtol=1e-7)
                current_state = temp_res.y[:, -1]
        
        # Integrate for 2.5 hours
        res = solve_ivp(nb.ode, (t_eval[0], t_eval[-1]), current_state, 
                    args=(b,), rtol=1e-7, t_eval=t_eval)
        
        current_state = res.y[:, -1]
        last_time = start_time + energy_window
        
        # Extract positions
        xs, ys, zs = res.y[0,:], res.y[1,:], res.y[2,:]
        xe, ye, ze = res.y[3,:], res.y[4,:], res.y[5,:]
        xp, yp, zp = res.y[9,:], res.y[10,:], res.y[11,:]
        
        # Calculate energies every 10th point
        for i in range(0, len(res.t), 10):
            b[dustidx].x, b[dustidx].y, b[dustidx].z = xp[i], yp[i], zp[i]
            b[1].x, b[1].y, b[1].z = xe[i], ye[i], ze[i]
            b[0].x, b[0].y, b[0].z = xs[i], ys[i], zs[i]

            if d.illuminated(b[dustidx], b[1], b[0]):
                E_recv = (Lsun * np.pi * b[dustidx].r**2) / (4 * np.pi * nb.reldist(b[0], b[dustidx])**2)
                E_deliv = E_recv * d.skyfraction(Rearth, nb.reldist(b[dustidx], b[1]))
            else:
                E_recv = E_deliv = 0

            E_removed = E_recv if d.shade(b[dustidx], b[1], b[0]) else 0
            E_change = E_deliv - E_removed
            
            time_hours = (res.t[i] - t_eval[0]) / hour
            print(f"  t={float(time_hours):.2f}h: E_recv={float(E_recv):.4}, E_deliv={float(E_deliv):.4}, E_removed={float(E_removed):.4}, E_net={float(E_change):.4}, Dist={float(np.sqrt((xp[i]-xe[i])**2+(yp[i]-ye[i])**2+(zp[i]-ze[i])**2)):.4}")
    '''
    
    framedat = []
    n_bodies = len(b)
    positions = res.y[:3*n_bodies, :].reshape(n_bodies, 3, -1)
    velocities = res.y[3*n_bodies:, :].reshape(n_bodies, 3, -1)
    
    # The below set of for loops will likely be very time consuming when we add more dust particles

    for i in range(len(res.t)):
        # Start each row with the time
        frame_row = [n_bodies, res.t[i]]
        
        # Loop through each body
        for j in range(n_bodies):
            # Create the list of values for this body
            body_values = [
                b[j].m, 
                b[j].r,
                positions[j, 0, i], 
                positions[j, 1, i], 
                positions[j, 2, i],
                velocities[j, 0, i], 
                velocities[j, 1, i], 
                velocities[j, 2, i], 
                0
            ]
            
            # Add each value from this body to the frame row
            for value in body_values:
                frame_row.append(value)
    
        # Add this complete frame to the framedat
        framedat.append(frame_row)

    framedat = np.array(framedat)
    complete_data = framedat.flatten()
    # complete_data: [4.0, time0, body_data..., time1, body_data..., ...]
    complete_data.tofile('earthsat.bin')
    
    # fbin = 'earthsat.bin'
    # if fbin: framedat.tofile(fbin)
    
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
