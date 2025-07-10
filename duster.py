import numpy as np
import pandas as pd
from scipy.stats import qmc
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter as smoo
import time
import pylab as pl
import os
import myfig
import attnmie
from constants import *
from myparms import *
import argparse
import sys
import nbody as nb
import scatter as sc

deg = np.pi/180 # deg in rad so 1*deg is a degre in radians

# conveniences, maybe, for dtype in nbody.py
def pos(p): return np.array([p.x,p.y,p.z])
def vel(p): return np.array([p.vx,p.vy,p.vz])
def posvelrel(ba,bb):
    return np.array([ba.x-bb.x,ba.y-bb.y,ba.z-bb.z]),np.array([ba.vx-bb.vx,ba.vy-bb.vy,ba.vz-bb.vz])
def posrel(ba,bb): return np.array([ba.x-bb.x,ba.y-bb.y,ba.z-bb.z])
def velrel(ba,bb): return np.array([ba.vx-bb.vx,ba.vy-bb.vy,ba.vz-bb.vz])

def unitvec(a): return a/np.linalg.norm(a)

def zenithframe(planet,star): # frame with z out of orbital plane, x is planet-sun dir
    ex = unitvec(posrel(star,planet))
    ez = unitvec(np.cross(velrel(planet,star),ex))
    ey = unitvec(np.cross(ez,ex)) # y is opposite planet motion...
    return ex,ey,ez

def orbels(sat,ctrlmass,ez=np.array([0.,0.,1.])): # a,e,i but i is rel to bg coordinates not anything sensible
    r,v = posvelrel(sat,ctrlmass)
    m = ctrlmass.m
    E = 0.5*np.sum(v**2)-GNewt*m/np.linalg.norm(r)
    L = np.cross(r,v)
    asemi = -GNewt*m/(2*E)
    ecc = np.linalg.norm(np.cross(v,L)/(GNewt*m)-unitvec(r))
    inc = np.arccos(np.dot(unitvec(L),ez))
    return asemi,ecc,inc

# gen quasirandom numbers on celestial sphere
def qrng_sky(n):
    sampler = qmc.Halton(d=2,scramble=False)
    sample = sampler.random(n)  # Generates n pts hopefully not 2^ns
    phi = 2 * np.pi*sample[:, 0]  # Longitude in [0, 2*pi]
    theta = np.arccos(2*sample[:, 1]-1)  # Latitude in [0, pi]
    return theta,phi

# given distance rel to mars center and its radius of Mars, what frac of total sky is Mars covering?
def skyfraction(Rad,dist):
    return 0.5*(1-np.sqrt(1-Rad**2/dist**2))

# given position rel to mars center, find angular size of Mars on the sky:
def skyradius(Rad,dist):
    return np.arcsin(Rad/dist)

def illumfraction(dust,planet,star):
    # returns scattering angle of light from star, scattered by dust toward center of planet, and
    # the fraction of dust's sky is subtended by planet. negative if planet 
    erad = unitvec(posrel(dust,star))
    eplanet = unitvec(posrel(planet,dust))
    costheta = np.dot(erad,eplanet)
    theta = np.arccos(costheta) # zero => no deflection
    distplanet = nb.pairsep(planet,dust)
    skyfrac = skyfraction(planet.r,distplanet)
    planetang = skyradius(planet.r,distplanet) # radial size of planet in dust's sky
    starang = np.pi-theta
    dust_illum = np.float64(starang > planetang)
    return theta,skyfrac,dust_illum

def illuminated(dust,planet,star):
    # returns 1 if vector ray (from dust) not shaded by planet
    estar = unitvec(posrel(star,dust))
    eplanet = unitvec(posrel(planet,dust))
    starang = np.arccos(np.dot(estar,eplanet))
    planetang = skyradius(planet.r,nb.pairsep(planet,dust)) # radial size of planet in dust's sky
    return starang > planetang

def scatters_to_planet(ray,dust,planet):
    # if ray from dust hits planet returns 1 else 0
    eplanet = unitvec(posrel(planet,dust))
    costheta = np.dot(ray,eplanet)
    distplanet = nb.pairsep(planet,dust)
    cosprad = np.cos(skyradius(planet.r,distplanet)) # cos of radial size of planet in dust's sky
    return costheta > cosprad

def shade(dust,planet,star):
    # fraction of stellar photons deflected that would have otherwise hit the planet
    erad = -unitvec(posrel(star,dust))
    eplanet = unitvec(posrel(planet,dust))
    costheta = np.dot(erad,eplanet)
    distplanet = nb.pairsep(planet,dust)
    cosbeta = np.cos(skyradius(planet.r,distplanet)) # cos of radial size of planet in dust's sky
    # worry about penumbra etc?
    # check this!
    # theta = ang. of planet relative to radiation flow. theta=0 means we're shading!
    # beta = ang size of the planet. if theta < beta, we are shading some part of planet
    # exit('check this! in shade')
    return costheta > cosbeta

def frame_out_check(i,nf,nt):
    if nf >= nt: return True
    every = nt/nf
    return True if int(i+1) % int(every) == 0 else False

frame_dat = []

def frame_start(L):
    global frame_dat
    frame_dat = []
    fram_parms = {'L':L}
    pl.clf()
    pl.style.use('dark_background')
    pl.gca().set_aspect('equal')
    pl.xlim(-L/2,L/2); pl.ylim(-L/2,L/2)
    cir = pl.Circle((0, 0), 1.0, color='#cc8888',fill=True)
    pl.gca().add_patch(cir)
    return

def frame_add(dust,planet,star,size=10,interp=False):
    global frame_dat
    dustuse = dust if dust.shape else [dust]
    for id,dustthis in enumerate(dustuse):
        illum = illuminated(dustthis,planet,star)
        ex,ey,ez = zenithframe(planet,star)
        dr = posrel(dustthis,planet)
        dx,dy,dz = np.dot(dr,ex)/planet.r,np.dot(dr,ey)/planet.r,np.dot(dr,ez)/planet.r
        if interp:
            frame_dat.append([id,dx,dy,dz])
        if illum:
            pl.scatter(dy,dz,s=size,c='#ffffaa',zorder=99)

def do_interp(x,y,z):
    r,p = np.sqrt(x**2+y**2+z**2),np.arctan2(y,x)
    h = np.arccos(z/r)
    idx = np.argsort(p)
    r,h,p,n = r[idx],h[idx],p[idx],3
    r = np.concatenate((r[-n:],r,r[:n])) # wrap
    h = np.concatenate((h[-n:],h,h[:n]))
    p = np.concatenate((p[-n:]-2*np.pi,p,p[:n]+2*np.pi))
    kind = 'linear'
    fr = interp1d(p, r, kind=kind)
    fh = interp1d(p, h, kind=kind)
    fp = interp1d(p, p, kind=kind)
    ta = np.linspace(-np.pi,np.pi,500) # ta as an array in phi
    ra,ha,pa = fr(ta),fh(ta),fp(ta)
    ra = smoo(ra, window_length=len(ra)//2, polyorder=2, mode='wrap')
    ha = smoo(ha, window_length=len(ha)//2, polyorder=2, mode='wrap')
    xa,ya,za = ra*np.cos(pa)*np.sin(ha),ra*np.sin(pa)*np.sin(ha),ra*np.cos(ha)
    msk = ~((xa < 0) & (np.sqrt(ya**2+za**2)<1.0)) 
    xa,ya,za = xa[msk],ya[msk],za[msk]
    return xa,ya,za
    
def frame_end(fframebase,fctr, interp=False):
    global frame_dat
    pl.xlabel('y (planetary radii)')
    pl.ylabel('z (planetary radii)')
    if interp:
        id,x,y,z = np.array(frame_dat).T
        ids = np.unique(id)
        for idthis in ids:
            mski = id == idthis
            xa,ya,za = x[mski],y[mski],z[mski]
            xa,ya,za = do_interp(xa,ya,za)
            pl.scatter(ya,za,s=5,c='#ffaaff',zorder=99)
        frame_dat = []
    fout = fframebase+f'{fctr:03d}.png'
    pl.savefig(fout)
    return fout

# ---- Main -----
if __name__ == '__main__':

    verbose = True
    plotmodes = ['orbit','illum','movie']
    plotmode = plotmodes[0]
    
    #parser = argparse.ArgumentParser(description='heating versus cooling circular orbits')
    #args = parser.parse_args()
    #seed = args.seed

    fil = 'solarsystem.csv'
    dfsolsys = pd.read_csv(fil)

    ndust = 1
    nb = 2 # sun-mars only
    epoch = dfsolsys.loc[0,'jd']
    sun = dfsolsys.iloc[0]
    mars = dfsolsys.iloc[5]
    phobos = dfsolsys.iloc[6]
    deimos = dfsolsys.iloc[7]

    b = np.zeros(nb+ndust,dtype=nb.bodyt).view(np.recarray)
    b[0] = (sun.m,sun.r,sun.x,sun.y,sun.z,sun.vx,sun.vy,sun.vz,0,0,0,0,0)
    b[1] = (mars.m,mars.r,mars.x,mars.y,mars.z,mars.vx,mars.vy,mars.vz,0,0,0,0,0)
    star,planet = b[0],b[1]

    tilt = 25.19*deg

    a,e,i = orbels(planet,star)
    print('Mars-Sun orbels:',a/AU,e,i*180/np.pi)
    Pplanet = 2*np.pi*np.sqrt(a**3/GNewt/star.m)

    # get spin of planet start with solstice date
    jdsummer = 2460824.44993165 # my estimate based on moon inclinations
    while jdsummer < sun.jd: jdsummer += Pplanet/day # check this or use fmod
    daysleft = np.fmod(jdsummer - sun.jd, Pplanet/day) # until next summer
    phi = daysleft * day / Pplanet * 2 * np.pi # in zenith frame (e_y = -e_v), phi is positive
    ex,ey,ez = zenithframe(planet,star) # orbit frame
    ezplanet = np.cos(phi)*np.sin(tilt)*ex + np.sin(phi)*np.sin(tilt)*ey + np.cos(tilt)*ez
    # recheck:
    ex,ey,ez = zenithframe(planet,star) # orbit frame
    es = unitvec(np.array([np.dot(ex,ezplanet),np.dot(ey,ezplanet),0])) # spin projected to orbital plane
    phisummer = np.arctan2(es[1],es[0]) # y,x where x is dir to sun, so zero here summer
    nextsummer = epoch + Pplanet * phisummer /2/np.pi / day
    print('JD summer on mars:',nextsummer)
    print(pd.to_datetime(nextsummer,unit='D',origin='julian'))

    aphobos,ephobos,iphobos = orbels(phobos,mars,ez=ezplanet)
    adeimos,edeimos,ideimos = orbels(deimos,mars,ez=ezplanet)
    print("Phobos orbels (R_mars, deg):",aphobos/mars.r,ephobos,iphobos/deg)
    print("Deimos orbels (R_mars, deg):",adeimos/mars.r,edeimos,ideimos/deg)
    
    # my dust prototype at phobos
    dusti = nb
    b[dusti] = (0,0,phobos.x,phobos.y,phobos.z,phobos.vx,phobos.vy,phobos.vz,0,0,0,0,0)
    if ndust > 1: b[-1] = (0,0,deimos.x,deimos.y,deimos.z,deimos.vx,deimos.vy,deimos.vz,0,0,0,0,0)
    dust = b[dusti:]

    a,e,i = orbels(dust[0],planet,ez=ezplanet)
    print('dust-Mars orbels:',a/planet.r,e,i*180/np.pi)
    Pdust = 2*np.pi*np.sqrt(a**3/GNewt/planet.m)

    # scattering 
    nolivine = 1.67 - 1e-3j; rhoolivine = 2.7 # Fabian, et al. Astron. Astrophys. 378, 228 (2001).
    rhop = rhoolivine
    rp = 1*micron
    rp = 20*micron
    #rp = 1.0*km 
    mp = 4*np.pi/3*rhop*rp**3
    if rp < 20*micron:
        mu = np.linspace(-1,1,5000)
        qext,qsca,g,pf = attnmie.do_mie(rp,nolivine,mu_pf=mu)
        qabs = qext-qsca
        pf = interp1d(np.arccos(mu),pf)
    else:
        pf = False
        if rp == 20*micron:
            qext,qsca,qabs,g = 1.33,0,0.3,0.7 # opaque
        else:
            qext,qsca,qabs,g = 0,0,0,0 # opaque

    nb.acc(b) # change this to nb.accTotal?
    print(np.array([b.ax,b.ay,b.az]).T)
    quit()
            
    dust.m = mp; dust.r = rp
    dust.Q = qext-g*qsca; dust.eta = 1/3
    pfd03 = sc.phase_function_D03
    gpf,nupf = g,0.1 # nu is 0 for Rayliegh, 1 for HG
    if pf: # test phase fun stuff
        h,p = qrng_sky(100000)
        f = pf(h)
        tot = np.mean(f)*4*np.pi
        print('mie:',tot)
        f = pfd03(h,gpf,nupf)
        tot = np.mean(f)*4*np.pi
        print('D03',tot)
        hh = np.arccos(mu)
        pl.plot(hh,pf(hh),'-k')
        pl.plot(hh,pfd03(hh,gpf,nupf),'-m')
        pl.savefig('tmp.pdf')
        os.system('convert tmp.pdf ~/www/tmp.jpg')

    print("q's:",qext,qsca,qabs)
    
        
        
    #for i in range(1,ndust): b[ti+i]=b[ti].copy()
    #b[ti:].vx += np.random.normal(0,0.1*km,b.shape[0]-nb)
    #b[ti:].vy += np.random.normal(0,0.1*km,b.shape[0]-nb)
    #b[ti:].vz += np.random.normal(0,0.1*km,b.shape[0]-nb)

    # dust dist, speed in Martian frame
    r = nb.pairsep(dust[0],planet)
    v = relspeed(dust[0],planet)
    a,_,_ = orbels(dust[0],planet)
    P = 2*np.pi*np.sqrt(a**3/GNewt/planet.m)
    print('orbital period dust (hours):',P/hour)
    Pest = 2*np.pi*r/v
    print('orb period dust est (hours):',Pest/hour)


    trun, nsteps = 1*Pplanet, 1000
    cutfac = 1
    trun, nsteps = trun/cutfac, nsteps//cutfac
    tsub = trun / nsteps
    ntsub = int(40 * np.ceil(tsub / P))

    if 'movie' in plotmode:
        fframelis = []
        pid = str(os.getpid()) 
        fframebase = 'duster'+pid
        fctr = 0
        nframes = 60//cutfac
            
    xlis,y1lis,y2lis = [],[],[]
    bsave = b.copy()
    L = 2.2*nb.pairsep(bsave[-1],bsave[1])/planet.r
    interp=True
    if 'movie' in plotmode:
        frame_start(L)
    tnow = 0.0
    for i in range(nsteps):
        nb.steps(b,tnow,tsub,ntsub,nb.acc)
        tnow = tsub*(i+1)
        ad,ed,id = orbels(dust[0],planet,ez=ezplanet)
        ap,ep,ip = orbels(planet,star)
        print(f'orbels Mars: {ap/AU:6.4f} au, {ep:6.4f}, {ip/deg:6.4f} deg') 
        print(f'orbels dust: {ad/planet.r:6.2f} Rp, {ed:6.4f}, {ip/deg:6.4f} deg') 
        if 'movie' in plotmode:
            frame_add(dust,planet,star,interp=interp)
            if frame_out_check(i,nframes,nsteps):
                fout = frame_end(fframebase,fctr,interp=interp)
                fframelis.append(fout)
                fctr += 1
                frame_start(L)
        elif plotmode == 'orbit':
            rrel,vrel = posvelrel(dust,planet)
            pl.plot(rrel[0]/planet.r,rrel[1]/planet.r,'.k')
            print(i,tnow/year,'rrel/R_Planet:',nb.pairsep(dust,planet)/planet.r)
        else:
            rsep = nb.pairsep(dust,planet)
            ang,illfrac,islit = illumfraction(dust,planet,star)
            illfrac *= islit
            fac = pf(ang,gpf,nupf) # Draine 2003; nu=0 (HG)->1 (Rayleigh)
            # pf integrates to unity over the surface of a sphere, is that right?
            # here assume fac is the average phase fn val toward the planet. then
            # the integrated light (rel to total amount scattered) is
            #pl.plot(tnow/hour,illfrac,'.k')
            dustill = fac * illfrac * 4*np.pi
            print('scattering dir, Sun to mars form dust (deg):',ang*180/np.pi)
            pl.plot(tnow/hour,illfrac,',k')
            pl.plot(tnow/hour,dustill,',k')
            xlis.append(tnow/hour); y1lis.append(illfrac);y2lis.append(dustill)
            ex,ey,ez = zenithframe(planet,star) # frame with z out of orbital plane, x is planet-sun dir
            edp = unitvec(posrel(dust,planet))
            ang = np.arccos(np.dot(ex,edp))
            #pl.plot(tnow/hour,ang,',m')
    #xlis,ylis = np.array(xlis),np.array(ylis)
    #tlis,alis = np.array(tlis),np.array(alis)
    #elis,ilis = np.array(elis),np.array(ilis)
    if 'movie' in plotmode:
        print('files: '+' '.join(fframelis))
        os.system('convert '+' '.join(fframelis)+' duster.gif')
        os.system('cp duster.gif ~/www/tmp.gif')
        os.system('rm '+' '.join(fframelis))
    else:
        pl.plot(xlis,y1lis,'-k')
        pl.plot(xlis,y2lis,'-k')
        out = 'duster.pdf'
        pl.savefig(out)
        os.system('convert '+out+' ~/www/tmp.jpg')
    
def calcspin(sun,mars,phobos,deimos,tilt): # calc'ing spin axis from moon data, since jpl didn't give me this..
    ezplanet = np.array([np.sin(tilt),0.0,np.cos(tilt)])
    incdei = 1.788*deg
    incpho = 1.075*deg
    def f(phi):
        ezpuse = np.array( [np.cos(phi)*np.sin(tilt), np.sin(phi)*np.sin(tilt), np.cos(tilt)])
        _,_,ipho = orbels(phobos,mars,ezpuse)
        _,_,idei = orbels(deimos,mars,ezpuse)
        return ((idei-incdei)**2 + (ipho-incpho)**2)
    phi = minimize_scalar(f,[-np.pi/2,np.pi/2], bounds=[-np.pi,np.pi]).x
    ezplanet = np.array( [np.cos(phi)*np.sin(tilt), np.sin(phi)*np.sin(tilt), np.cos(tilt)])
    # a guess of spin axiz based on moon orbits w/known inc?
    return ezplanet

