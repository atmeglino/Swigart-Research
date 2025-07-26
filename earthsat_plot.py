import numpy as np
import pandas as pd
from scipy.stats import qmc
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter as smoo
from scipy.spatial.transform import Rotation as Ro
import time
import pylab as pl
import os
import myfig
import attnmie
from constants import *
from myparms import *
import argparse
import sys
from nbody import *
import scatter as sc

deg = np.pi/180 # deg in rad so 1*deg is a degre in radians

# conveniences, maybe, for dtype in nbody.py
def pos(p): return np.array([p.x,p.y,p.z])
def vel(p): return np.array([p.vx,p.vy,p.vz])
def posvelrel(ba,bb):
    return np.array([ba.x-bb.x,ba.y-bb.y,ba.z-bb.z]),np.array([ba.vx-bb.vx,ba.vy-bb.vy,ba.vz-bb.vz])
def posrel(ba,bb): return np.array([ba.x-bb.x,ba.y-bb.y,ba.z-bb.z])
def velrel(ba,bb): return np.array([ba.vx-bb.vx,ba.vy-bb.vy,ba.vz-bb.vz])

def vec3d(x,y,z): return np.array([x,y,z])
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
    distplanet = pairsep(planet,dust)
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
    planetang = skyradius(planet.r,pairsep(planet,dust)) # radial size of planet in dust's sky
    return starang > planetang

def scatters_to_planet(ray,dust,planet):
    # if ray from dust hits planet returns 1 else 0
    eplanet = unitvec(posrel(planet,dust))
    costheta = np.dot(ray,eplanet)
    distplanet = pairsep(planet,dust)
    cosprad = np.cos(skyradius(planet.r,distplanet)) # cos of radial size of planet in dust's sky
    return costheta > cosprad

def shade(dust,planet,star):
    # fraction of stellar photons deflected that would have otherwise hit the planet
    erad = -unitvec(posrel(star,dust))
    eplanet = unitvec(posrel(planet,dust))
    distplanet = pairsep(planet,dust)
    costheta = np.dot(erad,eplanet)
    cosprad = np.cos(skyradius(planet.r,distplanet)) # cos of radial size of planet in dust's sky
    # worry about penumbra etc?
    return costheta < cosprad

def frame_out_check(i,nf,nt):
    if nf >= nt: return True
    every = nt/nf
    return True if int(i+1) % int(every) == 0 else False

frame_dat = []

def frame_start(L,color='#cc8888'):
    global frame_dat
    frame_dat = []
    fram_parms = {'L':L}
    pl.clf()
    pl.style.use('dark_background')
    pl.gca().set_aspect('equal')
    pl.xlim(-L/2,L/2); pl.ylim(-L/2,L/2)
    return

iddustoffset = 10

def frame_add(dust,planet,star):
    global frame_dat,iddustoffset
    dustuse = dust if dust.shape else [dust]
    for id,dustthis in enumerate(dustuse): # now this can be done as arrays....
        iduse = id if (dustthis.m > 1e15) else id+iddustoffset
        if dustthis.m > 1e22: iduse = -1 # moon, as in earth-moon
        illum = illuminated(dustthis,planet,star)
        ex,ey,ez = zenithframe(planet,star)
        dr = posrel(dustthis,planet)
        dv = velrel(dustthis,planet)
        dx,dy,dz = np.dot(dr,ex)/planet.r,np.dot(dr,ey)/planet.r,np.dot(dr,ez)/planet.r
        dvx,dvy,dvz = np.dot(dv,ex)/km,np.dot(dv,ey)/km,np.dot(dv,ez)/km
        frame_dat.append([iduse,dx,dy,dz,dvx,dvy,dvz])


def do_interp(x,y,z,mask=True):
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
    ta = np.linspace(-np.pi,np.pi,501) # ta as an array in phi
    ta = ta[:-1]
    ra,ha,pa = fr(ta),fh(ta),fp(ta)
    ra = smoo(ra, window_length=len(ra)//2, polyorder=2, mode='wrap')
    ha = smoo(ha, window_length=len(ha)//2, polyorder=2, mode='wrap')
    xa,ya,za = ra*np.cos(pa)*np.sin(ha),ra*np.sin(pa)*np.sin(ha),ra*np.cos(ha)
    if mask:
        msk = ~((xa < 0) & (np.sqrt(ya**2+za**2)<1.0)) 
        xa,ya,za = xa[msk],ya[msk],za[msk]
    return xa,ya,za
        
def do_interpx(x,y,z,vx,vy,vz):
    r = np.sqrt((x)**2+(y)**2+(z)**2)
    ez = np.mean(np.cross(np.array([x,y,z]).T,np.array([vx,vy,vz]).T),axis=0) # ang mo
    ez = unitvec(ez)
    ee = np.argmin(ez)
    ey = unitvec(np.cross(np.array([1,0,0] if ee==0 else ([0,1,0] if ee==1 else [0,0,1])),ez))
    ex = np.cross(ey,ez)
    rr = np.array([x,y,z])
    xa,ya,za = np.dot(ex,rr),np.dot(ey,rr),np.dot(ez,rr)
    xa,ya,za = do_interp(xa,ya,za,mask=False)
    n = len(xa)
    print(xa.shape)
    xa,ya,za = (np.tile(ex,(n,1)).T*xa)+(np.tile(ey,(n,1)).T*ya)+(np.tile(ez,(n,1)).T*za) 
    msk = ~((xa < 0) & (np.sqrt(ya**2+za**2)<1.0)) 
    xa,ya,za = xa[msk],ya[msk],za[msk]
    return xa,ya,za
        
def do_ellipse(x,y,z,vx,vy,vz):
    xc,yc,zc = np.mean(x),np.mean(y),np.mean(z)
    xc,yc,zc = 0,0,0
    r = np.sqrt((x-xc)**2+(y-yc)**2+(z-zc)**2)
    rm = np.mean(r)
    ez = np.mean(np.cross(np.array([x,y,z]).T,np.array([vx,vy,vz]).T),axis=0) # ang mo
    ez = unitvec(ez)
    ee = np.argmin(ez)
    ey = unitvec(np.cross(np.array([1,0,0] if ee==0 else ([0,1,0] if ee==1 else [0,0,1])),ez))
    ex = np.cross(ey,ez)
    n = 501
    phi = np.linspace(0,2*np.pi,n); n -= 1; phi = phi[:n] # rtfm linspace
    x,y = rm*np.cos(phi),rm*np.sin(phi)
    #ex = np.array([0,1,0])
    #ey = np.array([0,0,1])
    #print((np.tile(ey,(n,1)).T*y).shape)
    xa,ya,za = (np.tile(ex,(n,1)).T*x)+(np.tile(ey,(n,1)).T*y) #+rc
    xa += xc; ya += yc; za += zc
    msk = ~((xa < 0) & (np.sqrt(ya**2+za**2)<1.0)) 
    xa,ya,za = xa[msk],ya[msk],za[msk]
    return xa,ya,za

def frame_end(fframebase,fctr,planetcolor='#ffaaaa',moonplot='interp',dustplot='interp',timestamp='',polarang=-99):
    global frame_dat, iddustoffset
    cir = pl.Circle((0, 0), 1.0, color=planetcolor,fill=True)
    pl.gca().add_patch(cir)
    npole = np.array([0,0,1.2])
    tilt = tiltEarth
    r = Ro.from_quat([np.sin(tilt/2),0,0, np.cos(tilt/2)])
    npole = r.inv().apply(npole)
    r = Ro.from_quat([0,0,np.sin(polarang/2), np.cos(polarang/2)])
    npole = r.inv().apply(npole)
    xx,yy,zz = np.linspace(0,npole[0],400),np.linspace(0,npole[1],400),np.linspace(0,npole[2],400)
    rr,rp = np.sqrt(xx**2+yy**2+zz**2),np.sqrt(yy**2+zz**2)
    msk = (rr>1) if (npole[0]>0) else (rp>1)
    pl.plot(yy[msk],zz[msk],'-w') # north pole
    msk = (rp>1) if (npole[0]>0) else (rr>1)
    pl.plot(-yy[msk],-zz[msk],'-w') # south pole

    xeq = unitvec(np.cross(vec3d(0,-1,0),npole))
    yeq = unitvec(np.cross(npole,xeq))
    thet = np.linspace(0,2*np.pi,800).reshape(800,1)
    circ = np.cos(thet)*xeq + np.sin(thet)*yeq
    xx,yy,zz = circ[:,0],circ[:,1],circ[:,2]
    
    #idx = np.argmax(xx)
    #xx,yy,zz = np.roll(xx,idx),np.roll(yy,idx),np.roll(zz,idx)
    msk = xx > 0
    pl.plot(yy[msk],zz[msk],'-w') # equator
    pl.xlabel('y (planetary radii)')
    pl.ylabel('z (planetary radii)')
    if True:
        id,x,y,z,vx,vy,vz = np.array(frame_dat).T
        ids = np.unique(id)
        for idthis in ids:
            is_moon = True if idthis < iddustoffset else False
            is_dust = ~is_moon
            mski = id == idthis
            xa,ya,za = x[mski],y[mski],z[mski]
            vxa,vya,vza = vx[mski],vy[mski],vz[mski]
            if (is_moon and 'interp' in moonplot) or (is_dust and 'interp' in dustplot):
                xa,ya,za = do_interpx(xa,ya,za,vxa,vya,vza)
            if (is_moon and 'ring' in moonplot) or (is_dust and 'ring' in dustplot):
                xa,ya,za = do_ellipse(xa,ya,za,vxa,vya,vza)
            c,a = ('#aaaaaa',1) if is_moon else ('#ffffaa',0.1)
            if not (is_moon and ('off' in moonplot)):
                pl.scatter(ya,za,s=5,c=c,alpha=a,zorder=99)
        frame_dat = []
    if timestamp: pl.figtext(0.35,0.15,timestamp,color='w')
    fout = fframebase+f'{fctr:03d}.png'
    pl.savefig(fout)
    return fout

# ---- Main -----
if __name__ == '__main__':

    verbose = True
    plotmodes = ['orbit','illum','movie']
    plotmode = plotmodes[-1]
    
    fil = 'earthsat.bin'
    fout = 'earthsat.gif'
    #parser = argparse.ArgumentParser(description='heating versus cooling circular orbits')
    #parser.add_argument('fil', metavar='FILENAME', type=str, nargs=1, help='input binary file') # one arg
    #parser.add_argument('-o','--outfile', metavar='OUTFIL', type=str, nargs=1, default=[fout], help='output file')
    #args = parser.parse_args()
    #fil = args.fil[0]
    #fout = args.outfile[0]
    print('reading',fil,' and dumping to',fout)

    dat = np.fromfile(fil,dtype=np.float64)
    nhd,nel = 2,9
    nb = int(dat[0])
    ncol = (nhd+nb*nel)
    nt = len(dat)//ncol
    print('nbodies, nsamps: ',nb,nt)
    dat = dat.reshape(nt,ncol).copy()
    
    L = 5 #*planet.r
    dustplot='ring'
    moonplot='ring'
    dustplot = 'interp'; moonplot = 'interp'
    if 'movie' in plotmode:
        fframelis = []
        pid = str(os.getpid()) 
        fframebase = 'earthsat'+pid
        fctr = 0
        cutfac = 1
        nframes = 60//cutfac
        frame_start(L)
            
    xlis,y1lis,y2lis = [],[],[]

    print(dat.shape)
    t0 = dat[0,1]
    nsteps = dat.shape[0]
    for i in range(nsteps):
        t = dat[i,1]
        b = np.zeros(nb,dtype=bodyt).view(np.recarray)
        for j in range(nb):
            b[j].m,b[j].r = dat[i,nhd+nel*j],dat[i,nhd+nel*j+1]
            b[j].x,b[j].y,b[j].z = dat[i,nhd+nel*j+2],dat[i,nhd+nel*j+3],dat[i,nhd+nel*j+4]
            b[j].vx,b[j].vy,b[j].vz = dat[i,nhd+nel*j+5],dat[i,nhd+nel*j+6],dat[i,nhd+nel*j+7]
            b[j].L = dat[i,nhd+nel*j+8]
        star,planet,dust = b[0],b[1],b[2:]
        planetcolor = '#cc8888' 
        if planet.m > 5e27:
            planetcolor = '#88aadd'
            moonplot='off'
        if 'movie' in plotmode:
            print('time:',(t-t0)/year)
            frame_add(dust,planet,star)
            if frame_out_check(i,nframes,nsteps):
                delt = (t - TmarchequinoxEarth2025JD*day) % year
                polarang = 2*np.pi*delt/year
                print(f"polar ang {polarang/deg}")
                date = pd.to_datetime(t/day,unit='D',origin='julian')
                date = date.strftime("%Y-%m-%d %H:%M:%S")
                fout = frame_end(fframebase,fctr,planetcolor=planetcolor,moonplot=moonplot,\
                                 dustplot=dustplot,timestamp=date,polarang=polarang)
                fframelis.append(fout)
                fctr += 1
                frame_start(L)

    if 'movie' in plotmode:
        print('files: '+' '.join(fframelis))
        os.system('convert '+' '.join(fframelis)+' earthsat.gif')
        os.system('rm '+' '.join(fframelis))
        os.system('cp earthsat.gif ~/Swigart-Research/tmp.gif')
    else:
        if len(xlis):
            pl.plot(xlis,y1lis,'-k')
            pl.plot(xlis,y2lis,'-k')
        out = 'earthsat.pdf'
        pl.savefig(out)
        os.system('convert '+out+' ~/Swigart-Research/tmp.jpg')
    
