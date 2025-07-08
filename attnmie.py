#
import numpy as np
from scipy.optimize import root_scalar
import scatter as sc
import miepython

from constants import *

def do_mie(dustradius, refracindex, mu_pf = -99):  # mu_pf and dustradius cannot both be arrays...
    do_pf =  np.min(mu_pf) >= -1 and np.max(mu_pf) <= 1
    nm = micron*1e-3
    wavlenmin = 0.3*micron
    wavlenmax = 2.0*micron
    nw = 1000
    wavlen = np.linspace(wavlenmin,wavlenmax,nw)
    spec = 1/wavlen**5/(np.exp(hplanck*clight/kB/Tsun/wavlen)-1)
    spec /= np.max(spec)
    if np.isscalar(dustradius):
        qextbar, qscabar,gbar, pfbar = 0.,0.,0.,0.
    else:
        qextbar, qscabar,gbar = np.zeros(dustradius.shape), np.zeros(dustradius.shape), np.zeros(dustradius.shape)
    if do_pf:
        if np.isscalar(dustradius):
            if np.isscalar(mu_pf):
                pfbar = 0
            else:
                pfbar = np.zeros(mu_pf.shape)
        else:
            if np.isscalar(mu_pf):
                pfbar = np.zeros(dustradius.shape)
            else:
                exit('pf fail try using np.outer?')
    wei = 0
    for wv,sp in zip(wavlen,spec):
        wei += sp
        sizparm = 2*np.pi*dustradius/wv
        qext, qsca, _, g = miepython.mie(refracindex,sizparm)
        qextbar += sp * qext
        qscabar += sp * qsca
        gbar += sp * g
        if do_pf:
            pfbar += sp * miepython.i_unpolarized(refracindex,sizparm,mu_pf,'one')
    qext = qextbar/wei
    qsca = qscabar/wei
    g = gbar/wei
    if do_pf:
        pf = pfbar/wei
        return qext, qsca, g, pf
    else:
        return qext, qsca, g

def L1beta(Ms,Me,a,beta,G):
    Ome2 = G*Ms/a**3
    def ell1b(x,Ms,Me,a,beta,G):
        M = Ms+Me; Medsun,dearth = Me/M*a,Ms/M*a
        return ((G*Ms*(1-beta)/(a-x)**2)-(G*Me/x**2))/(dearth-x) - G*(Ms+Me)/a**3
    x0 = a*(Me/Ms/3)**(1/3)
    res = root_scalar(ell1b,args=(Ms,Me,a,beta,G),x0=x0,x1=2*x0,rtol=1e-8)
    return res.root

def get_d1(Ms,Me,a,beta,G):
    if np.isscalar(beta):
        if beta >= 0.95:
            return -1e99
        else:
            return L1beta(Ms,Me,a,beta,G)
    d1 = np.zeros(len(beta))
    for i in range(len(beta)):
        if beta[i]>=0.95: d1[i] = -1e99
        else: d1[i] = L1beta(Ms,Me,a,beta[i],G)
    return d1
    
def estL1pt(b,fac=1):
    x1,y1,z1 = (b[0].x-b[1].x),(b[0].y-b[1].y),(b[0].z-b[1].z) # Sun in Earth frame
    r1 = fac*0.01*AU/np.sqrt(x1**2+y1**2+z1**2); x1*=r1; y1*=r1; z1*=r1; # L1 in Earth frome
    return x1+b[1].x, y1+b[1].y, z1+b[1].z,
    
def norm2(x,y,z): return x**2+y**2+z**2

def skyang(b1,b2,bobs): # ang between b1 & b2 from b"obs"'s perspective
    dx1,dy1.dz1=(b1.x-bobs.x),(b1.y-bobs.y),(b1.z-bobs.z)
    dx2,dy2.dz2=(b2.x-bobs.x),(b2.y-bobs.y),(b2.z-bobs.z)
    cosang = (dx1*dx2 + dy1*dy2 + dz1*dz2)/np.sqrt(norm2(dx1,dy1,dz1)*norm2(dx2,dy2,dz2))
    return np.arccos(min(max(cosang,-1),1))
        
def skycoords(b,tracerindex=3,loc='C'):
    # this is overkill. create a basis, ux,uy,uz ux is earth-to-sun, uy is toward earth velocity, z is "up"
    ex,ey,ez = b[1].x,b[1].y,b[1].z # earth pos (loc=='C' for center)
    if loc in 'NSEW':
        if loc == 'N': ez += b[1].r
        elif loc == 'S': ez -= b[1].r
        else:
            sgn = 1 if loc=='W' else -1 
            evx,evy = b[1].vx,b[1].vy; fac = b[1].r/np.sqrt(evx**2+evy**2)
            ex += sgn*evx*fac; ey += sgn*evy*fac; 
    dx,dy,dz = (b[tracerindex:].x-ex),(b[tracerindex:].y-ey),(b[tracerindex:].z-ez)
    rp = np.sqrt(dx**2 + dy**2 + dz**2) 
    upx,upy,upz = dx/rp,dy/rp,dz/rp
    dxs,dys,dzs = (b[0].x-ex),(b[0].y-ey),(b[0].z-ez) # sun pos in earth frame
    re = np.sqrt(dxs**2+dys**2+dzs**2)
    uxx,uxy,uxz = dxs/re,dys/re,dzs/re # unit vec, ux is x-like, dir of sun 
    uzx,uzy,uzz = (uxy*b[1].vz-uxz*b[1].vy),(uxz*b[1].vx-uxx*b[1].vz),(uxx*b[1].vy-uxy*b[1].vx)
    unorm = np.sqrt(uzx**2+uzy**2+uzz**2); uzx,uzy,uzz = uzx/unorm,uzy/unorm,uzz/unorm # uz is z-like, perp to orb plane 
    uyx,uyy,uyz = (uzy*uxz-uzz*uxy),(uzz*uxx-uzx*uxz),(uzx*uxy-uzy*uxx) 
    unorm = np.sqrt(uyx**2+uyy**2+uyz**2); uyx,uyy,uyz = uyx/unorm,uyy/unorm,uyz/unorm # uy is y-like, 
    th = np.pi/2 - np.arccos(upx*uzx + upy*uzy + upz*uzz) # pi/2 - cosine angle, angle in z dir  
    ph = np.pi/2 - np.arccos(upx*uyx + upy*uyy + upz*uyz) #
    # check if not near sun?
    cossp = upx*uxx + upy*uxy + upz*uxz
    th = np.where(cossp<0,np.pi-th,th)
    ph = np.where(cossp<0,np.pi-ph,ph)
    return th,ph

def earthsunframe(b,tracerindex=3):
    # this is overkill. create a basis, ux,uy,uz ux is earth-to-sun, uy is toward earth velocity, z is "up"
    ex,ey,ez = b[1].x,b[1].y,b[1].z # earth pos (loc=='C' for center)
    dx,dy,dz = (b[tracerindex:].x-ex),(b[tracerindex:].y-ey),(b[tracerindex:].z-ez)
    dxs,dys,dzs = (b[0].x-ex),(b[0].y-ey),(b[0].z-ez) # sun pos in earth frame
    re = np.sqrt(dxs**2+dys**2+dzs**2); uxx,uxy,uxz = dxs/re,dys/re,dzs/re # ux, x-like, dir of sun
    uzx,uzy,uzz = (uxy*b[1].vz-uxz*b[1].vy),(uxz*b[1].vx-uxx*b[1].vz),(uxx*b[1].vy-uxy*b[1].vx) # np.cross?
    unorm = np.sqrt(uzx**2+uzy**2+uzz**2); uzx,uzy,uzz = uzx/unorm,uzy/unorm,uzz/unorm # uz, perp to orb plane 
    uyx,uyy,uyz = (uzy*uxz-uzz*uxy),(uzz*uxx-uzx*uxz),(uzx*uxy-uzy*uxx) 
    unorm = np.sqrt(uyx**2+uyy**2+uyz**2); uyx,uyy,uyz = uyx/unorm,uyy/unorm,uyz/unorm # uy is y-like
    x,y,z = uxx*dx+uxy*dy+uxz*dz, uyx*dx+uyy*dy+uyz*dz, uzx*dx+uzy*dy+uzz*dz # np.dot, right?
    return x,y,z

def cossep(bA,bB,b0):
    xA,yA,zA = (bA.x-b0.x),(bA.y-b0.y),(bA.z-b0.z),
    xB,yB,zB = (bB.x-b0.x),(bB.y-b0.y),(bB.z-b0.z),
    cosp = (xA*xB+yA*yB+zA*zB)/np.sqrt((xA**2+yA**2+zA**2)*(xB**2+yB**2+zB**2))
    return cosp

def overlap(rbig,rsmall,offset): #fraction of big circle that is partially overlapped by little circle
    R,r,d = rbig,rsmall,offset # wolfram mathworld
    drRfac,dRrfac = (d**2+r**2-R**2)/(2*d*r), (d**2+R**2-r**2)/(2*d*R)
    drRfac = np.where(np.abs(drRfac)>1,drRfac/np.abs(drRfac),drRfac)
    dRrfac = np.where(np.abs(dRrfac)>1,dRrfac/np.abs(dRrfac),dRrfac)
    drRyuk = (-d+r+R)*(d+r-R)*(d-r+R)*(d+r+R)
    drRyuk = np.where(drRyuk<0,0,drRyuk)
    return (r**2*np.arccos(drRfac)+R**2*np.arccos(dRrfac)-0.5*np.sqrt(drRyuk))/(np.pi*R**2)

def radredux(b):
    des,det,dst = pairsep(b[0],b[1]),pairsep(b,b[1]),pairsep(b,b[0])
    f = np.zeros(len(b))
    msk = (b.m<1e11)&(des>0)&(det>0)&(det<des) # should be all tracers
    re,rs = b[1].r,b[0].r
    rshadow = rs*det[msk]/(des-det[msk])
    rshctr = des*np.sin(np.arccos(cossep(b[msk],b[1],b[0]))) # loc of tracer shadow in plane of earth disk
    f[msk] = np.where((rshadow>re)&(rshctr<(rshadow-re)),re**2/rshadow**2,f[msk])
    f[msk] = np.where((rshadow>re)&(rshctr>(rshadow-re)),overlap(rshadow,re,rshctr),f[msk])
    f[msk] = np.where((rshadow<re)&(rshctr<(re-rshadow)),rshadow**2/re**2,f[msk])
    f[msk] = np.where((rshadow<re)&(rshctr>(re-rshadow)),overlap(re,rshadow,rshctr),f[msk])
    return f

def radreduxMC(b):
    N = 5000; xg = np.linspace(-1,1,N)
    dX,dY = np.meshgrid(xg,xg); msk = dX**2+dY**2 < 1; dX,dY = dX[msk].flatten(), dY[msk].flatten()
    dA = 4/N**2+0*dX
    re,rs = b[1].r,b[0].r
    f = np.zeros(len(b))
    for i in range(len(b)):
        bi = b[i]
        if (bi.m>1e11): continue
        des,det,dst = pairsep(b[0],b[1]),pairsep(bi,b[1]),pairsep(bi,b[0])
        rshadow = rs*det/(des-det)
        rshctr = des*np.sin(np.arccos(cossep(bi,b[1],b[0])))
        xi,yi,ai = dX*rshadow+rshctr,dY*rshadow,dA*rshadow**2
        f[i] = (np.sum(ai[xi**2+yi**2<re**2])/(np.pi*rshadow**2))
    return f

def ovrlap(d,re,rs): # overlapping area of circles rs > re displaced from centers by distance d.
    rootterm = (d+re+rs)*(d-re+rs)*(d+re-rs)*(-d+re+rs)
    rootterm = np.where(rootterm<=0,0.0,rootterm)
    drsre = (d**2+rs**2-re**2)/(2*d*rs)
    drsre = np.where(drsre > 1.0,1.0,drsre)
    drsre = np.where(drsre < -1.0,-1.0,drsre)
    drers = (d**2+re**2-rs**2)/(2*d*re)
    drers = np.where(drers > 1.0,1.0,drers)
    drers = np.where(drers < -1.0,-1.0,drers)
    return re**2*np.arccos(drers)+rs**2*np.arccos(drsre)-0.5*np.sqrt(rootterm)

def attngeo(rp,ang,d1,asemi=AU):
    global Rearth, Rsun
    rs = Rsun*d1/(asemi-d1) # radius of shadow
    re = Rearth # yes
    d = ang*d1*asemi/(asemi-d1) # distance shadow center from Earth center, projected onto plane at Earth loc
    # think thru: |----*-------------Sun  |< 
    #if np.isscalar(ang) and np.isscalar(rp) and np.isscalar(d1):
    if np.isscalar(ang) and np.isscalar(d1):
        fs = np.where(d<=np.abs(rs-re),min(re**2/rs**2,1.0),0) # covers cases shadow/earth overlap completely or not at all....
        if (d<rs+re)&(d>np.abs(rs-re)):
            fs = ovrlap(d,rs,re)/(np.pi*rs**2)
    else:
        fs = np.where(d <= np.abs(rs-re), np.where(re>rs,1.0,re**2/rs**2), 0)
        msk = (d<rs+re)&(d>np.abs(rs-re));
        if (msk.any()):
            fs[msk] = ovrlap(d[msk],re,rs[msk])/(np.pi*rs[msk]**2)
    return fs*rp**2/Rearth**2

def attngeoold(rp,ang,d1,asemi=AU):
    global Rearth, Rsun
    rs = Rsun*d1/(asemi-d1) # radius of shadow
    re = Rearth # yes
    d = ang*d1*asemi/(asemi-d1) # distance shadow center from Earth center, projected onto plane at Earth loc
    # think thru: |----*-------------Sun  |< 
    if np.isscalar(ang) and np.isscalar(rp) and np.isscalar(d1):
        if d > rs + re:
            fs = 0.0
        else:
            if rs >= re:
                if d <= rs - re: # earth in penumbra
                    fs = re**2/rs**2
                else:
                    fs = ovrlap(d,re,rs)/(np.pi*rs**2)
            else:
                if d <= re - rs: # shadow falls completely on the Earth
                    fs = 1.0
                else:
                    fs = ovrlap(d,rs,re)/(np.pi*rs**2)
        fs = np.where(d<=np.abs(rs-re),min(re**2/rs**2,1.0),0) # covers cases shadow/earth overlap completely or not at all....
        if (d<rs+re)&(d>np.abs(rs-re)):
            fs = ovrlap(d,rs,re)/(np.pi*rs**2)
    else:
        fs = np.where(d <= np.abs(rs-re), np.where(re>rs,1.0,re**2/rs**2), 0)
        msk = (d<rs+re)&(d>np.abs(rs-re));
        if (msk.any()):
            fs[msk] = ovrlap(d[msk],re,rs[msk])/(np.pi*rs[msk]**2)
    return fs*rp**2/d1**2*asemi**2/Rsun**2

def attnscatter(rp,qext,ang,d1,asemi=AU,Ndust=1):
    return Ndust*qext*attngeo(rp,ang,d1,asemi=AU)

def ampscatter(rp,qext,qsca,ang,pfang,d1,asemi=AU,re=Rearth,Ndust=1):
    fracscattered = qsca * (rp/re)**2
    fracreturned2earth = np.pi*(rp/d1)**2 * pfang * fracscattered
    return fracreturned2earth

def britensca(rp,th,g,nu,d1=0.01*AU,asemi=AU):
    global Rearth, Rsun
    delOm = np.pi*(Rearth/d1)**2
    return rp**2/d1**2*sc.phase_function_D03(th,g,nu)*delOm*asemi**2/Rsun**2


def index_of_refrac_mixture(nglass,fillfac):
    ef = (nglass.real)**2
    fff = lambda x: (1-fillfac)*(1-x)/(1+2*x)+fillfac*(ef-x)/(ef+2*x)
    neff = np.sqrt(root_scalar(fff,x0=1,x1=ef,rtol=1e-8).root)
    return neff



