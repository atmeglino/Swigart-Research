from numpy import pi

# cgs...
GNewt = 6.67403e-8

km = 1e5
kg = 1e3
meter = 1e2
hour = 3600.0
day = 24*hour
yr = 365.25*day 
year = yr

degree = 3.1415926535897932385/180

Rearth = 637800000.0
REarth = Rearth
Mearth = 5.972168398723462e+27
Mearth = 5.9724100046898e+27 # from JPL Horizons on 2024/12/22
MEarth = Mearth
PspinEarth = 23.9344695944*hour # from JPL Horizons
PorbitEarth = 365.25636*day # orbital period
tiltEarth = 23.4392911*degree # radians
TjunesolsticeEarth2025JD = 2460846.61251; # where did i get this?
TmarchequinoxEarth2025JD = 2460754.8756944 # print(2460754.8756944 + 3*hour/day)
J2tildeEarth = 0.108263e-2
MagmoEarth = 7.94e22 # A m^2
MagmoEarthlalo = (85.762,139.298) # lat, lon in degrees of magnetic north

AU =  1.495978707e13
pc = 3.8057e18
Rsun = 69570000000.0
Rsolar = Rsun
Msun = 1.9884098709818164e+33
Msun = 1.9884903130783e+33 # from JPL Horizons on 2024/12/22
Msolar = Msun
Lsun = 3.826e33
Lsolar = Lsun
Tsun = 5800.0

clight = 2.99792458e10 
uswind = 450e5
micron = 1e-4 # cgs
kB = 1.3807e-16
hplanck = 6.6261e-27
mu0 = 4*3.1415926535897932385*1e-7 # vacuum permeability 


nsalt = 1.5 - 1e-6j; rhosalt = 2.0
nglass = 1.5 - 1e-8j; rhoglass = 2.7
nal = 1.37 - 7.6j; rhoal = 2.7

#coal dust
rhocoal = 1.4; ncoal = 1.8 - 0.2j

# nlunar dust
rholunar =  2.7; nlunar = 1.7 - 1.0e-3j
rhoolivine =  2.7; nolivine = 1.7 - 1.0e-3j

# gold spheres
rhogold = 19.3; ngold = 0.27 - 2.9j