import numpy as np 
from math import pi
from scipy.integrate import ode
from scipy.integrate import odeint
from scipy.optimize import brentq
import Physical_Const as phys
from pylab import *
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.artist as pltart
from scipy.interpolate import interp1d,InterpolatedUnivariateSpline
from scipy.optimize import curve_fit

import argparse as ap

parser=ap.ArgumentParser()
parser.add_argument("file_EoS", help="input EoS", type=str)
parser.add_argument("file_Mcont", help="input mass const", type=str)
parser.add_argument("file_kep", help="input kep seq", type=str)
parser.add_argument("--mass", help="mass WD", type=float)
args=parser.parse_args()

Masswd0 = args.mass

#----Constants------
hbar=phys.hbar
h=hbar*(2.0*np.pi)
c=phys.c
G=phys.G
Msun=phys.Msun
sigma=phys.sigma
me=phys.me
Rsun = 6.95e10
mu = phys.mu

Sigma=np.power(me,4.0)*np.power(c,3.0)/(8.0*np.power(np.pi,2.0)*np.power(hbar,3.0))
SigmaP = c**2 * Sigma
MA=2.0
Sigma02=(mu*MA*np.power(me*c/hbar,3))/(3*np.power(np.pi,2))
Rdim=c/np.sqrt(Sigma02*G)
Cgrav = G * Msun / c**2
#Rdim = 0.01*Rsun
#Sigma02 = c**2 / ( Rdim**2 * G)
Mdim = Rdim / Cgrav
Jdim = G / c**3

rhobeta=np.log10(3.9e10)

#br=c/np.sqrt(Sigma*G)
#Mdim=(br*np.power(c,2.0)/G)/Msun

#---read EoS----------
rhoEoS,xeEoS,PEoS,PsatEoS,chip_EoS=np.loadtxt(args.file_EoS,usecols=(0,1,2,3,4),unpack=True)
rhoi=np.log10(rhoEoS[0]); rhof=np.log10(rhoEoS[-1])
xemin=xeEoS[0]; xemax=xeEoS[-1]

#---read keplerian and static sequence-----
rhowd_kep,pwd_kep,Mstawd_kep,Rstawd_kep,Mrotwd_kep,Rrotwd_kep,Jwd_kep,Omegawd_kep,Qwd_kep,Omegakwd_kep=np.loadtxt(args.file_kep,usecols=(0,1,2,3,4,5,6,7,8,9),unpack=True)


NNdat=len(rhowd_kep)
Mmax_sta=0.0; Nsta = 0
Mmax_rot=0.0; Nrot = 0
for i in range(NNdat):
    if Mmax_sta<Mstawd_kep[i]:
        Mmax_sta = Mstawd_kep[i]
        Nsta = i
    if Mmax_rot<Mrotwd_kep[i]:
        Mmax_rot = Mrotwd_kep[i]
        Nrot = i

print "#############################################################"
print "Maximun static mass: ", Mmax_sta*Mdim, " Msun\t Density ( [g / cm^3]):", np.power(10.0,rhowd_kep[Nsta]), "\tRadius: ", Rstawd_kep[Nsta]*Rdim, " [cm]"
print "Maximun rotating mass: ", Mmax_rot*Mdim, " Msun\t Density ([g / cm^3]):", np.power(10.0,rhowd_kep[Nrot]), "\tRadius: ", Rrotwd_kep[Nrot]*Rdim, " [cm]"

print "#############################################################"

Kep_Omega=interp1d( [Mrotwd_kep[j]*Mdim for j in range(Nrot+1)] , [Omegawd_kep[j] for j in range(Nrot+1)],kind='cubic' )
Kep_Req=interp1d( [ Mrotwd_kep[j]*Mdim for j in range(Nrot+1)], [ Rrotwd_kep[j] for j in range(Nrot+1)], kind='cubic' )
Kep_j=interp1d( [ Mrotwd_kep[j]*Mdim for j in range(Nrot+1)], [ Jwd_kep[j] for j in range(Nrot+1)], kind='cubic' )

Omegawd_0= Kep_Omega(Masswd0)
Reqwd_0= Kep_Req(Masswd0)
jwd_0 = Kep_j(Masswd0)


print "Mass0 = ", Masswd0, "\tOmega0 = ", Omegawd_0*Rdim/c, "\t Req0 = ", Reqwd_0*Rdim
print "############################################################"

#---Algoritm first derivativa----------
def factorD(x,x0,x1,x2,x3):
    den = 3.0 * x * x - 2.0 * x * ( x1 + x2 + x3 ) + x1 * x2 + x2 * x3 + x1 * x3
    num = ( x0 - x1 ) * ( x0 - x2 ) * ( x0 - x3 )
    return den/num

def fit_derivative(xpoints,ypoints):
    dd = [  ( ypoints[1] - ypoints[0] ) / ( xpoints[1] - xpoints[0] )  ]
    dd.append( ( ypoints[3] - ypoints[0] ) / ( xpoints[3] - xpoints[0] ) )
    for i in range(2,len(xpoints)-2):
        xi = xpoints[i]
        x0 = xpoints[i-2]
        x1 = xpoints[i-1]
        x2 = xpoints[i+1]
        x3 = xpoints[i+2]
        MMD = ypoints[i-2]*factorD(xi,x0,x1,x2,x3) + ypoints[i-1]*factorD(xi,x1,x0,x2,x3) + ypoints[i+1]*factorD(xi,x2,x0,x1,x3) + ypoints[i+2]*factorD(xi,x3,x0,x1,x2)
        dd.append(MMD)
    dd.append(  ( ypoints[-2] - ypoints[-3] ) / ( xpoints[-2] - xpoints[-3] )  )
    dd.append(  ( ypoints[-1] - ypoints[-2] ) / ( xpoints[-1] - xpoints[-2] )  )
    return dd

def Interp1D_ll(x,xpoints,ypoints):
   iix=-1;
   for i in range(len(xpoints)-1):
       if x<=xpoints[i] and x>=xpoints[i+1]:
          iix=i;
          break;
   if iix<0:
       if x>xpoints[0]:
           iix=0; 
       else:
           iix=len(xpoints)-2;
   mm =(ypoints[iix]-ypoints[iix+1])/(xpoints[iix]-xpoints[iix+1]);
   return mm * ( x - xpoints[iix] ) + ypoints[iix]

#----read constant mass sequence--------------------
rhowd,pwd,Mstawd,Rstawd,Mrotwd,Rrotwd,Jwd,Omegawd,Qwd,Omegakwd=np.loadtxt(args.file_Mcont,usecols=(0,1,2,3,4,5,6,7,8,9),unpack=True)

NN = len(Jwd)
Jadim = 1e50
JJdim = Jadim * Jdim  / Rdim**2
NNdat=len(rhowd)
jmin = min(Jwd)
jmin/=JJdim
jmax = max(Jwd)
jmax /= JJdim
jwd_0 /= JJdim

#DrhoDj_dats=fit_derivative( Jwd / JJdim , rhowd)
#---- DrhoDj------------
#DeDp = fit_derivative( P_RFMT,rho_RFMT) 
rho_j = interp1d( [Jwd[j] / JJdim for j in range(NN)] , [rhowd[j] for j in range(NN)],kind='linear' )
DrhoDj_dats=[]

for i in range(NN):
    jx =  Jwd[i] / JJdim
    if i==0:
        dx= (jx-Jwd[i+1]/JJdim)*0.001
        Der = (  rho_j( jx ) - rho_j( jx-dx) ) / (dx)
    elif i==NN-1:
        dx= (Jwd[i-1]/JJdim-jx)*0.001
        Der = ( rho_j( jx+dx) - rho_j( jx )) / dx
    else:
        dx= (Jwd[i+1]/JJdim-jx)*0.001
        Der = ( rho_j( jx+dx ) - rho_j( jx-dx ) ) / dx
    DrhoDj_dats.append(Der)

DRDj_dats=fit_derivative( Jwd / JJdim , Rrotwd / Reqwd_0)

Drho_dj=interp1d( [Jwd[j] / JJdim for j in range(NN)] , [DrhoDj_dats[j] for j in range(NN)],kind='linear' )

Req_j=interp1d( [ Jwd[j] / JJdim for j in range(NN)], [ Rrotwd[j] / Reqwd_0 for j in range(NN)], kind='linear' )
DReq_dj = interp1d( [ Jwd[j] / JJdim for j in range(NN)], [ DRDj_dats[j]  for j in range(NN)], kind='linear' )

Omega_j=interp1d( [ Jwd[j] / JJdim for j in range(NN)] , [Omegawd[j] / Omegawd_0 for j in range(NN)],kind='linear' )



x=np.linspace(0.1*min(jmin,jmax),1.1*max(jmin,jmax), 100)  
rhowd_2=[]
for xi in x:
    rhoww = Interp1D_ll(xi, [Jwd[j] / JJdim for j in range(NN)] , [rhowd[j] for j in range(NN)])
    rhowd_2.append(rhoww)

plt.plot(x,rhowd_2,'k:')
#plt.plot(x,rho_j(x),'k:')
plt.plot( Jwd/JJdim , rhowd,'o')
plt.show()

"""

plt.plot(x,Req_j(x),'k:')
plt.plot( Jwd/JJdim , Rrotwd / Reqwd_0,'o')
plt.show()

plt.plot(x,Omega_j(x),'k:')
plt.plot( Jwd/JJdim , Omegawd / Omegawd_0,'o')
plt.show()

plt.plot(x,Drho_dj(x),'k:')
plt.plot( Jwd/JJdim , DrhoDj_dats,'o')
plt.show()
"""

"""
plt.plot(x,DReq_dj(x),'k:')
plt.plot( Jwd/JJdim , DRDj_dats,'o')
plt.show()
"""

#EoS-----------------------------------
#---degenerate pressure-------------
def Pch_EoS(x):
    pWD=SigmaP*(x*np.sqrt(1.0+np.power(x,2.0))*(2.0*np.power(x,2.0)/3.0-1.0)+np.log(np.sqrt(1.0+np.power(x,2.0))+x))
    return pWD

def func(x, b):
    return (np.power(10.0,rhof)/Sigma02)*np.power(x,b)/np.power(xemax,b)

#---find best fit-----
popt, pcov = curve_fit(func, xeEoS , rhoEoS / Sigma02)
afit=popt[0]
#bfit=popt[1]
NNum=100

log_xXe=np.linspace(np.log10(xeEoS[0]*0.1),np.log10(xeEoS[-1]*1.5),NNum)
xx_xe = np.power(10.0,log_xXe)
P_RFMT=[]; i=0; rho_RFMT=[];
#Create grid to fit fuction and calculate denrivatives EoS-------
for xi in xx_xe:
    PP = Pch_EoS(xi)
    P_RFMT.append( PP / ( c**2*Sigma02 ) )
    rho_RFMT.append( func(xi,afit) )

EoS_RFMT=interp1d( rho_RFMT , P_RFMT ,kind='cubic' )
EoS_RFMT_02 = interp1d( P_RFMT , rho_RFMT, kind = 'cubic')
#EoS_RFMT_02 = interp1d( p_grid , rho_gridp,kind='cubic')


#----TOV Equation: Static sequence----------
#ec1: dMdr ; ec2: dnudr ; ec3: dP/dr
#---- HT first two equations----------------
#ec4 : djdr ; ec5 : domegadr
def TOV(r,y):
    mns = y[0]
    pns = y[1]
    nuns = y[2]
    ec2 = 2.0 * ( 4.0 * np.pi * np.power(r,3.0) * pns + mns ) / ( r * ( r - 2.0 * mns ) )
    ec1 = 4.0 * np.pi * np.power(r,2.0) * EoS_RFMT_02(pns)
    ec3 = - 0.5 * ec2 * ( pns + EoS_RFMT_02(pns) )
    ec4 = y[3]
    ec5 = - 4.0 * y[3] / r + 4.0 * np.pi * np.power(r,2.0) * (pns + EoS_RFMT_02(pns) ) * ( y[3] + 4.0 * y[4] / r ) / ( r - 2.0 * mns )
    return array([ec1,ec3,ec2,ec5,ec4])

Static=ode(TOV).set_integrator('dopri5',atol=1e-9)

#----Loop to calculate Static sequence--------
rf=0.1
r0=1e-9
def StaticSeq(y0,drr):
    #r0 = drr*1e-2
    Static.set_initial_value(y0,r0)
    while Static.successful() and Static.t<rf and np.log10(EoS_RFMT_02(Static.y[1])*Sigma02) > 3.0 and Static.y[1]>0.0:
        Static.integrate(Static.t+drr)
    mstar=Static.y[0]
    rstar=Static.t
    nustar = np.log( 1.0 - 2.0 * Static.y[0] / Static.t ) - Static.y[2]
    Jstar = np.power( Static.t , 4.0 ) * Static.y[3] / 6.0
    omegastar = Static.y[4] + 2.0 * Jstar / np.power(Static.t,3.0)      
    m2i = (1.0-2.0*y0[0]/r0) * np.exp(-nustar) * (4.0 * np.pi / 15.0 ) * ( EoS_RFMT_02(y0[1]) + y0[1] ) * ( 2.0 + np.power(10.0,DeDpF ( np.log10(y0[1]) )) ) * np.power(r0,5.0)
    p0i = 1.0 / 3.0 * np.power(r0,2.0) * ( 1.0 - 2.0 * y0[0] / r0 ) * np.exp(-nustar)  
    return [mstar,rstar,nustar,Jstar,omegastar,m2i,p0i]


def muWD(Bs,Rwd):
    return Bs*np.power(Rwd,3.0)

def TdipIII(R_wd,Omega_wd):
    return np.power(R_wd,6.0) * np.power(Omega_wd,3.0)

def AdipIII(tau,Omegawd_0,Rwd_0,Bs):
    return (1.0/c**3) * np.power(Omegawd_0,3.0) * muWD(Bs,Rwd_0)**2 * tau  / Jadim

def Torque_dip(t,y,AAdip):
    Reqx = Interp1D_ll( y[0], [ Jwd[j] / JJdim for j in range(NN)] , [ Rrotwd[j] / Reqwd_0 for j in range(NN)] )
    #Reqx = Req_j(y[0])
    Omegax = Interp1D_ll( y[0], [ Jwd[j] / JJdim for j in range(NN)] , [ Omegawd[j] / Omegawd_0 for j in range(NN)] )
   # Omegax = Omega_j(y[0])
    print Omegax,Reqx
    ec1 = np.log(10.0)*np.power(10.0,t)*AAdip*TdipIII(Reqx,Omegax)
    return np.array([ec1])

Bs0=1e6
tauD= 1.0 / AdipIII(1.0,Omegawd_0,Reqwd_0,Bs0)
A0 = AdipIII(tauD, Omegawd_0, Reqwd_0,Bs0) 

def RungheKut(t,y,h,fx):
    dsy1 = Torque_dip(t,y,fx)
    print dsy1
    dsy2 = Torque_dip( t+h/2.0 , y + dsy1[0] * h / 2.0 , fx )
    print dsy2
    dsy3 = Torque_dip( t + h / 2.0 , y + dsy2[0] * h / 2.0 , fx )
    print dsy3
    dsy4=Torque_dip( t + h  , y + dsy3[0] * h , fx )
    print dsy4
    jnew=y[0]+1.0/6.0*h*(dsy1[0]+2.0*(dsy2[0]+dsy3[0])+dsy4[0])
    return [jnew]

i=0; yy=[jwd_0]
t0=1e-4
dt = 1e-2
while rho_j(yy[0])<rhobeta:
    yynew=RungheKut(t0+i*dt,yy,dt,A0)
         #print yynew
    yy=yynew
    i=i+1
print yy[0]*Mdim
