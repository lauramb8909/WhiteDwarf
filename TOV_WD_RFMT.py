import numpy as np 
from math import pi
from scipy.integrate import ode
from scipy.integrate import odeint
from scipy.optimize import brentq
import Physical_Const as phys
from scipy.interpolate import interp1d,InterpolatedUnivariateSpline
from scipy.optimize import curve_fit


from pylab import *
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.artist as pltart

import argparse as ap

parser=ap.ArgumentParser()
parser.add_argument("file_EoS", help="input EoS", type=str)
args=parser.parse_args()


#----Constants------
hbar   = phys.hbar
h      = hbar*(2.0*np.pi)
c      = phys.c
G      = phys.G
Msun   = phys.Msun
Rsun   = phys.Rsum
sigma  = phys.sigmaSB
me     = phys.me
mu     = phys.mu
pii    = np.pi

Sigma   = np.power(me,4.0)*np.power(c,3.0)/(8.0*np.power(pi,2.0)*np.power(hbar,3.0))
SigmaP  = c**2 * Sigma

MA      = 2.0
Sigma02 = (mu*MA*np.power(me*c/hbar,3))/(3*np.power(pi,2))
Rdim    = c/np.sqrt(Sigma02*G)
Cgrav   = G * Msun / c**2
Mdim    = Rdim / Cgrav

#Rdim = 0.01*Rsun
#Sigma02 = c**2 / ( Rdim**2 * G)
#br=c/np.sqrt(Sigma*G)
#Mdim=(br*np.power(c,2.0)/G)/Msun

rhonuc=2.7e14

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
       if x>=xpoints[i] and x<=xpoints[i+1]:
          iix=i;
          break;
   if iix<0:
       if x<xpoints[i]:
           iix=0; 
       else:
           iix=len(xpoints)-2;
   mm =(ypoints[iix]-ypoints[iix+1])/(xpoints[iix]-xpoints[iix+1]);
   return mm * ( x - xpoints[iix] ) + ypoints[iix]



#---read EoS----------
rhoEoS, xeEoS, PEoS, PsatEoS, chip_EoS = np.loadtxt(args.file_EoS,usecols=(0,1,2,3,4),unpack=True)
rhoi  = np.log10( rhoEoS[0] )
rhof  = np.log10( rhoEoS[-1] )
xemin = xeEoS[0]
xemax = xeEoS[-1]

#---find best fit-----
popt, pcov = curve_fit(func, xeEoS , rhoEoS / Sigma02)
afit = popt[0]
#bfit=popt[1]
print(afit)#, bfit


log_xXe=np.linspace(np.log10(xeEoS[0]*0.5),np.log10(xeEoS[-1]*1.5),NN)
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

#---degenerate pressure-------------
def Pch_EoS(x):
    pWD = SigmaP*(x*np.sqrt(1.0+np.power(x,2.0))*(2.0*np.power(x,2.0)/3.0-1.0)+np.log(np.sqrt(1.0+np.power(x,2.0))+x))
    return pWD

def func(x, b):
    return (np.power(10.0,rhof)/Sigma02)*np.power(x,b)/np.power(xemax,b)



"""
#----Plot fit EoS------
x=np.linspace(rhoi,rhof,100)
Pmin = np.log10(P_RFMT[0]) ; Pmax=np.log10(P_RFMT[-2])
xp = np.linspace(Pmin,Pmax,100)
plt.loglog( rhoEoS/Sigma02, Pch_EoS(xeEoS)/ ( c**2*Sigma02 ),'o')
plt.loglog( np.power(10,x)/Sigma02, EoS_RFMT(np.power(10,x)/Sigma02),'b-')
plt.loglog( EoS_RFMT_02(np.power(10,xp)), np.power(10,xp) ,'r-')
plt.loglog( rho_RFMT, P_RFMT,'k:')
plt.show()
exit()
"""



"#---- DrhoDp------------
#DeDp = fit_derivative( P_RFMT,rho_RFMT) 
DeDp_2=[]
DxeF=0.01
for i in range(NN):
    dx=xx_xe[i]*DxeF
    if i==0:
        Der = (func(xx_xe[i]+dx,afit)-rho_RFMT[i]) / (Pch_EoS(xx_xe[i]+dx)/(c**2*Sigma02)-P_RFMT[i])
    else:
        Der = (func(xx_xe[i]+dx,afit)-func(xx_xe[i]-dx,afit)) / (Pch_EoS(xx_xe[i]+dx)/(c**2*Sigma02)-Pch_EoS(xx_xe[i]-dx)/(c**2*Sigma02))
    DeDp_2.append(Der)"

DeDpF = interp1d(np.log10(P_RFMT),  np.log10(DeDp_2),kind='cubic')

"""
#---Plot derivative EoS-----
Pmin=np.log10(P_RFMT[0]) ; Pmax=np.log10(P_RFMT[-1])
x=np.linspace(Pmin,Pmax,100)
plt.loglog(P_RFMT,DeDp_2,'o')
#plt.loglog(P_RFMT,DeDp_2,'o')
plt.loglog(np.power(10,x),np.power(10,DeDpF(x)),'k-')
plt.show()
exit()
"""

#----TOV Equation: Static sequence----------
#ec1: dMdr ; ec2: dnudr ; ec3: dP/dr
#---- HT first two equations----------------
#ec4 : djdr ; ec5 : domegadr
def TOV(r,y):
    mns  = y[0]
    pns  = y[1]
    nuns = y[2]

    rhons = EoS_RFMT_02(pns)

    r2 = r*r
    r3 = r2*r

    ec2 = 2.0 * ( 4.0 * pii * r3 * pns + mns ) / ( r * ( r - 2.0 * mns ) )
    ec1 = 4.0 * pii * r2 * rhons
    ec3 = - 0.5 * ec2 * ( pns + rhons )
    ec4 = y[3]
    ec5 = - 4.0 * y[3] / r + 4.0 * pii * r2 * ( pns + rhons ) * ( y[3] + 4.0 * y[4] / r ) / ( r - 2.0 * mns )
    
    return array([ec1,ec3,ec2,ec5,ec4])


#----TOV Equation: Static sequence----------
#ec1: dMdr ; ec2: dnudr ; ec3: dP/dr
#---- HT first two equations (first order equations)----------------
#ec4 : djdr ; ec5 : domegadr
def TOV_HT_02(r,y):
    mwd = y[0];
    nuwd = y[2];
    pwd = y[1];
    ec2 = 2.0 * ( 4.0 * np.pi * np.power(r,3.0) * pwd + mwd ) / ( r * ( r - 2.0 * mwd ) ); #dpdr
    ec1 = 4.0 * np.pi * np.power(r,2.0) * EoS_RFMT_02(pwd); # dmdr
    ec3 = -0.5 * ec2 * ( pwd + EoS_RFMT_02(pwd) )
    ec4 = y[3]
    ec5 = -4.0 * y[3] / r + 4.0 * np.pi * np.power(r,2.0) * (pwd + EoS_RFMT_02(pwd)) * ( y[3] + 4.0 * y[4] / r ) / ( r - 2.0 * mwd)
    ec6 = 4.0 * np.pi * np.power(r,2.0) * ( pwd + EoS_RFMT_02(pwd) ) * y[6] * np.power(10.0,DeDpF(np.log10(pwd))) + (np.power(r,3.0) / 12.0 ) * np.exp(-nuwd) * np.power(y[3],2.0) * ( r - 2.0 * mwd) + 8.0*np.power(y[4],2.0) * np.pi * (pwd + EoS_RFMT_02(pwd)) * (np.power(r,4) / 3.0 ) * np.exp(-nuwd)
    ec7 = -y[5] * ( 1.0 + 8.0 * np.pi * np.power(r,2.0) * pwd) / np.power( r - 2.0 * mwd,2.0 ) - y[6] * 4.0 * np.pi * np.power(r,2.0) * (pwd + EoS_RFMT_02(pwd)) / ( r - 2.0 * mwd) + ( 2.0 / 3.0 ) * np.power(y[4],2.0) * r * np.exp(-nuwd) * ( 1.0 - ( 4.0 * np.pi * np.power(r,3) * pwd + mwd) / ( r - 2.0 * mwd ) + r * y[3] / y[4] + np.power( r * y[3]/y[4],2.0) / 8.0)
    return array([ec1,ec3,ec2,ec5,ec4,ec6,ec7])


#----HT-TOV Equation: Static sequence----------
#ec1: dMdr ; ec2: dnudr ; ec3: dP/dr
def TOV_HT(r,y):
    mwd  = y[0]
    nuwd = y[2]
    pwd  = y[1]


    ec1 = 4.0 * np.pi * np.power(r,2.0) * EoS_RFMT_02(pwd); # dmdr
    ec2 = 2.0 * ( 4.0 * np.pi * np.power(r,3.0) * pwd + mwd ) / ( r * ( r - 2.0 * mwd ) ) #dpdr
    ec3 = - 0.5 * ec2 * ( EoS_RFMT_02(pwd) + pwd ); # dnudr
    ec4 = y[3]
    ec5 = -4.0 * y[3] / r + 4.0 * np.pi * np.power(r,2.0) * ( pwd + EoS_RFMT_02(pwd) ) * ( y[3] + 4.0 * y[4] / r ) / ( r - 2.0 * mwd ) # dbaromegadr
    ec6 = 4.0 * np.pi * np.power(r,2.0) * ( pwd + EoS_RFMT_02(pwd) ) * y[6] * np.power(10.0,DeDpF(np.log10(pwd))) + ( np.power(r,3.0) / 12.0 ) * np.exp(-nuwd) * np.power(y[3],2.0) * ( r - 2.0 * mwd ) + 8.0 * np.power(y[4],2.0) * np.pi * ( pwd + EoS_RFMT_02(pwd) ) * ( np.power(r,4) / 3.0) * np.exp(-nuwd);
    ec7 = -y[5] * (1.0 + 8.0 * np.pi * np.power(r,2.0) * pwd ) / np.power(r - 2.0 * mwd,2.0) - y[6] * 4.0 * np.pi * np.power(r,2.0) * ( pwd + EoS_RFMT_02(pwd) ) / ( r - 2.0 * mwd ) + (2.0/3.0) * np.power(y[4],2.0) * r * np.exp(-nuwd) * ( 1.0 - ( 4.0 * np.pi * np.power(r,3) * pwd + mwd ) / ( r - 2.0 * mwd ) + r * y[3] / y[4] + np.power( r * y[3] / y[4],2.0) / 8.0 )
    ec8 = -ec2 * y[8] + 0.5*( 2.0 + r * ec2 ) * ( ( 8.0 * np.pi / 3.0) * r * np.power(y[4],2.0) * (pwd + EoS_RFMT_02(pwd)) + 1.0/6.0 * ( r - 2.0 * mwd) * np.power(y[3],2)) * np.exp(-nuwd) * np.power(r,2.0)
    ec9 = y[8] * ( -ec2 + 4.0 * ( 2.0 * np.pi * (pwd + EoS_RFMT_02(pwd)) * np.power(r,3.0) - mwd) / (np.power(r,2.0) * ( r - 2.0 * mwd)*ec2)) - 2.0 * y[7] / (y[0] + 4.0 * np.pi * np.power(r,3.0) * pwd) + (1.0/6.0) * np.exp(-nuwd) * np.power(r,3.0) * np.power(y[3],2.0) * (0.5 * ( r - 2.0 * mwd) * ec2-1.0/ ( r * ec2 ) ) + (8.0/3.0) * np.pi * np.power(r,3.0) * np.power(y[4],2.0) * (pwd + EoS_RFMT_02(pwd)) * np.exp(-nuwd) * (0.5 * r * ec2 + 1.0 / ( ( r - 2.0 * mwd ) * ec2 ) )
    ec10 = -y[10] * 2.0 * (4.0 * np.pi * np.power(r,3.0) * pwd + mwd ) / ( r * ( r - 2.0 * mwd) )
    ec11 = y[10] * (0.5 / ( r * ( mwd + 4.0 * np.pi * np.power(r,3.0) * pwd ) ) * ( 8.0 * np.pi * np.power(r,3.0) * (pwd + EoS_RFMT_02(pwd)) - 4.0 *mwd ) - ec2 ) - 2.0 * y[9] / ( mwd + 4.0 * np.pi * np.power(r,3.0) * pwd)
    return array([ec1,ec3,ec2,ec5,ec4,ec6,ec7,ec8,ec9,ec10,ec11])

#----Keplerian sequence--------------------------   
def OmegaK(Mwd,Rwd,x,jwd,qwd):
    F = ( 15.0 / 32.0 ) * ( ( 1.0 / np.power(x,3) ) - 2.0 ) * np.log( 1.0 / ( 1.0 - 2.0 * x ) )
    F1 = np.power(x,1.5)
    F4 = (48.0*np.power(x,7.0) - 80.0 * np.power(x,6.0) + 4.0*np.power(x,5.0) - 18.0 * np.power(x,4.0) + 40.0*np.power(x,3.0) + 10.0*np.power(x,2.0) + 15.0*x - 15.0 ) / ( 1.0 - 2.0 * x )
    F2 =  F4 / np.power(4.0*x,2.0) + F
    F3 =  ( 5.0 / np.power(4.0*x,2.0) ) * ( 6.0 * np.power(x,4.0) - 8.0 * np.power(x,3.0) - 2.0 * np.power(x,2) - 3.0 * x + 3.0 ) / ( 1.0 - 2.0 * x ) - F
    F6 = 1.0 + ( -jwd * F1 + np.power(jwd,2.0) * F2 + qwd * F3 )
    if F6<0.0:
      F6=1.0 - ( -jwd * F1 + np.power(jwd,2.0) * F2 + qwd * F3)
    return np.sqrt(Mwd/np.power(Rwd,3.0))*(F6)

def Q22(x):
    if x>1e3:
      A=168.0/(85.0*np.power(x,15.0))+128.0/(65.0*np.power(x,13.0))+280.0/(143.0*np.power(x,11.0))+64.0/(33.0*np.power(x,9.0))+40.0/(21.0*np.power(x,7.0))+64.0/(33.0*np.power(x,5.0))+8.0/(5.0*np.power(x,3.0))
    else:
      A=(np.power(x,2.0)-1)*(3.0/2.0)*(np.log(x+1.0)-np.log(x-1.0))-(3.0*np.power(x,3.0)-5.0*x)/(np.power(x,2.0)-1.0)
    return A

def Q12(x):
    if x>1e3:
      A=866483.0/(3734016.0*np.power(x,15.0))+95923.0/(384384.0*np.power(x,13.0))+52069.0/(192192.0*np.power(x,11.0))+2749.0/(9240.0*np.power(x,9.0))+139.0/(420.0*np.power(x,7.0))+13.0/(35.0*np.power(x,5.0))+2.0/(5.0*np.power(x,3.0))
    else:
      A=(3.0/2.0)*x*np.sqrt(np.power(x,2.0)-1)*(np.log(x-1.0)-np.log(x+1.0))+(3.0*np.power(x,2.0)-2.0)/(np.sqrt(np.power(x,2.0)-1.0))
    return A

def Awd(jwd,Mwd,Rwd,h2h,h2p,v2h,v2p):
    A1 = np.array ( [ [ -Q22( Rwd/Mwd - 1.0 ) , h2h ] , [ -2.0 * ( Mwd/Rwd ) * Q12( Rwd/Mwd - 1.0 ) / np.sqrt ( 1.0 - 2.0 * Mwd / Rwd ) , v2h ] ] )
    A2 = np.array ( [ np.power( jwd / Rwd**2,2.0) * (Rwd/Mwd+1.0) - h2p, - np.power(jwd/Rwd**2,2.0) - v2p ] )
    return  np.linalg.solve(A1, A2)

"""
def Awd(jwd,Mwd,Rwd,h2h,h2p,v2h,v2p):
    A1=(-(np.power(jwd,2.0)/np.power(Rwd,4.0))*(1.0+(Rwd/Mwd+1.0)*v2h/h2h)-v2p+h2p*v2h/h2h)/(v2h*Q22(Rwd/Mwd-1.0)/h2h-2.0*(Mwd/Rwd)*Q12(Rwd/Mwd-1.0)/np.sqrt(1.0-2.0*Mwd/Rwd))
    A2=(A1*Q22(Rwd/Mwd-1.0)+np.power(jwd,2.0)*np.power(1.0/Rwd,4.0)*(Rwd/Mwd+1.0)-h2p)/h2h
    return array([A1,A2]) 
"""


Static=ode(TOV).set_integrator('dopri5',atol=1e-9)
rot_HT01=ode(TOV_HT_02).set_integrator('lsoda',method='bdf',atol=1e-9)
test=ode(TOV_HT).set_integrator('dopri5',atol=1e-9)


#----Loop to calculate Static sequence--------
rf=0.1
r0=1e-7
def StaticSeq(y0,drr):
    r0 = drr

    Static.set_initial_value(y0,r0)
    rhons = EoS_RFMT_02( Static.y[1] ) *Sigma02

    while Static.successful() and Static.t<rf and rhons > 1e3 and Static.y[1]>0.0:
        Static.integrate(Static.t+drr)
        rhons = EoS_RFMT_02( Static.y[1] ) *Sigma02

    mstar  = Static.y[0]
    rstar  = Static.t
    nuc    = np.log( 1.0 - 2.0 * Static.y[0] / Static.t ) - Static.y[2]
    Jstar  = np.power( rstar , 4.0 ) * Static.y[3] / 6.0
    omegastar = Static.y[4] + 2.0 * Jstar / np.power(rstar,3.0)
      
    m2i = (1.0-2.0*y0[0]/r0) * np.exp(-nustar) * (4.0 * np.pi / 15.0 ) * ( EoS_RFMT_02(y0[1]) + y0[1] ) * ( 2.0 + np.power(10.0,DeDpF ( np.log10(y0[1]) )) ) * np.power(r0,5.0)
    p0i = 1.0 / 3.0 * np.power(r0,2.0) * ( 1.0 - 2.0 * y0[0] / r0 ) * np.exp(-nustar)  

    return [mstar,rstar,nustar,Jstar,omegastar,m2i,p0i]

#----Loop to calculate slow rotating configuration (HT: first order equations)-------
#return
#0: mass ; 1: radius ; 2: nuS ; 3: Angular momentum ; 4: Angular Velocity
# 6....: funtions at the surface 
def MassRadius_01(y0, Rs2):
    drM1 = Rs2 * 5e-4
    rot_HT01.set_initial_value(y0,r0)
    while rot_HT01.successful() and rot_HT01.t<rf and EoS_RFMT_02(rot_HT01.y[1])*Sigma02>1e3 and rot_HT01.y[1]>0.0:
        rot_HT01.integrate(rot_HT01.t+drM1)
    mstar = rot_HT01.y[0]
    rstar = rot_HT01.t
    nustar = np.log( 1.0 - 2.0 * mstar / rstar ) - rot_HT01.y[2]
    Jstar = np.power(rstar,4.0) * rot_HT01.y[3] / 6.0
    omegastar = rot_HT01.y[4] + 2.0 * Jstar / np.power(rstar,3.0)
    omebar = rot_HT01.y[4]; fstar = rot_HT01.y[3]; 
    m0star = rot_HT01.y[5]; p0star = rot_HT01.y[6];
    return [mstar,rstar,nustar,Jstar,omegastar,omebar,fstar,m0star,p0star,rot_HT01.y[1]]

#----Loop to calculate slow rotating configuration-------
#return
#0: mass ; 1: radius ; 2: nuS ; 3: Angular momentum ; 4: Angular Velocity
# 5: Quadrupole momentum ; 6....: funtions at the surface 
def MassRadius(y0, Rs2):
    drM1 = Rs2 * 1e-5
    test.set_initial_value(y0,r0)
    while test.successful() and test.t<rf and EoS_RFMT_02(test.y[1])*Sigma02>1e3 and test.y[1]>0.0:
        test.integrate(test.t+drM1)
    mstar = test.y[0]
    rstar = test.t
    nustar = np.log( 1.0 - 2.0 * test.y[0] / test.t ) - test.y[2]
    Jstar = np.power(test.t,4.0) * test.y[3] / 6.0
    omegastar = test.y[4] + 2.0 * Jstar / np.power(test.t,3.0)
    AAs = Awd( Jstar, mstar, rstar, test.y[10], test.y[8], test.y[9], test.y[7] )
    Qstar = np.power(Jstar,2.0) / mstar + (8.0/5.0) * np.power(mstar,3) * AAs[0]
    omebar = test.y[4]; fstar = test.y[3]; 
    m0star = test.y[5]; p0star = test.y[6];
    v2star = test.y[7]; h2star = test.y[8]
    return [mstar,rstar,nustar,Jstar,omegastar,Qstar,omebar,fstar,m0star,p0star,v2star,h2star,test.y[1],test.y[9],test.y[10],AAs[1],AAs[0]]

#----Loop to calculate slow rotating configuration with specific angular velocity-------
#return
#0: pressure ; 1: mass stattic ; 2: radius static ; 3: mass rotating ; 4: radius rotating
#5: Angular Momentum ; 6: Angular velocity ; 7: Quadrupole momentum ; 8: keplerian momentum....
def Omegacont(OmegaS,xcc,nuSS,omegaSS,mcSS, pcSS, r0, Rs2):
    mm0 = ( 4.0 / 3.0 ) * np.pi * r0**3 * EoS_RFMT_02(xcc)
    MR = MassRadius( [ mm0, xcc, nuSS, 0.0, OmegaS/omegaSS, mcSS*np.power(OmegaS/omegaSS,2.0), pcSS*np.power(OmegaS/omegaSS,2.0), 2.0 * np.pi * pcSS * np.power(OmegaS*r0/omegaSS,2.0) * ( xcc + EoS_RFMT_02(xcc) ) - 2.0 * np.pi * (xcc+EoS_RFMT_02(xcc)/3.0) * np.power(r0,4.0), np.power(r0,2.0), -2.0 * np.pi * (xcc+EoS_RFMT_02(xcc)/3.0) * np.power(r0,4.0), np.power(r0,2.0) ] , Rs2)
    RR = MR[1] * (1.0 + ( ( MR[1] / MR[0] - 2.0 ) / ( 1.0 + 4.0 * np.pi * np.power(MR[1],3.0) * MR[12] / MR[0] ) ) * ( MR[9] + ( 0.5 * ( MR[14] * MR[15] + MR[11] ) + 1.0 / 6.0 * ( np.power(MR[1],3.0) * np.power(MR[6],2.0) / ( MR[1] - 2.0 * MR[2] ) ) ) ) )
    MM = MR[0] + MR[8] + np.power(MR[3],2.0) / np.power(MR[1],3.0)
   # MR1 = MassRadius_01( [ mm0, xcc, nuSS, 0.0, OmegaS/omegaSS, mcSS*np.power(OmegaS/omegaSS,2.0), pcSS*np.power(OmegaS/omegaSS,2.0) ], Rs2)
    #MM1 = MR1[0] + MR1[7] + np.power(MR1[3],2.0) / np.power(MR1[1],3.0)
    OmegaKK = OmegaK( MM , RR, MR[0]/RR,  MR[3]/np.power(MR[0],2.0),0.0*MR[5]/np.power(MR[0],3.0) )
    return (xcc,MR[0],MR[1],MM,RR,MR[3],MR[4],MR[5],OmegaKK,MR[11],MR[15],MR[16])



dr=1e-7
#rhoc = arange(6.5,rhof,0.1)
rhoc=[rhof]
xc = EoS_RFMT(np.power(10.0,rhoc)/Sigma02)
#---Build static sequence-------
"""
MMsta=[]; RRsta=[]; nuSur=[]; omegaSur=[]; OmeK=[];m2ic=[];p0ic=[];
i=0
for xcc in xc:
   m00 = ( 4.0 / 3.0 ) * np.pi * (dr*1e-3)**3 * np.power(10.0,rhoc[i])/Sigma02
   M1=StaticSeq([m00,xcc,0,0,1],dr)
   MMsta.append(Mdim*M1[0])
   RRsta.append(M1[1])
   print rhoc[i] , MMsta[-1],RRsta[-1]
   dr = RRsta[-1]*1e-4
   print dr
   i+=1

exit()
"""
Kep=[]
KepNew=[]

for i in range(len(xc)):
    m00 = ( 4.0 / 3.0 ) * np.pi * (r0)**3 * np.power(10.0,rhoc[i])/Sigma02
    M1=StaticSeq([m00,xc[i],0,0,1],dr)
    print M1
    OOp=Omegacont(OmegaK( M1[0], M1[1], M1[0]/M1[1] , 0 , 0 ), xc[i] , M1[2] , M1[4] , M1[5] , M1[6] , r0  , M1[1])
    print  OOp[6],OOp[8]
    KepNew.append(OOp)
    cut=0
    while np.abs(OOp[8]-OOp[6])>1e-3 and cut<80:
        OOp=Omegacont(OOp[8],xc[i],M1[2],M1[4],M1[5],M1[6],r0,M1[1])
        cut=cut+1
        print OOp[6],OOp[8], M1[0]*Mdim,OOp[1]*Mdim, OOp[3]*Mdim
    OOpnew = [ rhoc[i] ]
    for j in range(len(OOp)):
        OOpnew.append(OOp[j])
    Kep.append(OOpnew)
    dr_2 = OOp[2]/5000.0
    if dr_2<dr:
        dr = dr_2
    np.savetxt('KepSeqWD_RMTF_2.dat',Kep,fmt='%1.4e')

exit()




