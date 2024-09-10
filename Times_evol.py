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
import ions as ions
import ion_electron as ie

import argparse as ap

parser=ap.ArgumentParser()
parser.add_argument("file_EoS", help="input EoS", type=str)
parser.add_argument("file_Mcont", help="input mass const", type=str)
parser.add_argument("file_kep", help="input kep seq", type=str)
parser.add_argument("file_SecInst", help="input sec", type=str)
parser.add_argument("file_ignition", help="T ignition", type=str)
parser.add_argument("--mass", help="mass WD", type=float)
args=parser.parse_args()

Masswd0 = args.mass

#----Constants------
hbar=phys.hbar
h=hbar*(2.0*np.pi)
c=phys.c
G=phys.G
Msun=phys.Msun
Rsun = 6.95e10
mu = phys.mu
yr = 60.0*60.0*24.0*364.0
qe = phys.qe 
mu = phys.mu
sigmaSB =  phys.sigmaSB
kappa =  phys.kappa 
ev2erg = phys.ev2erg 
me_mev=phys.me
mevtoerg = phys.mevtoerg
me = me_mev *mevtoerg / c**2

arad = 4.0*sigmaSB/c
avo = 1.0/mu
h = 2.5e0 * hbar * np.pi
Rbohr = hbar*hbar/(me * qe * qe)

mpl = np.sqrt(hbar * c / G) # planck mass
Rdim = np.power( mpl / me , 2.0) * hbar / (mpl * c)
Sigma02 = np.power( c / Rdim,2.0) / G

Sigma=np.power(me,4.0)*np.power(c,3.0)/(8.0*np.power(np.pi,2.0)*np.power(hbar,3.0))
SigmaP = c**2 * Sigma
MA=2.0
#Sigma02=(mu*MA*np.power(me*c/hbar,3))/(3*np.power(np.pi,2))
#Rdim=c/np.sqrt(Sigma02*G)
Cgrav = G * Msun / c**2
#Rdim = 0.01*Rsun
#Sigma02 = c**2 / ( Rdim**2 * G)
Mdim = Rdim / Cgrav
Jdim = G / c**3

Tdim = np.power(c**5 / ( 16.0**2 * np.pi**2 * G * sigmaSB * Rsun**2) ,1.0/4.0)
rhobeta=np.log10(3.97e10)

#br=c/np.sqrt(Sigma*G)
#Mdim=(br*np.power(c,2.0)/G)/Msun

#---read EoS----------
rhoEoS,xeEoS,PEoS,PsatEoS,chip_EoS=np.loadtxt(args.file_EoS,usecols=(0,1,2,3,4),unpack=True)
rhoi=np.log10(rhoEoS[0]); rhof=np.log10(rhoEoS[-1])
xemin=xeEoS[0]; xemax=xeEoS[-1]

#---read keplerian and static sequence-----
rhowd_kep,pwd_kep,Mstawd_kep,Rstawd_kep,Mrotwd_kep,Rrotwd_kep,Jwd_kep,Omegawd_kep,Qwd_kep,Omegakwd_kep=np.loadtxt(args.file_kep,usecols=(0,1,2,3,4,5,6,7,8,9),unpack=True)

#---read secular instability sequence-----
rhowd_seq,pwd_seq,Mstawd_seq,Rstawd_seq,Mrotwd_seq,Rrotwd_seq,Jwd_seq,Omegawd_seq,Qwd_seq,Omegakwd_seq=np.loadtxt(args.file_SecInst,usecols=(0,1,2,3,4,5,6,7,8,9),unpack=True)

#---read ignition line-------
rho_ign,T_ign = np.loadtxt(args.file_ignition,usecols=(0,1),unpack=True)
TauCC=interp1d(T_ign,rho_ign,)

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

NNseq=len(rhowd_seq)
for i in range(len(rhowd_seq)):
    if rhowd_seq[i]>rhobeta:
        NNseq=i
        break;

MMseq = Mrotwd_seq[NNseq]*Mdim

print "#############################################################"
print "Maximun static mass: ", Mmax_sta*Mdim, " Msun\t Density ( [g / cm^3]):", np.power(10.0,rhowd_kep[Nsta]), "\tRadius: ", Rstawd_kep[Nsta]*Rdim, " [cm]"
print "Maximun rotating mass: ", Mmax_rot*Mdim, " Msun\t Density ([g / cm^3]):", np.power(10.0,rhowd_kep[Nrot]), "\tRadius: ", Rrotwd_kep[Nrot]*Rdim, " [cm]"

print "#############################################################"

Kep_Omega=interp1d( [Mrotwd_kep[j]*Mdim for j in range(Nrot+1)] , [Omegawd_kep[j] for j in range(Nrot+1)],kind='linear' )
Kep_rho=interp1d( [Mrotwd_kep[j]*Mdim for j in range(Nrot+1)] , [rhowd_kep[j] for j in range(Nrot+1)],kind='linear' )
Kep_Req=interp1d( [ Mrotwd_kep[j]*Mdim for j in range(Nrot+1)], [ Rrotwd_kep[j] for j in range(Nrot+1)], kind='linear' )
Kep_j=interp1d( [ Mrotwd_kep[j]*Mdim for j in range(Nrot+1)], [ Jwd_kep[j] for j in range(Nrot+1)], kind='linear' )

Sta_rho=interp1d( [Mstawd_kep[j]*Mdim for j in range(Nsta+1)] , [rhowd_kep[j] for j in range(Nsta+1)],kind='linear' )
Seq_rho=interp1d( [Mrotwd_seq[j]*Mdim for j in range(NNseq)] , [rhowd_seq[j] for j in range(NNseq)],kind='linear' )

Omegawd_0= Kep_Omega(Masswd0)
Reqwd_0= Kep_Req(Masswd0)
jwd_0 = Kep_j(Masswd0)
rhowd_0 = Kep_rho(Masswd0)


print "Mass0 = ", Masswd0, "\tOmega0 = ", Omegawd_0*c/Rdim, "\t Req0 = ", Reqwd_0*Rdim
print "tau_dyn", 1.0/(24.0*np.pi*G*np.power(10.0, rhowd_0))
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
rhowd,pwd,Mstawd,Rstawd,Mrotwd,Rrotwd_eq,Rrotwd_pol,Jwd,Omegawd,Qwd,Omegakwd=np.loadtxt(args.file_Mcont,usecols=(0,1,2,3,4,5,6,7,8,9,10),unpack=True)

print np.power(10.0,min(rhowd)), np.power(10.0,max(rhowd))
print "rho_c\t Mass_Sta\t Rsta\t Req\t Rpol \t J\t Omega\t Period\t DeltaM"
print "######################################################"
for i in range(len(rhowd)):
    print rhowd[i],"\t", Mstawd[i]*Mdim,"\t",  Rstawd[i]*Rdim/1e5,"\t",  Rrotwd_eq[i]*Rdim/1e5,"\t",  Rrotwd_pol[i]*Rdim/1e5,"\t",  Jwd[i]*Rdim**2,"\t", (Omegawd[i]*c/Rdim),"\t",  2.0*np.pi/(Omegawd[i]*c/Rdim),"\t",  Mrotwd[i]/Mstawd[i]-1.0,"\t", Mrotwd[i]*Mdim 

#exit()

NN = len(Jwd)
#print NN
if args.mass<MMseq:
    if args.mass<Mmax_sta*Mdim:
        rhobeta =Sta_rho(args.mass)
    else:
        rhobeta = Seq_rho(args.mass)
        for i in range(NN):
            if rhowd[i]>rhobeta:
                NN = i-1
                break;
#else:
#    rhobeta=rhowd[-1]

#print NN, rhobeta
#---Initial paramenter----------
Omegawd_0= Omegawd[0]
Reqwd_0 = (1.0 / 3.0) * (Rrotwd_pol[0]+2.0*Rrotwd_eq[0])
jwd_0 = Jwd[0]
rhowd_0 = rhowd[0]

Jadim = 1e50
JJdim = Jadim * Jdim  / Rdim**2
NNdat = len(rhowd)
jmin = min(Jwd)
jmin/=JJdim
jmax = max(Jwd)
jmax /= JJdim
jwd_0 /= JJdim
#jwd_0 = jmax

print "jwd_0 =" , jwd_0," 1e50 g cm s^-1"
print "Min ang: ",jmin,"\t Max ang: ", jmax
print "########################################################################"

DrhoDj_dats=fit_derivative( [ Jwd[i] / JJdim for i in range(NN) ] , [ rhowd[i] for i in range(NN)])
#---- DrhoDj------------
#rho_j=interp1d( [Jwd[j] / JJdim for j in range(NN)] , [rhowd[j] for j in range(NN)],kind='linear' )

"""
DrhoDj_dats=[]

for i in range(NN):
    jx =  Jwd[i] / JJdim
    if i==0:
        dx= (jx-Jwd[i+1]/JJdim)*0.01
        rhojx1 = Interp1D_ll(jx, [Jwd[j] / JJdim for j in range(NN)] , [rhowd[j] for j in range(NN)])
        rhojx2 = Interp1D_ll(jx-dx, [Jwd[j] / JJdim for j in range(NN)] , [rhowd[j] for j in range(NN)])
        Der = (  rhojx1 - rhojx2 ) / (dx)
    elif i==NN-1:
        dx= (Jwd[i-1]/JJdim-jx)*0.01
        rhojx1 = Interp1D_ll(jx, [Jwd[j] / JJdim for j in range(NN)] , [rhowd[j] for j in range(NN)])
        rhojx2 = Interp1D_ll(jx+dx, [Jwd[j] / JJdim for j in range(NN)] , [rhowd[j] for j in range(NN)])
        Der = (  rhojx2 - rhojx1 ) / (dx)
    else:
        dx= (Jwd[i+1]/JJdim-jx)*0.01
        rhojx1 = Interp1D_ll(jx-dx, [Jwd[j] / JJdim for j in range(NN)] , [rhowd[j] for j in range(NN)])
        rhojx2 = Interp1D_ll(jx+dx, [Jwd[j] / JJdim for j in range(NN)] , [rhowd[j] for j in range(NN)])
        Der = (  rhojx2 - rhojx1 ) / (dx)
    DrhoDj_dats.append(Der)
"""

#DRDj_dats=fit_derivative( Jwd / JJdim , Rrotwd / Reqwd_0)

Drho_dj=interp1d( [ Jwd[j] / JJdim for j in range(NN)] , [DrhoDj_dats[j] for j in range(NN)],kind='linear' )

Req_j=interp1d( [ Jwd[j] / JJdim for j in range(NN)], [ (1.0/3.0) * (Rrotwd_pol[j] + 2.0*Rrotwd_eq[j]) / Reqwd_0 for j in range(NN)], kind='linear' )
#DReq_dj = interp1d( [ Jwd[j] / JJdim for j in range(NN)], [ DRDj_dats[j]  for j in range(NN)], kind='linear' )

Omega_j=interp1d( [ Jwd[j] / JJdim for j in range(NN)] , [Omegawd[j] / Omegawd_0 for j in range(NN)],kind='linear' )


x=np.linspace(min(jmin,jmax),max(jmin,jmax), 1000)  
rhowd_2=[]
omegawd_2=[]
Rwd_2=[]
Drhodj_2=[]
for xi in x:
    rhoww = Interp1D_ll(xi, [Jwd[j] / JJdim for j in range(NN)] , [rhowd[j] for j in range(NN)])
    omegaww = Interp1D_ll(xi, [Jwd[j] / JJdim for j in range(NN)] , [ Omegawd[j] / Omegawd_0 for j in range(NN)])
    Rww = Interp1D_ll(xi, [Jwd[j] / JJdim for j in range(NN)] , [ (1.0/3.0) * (Rrotwd_pol[j] + 2.0*Rrotwd_eq[j])  / Reqwd_0 for j in range(NN)])
    omegawd_2.append(omegaww)
    rhowd_2.append(rhoww)
    Rwd_2.append(Rww)
    Drhodj_2.append( Interp1D_ll( xi, [ Jwd[j] / JJdim for j in range(NN)] , DrhoDj_dats )
)


"""
plt.subplot(1,3,1)
plt.plot(x,rhowd_2,'k:')
#plt.plot(x,rho_j(x),'k:')
plt.plot( Jwd/JJdim , rhowd,'o')
plt.xlabel(r'$j_{wd}$')
plt.ylabel(r'$\mathrm{log}\, (\rho\,)$')

plt.subplot(1,3,2)
plt.plot(x,Rwd_2,'k:')
#plt.plot(x,Req_j(x),'k:')
plt.plot( Jwd/JJdim , (1.0/3.0)*(Rrotwd_pol+2.0*Rrotwd_eq) / Reqwd_0,'o')
plt.xlabel(r'$j_{wd}$')
plt.ylabel(r'$R\, /\, R_{wd,0}$')


plt.subplot(1,3,3)
plt.plot(x,omegawd_2,'k:')
#plt.plot(x,Omega_j(x),'k:')
plt.plot( Jwd/JJdim , Omegawd / Omegawd_0,'o')
plt.xlabel(r'$j_{wd}$')
plt.ylabel(r'$\Omega\, /\, \Omega_{wd,0}$')

plt.show()

#plt.plot(x,Drho_dj(x),'k:')
plt.plot( [Jwd[i]/JJdim for i in range(NN)] , DrhoDj_dats,'o')
plt.plot( x, Drhodj_2, 'k-')
plt.xlabel(r'$j_{wd}$')
plt.show()

#plt.plot(x,DReq_dj(x),'k:')
#plt.plot( Jwd/JJdim , DRDj_dats,'o')
#plt.show()
"""

def Integration(xpoints, ypoints):
    Npoints = len(xpoints)
    Sum = 0.0
    for i in range(Npoints-1):
        Deltax = 0.5*(xpoints[i+1] - xpoints[i])
        ff = ypoints[i+1] + ypoints[i]
        Sum += ff*Deltax
    return Sum


#plt.semilogy( Jwd/JJdim , (  np.power(Omegawd_0,3.0) / np.power(Omegawd,3.0) ) * (np.power( Reqwd_0,6.0)  / np.power( (1.0/3.0)*(Rrotwd_pol+2.0*Rrotwd_eq),6.0) ), 'o') 
#plt.show()

x_B=np.linspace(6,10,10)
timescale= []
#FN = open('timesscales_lossangular_146.dat','w')
for xxB in x_B:
    Bs0 = np.power(10.0,xxB)
    Dimi = ( (2.0/(3.0*c**3))  * Bs0*Bs0 * np.power(Omegawd_0*c/Rdim,3.0) * np.power(Reqwd_0*Rdim,6.0) / Jadim ) * 1e6*yr
    II1 = Integration([ Jwd[j] / JJdim for j in range(NN)] , [ np.power( Omegawd_0 / Omegawd[j],3.0) * (Reqwd_0**6 / np.power( (1.0/3.0) * (Rrotwd_pol[j] + 2.0*Rrotwd_eq[j]),6.0))   for j in range(NN)])
    II2 = Integration([ Jwd[j] / JJdim for j in range(NN)] , [ np.power( Omegawd_0 / Omegawd[j],3.0) * (Reqwd_0**2 / np.power( (1.0/3.0) * (Rrotwd_pol[j] + 2.0*Rrotwd_eq[j]),2.0))   for j in range(NN)])
    timescale.append([xxB,-II1 / ( Dimi ),-II2 / ( Dimi )])
    #np.savetxt(FN,[[xxB, -II1/ ( Dimi ), -II2 / ( Dimi )]],fmt='%1.7e')
#FN.close()

print "Total time: ",  -II1 / ( Dimi )
print "Total time_flux: ",  -II2 / ( Dimi )

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

#-----Helmotz EoS----

#ionmax=2
xmass=[0.5,0.5]
aion=[12.0,16.0]
zion=[6.0,8.0]

imax=541 ; jmax=201
Ndats=imax*jmax

#Temperature range
tlo = 3.0e0 ; thi = 13.0e0
tstp  = (thi - tlo)/float(jmax-1)
tstpi = 1.0e0/tstp
#density range
dlo = -12.0e0 ; dhi = 15.0e0
dstp  = (dhi - dlo)/float(imax-1)
dstpi = 1.0e0/dstp

# ---- Read Table EoS ----
dats=[]
with open("../WD_EoS/helmholtz/helm_table.dat","r") as theFile:            
    for line in theFile:
           p1 = line.split('\n')
           p2 = p1[0].split(' ')
           line2=[]
           for pp in p2:
               if pp!='' and pp!=' ':
                  line2.append(float(pp))
           dats.append(line2)
  
#Free energy
T_grid=[] ;rho_grid=[] ; FE=[]

#0: free energy; 1: df_drho; 2: df_dt; 3: df_drhorho; 4: df_dtdt
#5: df_drhodt; 6: df_drhodrhodt; 7: df_drhodtdt; 8: df_drhodrhodtdt;
k=0

for j in range(jmax):
    T_grid.append(np.power(10.0,tlo+(j)*tstp))
    AA=[]
    for i in range(imax):
        rho_grid.append(np.power(10.0,dlo+(i)*dstp))
        AA.append([dats[k][0],dats[k][2],dats[k][4],dats[k][1],dats[k][3],dats[k][5],dats[k][6],dats[k][7],dats[k][8]])
        k+=1
    FE.append(AA)

dpr_e=[]
#Pressure derivaties
#0: dpr_df; 1: dpr_df_rho; 2: dpr_df_t; 3: dpr_dfdt; 
for j in range(jmax):
    AA=[]
    for i in range(imax):
        AA.append([dats[k][0],dats[k][2],dats[k][1],dats[k][3]])
        k+=1
    dpr_e.append(AA)

#electron chemical dats
eta_e=[]
#0: electron chemical; 1: deta_drho; 2: deta_dt; 3: deta_drhodt; 
for j in range(jmax):
   AA=[]
   for i in range(imax):
      AA.append(dats[k])
      k+=1
   eta_e.append(AA)

#number density dats
xf=[]
#0: number density; 1: dxf_drho; 2: dxf_dt; 3: dxf_drhodt; 
for j in range(jmax):
   AA=[]
   for i in range(imax):
      AA.append(dats[k])
      k+=1
   xf.append(AA)

deltaT=[] ; deltaT2=[]
deltaT_inv=[]; deltaT2_inv=[]; deltaT3_inv=[]

deltarho=[] ; deltarho2=[]
deltarho_inv=[]; deltarho2_inv=[]; deltarho3_inv=[]

for j in range(jmax-1):
    dth  = T_grid[j+1]-T_grid[j]
    dt2  = dth * dth; dti  = 1.0e0/dth
    dt2i  = 1.0e0/dt2; dt3i  = dt2i*dti
    deltaT.append(dth)
    deltaT2.append(dt2)
    deltaT_inv.append(dti)
    deltaT2_inv.append(dt2i)
    deltaT3_inv.append(dt3i)

for i in range(imax-1):
    dd  = rho_grid[i+1]-rho_grid[i]
    dd2  = dd * dd; ddi  = 1.0e0/dd
    dd2i  = 1.0e0/dd2; dd3i  = dd2i*ddi
    deltarho.append(dd)
    deltarho2.append(dd2)
    deltarho_inv.append(ddi)
    deltarho2_inv.append(dd2i)
    deltarho3_inv.append(dd3i)

#--function for interpolation------
# quintic hermite polynomial statement functions
# psi0 and its derivatives
def psi0(z):
    return np.power(z,3.0) * ( z * (-6.0e0*z + 15.0e0) -10.0e0) + 1.0e0
      
def dpsi0(z):
    return np.power(z,2.0) * ( z * (-30.0e0*z + 60.0e0) - 30.0e0)

def ddpsi0(z):
    return z* ( z*( -120.0e0*z + 180.0e0) -60.0e0)

# psi1 and its derivatives
def psi1(z):
    return z* ( np.power(z,2.0) * ( z * (-3.0e0*z + 8.0e0) - 6.0e0) + 1.0e0)
     
def dpsi1(z):
    return z*z * ( z * (-15.0e0*z + 32.0e0) - 18.0e0) +1.0e0
      
def ddpsi1(z):
    return z * (z * (-60.0e0*z + 96.0e0) -36.0e0)

# psi2  and its derivatives
def psi2(z):
    return 0.5e0*z*z*( z* ( z * (-z + 3.0e0) - 3.0e0) + 1.0e0)

def dpsi2(z):
    return 0.5e0*z*( z*(z*(-5.0e0*z + 12.0e0) - 9.0e0) + 2.0e0)

def ddpsi2(z):
    return 0.5e0*(z*( z * (-20.0e0*z + 36.0e0) - 18.0e0) + 2.0e0)

#biquintic hermite polynomial statement function
def h5(fi,wt,wmt,wd,wmd):
    A1=(fi[1]*wd[0] + fi[2]*wmd[0])*wt[0] + (fi[3]*wd[0] + fi[4]*wmd[0])*wmt[0] 
    A2=(fi[5]*wd[0] + fi[6]*wmd[0])*wt[1] + (fi[7]*wd[0] + fi[8]*wmd[0])*wmt[1] 
    A3=(fi[9]*wd[0] + fi[10]*wmd[0])*wt[2] + (fi[11]*wd[0] + fi[12]*wmd[0])*wmt[2] 

    B1=(fi[13]*wd[1] + fi[14]*wmd[1])*wt[0] + (fi[15]*wd[1] + fi[16]*wmd[1])*wmt[0]  
    B2=(fi[17]*wd[2] + fi[18]*wmd[2])*wt[0] + (fi[19]*wd[2] + fi[20]*wmd[2])*wmt[0]    
    
    C1=(fi[21]*wd[1] + fi[22]*wmd[1])*wt[1] + (fi[23]*wd[1] + fi[24]*wmd[1])*wmt[1]  
    C2=(fi[25]*wd[2] + fi[26]*wmd[2])*wt[1] + (fi[27]*wd[2] + fi[28]*wmd[2])*wmt[1]  
    
    D1=(fi[29]*wd[1] + fi[30]*wmd[1])*wt[2] + (fi[31]*wd[1] + fi[32]*wmd[1])*wmt[2] 
    D2=(fi[33]*wd[2] + fi[34]*wmd[2])*wt[2] + (fi[35]*wd[2] + fi[36]*wmd[2])*wmt[2] 
    return A1+A2+A3+B1+B2+C1+C2+D1+D2

#cubic hermite polynomial statement functions
# psi0 & derivatives
def xpsi0(z):
    return z * z * (2.0e0*z - 3.0e0) + 1.0

def xdpsi0(z):
    return z * (6.0e0*z - 6.0e0)

# psi1 & derivatives
def xpsi1(z):
    return z * ( z * (z - 2.0e0) + 1.0e0)

def xdpsi1(z):
    return z * (3.0e0*z - 4.0e0) + 1.0e0

# bicubic hermite polynomial statement function
def h3(fi,wt,wmt,wd,wmd):
  A1=(fi[1]*wd[0] + fi[2]*wmd[0])*wt[0] + (fi[3]*wd[0] + fi[4]*wmd[0])*wmt[0] 
  A2=(fi[5]*wd[0] + fi[6]*wmd[0])*wt[1] + (fi[7]*wd[0] + fi[8]*wmd[0])*wmt[1] 
  
  B1=(fi[9]*wd[1] + fi[10]*wmd[1])*wt[0] + (fi[11]*wd[1] + fi[12]*wmd[1])*wmt[0]  
    
  C1=(fi[13]*wd[1] + fi[14]*wmd[1])*wt[1] + (fi[15]*wd[1] + fi[16]*wmd[1])*wmt[1]  
  return A1+A2+B1+C1


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


#--- Heat Capacity ---
def cv_gas(rho, T, agas, xxgas,zgas):
    #average atomic weight and charge
    ration=0.0; XmassT=0.0; ZionT=0.0; AionT=0.0
    for i in range(len(agas)):
        ration+=xxgas[i]/agas[i]
        XmassT+=xxgas[i]*zgas[i]/agas[i]
    abar = 1.0e0/ration
    zbar   = abar * XmassT
#print "abar= ", abar, " zbar= ", zbar
    ytot1 = 1.0e0/abar
    ye    = max(1.0e-16,ytot1 * zbar) 
#ye=0.5
#print "Ye=", ye

    deni=1.0/rho
    tempi=1.0/T
    kT= kappa * T
    kT_inv= 1.0 / kT

#   print "log Temperature:", np.log10(T), "log rho: ", np.log10(rho)
#radiation:
    prad=(1.0/3.0) * arad * np.power(T,4.0)
    dprad_dT = 4.0 * prad * tempi

    erad=3.0*prad * deni
    derad_dT = 3.0 * dprad_dT * deni
    
    srad = ( prad * deni + erad) * tempi

#ion:
    xini = rho * avo * ytot1
    dxini_dd = avo * ytot1
    dxini_da = -xini * ytot1

    pion = xini * kT
    dpion_dT = pion * tempi
 
    eion = 1.5 * pion * deni
    deion_dT = 1.5 * dpion_dT * deni

# sackur-tetrode equation for the ion entropy of a single ideal gas characterized by abar
    x  = abar*abar*np.sqrt(abar) * deni/avo
    sioncon = (2.0 * np.pi * mu * kappa) / (h*h)
    s = sioncon * T
    z = x * s * np.sqrt(s)
    y = np.log(z)

#        y       = 1.0d0/(abar*kt)
#        yy      = y * sqrt(y)
#        z       = xni * sifac * yy
#        etaion  = log(z)
    kergavo=kappa*avo
    sion = (pion * deni + eion) * tempi + kergavo * ytot1 * y

#electron-positron:

#assume complete ionization
    xnem    = xini * zbar
    din = ye*rho
    if T>T_grid[-1]:
       print "Temperature too hot"
       return
    if T<T_grid[0]:
       print T, T_grid[0]
       print "Temperature too cold"
       return
    if din<rho_grid[0]:
       print "density too low"
       return
    if din>rho_grid[-1]:
       print "density too high"
       return

# hash locate this temperature and density
    jat = int((np.log10(T) - tlo)*tstpi)
    jat = max(0,min(jat,jmax-1))
    iat = int((np.log10(din) - dlo)*dstpi) 
    iat = max(0,min(iat,imax-1))
  
# access the table locations only once
    fi=np.zeros(37)   
    for i in range(int(36.0/4.0)):
        fi[4*i+1] = FE[jat][iat][i]
        fi[4*i+2] = FE[jat][iat+1][i]
        fi[4*i+3] = FE[jat+1][iat][i]
        fi[4*i+4] = FE[jat+1][iat+1][i]
 #  print din
 #  print T_grid[jat], rho_grid[iat], rho_grid[iat+1]
   
   
# various differences
    xt  = max( (T- T_grid[jat])*deltaT_inv[jat], 0.0e0)
    xd  = max( (din - rho_grid[iat])*deltarho_inv[iat], 0.0e0)
    mxt = 1.0e0 - xt
    mxd = 1.0e0 - xd

# the six density and six temperature basis functions
    si0t =   psi0(xt)
    si1t =   psi1(xt)*deltaT[jat]
    si2t =   psi2(xt)*deltaT2[jat]

    si0mt =  psi0(mxt)
    si1mt = -psi1(mxt)*deltaT[jat]
    si2mt =  psi2(mxt)*deltaT2[jat]

    si0d =   psi0(xd)
    si1d =   psi1(xd)*deltarho[iat]
    si2d =   psi2(xd)*deltarho2[iat]

    si0md =  psi0(mxd)
    si1md = -psi1(mxd)*deltarho[iat]
    si2md =  psi2(mxd)*deltarho2[iat]

# derivatives of the weight functions
    dsi0t =   dpsi0(xt)*deltaT_inv[jat]
    dsi1t =   dpsi1(xt)
    dsi2t =   dpsi2(xt)*deltaT[jat]

    dsi0mt = -dpsi0(mxt)*deltaT_inv[jat]
    dsi1mt =  dpsi1(mxt)
    dsi2mt = -dpsi2(mxt)*deltaT[jat]

    dsi0d =   dpsi0(xd)*deltarho_inv[iat]
    dsi1d =   dpsi1(xd)
    dsi2d =   dpsi2(xd)*deltarho[iat]

    dsi0md = -dpsi0(mxd)*deltarho_inv[iat]
    dsi1md =  dpsi1(mxd)
    dsi2md = -dpsi2(mxd)*deltarho[iat]

# second derivatives of the weight functions
    ddsi0t =   ddpsi0(xt)*deltaT2_inv[jat]
    ddsi1t =   ddpsi1(xt)*deltaT_inv[jat]
    ddsi2t =   ddpsi2(xt)

    ddsi0mt =  ddpsi0(mxt)*deltaT2_inv[jat]
    ddsi1mt = -ddpsi1(mxt)*deltaT_inv[jat]
    ddsi2mt =  ddpsi2(mxt)

#        ddsi0d =   ddpsi0(xd)*deltarho2_inv[iat]
#        ddsi1d =   ddpsi1(xd)*deltarho_inv[iat]
#        ddsi2d =   ddpsi2(xd)

#        ddsi0md =  ddpsi0(mxd)*deltarhoi_inv[iat]
#        ddsi1md = -ddpsi1(mxd)*deltarho_inv[iat]
#        ddsi2md =  ddpsi2(mxd)

# the free energy
    free  = h5(fi, [si0t,si1t,si2t],[si0mt,si1mt,si2mt],[si0d,si1d,si2d],[si0md,si1md,si2md])
# derivative with respect to density
    df_d  = h5(fi,[si0t,si1t,si2t],[si0mt,si1mt,si2mt],[dsi0d,dsi1d,dsi2d],[dsi0md,dsi1md,dsi2md])
# derivative with respect to temperature
    df_t = h5(fi,[dsi0t,dsi1t,dsi2t],[dsi0mt,dsi1mt,dsi2mt],[si0d,si1d,si2d],[si0md,si1md,si2md])   
   
# derivative with respect to density**2
#  df_dd = h5(fi,[si0t,si1t,si2t],[si0mt,si1mt,si2mt],[ddsi0d,ddsi1d,ddsi2d],[ddsi0md,ddsi1md,ddsi2md]

# derivative with respect to temperature**2
    df_tt = h5(fi,[ddsi0t,ddsi1t,ddsi2t],[ddsi0mt,ddsi1mt,ddsi2mt],[si0d,si1d,si2d],[si0md,si1md,si2md])
             
#derivative with respect to temperature and density
    df_dt = h5(fi,[dsi0t,dsi1t,dsi2t],[dsi0mt,dsi1mt,dsi2mt],[dsi0d,dsi1d,dsi2d],[dsi0md,dsi1md,dsi2md])
   
# the desired electron-positron thermodynamic quantities

    x       = din * din
    pele    = x * df_d

    x       = ye * ye
    sele    = -df_t * ye
    dsep_dt  = -df_tt * ye

    eele    = ye*free + T * sele
    deep_dt  = T * dsep_dt

# coulomb section:
# uniform background corrections only from yakovlev & shalybkov 1989
# lami: average ion seperation
# plasg: plasma coupling parameter

    a1 = -0.898004e0
    b1 =  0.96786e0
    c1 =  0.220703e0
    d1 = -0.86097e0
    e1 =  2.5269e0
    a2 =  0.29561e0
    b2 =  1.9885e0
    c2 =  0.288675e0
    third =  1.0e0/3.0e0
    esqu  =  qe * qe

    z = (4.0/3.0) * np.pi
    s = z * xini
    ds_dd = z * dxini_dd
    ds_da = z *dxini_da

    lami = np.power( 1.0/s ,1.0/3.0)
    inv_lami = 1.0 / lami
    z = -1.0/3.0 * lami
    lami_dd = -(1.0/3.0)* lami * ds_dd / s
    lami_da = -(1.0/3.0)* lami * ds_da / s

    plasg = zbar*zbar*esqu*kT_inv*inv_lami
    z = - plasg * inv_lami
    plasg_dd  = z * lami_dd
    plasg_da  = z * lami_da
    plasg_dT  = -plasg*kT_inv * kappa
    plasg_dz  = 2.0e0 * plasg/zbar
    
# yakovlev & shalybkov 1989 equations 82, 85, 86, 87
    if (plasg >= 1.0):
        x = np.power(plasg,0.25)
        y = avo * ytot1 * kappa
        ecoul = y * tempi * (a1*plasg + b1*x + c1/x + d1)
        pcoul = third * deni * ecoul
        scoul = -y * (3.0e0*b1*x - 5.0e0*c1/x + d1 * (np.log(plasg) - 1.0e0) - e1)

        y  = avo*ytot1*kT*(a1 + 0.25e0/plasg*(b1*x - c1/x))
        decoul_dt = y * plasg_dT + ecoul/T



# yakovlev & shalybkov 1989 equations 102, 103, 104
    elif (plasg < 1.0):
        x = plasg*np.sqrt(plasg)
        y = plasg**b2
        z = c2 * x - third * a2 * y
        pcoul = -pion * z
        ecoul = 3.0e0 * pcoul/rho
        scoul = -avo/abar*kappa*(c2*x -a2*(b2-1.0e0)/b2*y)

        s  = 1.5e0*c2*x/plasg - third*a2*b2*y/plasg
        dpcoul_dt = -dpion_dT*z - pion*s*plasg_dT
       
        s        = 3.0e0/rho
        decoul_dt = s * dpcoul_dt
       

    x   = prad + pion + pele + pcoul
    y   = erad + eion + eele + ecoul
    z   = srad + sion + sele + scoul

    
    if (x < 0.0 or y < 0.0):
       print 'coulomb corrections are causing a negative pressure'
       print 'setting all coulomb corrections to zero'

       pcoul    = 0.0e0
       ecoul    = 0.0e0
       decouldt = 0.0e0
       scoul    = 0.0e0
 
    pcoul    = 0.0e0
    ecoul    = 0.0e0
    decouldt = 0.0e0
    scoul    = 0.0e0

    pgas   =  pion + pele + pcoul
    egas   =  eion + eele + ecoul
    sgas  =  sion + sele + scoul

   # degasdt = deion_dT + decoul_dt + deep_dt 
    degasdt = deion_dT + deep_dt #+ decoul_dt
    pres= pgas + prad
    ener= egas + erad
    entr= sgas + srad

    denerdt = derad_dT + degasdt 
    #denerdt =  degasdt
    cv_gas = abar * denerdt / (kappa*avo) + ions.Cvii(abar,zbar,rho,T) + ie.Cvie(abar,zbar,rho,T)
    return cv_gas

#print cv_gas(1e6,1e8,aion,xmass,zion)
cv_dat=[]
Tpon_dat=np.linspace(4.0,10.0,200)
for ti in Tpon_dat:
    cvv= cv_gas(1e9,np.power(10.0,ti),aion,xmass,zion)
    cv_dat.append(cvv)

#print cv_dat

#---angular momentum torque ---  

def muWD(Bs,Rwd):
    return Bs*np.power(Rwd,3.0)

def TdipIII(R_wd,Omega_wd):
    return np.power(R_wd,6.0) * np.power(Omega_wd,3.0)

def AdipIII(tau,Omegawd_0,Rwd_0,Jwd_0,Bs):
    return (1.0/(3.0*c**3)) * (1.0 + np.sin(np.pi*0.5)**2) * np.power(Omegawd_0,3.0) * np.power(muWD(Bs,Rwd_0),2.0) * tau  / Jwd_0


def Torque_dip(t,y,ffx):
    AAdip = ffx[0]
    cc = ffx[1]
    Reqx = Interp1D_ll( y[0], [ Jwd[j] / JJdim for j in range(NN)] , [  (1.0/3.0) * (Rrotwd_pol[j] + 2.0*Rrotwd_eq[j]) / Reqwd_0  for j in range(NN)] )
    #Reqx = Req_j(y[0])
    Omegax = Interp1D_ll( y[0], [ Jwd[j] / JJdim for j in range(NN)] , [ Omegawd[j] / Omegawd_0 for j in range(NN)] )
   # Omegax = Omega_j(y[0])
    if cc==0:
        ec1 = -np.log(10.0)*np.power(10.0,t)*AAdip*TdipIII(Reqx,Omegax)
    else:
        BB = 1.0 / Reqx
        ec1 =  -np.log(10.0)*np.power(10.0,t)*AAdip*TdipIII(Reqx,Omegax) * BB**4
    drhodj =  Interp1D_ll( y[0], [ Jwd[j] / JJdim for j in range(NN)] , DrhoDj_dats )
    rhowdx =  Interp1D_ll( y[0], [ Jwd[j] / JJdim for j in range(NN)] , [ rhowd[j]  for j in range(NN)] )
    #print rhowdx, Omegax, Reqx
    ec2 = y[1] * drhodj * np.log(10.0) * ec1 / cv_gas( np.power(10.0,rhowdx), Tdim*y[1], aion, xmass,zion)
    return np.array([ec1,ec2])

 
def RungheKut(t,y,h,fx):
    dsy1 = Torque_dip(t,y,fx)
    #print dsy1
    dsy2 = Torque_dip( t + h/2.0 , y + dsy1[0] * h / 2.0 , fx )
    #print dsy2
    dsy3 = Torque_dip( t + h / 2.0 , y + dsy2[0] * h / 2.0 , fx )
    #print dsy3
    dsy4=Torque_dip( t + h  , y + dsy3[0] * h , fx )
    #print dsy4
    jnew = y[0] + (1.0 / 6.0) * h * ( dsy1[0] + 2.0 * ( dsy2[0] + dsy3[0]) + dsy4[0] )
    Tnew = y[1] + (1.0 / 6.0) * h * ( dsy1[1] +2.0*(dsy2[1]+dsy3[1])+dsy4[1])
    return [jnew,Tnew]

#dt = 1e-2
#rho_x = rhowd_0 
#Tem_x = 1e8/Tdim

xnamefile = 'TevolutionWD_Rnewv3'
def Tevolution(jwd0,rhowd0,T0,Bs,cc):
    Bs0=Bs
    tauD= 1.0 / AdipIII( 1.0, Omegawd_0*c/Rdim, Reqwd_0*Rdim, jwd_0*JJdim*Rdim**2/Jdim, Bs0)
    A0 = AdipIII( tauD, Omegawd_0*c/Rdim, Reqwd_0*Rdim, Jadim,Bs0 )
    if cc==0:
        xname = '_'.join([xnamefile,str(int(np.log10(T0))),str(int(args.mass*100.0))]) 
    else:
        xname = '_'.join([xnamefile,'fluxconst',str(int(np.log10(T0))),str(int(args.mass*100.0))]) 

    xxname = '.'.join([xname,'dat'])
    print '......', xxname
    #FN = open(xxname,'w')
    yy=[jwd0, T0/Tdim]
    rho_x = rhowd0
    t0=-5.0
    evol_1 = [ [np.power(10.0,t0)*tauD, rho_x, yy[0], Omegawd_0, T0/Tdim, Reqwd_0 ] ]
    i=1;  
    dt =1e-2
    while rho_x<rhobeta:
        yynew=RungheKut(t0+i*dt,yy,dt,[A0,cc])
        rho_x = Interp1D_ll( yynew[0], [ Jwd[j] / JJdim for j in range(NN)] , [ rhowd[j]  for j in range(NN)] )
        yy=yynew
        Omegaxnew = Interp1D_ll( yynew[0], [ Jwd[j] / JJdim for j in range(NN)] , [ Omegawd[j] / Omegawd_0 for j in range(NN)] )
        Reqnew = Interp1D_ll( yynew[0], [ Jwd[j] / JJdim for j in range(NN)] , [   (1.0/3.0) * (Rrotwd_pol[j] + 2.0*Rrotwd_eq[j]) / Reqwd_0  for j in range(NN)] )
        evol_1.append([ np.power(10.0,(t0+i*dt))*tauD , rho_x ,yy[0] , Omegawd_0 * Omegaxnew , yy[1], Reqnew * Reqwd_0])
        i=i+1
    #np.savetxt(FN,evol_1,fmt='%1.7e')
    #FN.close()
    return evol_1

Temperature=ode(Torque_dip).set_integrator('dopri5',atol=1e-7)
#'lsoda',method='bdf',atol=1e-7
def Tevolution_2(y0,drr,rhowd0,tauD,A0,cc):
    T0 = y0[1]*Tdim
        #r0 = drr*1e-2
    if cc==0:
        xname = '_'.join([xnamefile,str(int(np.log10(T0))),str(int(args.mass*100.0))]) 
    else:
        xname = '_'.join([xnamefile,'fluxconst',str(int(np.log10(T0))),str(int(args.mass*100.0))]) 

    xxname = '.'.join([xname,'dat'])
   # FN = open(xxname,'w')
    print '......', xxname
    t0 = -5
    rho_x=rhowd0
    evol_1=[ [np.power(10.0,t0)*tauD, rho_x, y0[0], Omegawd_0, y0[1] , Reqwd_0] ]
    i=1; 
    Temperature.set_initial_value( y0,t0 ).set_f_params([A0, cc])
    xj = y0[0]
    while Temperature.successful() and   xj>jmin:
   # rho_x<rhobeta :
        Temperature.integrate(Temperature.t+drr)
        xj = Temperature.y[0]
        #print Static.y[1] * Sigma02 * c**2
        rho_x = Interp1D_ll( Temperature.y[0], [ Jwd[j] / JJdim for j in range(NN)] , [ rhowd[j]  for j in range(NN)] )
        if rho_x<TauCC(np.log10(Temperature.y[1]*Tdim)):
           Omegaxnew = Interp1D_ll( Temperature.y[0], [ Jwd[j] / JJdim for j in range(NN)] , [ Omegawd[j] / Omegawd_0 for j in range(NN)] )
        #print Omegaxnew 
           Reqnew = Interp1D_ll( Temperature.y[0], [ Jwd[j] / JJdim for j in range(NN)] , [  (1.0/3.0) * (Rrotwd_pol[j] + 2.0*Rrotwd_eq[j]) / Reqwd_0 for j in range(NN)] )
           evol_1.append([ np.power(10.0,(Temperature.t))*tauD , rho_x ,Temperature.y[0] , Omegawd_0 * Omegaxnew , Temperature.y[1], Reqnew * Reqwd_0])
        else:
            return np.power(10.0,(Temperature.t))*tauD/(1e6*yr) 
        #print [ np.power(10.0,(Temperature.t))*tauD , rho_x ,Temperature.y[0] , Omegawd_0 * Omegaxnew , Temperature.y[1], Reqnew * Reqwd_0]
       # np.savetxt(FN,[[ np.power(10.0,(Temperature.t))*tauD , rho_x ,Temperature.y[0] , Omegawd_0 * Omegaxnew , Temperature.y[1], Reqnew * Reqwd_0]],fmt='%1.7e')
    #FN.close()
    return evol_1

x_B=np.linspace(6,10,10)
timescale= []
FN = open('timesscales_142_T8.dat','w')
for xxB in x_B:
    Bs0 = np.power(10.0,xxB)
    tauD= 1.0 / AdipIII( 1.0, Omegawd_0*c/Rdim, Reqwd_0*Rdim, jwd_0*JJdim*Rdim**2/Jdim , Bs0)
    A0 = AdipIII( tauD, Omegawd_0*c/Rdim, Reqwd_0*Rdim, Jadim,Bs0 )
    Evol1 = Tevolution_2( [ jwd_0, 1e8/Tdim ] , 5e-3, rhowd_0, tauD , A0 , 0.0 )
    Evol1_flux = Tevolution_2( [ jwd_0 , 1e8/Tdim ] , 5e-3 , rhowd_0 , tauD , A0 , 1.0)
    timescale.append([xxB,Evol1,Evol1_flux])
    np.savetxt(FN,[[xxB,Evol1,Evol1_flux]],fmt='%1.7e')
    #FN.close()

for i in range(len(timescale)):
    print [timescale[i][j] for j in range(3)]
exit()

plt.plot([Evol1[j][0] for j in range(len(Evol1)-1)],[ Omegawd_0*Evol1[j][3]*c/Rdim for j in range(len(Evol1)-1)],'ro')
plt.plot([Evol1_flux[j][0] for j in range(len(Evol1_flux)-1)],[ Omegawd_0*Evol1_flux[j][3]*c/Rdim for j in range(len(Evol1_flux)-1)],'bo')
plt.show()

exit()

plt.semilogy([ np.log10( Evol1[j][0] / (1e6*yr)) for j in range(len(Evol1)-1)],[ Tdim*Evol1[j][4] for j in range(len(Evol1)-1)],'k-')
plt.semilogy([ np.log10( Evol1_flux[j][0] / (1e6*yr)) for j in range(len(Evol1_flux)-1)],[ Tdim*Evol1_flux[j][4] for j in range(len(Evol1_flux)-1)],'k--')
plt.show()


Evol2 = Tevolution_2( [ jwd_0 , 1e7/Tdim ] , 1e-2 , rhowd_0 , 0.0)
Evol2_flux = Tevolution_2( [ jwd_0 , 1e7/Tdim ] , 5e-3 , rhowd_0 , 1.0)

plt.semilogy([ np.log10( Evol2[j][0] / (1e6*yr)) for j in range(len(Evol2)-1)],[ Tdim*Evol2[j][4] for j in range(len(Evol2)-1)],'b-')
plt.semilogy([ np.log10( Evol2_flux[j][0] / (1e6*yr)) for j in range(len(Evol2_flux)-1)],[ Tdim*Evol2_flux[j][4] for j in range(len(Evol2_flux)-1)],'b--')
plt.show()

Evol3 = Tevolution_2( [ jwd_0 , 1e6/Tdim ] , 1e-2 , rhowd_0 , 0.0)

Evol3_flux = Tevolution_2( [ jwd_0 , 1e6/Tdim ] , 1e-2 , rhowd_0 , 1.0)

plt.semilogy([ np.log10( Evol3[j][0] / (1e6*yr) ) for j in range(len(Evol3)-1)],[ Tdim*Evol3[j][4] for j in range(len(Evol3)-1)],'r-')

plt.semilogy([ np.log10( Evol3_flux[j][0] / (1e6*yr)) for j in range(len(Evol3_flux)-1)],[ Tdim*Evol3_flux[j][4] for j in range(len(Evol3_flux)-1)],'r--')

"""
x = np.linspace(0.0,3.0,100)
plt.loglog( np.power(10.0,x) , -ii / ( dimi * np.power( 10.0,2*x) ) , 'k-')
plt.show()
exit()
"""


exit()
