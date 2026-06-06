import numpy as np 
from math import pi
from scipy.integrate import ode,quad,quadrature
from scipy.optimize import brenth,brentq,fsolve,root,bisect
import Physical_Const as phys
from scipy import integrate
from scipy.interpolate import interp1d
import dfermiDirac as FD

hbar     = phys.hbar
c        = phys.c
G        = phys.G
sigma    = phys.sigmaSB
me_mev   = phys.me
mevtoerg = phys.mevtoerg
mu       = phys.mu
kappaB   = phys.kappa

h    = hbar*(2.0*np.pi)
me   = me_mev * mevtoerg / c**2
c2   = c*c
arad = 4.0*sigma / c
Tr   = me * c2 / kappaB
pi05 = np.sqrt(np.pi)


def lambdae(t):
    return ( hbar / ( me*c) ) * np.sqrt(2.0*np.pi / t)

#electrons+positrons
###################################################################################################################################################
def PUelec(xi,t):
    Int05 = FD.dfermi(1.0/2.0,xi,t)[0]
    Int15 = FD.dfermi(3.0/2.0,xi,t)[0]
    Int25 = FD.dfermi(5.0/2.0,xi,t)[0]
    
    le  = lambdae(t)
    le3 = le*le*le
    Ee  = me*c2
    
    ne  = 4.0 / ( pi05 * le3 ) * ( Int05 + t * Int15 )
    Pe  = 8.0 * t * Ee / ( 3.0*pi05*le3 ) * ( Int15 + 0.5 * t * Int25 )
    Ue  = 4.0 * t * Ee / ( pi05 * le3 ) * ( Int15 + t*Int25 )
    
    return [ ne, Pe, Ue ] 

def PUpos(xi,t):
    Int05=FD.dfermi(1.0/2.0,-xi-2.0/t,t)[0]
    Int15=FD.dfermi(3.0/2.0,-xi-2.0/t,t)[0]
    Int25=FD.dfermi(5.0/2.0,-xi-2.0/t,t)[0]
    
    le  = lambdae(t)
    le3 = le*le*le
    Ee  = me*c2

    np  = 4.0 / ( pi05 * le3 ) * ( Int05 + t * Int15 )
    Pp  = 8.0 * t * Ee / ( 3.0*pi05 * le3 ) * ( Int15 + 0.5 * t * Int25 )
    Up  = 4.0 * t *Ee / ( pi05 * le3 ) * ( Int15 + t * Int25 ) + 2.0*Ee*np

    return [np, Pp, Up]


#Radiation
##########################################################################
def PUrad(t):
    t4 = np.power(t*Tr,4.0)
    return [ arad*t4/3.0 , arad*t4]

#Ions
##########################################################################
def PUion(Nion,rho,t):
    PP = Nion * kappaB * Tr * t
    return [ PP, 3.0/2.0*PP ] 


def PUWtotal(rho,t):
    xic  = XXI(rho,t)
    
    rad  = PUrad(t)
    ion  = PUion( rho/(A*mu) , rho , t )
    ele  = PUelec(xic,t)
    pos  = PUpos(xic,t)
    
    Ptot = rad[0]+ion[0]+ele[1]+pos[1]
    Utot = rad[1]+ion[1]+ele[2]+pos[2] 
    
    return [ Ptot, Utot , c2*rho + Utot + Ptot ]


#Electron Chemical potential (neutrality condition)
###########################################################################
def xi_eq(xi,A,Z,rho,t):
    ne = PUelec(xi,t)[0]
    return  ne  * mu * A / ( Z * rho ) - 1.0

def XXI(A,Z,rho,t):
    xl=-10;  xr=10;
    i=0
    while xi_eq(xl,A,Z,rho,t)*xi_eq(xr,A,Z,rho,t)>0.0:
      
       if xi_eq(A,Z,xl,rho,t)>0.0:
          xr = xl; xl = xl*10.0
       else:
          xl = xr; xr = xr*10.0
           
    return brentq(xi_eq,xl,xr,args=(A,Z,rho,t))


# Heat capacity electrons and ions
############################################################################

def cv_ion():
    return 3.0/2.0


#electrons free energy
def fe(A,Z,rho,t):
    xip = XXI(A,Z,rho,t)
    ele = PUelec(xip,t)
    ne = ele[0]
    pe = ele[1]
    
    return ( xip*t + 1.0 )*me*c**2*ne - pe


def Cvienm(A,Z,rho,tt):
    
    ttc = np.log10(tt)
    dx = 0.01
    tic = [ ttc - 2*dx , ttc - dx, ttc, ttc + dx, ttc + 2*dx]
    
    Feic = np.zeros(5)
    for i,tcc in zip(range(5), tic):
        Feic[i] = fe(A,Z,rho,np.power(10.0,tcc)) 
     
    Df  = ( -Feic[4] + 8.0*Feic[3] - 8.0*Feic[1] + Feic[0] ) / (12.0*dx*np.log(10.0) )
    DDf = ( - Feic[4] + 16.0*Feic[3] - 30.0*Feic[2] + 16.0*Feic[1] - Feic[0]) /(12.0*np.power(np.log(10.0)*dx,2.0))
      
    Cvelec =  -( DDf - Df  ) 
        
    return ( A*mu / (kappaB*rho*Tr) ) * ( Cvelec / tt)



#Entropy
###################################################################################################################################################
def EntropyTotal(rho,t):
    xic=XXI(rho,t)
    nion= rho / (A*mu)
    le = lambdae(t)
    Ee = me*c2
    
    Sion = -( np.log( nion * np.power( np.sqrt(me/(A*mu)) * le,3.0) ) -1.0 - 3.0/2.0 ) / A
    
    ion = PUion( nion, rho , t )
    Sion2 = ion[0] / ( kappaB*t*Tr * A*nion )
    
    ele= PUelec(xi,t)
    pos = PUpos(xi,t)
    Sep = ( ele[2] + ele[1]  - Ee*(xic*t+1.0) * ele[0] ) / ( kappaB*t*Tr*A*nion )
    Sp =  ( pos[2] + pos[1] + Ee*(xic*t+1.0) * pos[0] )  / ( kappaB*t*Tr*A*ion )
    
    rad = PUrad(t)
    Srad = ( rad[1] + rad[0] ) / ( kappaB*t*Tr*A*nion )
    return [Sion , Srad , Entropy(rho,t), Sep , Sp]


def ftot(xi,t,rho):
    Pele= PUelec(xi,t)[1]
    Ppos = PUpos(xi,t)[1]
    return (xi+1.0/t)*Z/A - ( Pele + Ppos ) / ( t * (me/mu)*c2*rho)

def Entropy(rho,tt):
    global telec,Selec
    
    NNic = 100
    ttc = np.log10(tt)
    tic = np.linspace(ttc-0.5,ttc+0.5,NNic)
    Feic=[];
    
    for tcc in tic:
        Tc = np.power(10.0, tcc)
        xic=XXI(rho,Tc)
        Feic.append( ftot(xic,Tc,rho) )
        
    Selec=[]; telec=[];  i=2; #SS2=[]
    while i<NNic-3:
        tpoint=tic[i]
        Dpoint=tic[i+1]-tic[i-1]
        Ficpoint=Feic[i]
        telec.append(tpoint)
        Dfpoint=((-Feic[i+2]+8.0*Feic[i+1]-8.0*Feic[i-1]+Feic[i-2])/(12.0*Dpoint/2.0))/np.log(10.0)
        Selec.append(-(Ficpoint+Dfpoint))
        i=i+1
    Sfin=interp1d(telec,Selec)
    S_t=Sfin(ttc)
    return (S_t)

def dPdT(rho,tt):
    global telec,P
    NNic=50
    ttc=np.log10(tt)
    tic=np.linspace(ttc-0.5,ttc+0.5,NNic)
    Peic=[];
    for tcc in tic:
        Tc = np.power(10.0, tcc)
        Ptotal = PUWtotal(rho,Tc)[0]
        Peic.append( Ptotal )
    DPelec=[]; telec=[];  i=2; #SS2=[]
    while i<NNic-3:
        tpoint=tic[i]
        Dpoint=tic[i+1]-tic[i-1]
        Picpoint=Peic[i]
        telec.append(tpoint)
        DPpoint=((-Peic[i+2]+8.0*Peic[i+1]-8.0*Peic[i-1]+Peic[i-2])/(12.0*Dpoint/2.0))/np.log(10.0)
        DPelec.append(DPpoint/Picpoint)
        i=i+1
    Pfin=interp1d(telec,DPelec)
    DP_t=Pfin(ttc)
    return (DP_t)


def Ptot_eq(t,p,rho):
    rhoc = np.power(10.0, rho)
    return PUWtotal(rhoc,t)[0] / p - 1.0

def dTdrho(P,rho):
    global telec
    NNic=50
    ttc=np.log10(rho)
    tic=np.linspace(ttc-0.5,ttc+0.5,NNic)
    teic=[];
    for tcc in tic:
        teic.append(brentq(Ptot_eq,1e-4,100,args=(P,tcc)))
    Dtelec=[]; telec=[];  i=2; #SS2=[]
    while i<NNic-3:
        tpoint=tic[i]
        Dpoint=tic[i+1]-tic[i-1]
        #xic=XXI(rho,np.power(10.0,tpoint));
        ticpoint=teic[i]
        telec.append(tpoint)
        Dtpoint=((-teic[i+2]+8.0*teic[i+1]-8.0*teic[i-1]+teic[i-2])/(12.0*Dpoint/2.0))/np.log(10)
        Dtelec.append(Dtpoint/tpoint)
        i=i+1
    tfin=interp1d(telec,Dtelec)
    DP_t=tfin(ttc)
    #print telec[0],telec[-1],ttc
    return (DP_t)


def dPdrho(rho,tt):
    global telec,DPelec
    NNic=50
    rhoc=np.log10(rho)
    rhoic=np.linspace(rhoc-0.5,rhoc+0.5,NNic)
    Peic=[];
    for tcc in rhoic:
        Tc = np.power(10.0, tcc)
        Ptotal = PUWtotal(rho,Tc)[0]
        Peic.append( Ptotal )
        
    DPelec=[]; Pelec=[]; telec=[];  i=2; 
    while i<NNic-3:
        tpoint=rhoic[i]
        Dpoint=rhoic[i+1]-rhoic[i-1]
        Picpoint=Peic[i]
        Pelec.append(Picpoint)
        telec.append(tpoint)
        DPpoint=((-Peic[i+2]+8.0*Peic[i+1]-8.0*Peic[i-1]+Peic[i-2])/(12.0*Dpoint/2.0))/np.log(10.0)
        DPelec.append(DPpoint/Picpoint)
        i=i+1
    Pfin=interp1d(telec,DPelec)
    DP_t=Pfin(rhoc)
    return (DP_t)



