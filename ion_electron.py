import numpy as np 
import Physical_Const as phys
from scipy.interpolate import interp1d

##### Constants -----
hbar     = phys.hbar
c        = phys.c
G        = phys.G
sigma    = phys.sigmaSB
e        = phys.e
e_erg    = phys.e_erg
me_mev   = phys.me
mevtoerg = phys.mevtoerg
mu       = phys.mu
kappaB   = phys.kappa

h     = hbar*(2.0*np.pi)
me    = me_mev * mevtoerg / c**2
c2    = c*c
arad  = 4.0*sigma / c
Tr    = me * c2 / kappaB
RBohr = hbar**2 / (me*e_erg)



def Gamma(A,Z,rho,T):
    Rcell = np.power( 4.0/3.0 * np.pi * rho/(A*mu),-1.0/3.0)
    return (Z**2.0*e_erg)/(Rcell*kappaB*T)

def omegap(A,Z,rho):
    return 2.0*Z/(A*mu)*np.sqrt(np.pi*e_erg*rho)

def eta(A,Z,rho,T):
    return hbar*omegap(A,Z,rho)/(kappaB*T)



# ion-electron free energy [Haensel book equation 2.154)
def feinm(A,Z,rho,T):
    Z13   = np.power(Z, 1.0/3.0)
    Z12   = np.sqrt(Z)
    logZ  = np.log(Z)
    
    Rcell = np.power( (4.0 / 3.0) * np.pi * rho / (A*mu), -1.0/3.0)
    rs    = Rcell / ( Z13 * RBohr )
    xx    = np.power( rho*Z/(A*mu)*3.0*np.pi**2 , 1.0/3.0) * hbar / (me*c)
    xx2   = xx*xx
    
    cdh   = ( Z / np.sqrt(3) )*( np.power(1.0+Z,3.0/2.0) - 1.0 - np.power(Z,3.0/2.0) )
    cnfty = (18.0/175.0) * np.power(12.0/np.pi,2.0/3.0)
    ctf   = cnfty * np.power( Z13 , 7.0) * (1.0-1.0 / Z13 + 0.2/Z12);
    
    nu    = 1.16 + 0.08 * logZ;
    bb    = 0.2 + 0.078 * logZ**2
    aa    = 1.11 * np.power(Z,0.475)

    gamma   = Gamma(A,Z,rho,T)
    gammae  = gamma / np.power(Z13,5.0)
    gammae2 = np.sqrt(gammae)
    gammanu = np.power(gammae,nu)
    g1    = gammanu * ( 1.0 + 0.78  * (gammae2 / Z12) / ( 21.0 + gammae * np.power(Z/rs,3.0) ) )
    g2    = ( gammanu/ rs) * ( 1.0 + ( (Z-1) * np.power(rs,3.0) / ( 9.0* ( 1.0 + 4.0*rs**2) ) ) * ( 1.0 + 1.0/(0.001*Z**2+2.0*gammae) )  ) ;
    h2    = 1.0 / np.sqrt( 1.0 + xx2 )
    h1    = ( 1.0 + xx2/5.0 ) / ( 1.0 + 0.18 * xx / np.power(Z,1.0/4.0) + 0.37 / Z12 * xx2 + xx2/5.0 )
    
    return -gammae * ( cdh * gammae2 + ctf * aa * g1 * h1 ) / ( 1.0 + h2 * ( bb * gammae2 + aa * g2 ) );

def Cvienm(A,Z,rho,tt):

    ttc = np.log10(tt)
    dx = 0.001
    tic = [ ttc - 2.0*dx , ttc - dx, ttc, ttc + dx, ttc + 2.0*dx]
    log10 = np.log(10.0)
    
    Feic = np.zeros(5)
    for i,tcc in zip(range(5), tic):
        Feic[i] = feinm(A,Z,rho,np.power(10.0,tcc)) 
     
    Df  = ( Feic[0] - 8.0*Feic[1] + 8.0*Feic[3] - Feic[4] ) / (12.0*dx*log10 )
    DDf = ( - Feic[4] + 16.0*Feic[3] - 30.0*Feic[2] + 16.0*Feic[1] - Feic[0]) /(12.0*np.power(log10*dx,2.0))
      
    Cvfin =  -( DDf + Df  )
    
   
    return Cvfin

# ion-electron free energy (solid) [Haensel book equation 2.154)
## bcc lattice type
def feim(A,Z,rho,T):
    aa1 = 1.1866; aa2 = 0.684; aa3 = 17.9; aa4 = 41.5; q = 0.205
    
    xx   = np.power( rho*Z/(A*mu)*3.0*np.pi**2,1.0/3.0)*hbar/(me*c)
    xx2  = xx*xx
    gamm = Gamma(A,Z,rho,T)
    etax = eta(A,Z,rho,T)
     
    b1   = 1.0 - aa1 * np.power(Z,-0.267) + 0.27 / Z
    b2   = 1.0 + 2.25 / np.power(Z,1.0/3.0) * ( 1.0 + aa2*np.power(Z,5.0) + 0.222*np.power(Z,6.0) ) / ( 1.0 + 0.2222 * np.power(Z,6) )
    logZ = np.log(Z)
    b3   = aa4 / ( 1.0 + logZ )
    b4   = 0.395 * logZ + 0.347 * np.power(Z,-3.0/2.0)
    Af   = ( b3 + aa3 * xx2 ) / ( 1.0 + b4 * xx2 )
    #Q=np.power(np.log(1.0+np.exp(np.power(q*etax,2.0)))/np.log(np.exp(1.0)-(np.exp(1.0)-2.0)*np.exp(-np.power(q*etax,2.0))),1.0/2.0)
    Q    = np.sqrt( 1.0  + np.power(q*etax,2.0) )
    s    = 1.0 / ( 1.0 + 0.01 * np.power( logZ,3.0/2.0 ) + 0.097 / Z**2 )
    finf = 0.00352*np.power(Z,2.0/3.0) * b1 * np.sqrt( 1.0 + b2 / xx2 )
    return -finf * gamm * ( 1.0 + Af * np.power( Q / gamm, s) )



def Cviem(A,Z,rho,tt):
    
    ttc = np.log10(tt)
    
    dx = 0.001
    tic = [ ttc - 2*dx , ttc - dx, ttc, ttc + dx, ttc + 2*dx]
    log10 = np.log(10.0)
    
    Feic = np.zeros(5)
    for i,tcc in zip(range(5), tic):
        Feic[i] = feim(A,Z,rho,np.power(10.0,tcc)) 


    Df  = ( Feic[0] - 8.0*Feic[1] + 8.0*Feic[3] - Feic[4] ) / (12.0*dx*log10 )
    DDf = ( - Feic[4] + 16.0*Feic[3] - 30.0*Feic[2] + 16.0*Feic[1] - Feic[0]) /(12.0*np.power(log10*dx,2.0))
      
    Cvfin =  -( DDf + Df  )
    
    return Cvfin



def Cvie(A,Z,rho,T):
    gammapoint = Gamma(A,Z,rho,T)
    
    if gammapoint>=175.0:
       
        Cvpoint=Cviem(A,Z,rho,T)
    else:
       
        Cvpoint=Cvienm(A,Z,rho,T)
    return Cvpoint





