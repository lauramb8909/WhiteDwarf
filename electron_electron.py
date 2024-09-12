import numpy as np 
import Physical_Const as phys

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

def lambdae(t):
    return ( hbar / ( me*c) ) * np.sqrt(2.0*np.pi / t)

# Ichimaru, H. Iyetomi, & S. Tanaka, 1987 (ecuacion 3.87)
def feenr(A,Z,rho,T):
    tr    = T / Tr
    Rcell = np.power( (4.0 / 3.0) * np.pi * rho / (A*mu), -1.0/3.0)
    rs    = Rcell / ( np.power(Z,1.0/3.0) * RBohr )
    Tf    = ( np.sqrt(1.0+1.96e-4/rs**2) - 1.0 ) * Tr
    th    =  T / Tf
    th2   = th*th
    th3   = th*th2
    th4   = th2*th2
    th12  = np.sqrt(th)
    tan   = th12 * np.tanh(1.0/th12)
    tan2  =  np.tanh(1.0/th) 
    
    gamma =  Gamma(A,Z,rho,T)
    G12   = np.sqrt(gamma)
    
    a     =  tan2* np.power(9.0 / (4*np.pi**2),1.0/3.0 ) * ( ( 0.75 + 3.04363*th2 - 0.09227*th3 + 1.7035*th4 ) / ( 1.0 + 8.3105*th2 + 5.1105*th4))
    b     = ( 0.341308 + 12.070873*th2 + 1.148889*th4 ) * tan / ( 1.0 + 10.495346*th2 + 1.326623*th4 )
    e     = ( 0.539409 + 2.522206*th2 + 0.178484*th4 ) * th * tan2 / ( 1.0 + 2.555501*th2 + 0.146319*th4 )
    c     = ( 0.872496 + 0.025248*np.exp(-1.0/th) )*e
    d     = ( 0.614925 + 16.996055*th2 + 1.489056*th4) * tan / ( 1.0 + 10.109359*th2 + 1.22185*th4 )

    ce = c / e
    fact1 = b - ce*d 
    fact2 = a - ce
    dem = np.sqrt( 4.0 * e - d**2 )
    f1 = ce * gamma + ( 2.0 / e ) * fact1 * G12 
    f2 = ( 1.0 / e )*( fact2 - (d/e) * fact1 ) * np.log( abs( e*gamma + d*G12 + 1.0) )
    f3 = ( 2.0 / (e*dem) ) * ( d * fact2 + (2.0-d**2/e) * fact1 ) * (np.atan( ( 2.0 * e * G12 + d ) / dem ) - np.atan( d / dem ) )
    
    return -( f1 + f2 - f3) 


def feer(A,Z,rho,T):
    alphaf = 1.0 / 137.0

    xr  = np.power( rho*Z/(A*mu)*3.0*np.pi**2 , 1.0/3.0) * hbar / (me*c)
    xr2 = xr*xr
    xr4 = xr2*xr2
    xr5 = xr4*xr
    tr  = T / Tr
    tr2 = tr*tr
    tr4 = tr2*tr2

    L   = np.sqrt( 1.0 + xr2 ) * np.log( xr + np.sqrt( 1.0 + xr2 ) )
    f0  = (3.0/2.0)*L**2 / (1.0 + xr2) - 3.0*xr*L + 3.0*xr2/2.0 + xr4/2.0
    f2  = (np.pi**2/3.0) * ( 2.0*np.log( 2.0*xr2/tr ) + xr2 - 3.0*L/xr - 0.7046 )
    f4  = (np.pi**4/18.0) * ( 1.0 - 1.1 / xr2 - 3.7 / xr4 - 6.3*L/xr5 )
                       
    FF  = f0 + f2*tr2 +f4*tr4
    GG  = (3.0 * alphaf / ( 4.0 * np.pi ) ) * ( FF /(xr**3*tr) )
    
    return GG * Z



def Cveenr(A,Z,rho,tt):
    
    ttc = np.log10(tt)
    
    dx = 0.001
    tic = [ ttc - 2*dx , ttc - dx, ttc, ttc + dx, ttc + 2*dx]
    log10 = np.log(10.0)
    
    Feic = np.zeros(5)
    for i,tcc in zip(range(5), tic):
        Feic[i] = feenr(A,Z,rho,np.power(10.0,tcc)) 


    Df  = ( Feic[0] - 8.0*Feic[1] + 8.0*Feic[3] - Feic[4] ) / (12.0*dx*log10 )
    DDf = ( - Feic[4] + 16.0*Feic[3] - 30.0*Feic[2] + 16.0*Feic[1] - Feic[0]) /(12.0*np.power(log10*dx,2.0))
      
    Cvfin =  -( DDf + Df  )
    
    return Cvfin


def Cvee(A,Z,rho,tt):

    Z13 = np.power(Z,1.0/3.0)
    Rcell = np.power( (4.0 / 3.0) * np.pi * rho / (A*mu), -1.0/3.0)
    rs    = Rcell / ( Z13 * RBohr )
    gammae  = Gamma(A,Z,rho,tt )/ np.power(Z13,5.0)
    
    ttc = np.log10(tt)
    
    dx = 0.001
    tic = [ ttc - 2*dx , ttc - dx, ttc, ttc + dx, ttc + 2*dx]
    log10 = np.log(10.0)
    
    Feic = np.zeros(5)
    if (gammae>=0.07) and (rs<=0.13):
        for i,tcc in zip(range(5), tic):
            gammae  = Gamma(A,Z,rho,np.power(10.0,tcc)) / np.power(Z,5.0/3.0)
            eta = np.exp ( -1.0/(gammae/0.07-0.9)**2 - 1.0/(0.13/rs-0.9)**2) 
            Feic[i] = eta*feer(A,Z,rho,np.power(10.0,tcc)) + (1.0-eta)*feenr(A,Z,rho,np.power(10.0,tcc))
    else:
        for i,tcc in zip(range(5), tic):
            Feic[i] = feenr(A,Z,rho,np.power(10.0,tcc))


    Df  = ( Feic[0] - 8.0*Feic[1] + 8.0*Feic[3] - Feic[4] ) / (12.0*dx*log10 )
    DDf = ( - Feic[4] + 16.0*Feic[3] - 30.0*Feic[2] + 16.0*Feic[1] - Feic[0]) /(12.0*np.power(log10*dx,2.0))
      
    Cvfin =  -( DDf + Df  )
    
    return Cvfin



    