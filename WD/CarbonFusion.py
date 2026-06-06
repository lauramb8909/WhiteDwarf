import numpy as np 
import Physical_Const as phys


#----Constants------

hbar     = phys.hbar
c        = phys.c
G        = phys.G
sigma    = phys.sigmaSB
me_mev   = phys.me

ee       = phys.e_erg
mevtoerg = phys.mevtoerg
mu       = phys.mu
kappaB   = phys.kappa
Msun     = phys.Msun
Rsun     = phys.Rsun

h        = hbar*(2.0*np.pi)
me       = me_mev * mevtoerg / c**2
c2       = c*c
arad     = 4.0*sigma / c
Tr       = me * c2 / kappaB
pi05     = np.sqrt(np.pi)
RBohr    = hbar**2 / (me*ee)



def wplasma(Zbar,Abar,rho):
    return np.sqrt( 4.0 * np.pi * Zbar**2 * ee * rho / np.power( Abar * mu,2.0) )

def Screen(E):
    return  5.15e16 * np.exp( -0.428 * E - 3.0 * np.power(E,0.308) / ( 1.0 + np.exp( 0.613 * ( 8.0 - E ) ) ) )

def Reaction(rho,T,Zbar,Abar,XC):
    Cpyc=3.90 ; Cexp=2.638 ; Cpl=1.25 ; Ct=0.724 ; Csc=1.0754 ;  Lambda=0.5
    E = hbar * wplasma(Zbar,Abar,rho) / (1.602e-6)
    Rion = np.power( 4.0/3.0 * np.pi * rho / ( Abar * mu ) ,-1.0/3.0)
    lambdarho = hbar**2 / ( Abar * mu * Zbar**2 * ee ) * np.power( rho * XC / (2.0 * Abar*mu), 1.0/3.0)
    Rpyc0 =  0.5 * (1.602e-30) * Screen(E) * np.power(rho * XC / ( Abar*mu), 2.0) * hbar / ( Zbar**2 * ee * Abar * mu ) * 8.0 * Cpyc * 11.515 / np.power( lambdarho, Cpl) * np.exp( -Cexp / np.sqrt( lambdarho ))
    Ea = Abar * mu * np.power( Zbar**2 * ee / hbar , 2.0)
    Tnuc = T * np.sqrt( 1.0 + np.power( Ct * hbar * wplasma(Zbar,Abar,rho) / (T * kappaB) ,2.0 ) )
    taunuc = 3.0 * np.power( np.pi * 0.5, 2.0/3.0) * np.power( Ea / ( kappaB * Tnuc ) , 1.0/3.0)
    gg = 2.0/3.0 * ( T**2 + np.power( hbar * wplasma(Zbar,Abar,rho) / kappaB ,2.0) * ( Cpl +0.5 ) ) / ( T**2 + np.power( hbar * wplasma(Zbar,Abar,rho) /kappaB , 2.0) )
    Gammanuc = Zbar**2 * ee / ( Rion  * kappaB) 
    Pnuc = 8.0 * np.power( np.pi , 1.0/3.0 ) / ( np.sqrt(3.0) * np.power( 2.0 , 1.0/3.0 ) ) * np.power( Ea / ( kappaB * Tnuc ) , gg )
    phiii = np.sqrt( Gammanuc / T ) / np.power( Csc**4 / 9.0 + np.power( Gammanuc/T , 2.0 ) , 1.0/4.0)
    Fnuc = np.exp( -taunuc - Lambda * hbar * wplasma(Zbar,Abar,rho) / ( kappaB * T ) + Csc * Gammanuc/Tnuc * phiii * np.exp( -Lambda * hbar * wplasma(Zbar, Abar, rho) / ( kappaB * T ) ) )
    Epk = hbar * wplasma(Zbar,Abar,rho) + ( Zbar**2 * ee / Rion + kappaB * T / 3.0 * ( 3.0 * np.power( np.pi/2.0 , 2.0/3.0) * np.power( Ea/ ( kappaB * T ) , 1.0/3.0 ) ) ) * np.exp( -Lambda * hbar * wplasma(Zbar,Abar,rho) / ( kappaB * T ) )
    DRpyc = 0.5 * (1.602e-30) * Screen( Epk / 1.602e-6 ) * np.power( rho * XC / ( Zbar * Abar * mu),2.0) * ( hbar / ( ee * Abar * mu ) ) * Fnuc * Pnuc  
    Rpyc = Rpyc0 + DRpyc
    epsilon = ( 4.62 + 2.24 ) * 1.602e-6 * 0.5
    return Rpyc

def QCC(rho,T,Zbar,Abar,XC):
    Rpyc = Reaction(rho,T,Zbar,Abar,XC)
    epsilon = ( 4.62 + 2.24 ) * 1.602e-6 * 0.5
    return Rpyc*epsilon




