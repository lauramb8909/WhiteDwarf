import numpy as np 
from math import pi
from scipy.integrate import ode,quad,quadrature
from scipy.optimize import brentq
import Physical_Const as phys
from scipy import integrate
from scipy import misc
from scipy.interpolate import interp1d

#--Constans---

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

h    = hbar*(2.0*np.pi)
me   = me_mev * mevtoerg / c**2
c2   = c*c
arad = 4.0*sigma / c
Tr   = me * c2 / kappaB
pi05 = np.sqrt(np.pi)



#Coupling parameter
def Gamma(A,Z,rho,T):
    
    Rcell = np.power( 4.0/3.0*np.pi*rho/(A*mu), -1.0/3.0)
    
    return ( Z**2.0 * e_erg ) / ( Rcell * kappaB * T )

#def Tmelting(rho,T,A,Z):
#   return 175.0 - Gamma(A,Z,rho,T)

def omegap(A,Z,rho):
    return 2.0 * Z / (A*mu) * np.sqrt( np.pi * e_erg * rho )

def eta(A,Z,rho,T):
    return h * omegap(A,Z,rho) / ( kappaB * T )

#--heat capacity liquid (gamma<gamma_m) [Haensel book equation (2.82)]-----
def Cviinm(gamma):
    A1  = -0.9070 ; A2 = 0.62954; A3 = -np.sqrt(3.0) / 2.0 - A1 / np.sqrt(A2)
    B1  = 4.56e-3 ; B2 = 211.6 ; B3 = -1e-4 ; B4 = 4.62e-3

    gamma2 = gamma*gamma
    
    ft1 = 0.5 * np.power(gamma, 3.0/2.0) * ( -A1 * A2 / np.power( A2 + gamma, 3.0/2.0 ) + A3 * (gamma - 1.0) / np.power(1.0 + gamma, 2.0) )
    ft2 = -B1 * B2 * gamma2 / np.power( B2 + gamma , 2.0)
    ft3 = B3 * ( gamma2**2 - B4 * gamma2 ) / np.power( B4 + gamma2 , 2.0)
    
    return ft1 +ft2 + ft3
    

#----Solid
#Introducir correcion bajas temperaturas



#Haensel book equation 2.115
def fiim(A,Z,rho,T):
    eta_p = eta(A,Z,rho,T)
    #alph1_cvm = 0.932446 ; alph2_cvm = 0.334547 ;  alph3_cvm = 0.265764;
    alph = [0.932446 ,0.334547  ,0.265764 ]
    Aa_cvm = [ 1.0 , 0.1839 , 0.593586 , 0.0054814 , 5.01813e-4 , 0.0 , 3.9247e-7 , 0.0 , 5.8356e-11 ];
    Bb_cvm =[ 261.66 , 0.0 , 7.07997 , 0.0 , 0.0409484 , 3.97355e-4 , 5.1148e-5 , 2.19749e-6 ];
    alp6 = 4.757014e-3 * Aa_cvm[6] ; 
    alp8 = 4.7770935e-3 * Aa_cvm[8];
    
    A_cvm = 0.0 ;  B_cvm =0.0
    for i, Ac in zip(range(9), Aa_cvm):
        A_cvm += Ac * np.power(eta_p,i)
    for i, Bc in zip(range(8), Bb_cvm):
        B_cvm += Bc * np.power(eta_p,i)
        
    B_cvm += alp6 * eta_p**9
    B_cvm += alp8 * eta_p**11
    fterm_1 = 0
    for i, al in zip(range(3), alph):
        fterm_1 += np.log( 1.0 - np.exp ( -al * eta_p )) 
        
    gamma = Gamma(A,Z,rho,T)
    FnH = 10.9/gamma + 247.0/(2.0*gamma**2) + 1.765e5/(3.0*gamma**3)
    
    return fterm_1 - A_cvm / B_cvm

def Cviim_ion(A,Z,rho,tt): 
    ttc = np.log10(tt)
    dx = 0.01
    tic = [ ttc - 2*dx , ttc - dx, ttc, ttc + dx, ttc + 2*dx]
    log10 = np.log(10.0)
    
    Feic = np.zeros(5)
    for i,tcc in zip(range(5), tic):
        Feic[i] = fiim(A,Z,rho,np.power(10.0,tcc)) #fe(A,Z,rho,np.power(10.0,tcc)) 
     
    Df  = ( Feic[0] - 8.0*Feic[1] + 8.0*Feic[3] - Feic[4] ) / (12.0*dx*log10 )
    DDf = ( - Feic[4] + 16.0*Feic[3] - 30.0*Feic[2] + 16.0*Feic[1] - Feic[0]) /(12.0*np.power(log10*dx,2.0))
      
    Cvfin =  -( DDf + Df  ) - 3.0/2.0 
   
    
    return Cvfin

def Cviim(A,Z,rho,T):
    ft_1 = Cviim_ion(A,Z,rho,T)
    C1B1 = 0.0112;
    alphat = 0.426548 ; alphal = 0.88412;
    a1_cvm = 10.9 ; a2_cvm = 247.0 ;  a3_cvm = 1.765e5 ;
    eta_p = eta(A,Z,rho,T)
    gamma = Gamma(A,Z,rho,T)
    ft_2 = ( 2.0 * a1_cvm + 2.0 * C1B1 * a1_cvm * eta_p**2 + 4.0 * a1_cvm * C1B1**2 * eta_p**4 ) * np.exp( - C1B1 * eta_p**2 ) 
    ft_3 = ( 3.0 * a2_cvm + ( 4.0 - 3.0 / 2.0 ) * C1B1 * a2_cvm * eta_p**2 + 4.0 * a2_cvm * C1B1**2 * eta_p**4 ) * np.exp( - C1B1 * eta_p**2) 
    ft_4 = ( 4.0 * a3_cvm + 3.0 * C1B1 * a3_cvm * eta_p**2 + 4.0 * a3_cvm * C1B1**2 * eta_p**4 ) * np.exp( -C1B1 * eta_p**2 )
    return ft_1 #+ ft_2 / gamma + ft_3 / gamma**2 + ft_4 / gamma**3


#---Total Cv for ions-----
def Cvii(A,Z,rho,T):
    gammapoint = Gamma(A,Z,rho,T)
    
    if gammapoint > 175.0:
       Cvpoint = Cviim(A,Z,rho,T)
    else:
       Cvpoint = Cviinm( gammapoint )
    
    return Cvpoint




#exit()


