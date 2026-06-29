import numpy as np
import Physical_Const as phys

#----Constants------

hbar     = phys.hbar
c        = phys.c
G        = phys.G
SigmaSB  = phys.sigmaSB
me_mev   = phys.me
mp_mev   = phys.mp
e        = phys.e
e_erg    = phys.e_erg
mevtoerg = phys.mevtoerg
mu       = phys.mu 
kappaB   = phys.kappa
#Msun     = phys.Msun
#Rsun     = phys.Rsun
#Lsun     = phys.Lsun

h        = hbar*(2.0*np.pi)
c2       = c*c
me       = me_mev * mevtoerg / c2
mp       = mp_mev * mevtoerg / c2

arad     = 4.0*SigmaSB / c
Tr       = me * c2 / kappaB
pi05     = np.sqrt(np.pi)
RBohr    = hbar**2 / (me*e_erg)
yr       = (364)*24*60*60
avo      = 1.0 / mu

kappa0 = 4.34e24                        # opacity constant (Kramers-like)

#### Definitions

def Rion(A, rho):
    return ( ( 4.0 * np.pi / 3.0 ) * rho / ( A * mu ) ) ** (-1.0 / 3.0)

def wplasma(Z, A, rho):
    return ( 2.0 * Z /( A * mu ) ) * np.sqrt( np.pi * e_erg * rho )

def Gammanuc(Z, A, rho, T):
    return Z ** 2 * e_erg / ( Rion(A, rho) * kappaB * T )

def eta_plasma(A, Z, rho, T):
    return hbar * wplasma(Z, A, rho) / (kappaB * T)

def xx(rho, A, Z):
    # Relativity parameter x as a function of mass density
    return (rho * Z / (A * mu) * 3 * np.pi ** 2) ** (1 / 3) * hbar / (me * c)


########################################################

def II_cond(A, Z, rho, T):
    return (np.sqrt(np.pi / 3) * np.log(Z ** (1 / 3))
            + 2 / 3 * np.log(1.32 + 2.33 / np.sqrt(Gammanuc(Z, A, rho, T)))
            - 0.484 * (rho / 1e6) ** (2 / 3) / (1 + 1.018 * (rho / 1e6) ** (2 / 3)))

def kappa_liquid(A, Z, rho, T):
    """Electron thermal conductivity in the degenerate-gas regime (Gamma > 175)."""
    """ Nandkumar & Pethick, MNRAS 209, 1984"""
    return (2.346e15 * (rho / 1e6) * (T / 1e6) / (Z * (1 + 1.018 * (rho / 1e6) ** (2 / 3)))
            * (1 / II_cond(A, Z, rho, T)))

###############################################

u_2, c2_cond = 13, 29.98

def v_k_F(A, Z, rho):
    return c / np.sqrt(1 + 1 / xx(rho, A, Z) ** 2)

def G0_cond(A, Z, rho, T):
    return u_2 / np.sqrt(1 + (3 * u_2 * eta_plasma(A, Z, rho, T) / (np.pi ** 2 * c2_cond)) ** 2)

def G2_cond(A, Z, rho, T):
    eta = eta_plasma(A, Z, rho, T)
    return (eta / np.pi) ** 2 * (1 + (15 / (4 * np.pi ** 4 * c2_cond)) ** (2 / 3) * eta ** 2) ** (-1.5)

def u_ke(rho):
    return 2.0 * np.pi * np.log10(rho) / 25.6

aa_ke = [ 0.50998, 0.16533, 0.04904, -0.04479, -0.00038, 0.01222, 0.00235, -0.00646, -0.00054, 0.00134]
dd_k  = [ 0.71734, 0.36575, 0.08528, -0.06504, -0.00514, 0.02633, 0.00659, -0.00986, -0.00156, 0.00319]
bb_ke, ccsigma180, eesigma5000, ff_k = -0.01214, 0.40868, -0.04356, 0.82040

aak = [1.38810, 0.20019, 0.05281, -0.11228, -0.01393, -0.00360, 0.00065, -0.01419, -0.00047, -0.0014]
ddk = [1.72569, 0.37806, 0.06998,  -0.1277, -0.02128,  0.00428, 0.00197, -0.01730, -0.00173, -0.00043]
bbk, cckappa180, eek, ffkappa500 = 0.05435, 0.48610, 0.03393, 0.775

def Isigma180(u):
    return sum(aa_ke[o - 1] * np.sin(o * u) for o in range(1, 11)) + bb_ke * u * 12.8 / np.pi + ccsigma180

def Isigma5000(u):
    return sum(dd_k[o - 1] * np.sin(o * u) for o in range(1, 11)) + eesigma5000 * u * 12.8 / np.pi + ff_k

def vv_cond(A, Z, rho, T):
    g = Gammanuc(Z, A, rho, T)
    return 0.4481 + 17.3468 * g ** (-1 / 3) - 146.8104 * g ** (-2 / 3) + 195.1610 * g ** (-1)

def ww_cond(A, Z, rho, T):
    g = Gammanuc(Z, A, rho, T)
    g13 = g ** (-1 / 3)
    g23 = g13*g13
    g33 = g23*g13
    return 0.5893 + 12.503 * g ** (-1 / 3) - 95.9862 * g ** (-2 / 3) + 37.1996 * g ** (-1)

def Isigma_cond(A, Z, rho, T):
    u = u_ke(rho)
    v = vv_cond(A, Z, rho, T)
    return (1 - v) * Isigma180(u) + v * Isigma5000(u)

def Ikappa180(u):
    return sum(aak[o - 1] * np.sin(o * u) for o in range(1, 11)) + bbk * 12.8 / np.pi * u + cckappa180

def Ikappa5000(u):
    return sum(ddk[o - 1] * np.sin(o * u) for o in range(1, 11)) + eek * 12.8 / np.pi * u + ffkappa500

def Ikappa_cond(A, Z, rho, T):
    u = u_ke(rho)
    w = ww_cond(A, Z, rho, T)
    return (1 - w) * Ikappa180(u) + w * Ikappa5000(u)

def Fkappa_ke(A, Z, rho, T):
    return Isigma_cond(A, Z, rho, T) * G0_cond(A, Z, rho, T) + Ikappa_cond(A, Z, rho, T) * G2_cond(A, Z, rho, T)

def nu_k_ke(A, Z, rho, T):
    return e_erg * kappaB * T / hbar ** 2 * Fkappa_ke(A, Z, rho, T) / v_k_F(A, Z, rho)

def kappae(A, Z, rho, T):
    """Electron thermal conductivity in the degenerate-gas regime (Gamma < 175)."""
    """ Itoh et. al. ApJ 285, 1984, Itoh et al,418 ApJ 1993 """
    return (np.pi ** 2 * kappaB ** 2 * T * (Z / (A * mu) * rho)
            / (3 * me * np.sqrt(1 + xx(rho, A, Z) ** 2) * nu_k_ke(A, Z, rho, T)))

def kcondeeT(A,Z, rho, T):
    """Electron conductive opacity-equivalent, switching liquid/gas at Gamma=175."""
    if Gammanuc(Z, A, rho, T) <= 175:
        return kappa_liquid(A, Z, rho, T)
    else:
        return kappae(A, Z, rho, T)

# ---------------------------------------------------------------------------
# Total opacity, neutrino and photon-diffusion timescales
# ---------------------------------------------------------------------------
def opacity(A,Z,rho, T):
    """Radiative (Kramers-like) + electron-conduction opacity, combined as
    is standard for WD envelopes (harmonic-mean-style additive inverse
    conductivities)."""
    return 1 / (0.1*kappa0 * rho / T ** 3.5) + kcondeeT(A,Z,rho, T) * 3 * rho / (16 * SigmaSB * T ** 3)
