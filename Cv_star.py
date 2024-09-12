import numpy as np 
from scipy.optimize import brenth,brentq
from scipy.interpolate import interp1d

import Physical_Const as phys
import EoS_ionee as EOS
import ion_electron as ionee
import ions as ion
import electron_electron as ee


#----Constants------

hbar     = phys.hbar
c        = phys.c
G        = phys.G
sigma    = phys.sigmaSB
me_mev   = phys.me
e        = phys.e
qe       = phys.qe 
e_erg    = phys.e_erg
mevtoerg = phys.mevtoerg
mu       = phys.mu
kappaB   = phys.kappa
Msun     = phys.Msun

h     = hbar*(2.0*np.pi)
me    = me_mev * mevtoerg / c**2
c2    = c*c
arad  = 4.0*sigma / c
Tr    = me * c2 / kappaB
pi05  = np.sqrt(np.pi)
RBohr = hbar**2 / (me*e_erg)
avo   = 1.0/mu


###########################
imax  = 541 
jmax  = 201
Ndats = imax*jmax

#Temperature range
tlo   = 3.0e0 
thi   = 13.0e0
tstp  = (thi - tlo)/float(jmax-1)
tstpi = 1.0e0/tstp

#density range
dlo   = -12.0e0 
dhi   = 15.0e0
dstp  = (dhi - dlo)/float(imax-1)
dstpi = 1.0e0/dstp

# ---- Read Table EoS ----
dats=[]
with open("./helmholtzEoS/helm_table.dat","r") as theFile:            
    for line in theFile:
           p1 = line.split('\n')
           p2 = p1[0].split(' ')
           line2 = []
           for pp in p2:
               if pp!='' and pp!=' ':
                  line2.append(float(pp))
           dats.append(line2)
  
#Free energy
T_grid   = np.zeros(jmax)
rho_grid = np.zeros(imax)
FE = np.zeros( (jmax, imax, 9) ) 

#0: free energy; 1: df_drho; 2: df_dt; 3: df_drhorho; 4: df_dtdt
#5: df_drhodt; 6: df_drhodrhodt; 7: df_drhodtdt; 8: df_drhodrhodtdt;
k=0
for j in range(jmax):
    T_grid[j] = np.power(10.0,tlo+(j)*tstp)
    for i in range(imax):
        rho_grid[i] = np.power(10.0,dlo+(i)*dstp)
        FE[j,i,:]   = [ dats[k][0],dats[k][2],dats[k][4],dats[k][1],dats[k][3],dats[k][5],dats[k][6],dats[k][7],dats[k][8] ] 
        k+=1
      
  
deltaT      = np.zeros(jmax-1) 
deltaT2     = np.zeros(jmax-1)
deltaT_inv  = np.zeros(jmax-1)
deltaT2_inv = np.zeros(jmax-1)
deltaT3_inv = np.zeros(jmax-1)

deltarho      = np.zeros(imax-1) 
deltarho2     = np.zeros(imax-1) 
deltarho_inv  = np.zeros(imax-1) 
deltarho2_inv = np.zeros(imax-1) 
deltarho3_inv = np.zeros(imax-1) 

for j in range(jmax-1):
    dth  = T_grid[j+1]-T_grid[j]
    dti  = 1.0e0/dth
   
    deltaT[j]      = dth
    deltaT2[j]     = dth*dth
    deltaT_inv[j]  = dti 
    deltaT2_inv[j] = dti*dti
    deltaT3_inv[j] = dti*dti*dti

for i in range(imax-1):
    dd  = rho_grid[i+1]-rho_grid[i]
    ddi = 1.0e0/dd
    deltarho[i]      = dd
    deltarho2[i]     = dd*dd
    deltarho_inv[i]  = ddi
    deltarho2_inv[i] = ddi*ddi
    deltarho3_inv[i] = ddi*ddi*ddi

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
    A1 = (fi[1]*wd[0] + fi[2]*wmd[0])*wt[0] + (fi[3]*wd[0] + fi[4]*wmd[0])*wmt[0] 
    A2 = (fi[5]*wd[0] + fi[6]*wmd[0])*wt[1] + (fi[7]*wd[0] + fi[8]*wmd[0])*wmt[1] 
    A3 = (fi[9]*wd[0] + fi[10]*wmd[0])*wt[2] + (fi[11]*wd[0] + fi[12]*wmd[0])*wmt[2] 

    B1 = (fi[13]*wd[1] + fi[14]*wmd[1])*wt[0] + (fi[15]*wd[1] + fi[16]*wmd[1])*wmt[0]  
    B2 = (fi[17]*wd[2] + fi[18]*wmd[2])*wt[0] + (fi[19]*wd[2] + fi[20]*wmd[2])*wmt[0]    
    
    C1 = (fi[21]*wd[1] + fi[22]*wmd[1])*wt[1] + (fi[23]*wd[1] + fi[24]*wmd[1])*wmt[1]  
    C2 = (fi[25]*wd[2] + fi[26]*wmd[2])*wt[1] + (fi[27]*wd[2] + fi[28]*wmd[2])*wmt[1]  
    
    D1 = (fi[29]*wd[1] + fi[30]*wmd[1])*wt[2] + (fi[31]*wd[1] + fi[32]*wmd[1])*wmt[2] 
    D2 = (fi[33]*wd[2] + fi[34]*wmd[2])*wt[2] + (fi[35]*wd[2] + fi[36]*wmd[2])*wmt[2] 
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
  A1 = (fi[1]*wd[0] + fi[2]*wmd[0])*wt[0] + (fi[3]*wd[0] + fi[4]*wmd[0])*wmt[0] 
  A2 = (fi[5]*wd[0] + fi[6]*wmd[0])*wt[1] + (fi[7]*wd[0] + fi[8]*wmd[0])*wmt[1] 
  
  B1 = (fi[9]*wd[1] + fi[10]*wmd[1])*wt[0] + (fi[11]*wd[1] + fi[12]*wmd[1])*wmt[0]  
    
  C1 = (fi[13]*wd[1] + fi[14]*wmd[1])*wt[1] + (fi[15]*wd[1] + fi[16]*wmd[1])*wmt[1]  
  return A1+A2+B1+C1


#--- Heat Capacity ---
def cv_gas(rho, T, agas, xxgas,zgas):
    #average atomic weight and charge
    #ration=0.0; XmassT=0.0; ZionT=0.0; AionT=0.0
    #for i in range(len(agas)):
    #    ration+=xxgas[i]/agas[i]
    #    XmassT+=xxgas[i]*zgas[i]/agas[i]

    ration = (xxgas/agas).sum()
    XmassT = (xxgas*zgas/agas).sum()
        
    abar   = 1.0e0/ration
    zbar   = abar * XmassT
   
    ytot1  = 1.0e0/abar
    ye     = max(1.0e-16, ytot1 * zbar) 

    #print( "abar= ", abar, " zbar= ", zbar," Ye=", ye)
    
    deni   = 1.0/rho
    tempi  = 1.0/T
    kT     = kappaB * T
    kT_inv = 1.0 / kT

#   print( "log Temperature:", np.log10(T), "log rho: ", np.log10(rho) )
#radiation:
    prad     = (1.0/3.0) * arad * np.power(T,4.0)
    dprad_dT = 4.0 * prad * tempi

    erad     = 3.0 * prad * deni
    derad_dT = 3.0 * dprad_dT * deni
    
    srad = ( prad * deni + erad) * tempi

#ion:
    xini     = rho * avo * ytot1
    dxini_dd = avo * ytot1
    dxini_da = -xini * ytot1

    pion     = xini * kT
    dpion_dT = pion * tempi
 
    eion     = 1.5 * pion * deni
    deion_dT = 1.5 * dpion_dT * deni

# sackur-tetrode equation for the ion entropy of a single ideal gas characterized by abar
    x        = abar*abar*np.sqrt(abar) * deni/avo
    sioncon  = (2.0 * np.pi * mu * kappaB) / (h*h)
    s        = sioncon * T
    z        = x * s * np.sqrt(s)
    y        = np.log(z)

#        y       = 1.0d0/(abar*kt)
#        yy      = y * sqrt(y)
#        z       = xni * sifac * yy
#        etaion  = log(z)
    kergavo = kappaB*avo
    sion    = (pion * deni + eion) * tempi + kergavo * ytot1 * y


#assume complete ionization
    xnem    = xini * zbar
    din     = ye*rho
    if T>T_grid[-1]:
       print("Temperature too hot")
       return
    if T<T_grid[0]:
       print("Temperature too cold")
       return
    if din<rho_grid[0]:
       print("density too low")
       return
    if din>rho_grid[-1]:
       print("density too high")
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
        
 #  print( T_grid[jat], rho_grid[iat], rho_grid[iat+1])
   
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
    deep_dt = T * dsep_dt

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
    plasg_dT  = -plasg*kT_inv * kappaB
    plasg_dz  = 2.0e0 * plasg/zbar
    
# yakovlev & shalybkov 1989 equations 82, 85, 86, 87
    if (plasg >= 1.0):
        x = np.power(plasg,0.25)
        y = avo * ytot1 * kappaB
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
        scoul = -avo/abar*kappaB*(c2*x -a2*(b2-1.0e0)/b2*y)

        s  = 1.5e0*c2*x/plasg - third*a2*b2*y/plasg
        dpcoul_dt = -dpion_dT*z - pion*s*plasg_dT
       
        s        = 3.0e0/rho
        decoul_dt = s * dpcoul_dt
       

    x   = prad + pion + pele + pcoul
    y   = erad + eion + eele + ecoul
    z   = srad + sion + sele + scoul

    
    if (x < 0.0 or y < 0.0):
       print('coulomb corrections are causing a negative pressure')
       print('setting all coulomb corrections to zero')

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
    sgas   =  sion + sele + scoul

   # degasdt = deion_dT + decoul_dt + deep_dt 
    degasdt = deion_dT + deep_dt #+ decoul_dt
    pres= pgas + prad
    ener= egas + erad
    entr= sgas + srad

    #denerdt = derad_dT + degasdt 
    denerdt =  degasdt
    cv_gas = abar * denerdt / (kappaB*avo) #+ ion.Cvii(abar,zbar,rho,T) + ionee.Cvie(abar,zbar,rho,T)
    
    return cv_gas
