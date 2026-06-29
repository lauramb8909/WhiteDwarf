import numpy as np 
from scipy.integrate import ode, odeint
from scipy.optimize import brentq, curve_fit
import Physical_Const as phys
from scipy.interpolate import interp1d,InterpolatedUnivariateSpline

#----Constants------
pii    = np.pi




def TOV(r,y, EOS):
#----TOV Equation: Static sequence----------
#ec1: dMdr ; ec2: dnudr ; ec3: dP/dr
#---- HT first two equations----------------
#ec4 : djdr ; ec5 : domegadr

    mns  = y[0]
    pns  = y[1]
    nuns = y[2]

    if pns<1e-8:
       pns = 1e-8

    rhons = EOS(pns)

    r2 = r*r
    r3 = r2*r

    ec2 = 2.0 * ( 4.0 * pii * r3 * pns + mns ) / ( r * ( r - 2.0 * mns ) )
    ec1 = 4.0 * pii * r2 * rhons
    ec3 = - 0.5 * ec2 * ( pns + rhons )
    ec4 = y[3]
    ec5 = - 4.0 * y[3] / r + 4.0 * pii * r2 * ( pns + rhons ) * ( y[3] + 4.0 * y[4] / r ) / ( r - 2.0 * mns )
    
    return np.array( [ ec1, ec3, ec2, ec5, ec4] )


def TOV_HT(r,y, EOS, DEOS):  
#########################################
#----TOV Equation: Static sequence----------
#ec1: dMdr ; ec2: dnudr ; ec3: dP/dr
#---- HT first order equations in Omega----------------
#ec4 : djdr ; ec5 : domegadr
# ---- HT second order equations in Omega----------------
#ec6 : dm0dr ; ec7 : dpstardr
#ec8 : dvdr ; ec9 : dh2drup
#########################################

    mns  = y[0]
    pns  = y[1]
    nuns = np.exp(-y[2])

    if pns<1e-8:
       pns = 1e-8

    r2 = r*r
    r3 = r2*r 
    r4 = r2*r2
    pi4 = 4.0*pii  ; pi2 = 2.0*np.pi

    rhons = EOS(pns)
    DrhodP = np.power(10.0, DEOS( np.log10(pns) ) )
    rho2   = ( pns + rhons ) * y[6] * DrhodP
    p2     = y[6] * ( pns + rhons )
    p2l2   = -y[8] - ( 1.0 / 3.0) * r2 * nuns * y[4]**2
    
    ec1 = pi4 * r2 * rhons   # dmdr
    ec2 = 2.0 * ( pi4 * r3 * pns + mns ) / ( r * ( r - 2.0*mns ) )  #dnudr
    ec3 = -0.5 * ec2 * ( pns + rhons) #dpdr
    
    ec4 = y[3] #domegadr
    ec5 = - 4.0 * y[3] / r + 4.0 * pii * r * ( pns + rhons ) * ( y[3] * r + 4.0 * y[4] ) / ( r - 2.0 * mns )

    ### l=0 (second order in Omega)
    ec6 = pi4 * r2 * rho2  + ( r3 *nuns / 12.0 ) * y[3]**2.0 * ( r - 2.0*mns ) + (2.0 * pi4 / 3.0) *r4 *nuns * ( pns + rhons ) * y[4]**2.0    #dm0dr

    dh0 = y[5]*( 1.0 + 2.0 * pi4 * r2*pns ) / (r - 2.0 * mns)**2.0 + pi4 * r2 * p2 / ( r - 2.0*mns ) - r3*nuns * y[3]**2 / 12
     
    ec7 = -dh0 + ( nuns / 3.0) * ( y[4]**2 * ( 2.0*r - r2*ec2 ) + 2.0*r2 * y[3] * y[4] ) #dp0dr

    ####### l=2 (second order in Omega)

    m2  = -y[8] + (2.0*pi4 / 3.0)* r4 * nuns * ( rhons + pns ) * y[4]**2.0  + (1.0/6.0) * r3 * nuns * ( r-2*mns) * y[3]**2.0

    ec8 = y[8] * ( 1.0 / r - ec2 / 2.0 ) + m2 * ( 1.0 / r + ec2 / 2.0 ) # dvdr

    ec9 = -( y[8] + 2.0* y[7] ) /  (mns + pi4*r3*pns ) + ec8 * ( 1.0 + 2.0 / ( r*ec2) ) - 2.0*(m2/r2)*(1.0/ec2 + r) - 2.0*pi4*r*p2l2 / ( ( r - 2.0 * mns ) * ec2 ) - ( nuns /  6.0 ) * ( r2 / ec2 ) * y[4]**2.0  #dh2dr

    ec10 = -y[10]*ec2 #dvdr (homogenea)

    ec11 = -( y[10] + 2.0* y[9] ) /  (mns + pi4*r3*pns ) + ec10 * ( 1.0 + 2.0 / ( r*ec2) ) + 2.0*(y[10]/r2)*(1.0/ec2 + r) + 2.0*pi4*r*y[10] / ( ( r - 2.0 * mns ) * ec2 )  #dh2dr (homogenea)
    
    return np.array( [ ec1, ec3, ec2, ec5, ec4, ec6, ec7, ec8, ec9, ec10, ec11 ] )



def OmegaK( Mwd, Rwd, DM, jwd, qwd):
###---- APJ, 762:117 (14pp), 2013  (Boshkayev, Rueda, Ruffini and Siutsou)
    x  = Mwd / Rwd
    x2 = x*x
    x3 = x2*x
    x4 = x2*x2
    x5 = x4*x
    x6 = x3*x3
    x7 = x6*x

    x3_1 = 1.0/x3
    
    F = (15.0/32.0) * ( x3_1 - 2.0) * np.log( 1.0 / (1.0-2.0*x) )
    
    F1 = x**1.5
    F4 = ( 48.0 * x7 - 80.0*x6 + 4.0*x5 - 18.0 * x4 + 40.0 * x3 + 10.0 * x2 + 15.0 * x - 15.0 ) / ( 1.0 - 2.0*x )
    F2 = F4 / np.power( 4.0*x , 2.0) + F
    
    F3 = ( 5.0 / (16.0*x2) ) * ( 6.0*x4 - 8.0 * x3 - 2.0 * x2 - 3.0*x + 3.0 ) / ( 1.0 - 2.0*x ) - F
    
    F6 = 1.0 + DM/(2.0*Mwd) + ( -jwd * F1 + np.power(jwd,2.0) * F2 + qwd*F3 )
   
    if F6<0.0:
        F6 = 1.0 - ( -jwd * F1 + np.power(jwd,2.0) * F2 + qwd*F3 )
        
    return np.sqrt( Mwd / np.power(Rwd,3.0) ) * F6

#------ Functions to caculate the quadrupole
def Q22(x):
    
    if x>1e3: 
        x1  = 1.0/x
        x2  = x1 * x1
        x3  = x1 * x2
        x5  = x2 * x3
        x6  = x3 * x3
        x7  = x6 * x1 
        x9  = x6 * x3
        x11 = x9 * x2
        x12 = x6 * x6
        x13 = x12 * x1
        x15 = x12 * x3 
        A   = 168.0 * x15 / 85.0  + 128.0 * x13 / 65.0 + 280.0 * x11 / 143.0 + 64.0 * x9 / 33.0 + 40.0 * x7 / 21.0 + 64.0 * x5 / 35.0  + 8.0 * x3 / 5.0
    else:
        x2 = x*x 
        x3 = x2*x
        A   = (3.0/2.0)*( x2 - 1.0)*( np.log(x+1.0) - np.log(x-1.0) ) - ( 3.0*x3 - 5.0*x) /( x2 - 1.0 )
    return A


def Q12(x):
    
    if x>1e3:
        x1  = 1.0/x
        x2  = x1 * x1
        x3  = x1 * x2
        x5  = x2 * x3
        x6  = x3 * x3
        x7  = x6 * x1 
        x9  = x6 * x3
        x11 = x9 * x2
        x12 = x6 * x6
        x13 = x12 * x1
        x15 = x12 * x3 

        A = 866483.0 * x15 / 3734016.0 + 95923.0 * x13 / 384384.0 + 52069.0 * x11 / 192192.0 + 2749.0 * x9 / 9240.0 + 139.0 * x7 / 420.0 + 13.0 * x5 / 35.0 + 2.0 *x3 / 5.0
    else:
        x2 = x*x
        x3 = x2*x
        A = ( 3.0 / 2.0 ) * x * np.sqrt( x2 - 1.0 ) * ( np.log( x - 1.0 ) - np.log( x + 1.0 ) ) + ( 3.0*x2 - 2.0 ) / np.sqrt( x2 - 1.0 )
    return A

def Awd( jwd, Mwd, Rwd, h2h, h2p, v2h, v2p ):
    xx = Rwd / Mwd
    xj = jwd / Rwd**2
    
    A1 = np.array ( [ [ -Q22( xx - 1.0 ) , h2h ] , [ -(2.0) * Q12( xx - 1.0 ) / np.sqrt ( xx**2 - 2.0 * xx ) , v2h ] ] )
    A2 = np.array ( [ xj**2 * ( xx + 1.0 ) - h2p, - xj**2 - v2p ] )
    return  np.linalg.solve(A1, A2)

def IntCond_Static( rhoc, pc, drr ):
#--- Expasion around the center of the Static eqution 

    dr2    = drr * drr
    dr3    = dr2 * drr

    m0c    =  4.0 * np.pi * rhoc * dr3  / 3.0 
    Pi     = pc - 2.0 * np.pi * ( pc + rhoc ) * ( pc + rhoc/3.0 ) * dr2
    nuc    = 4.0*np.pi*(  pc + rhoc / 3.0 )*dr2 
    
    omegac  = 1.0 + 8.0 * np.pi * ( pc + rhoc ) * dr2 / 3.0
    Domegac = 16.0 * np.pi * ( pc + rhoc ) * drr / 3.0

        
    return np.array( [ m0c , Pi, nuc , Domegac , omegac ] )

def IntCond_Rotating( OmegaS, Oold, rhoc, pc, nu0, drr , rf , eos , deos ):
#--- Expasion around the center of the eqaution to star the integration

    dr2    = drr * drr
    dr3    = dr2 * drr
    dr4    = dr2 * dr2
    dr5    = dr4 * drr

    m0c    =  4.0 * np.pi * rhoc * dr3  / 3.0 
    Pi     = pc - 2.0 * np.pi * ( pc + rhoc ) * ( pc + rhoc/3.0 ) * dr2
    nuc    = nu0 + 4.0*np.pi*(  pc + rhoc / 3.0 )*dr2 
    
    omegac  = 1.0 + 8.0 * np.pi * ( pc + rhoc ) * dr2 / 3.0
    Domegac = 16.0 * np.pi * ( pc + rhoc ) * drr / 3.0

    omega0  = OmegaS / Oold
    omegac  = omegac * omega0
    Domegac =  Domegac *  omega0 
    
    m2i     =  np.exp(-nuc) * (4.0*np.pi/15.0) * ( pc + rhoc ) * ( 2.0 + deos ) * dr5 
    p0i     = (1.0/3.0) * dr2 * np.exp(-nuc) 
    m00     = m2i * omega0**2
    p00     = p0i * omega0**2
    
    v20     = 2.0 * np.pi * p00 * ( pc + rhoc ) * dr2 - 2.0 * np.pi * ( pc + rhoc/ 3.0 ) * dr4 
    h20     = dr2
    
    m20     = - 2.0 * np.pi * ( pc + rhoc / 3.0 ) * dr4
    p20     = dr2
    
    return np.array( [ m0c , Pi, nuc , Domegac , omegac , m00 , p00 , v20 , h20, m20 , p20 ] )

#----Loop to calculate Static sequence--------
def StaticSeq(y0,drr,rff, eos ):
    r0 = drr
    EoS_RFMT    = interp1d( eos[0] , eos[1] , kind='cubic' , bounds_error=False)
    EoS_RFMT_02 = interp1d( eos[1] , eos[0] , kind = 'cubic', bounds_error=False)
    rho1 = eos[0][0]

    Static = ode(TOV)
    Static.set_integrator('dopri5',atol=1e-9)
    Static.set_initial_value(y0,r0)
    Static.set_f_params(EoS_RFMT_02)

    rhons = EoS_RFMT_02( Static.y[1] ) 

    while Static.successful() and Static.t < rff :
        Static.integrate(Static.t+drr)
        if rhons > 10*rho1 and Static.y[1]>0.0:
            rhons = EoS_RFMT_02( Static.y[1] ) 
        else:
            break

    mstar  = Static.y[0]
    rstar  = Static.t
    nuc    = np.log( 1.0 - 2.0*mstar / rstar ) - Static.y[2]

    Jstar  = np.power( rstar , 4.0 ) * Static.y[3] / 6.0
    omegastar = Static.y[4] + 2.0 * Jstar / np.power(rstar,3.0)
      
    return [ mstar, rstar, nuc, Jstar, omegastar ]


def MassRadius( y0,drr,rff, eos, deos ):
    r0 = drr
    EoS_RFMT    = interp1d( eos[0] , eos[1] , kind='cubic' )
    EoS_RFMT_02 = interp1d( eos[1] , eos[0] , kind = 'cubic')
    DeDpF       = interp1d( np.log10( eos[1] ),  np.log10( deos ) , kind='cubic')
    rho1 = eos[0][0]
    p1   = eos[1][0]

    test=ode(TOV_HT).set_integrator('dopri5',atol=1e-7)
    test.set_initial_value(y0,r0)
    test.set_f_params(EoS_RFMT_02, DeDpF)

    rhons = EoS_RFMT_02( test.y[1] ) 

    while test.successful() and test.t < rff :
       
        test.integrate(test.t+drr)

        if rhons > 10*rho1 and test.y[1]>p1:
            rhons = EoS_RFMT_02( test.y[1] ) 
        else:
            test.integrate(test.t-drr)
            break
    
    mstar     = test.y[0]
    rstar     = test.t

    R2        = rstar * rstar
    R3        = R2 * rstar
    R4        = R2 * R2
    nuc       = test.y[2]
    Jstar     = R4 * test.y[3] / 6.0
    omegastar = test.y[4] + 2.0*Jstar / R3

    #rhons     = EoS_RFMT_02( test.y[1] )
   
    J2        = Jstar*Jstar
    Mass      =  mstar + test.y[5] + J2 / R3 

    dff       = TOV_HT(rstar,test.y, EoS_RFMT_02, DeDpF)
    epsilonz  = test.y[6] * rstar * ( rhons + test.y[1]) / abs( dff[1] ) #* ( rstar - 2.0 * mstar ) / ( mstar + 4.0*pii * R3 * test.y[1] )
    
    AAs       = Awd(Jstar,mstar,rstar,test.y[10],test.y[8],test.y[9],test.y[7])
    v2star    = test.y[7] + AAs[1]*test.y[9]
    h2star    = test.y[8] + AAs[1]*test.y[10]
    ppt       = -h2star - ( 1.0 / 3.0) * R2 * np.exp( -nuc ) * test.y[4]**2
    epsilont  = ppt * rstar * ( rhons + test.y[1]) / abs( dff[1] ) #* ( rstar - 2.0 * mstar ) / ( mstar + 4.0 * pii *R3 * test.y[1]  )
    #equatorial radius
    RR        =  rstar + epsilonz - 0.5*epsilont - 0.5 * rstar * ( h2star - v2star )
    
    Qstar     = J2 / mstar + (8.0/5.0) * mstar**3 * AAs[0]

    return [ mstar, rstar,nuc, Jstar, omegastar, Qstar, Mass, RR]


def MassRadius_Structure( y0,drr,rff, eos, deos ):
    r0 = drr
    EoS_RFMT    = interp1d( eos[0] , eos[1] , kind='cubic' )
    EoS_RFMT_02 = interp1d( eos[1] , eos[0] , kind = 'cubic')
    DeDpF       = interp1d( np.log10( eos[1] ),  np.log10( deos ) , kind='cubic')
    rho1 = eos[0][0]
    p1   = eos[1][0]

    test=ode(TOV_HT).set_integrator('dopri5',atol=1e-7)
    test.set_initial_value(y0,r0)
    test.set_f_params(EoS_RFMT_02, DeDpF)

    rhons = EoS_RFMT_02( test.y[1] ) 

    datai =  [  test.t ]
    datai.extend( test.y[:]  )
    data = [datai]

    while test.successful() and test.t < rff :
       
        test.integrate(test.t+drr)

        datai =  [  test.t ]
        datai.extend( test.y[:]  )
        

        if rhons > 10*rho1 and test.y[1]>p1:
            rhons = EoS_RFMT_02( test.y[1] ) 
            data.append( datai  )
        else:
            test.integrate(test.t-drr)
            break
    
    #mstar     = test.y[0]
    #rstar     = test.t

    #R2        = rstar * rstar
    #R3        = R2 * rstar
    #R4        = R2 * R2
    #nuc       = test.y[2]
    #Jstar     = R4 * test.y[3] / 6.0
    #omegastar = test.y[4] + 2.0*Jstar / R3

    #rhons     = EoS_RFMT_02( test.y[1] )
   
    #J2        = Jstar*Jstar
    #Mass      =  mstar + test.y[5] + J2 / R3 

    #dff       = TOV_HT(rstar,test.y, EoS_RFMT_02, DeDpF)
    #epsilonz  = test.y[6] * rstar * ( rhons + test.y[1]) / abs( dff[1] ) #* ( rstar - 2.0 * mstar ) / ( mstar + 4.0*pii * R3 * test.y[1] )
    
    #AAs       = Awd(Jstar,mstar,rstar,test.y[10],test.y[8],test.y[9],test.y[7])
    #v2star    = test.y[7] + AAs[1]*test.y[9]
    #h2star    = test.y[8] + AAs[1]*test.y[10]
    #ppt       = -h2star - ( 1.0 / 3.0) * R2 * np.exp( -nuc ) * test.y[4]**2
    #epsilont  = ppt * rstar * ( rhons + test.y[1]) / abs( dff[1] ) #* ( rstar - 2.0 * mstar ) / ( mstar + 4.0 * pii *R3 * test.y[1]  )
    #equatorial radius
    #RR        =  rstar + epsilonz - 0.5*epsilont - 0.5 * rstar * ( h2star - v2star )
    
    #Qstar     = J2 / mstar + (8.0/5.0) * mstar**3 * AAs[0]

    return data



def EoS_Mmax( rhoii, rhoff, dr, rf, eos):
#Rutine to find the Max Mass at J

    EoS_RFMT    = interp1d( eos[0] , eos[1] , kind='cubic' , bounds_error=False)
    EoS_RFMT_02 = interp1d( eos[1] , eos[0] , kind = 'cubic', bounds_error=False)

    ww = 0.6180399; zz = 1.0 - ww;

    xmin   = EoS_RFMT( rhoii )
    xmax   = EoS_RFMT( rhoff )
    
    x0  = xmin + 0.0*( xmax - xmin );
    xp  = x0 + 0.4*( xmax - x0 ) ;
    y0i = IntCond_Static( EoS_RFMT_02(xp) , xp, dr )
    mass_m  = StaticSeq( y0i , dr, rf, eos )[0]
  
    xnew =  xp + zz*( xmax - x0) 
    y0i  = IntCond_Static( EoS_RFMT_02(xnew), xnew, dr )
    mass_new  = StaticSeq( y0i , dr, rf, eos )[0]

    for i in range(100):
        
        if ( mass_new > mass_m ):
            x0   = xp;
            xp   = xnew
            xnew = ww*xnew + zz*xmax 
            
            mass_m = mass_new
    
            y0i = IntCond_Static( EoS_RFMT_02(xnew), xnew, dr )
            mass_new  = StaticSeq( y0i , dr, rf, eos )[0]

        else:
            xmax = xnew
            xnew = xp
            xp   =  ww*xp + zz*x0
            
            mass_new = mass_m;
         
            y0i = IntCond_Static( EoS_RFMT_02(xp), xp, dr )
            mass_m  = StaticSeq( y0i , dr, rf, eos )[0]
          
        if ( ( xmax - x0 )< 1e-9*( xp + xmax)  ): 
            break;
            
    if (mass_m>mass_new):
        xmax = xp;
    else:
        xmax = xnew;

    return [ EoS_RFMT_02(xmax), mass_m]


def EoS_RoTMmax( rhoii, rhoff,jwd, dr, rf, eos, deos):
#Rutine to find the Max Mass at J

    EoS_RFMT    = interp1d( eos[0] , eos[1] , kind='cubic' , bounds_error=False)
    EoS_RFMT_02 = interp1d( eos[1] , eos[0] , kind = 'cubic', bounds_error=False)
    DeDpF       = interp1d( np.log10( eos[1] ),  np.log10( deos ) , kind='cubic')

    ww = 0.6180399;zz = 1.0 - ww;

    xmin   = EoS_RFMT( rhoii )
    xmax   = EoS_RFMT( rhoff )
    
    x0  = xmin + 0.0*( xmax - xmin ) 
    xp  = x0 + 0.4*( xmax - x0 ) 
    y0i = IntCond_Static( EoS_RFMT_02(xp) , xp, dr )
    MS  = StaticSeq( y0i , dr, rf, eos )
    y0i = IntCond_Rotating( jwd, MS[3],  EoS_RFMT_02(xp), xp, MS[2], dr , rf , eos, DeDpF(np.log10(xp)) )
    mass_m   = MassRadius(y0i, dr, rf,  eos, deos )[6]

  
    xnew =  xp + zz*( xmax - x0) 
    y0i  = IntCond_Static( EoS_RFMT_02(xnew), xnew, dr )
    MS  = StaticSeq( y0i , dr, rf, eos )
    y0i = IntCond_Rotating( jwd, MS[3],  EoS_RFMT_02(xnew), xnew, MS[2], dr , rf , eos, DeDpF(np.log10(xnew)) )
    mass_new   = MassRadius(y0i, dr, rf,  eos, deos )[6]


    for i in range(100):


        if ( mass_new > mass_m ):
            x0   = xp;
            xp   = xnew
            xnew = ww*xnew + zz*xmax 
            
            mass_m = mass_new
    
            y0i = IntCond_Static( EoS_RFMT_02(xnew), xnew, dr )
            MS  = StaticSeq( y0i , dr, rf, eos )
            y0i = IntCond_Rotating( jwd, MS[3],  EoS_RFMT_02(xnew), xnew, MS[2], dr , rf , eos,  DeDpF(np.log10(xnew)) )
            mass_new   = MassRadius(y0i, dr, rf,  eos, deos )[6]

        else:
            xmax = xnew
            xnew = xp
            xp   =  ww*xp + zz*x0
            
            mass_new = mass_m;
         
            y0i = IntCond_Static( EoS_RFMT_02(xp), xp, dr )
            MS  = StaticSeq( y0i , dr, rf, eos )
            y0i = IntCond_Rotating( jwd, MS[3],  EoS_RFMT_02(xp), xp, MS[2], dr , rf , eos,  DeDpF(np.log10(xp)) )
            mass_m  = MassRadius(y0i, dr, rf,  eos, deos )[6]

          
        if ( ( xmax - x0 )< 1e-9*( xp + xmax)  ): 
            break;
            
    if (mass_m>mass_new):
        xmax = xp;
    else:
        xmax = xnew;

    return [ EoS_RFMT_02(xmax), mass_m]

