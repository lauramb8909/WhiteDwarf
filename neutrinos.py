import numpy as np 
from math import pi
import Physical_Const as phys
from pylab import *


h=phys.h
c=phys.c
me=phys.me
mu=phys.mu
kappaB=phys.kappaB

ee=2.310271142e-19

theta_w=np.arcsin(np.sqrt(0.2319))
Cv=1.0/2.0+2.0*np.sin(theta_w)**2.0
Cvv=1.0-Cv
Ca=1.0/2.0
Caa=1.0-Ca

def Lambda(T):
    return  T / 5.9302e9

def epsilon(rho,T):
    return np.power( rho/1e9 ,1.0/3.0) / Lambda(T)

def Gamma2(A,Z,rho):
   return  np.power(Z,2.0) * ee / ( kappaB * ( 1.0/np.power(4.0*np.pi*rho/(3.0*A*mu),1.0/3.0)))


###################################################################################################################################################
#Pair neutrino process
###################################################################################################################################################
def Qpair(rho,T,n):
    rho=rho/2.0
    a1=2.084e20;a2=1.872e21;a0=6.002e19
    if T<1e10: 
       b1pair=9.383e-1; b2pair=-4.141e-1; b3pair=5.829e-2; cpair=5.5924
    else:
       b1pair=1.2383; b2pair=-0.8141; b3pair=0.0; cpair=4.9923
    qpair=np.power(1.0+rho/(7.692e7*np.power(Lambda(T),3.0)+9.715e6*np.sqrt(Lambda(T))),-0.3)/(10.748*Lambda(T)**2+0.3967*np.sqrt(Lambda(T))+1.005)
    gpair=1.0-13.04*Lambda(T)**2.0+133.5*np.power(Lambda(T),4.0)+1534*np.power(Lambda(T),6.0)+918.6*np.power(Lambda(T),8.0);
    fpair=(a0+a1*epsilon(rho,T)+a2*epsilon(rho,T)**2.0)*np.exp(-cpair*epsilon(rho,T))/(epsilon(rho,T)**3+b1pair/Lambda(T)+b2pair/Lambda(T)**2.0+b3pair/np.power(Lambda(T),3))
    return 0.5*((Cv**2.0+Ca**2.0)+n*(Cvv**2.0+Caa**2.0))*(1+((Cv**2.0-Ca**2.0)+n*(Cvv*2.0-Caa**2.0))/((Cv**2.0+Ca**2.0)+n*(Cvv**2.0+Caa**2.0))*qpair)*gpair*np.exp(-2.0/Lambda(T))*fpair;

cc=[[1.008e11,0.0,0.0,0.0,0.0, 0.0,0.0],[8.156e10,9.728e8,-3.806e9,-4.384e9,-5.774e9,-5.249e9,-5.153e9],[1.067e11,-9.782e9,-7.193e9,-6.936e9,-6.893e9,-7.041e9,-7.193e9]]


###################################################################################################################################################
#Photoneutrino process
###################################################################################################################################################
def Qphoto_1(rho,T,n):
    rho=rho/2.0
    b4=6.290e-3;b5=7.483e-3; b6=3.061e-4
    if T>=1e7 and T<1e8:
       dphoto=0.5654+log10(T/1e7)
       tau=np.log10(T/1e7)
       cc=[[1.008e11,0.0,0.0,0.0,0.0,0.0,0.0],[8.156e10,9.728e8,-3.806e9,-4.384e9,-5.774e9,-5.249e9,-5.153e9],[1.067e11,-9.782e9,-7.193e9,-6.936e9,-6.893e9,-7.041e9,-7.193e9]]
       dd=[[0.0,0.0,0.0,0.0,0.0],[ -1.879e10,-9.667e9,-5.602e9,-3.370e9,-1.825e9],[-2.919e10,-1.185e10,-7.270e9,-4.222e9,-1.560e9]]
       AAphoto1=cc[0][0];AAphoto2=cc[1][0];AAphoto3=cc[2][0];
       i=1
       while i<6:
           AAphoto1=AAphoto1+ cc[0][i]*cos(5.0/3.0*np.pi*i*tau)+dd[0][i-1]*sin(5.0/3.0*np.pi*i*tau)
           AAphoto2=AAphoto2+ cc[1][i]*cos(5.0/3.0*np.pi*i*tau)+dd[1][i-1]*sin(5.0/3.0*np.pi*i*tau)
           AAphoto3=AAphoto3+ cc[2][i]*cos(5.0/3.0*np.pi*i*tau)+dd[2][i-1]*sin(5.0/3.0*np.pi*i*tau)
           i=i+1
    elif T>=1e8 and T<1e9:
       dphoto=1.5654
       tau=np.log10(T/1e8)
       cc=[[9.889e10,-4.524e8,-6.088e6,4.269e7,5.172e7,4.910e7,4.388e7],[1.813e11,-7.556e9,-3.304e9,-1.031e9,-1.764e9,-1.851e9,-1.928e9],[9.750e11,3.484e10,5.199e9,-1.695e9,-2.865e9,-3.395e9,-3.418e9]]
       dd=[[-1.135e8,1.256e8,5.149e7,3.436e7,1.005e7],[1.652e9 ,-3.119e9 ,-1.839e9,-1.458e9,-8.956e8],[-1.549e10,-9.338e9,-5.899e9,-3.035e9,-1.598e9]]
       AAphoto1=cc[0][0];AAphoto2=cc[1][0];AAphoto3=cc[2][0];
       i=1
       while i<6:
           AAphoto1=AAphoto1+ cc[0][i]*cos(5.0/3.0*np.pi*i*tau)+dd[0][i-1]*sin(5.0/3.0*np.pi*i*tau)
           AAphoto2= AAphoto2+cc[1][i]*cos(5.0/3.0*np.pi*i*tau)+dd[1][i-1]*sin(5.0/3.0*np.pi*i*tau)
           AAphoto3= AAphoto3+cc[2][i]*cos(5.0/3.0*np.pi*i*tau)+dd[2][i-1]*sin(5.0/3.0*np.pi*i*tau)
           i=i+1
    elif T>=1e9:
       dphoto=1.5654
       tau=np.log10(T/1e9)
       cc=[[9.581e10,4.107e8,2.305e8 ,2.236e8,1.580e8,2.165e8 ,1.721e8],[1.459e12,1.314e11,1.169e11,-1.765e11,-1.867e11,-1.983e11,-1.896e11],[ 2.424e11,-3.669e9,-8.691e9,-7.967e9 ,-7.932e9,-7.987e9,-8.333e9]]
       dd=[[4.724e8,2.976e8 ,2.242e8,7.937e7,4.859e7],[-7.094e11,-3.69711,-2.189e11,-1.273e11,-5.705e10],[-2.254e10,-1.551e10,-7.793e9,-4.489e9,-2.185e9]]
       AAphoto1=cc[0][0];AAphoto2=cc[1][0];AAphoto3=cc[2][0];
       i=1
       while i<6:
           AAphoto1= AAphoto1+cc[0][i]*cos(5.0/3.0*np.pi*i*tau)+dd[0][i-1]*sin(5.0/3.0*np.pi*i*tau)
           AAphoto2= AAphoto2+cc[1][i]*cos(5.0/3.0*np.pi*i*tau)+dd[1][i-1]*sin(5.0/3.0*np.pi*i*tau)
           AAphoto3= AAphoto3+cc[2][i]*cos(5.0/3.0*np.pi*i*tau)+dd[2][i-1]*sin(5.0/3.0*np.pi*i*tau)
           i=i+1
    qphoto=(0.666/np.power(1.0+2.045*Lambda(T),2.066))/(1.0+rho/(1.875e8*Lambda(T)+1.653e8*Lambda(T)**2+8.499e8*np.power(Lambda(T),3.0)-1.604e8*np.power(Lambda(T),4.0)));
    fphoto=(AAphoto1+AAphoto2*epsilon(rho,T)+AAphoto3*epsilon(rho,T))*np.exp(-dphoto*epsilon(rho,T))/(np.power(epsilon(rho,T),3.0)+b4/Lambda(T)+b5/Lambda(T)**2.0+b6/np.power(Lambda(T),3.0));
    return 0.5*((Cv**2.0+Ca**2.0)+n*(Cvv**2.0+Caa**2.0))*(1+((Cv**2.0-Ca**2.0)+n*(Cvv*2.0-Caa**2.0))/((Cv**2.0+Ca**2.0)+n*(Cvv**2.0+Caa**2.0))*qphoto)*rho*np.power(Lambda(T),5.0)*fphoto;

def Qphoto(rho,T,n):
    if T<1e7:
       Qp=0.0
    else:
       Qp=Qphoto_1(rho,T,n)
    return Qp


###################################################################################################################################################
#Plasmon decay
###################################################################################################################################################
def Qplasma(rho,T,n):
    rho=rho/2.0
    Gamma=np.sqrt(1.1095e11*rho/(T**2*np.sqrt(1.0+np.power(1.019e-6*rho,2.0/3.0))))
    fTT=2.4+0.6*np.sqrt(Gamma)+0.51*Gamma+1.25*np.power(Gamma,3.0/2.0);
    fL=(8.6*Gamma**2.0+1.35*np.power(Gamma,7.0/2.0))/(225.0-17.0*Gamma+Gamma**2.0);
    xx=(1.0/6.0*(17.5+np.log10(2.0*rho)-3.0*np.log10(T)))
    yy=(1.0/6.0*(-24.5+np.log10(2.0*rho)+3.0*np.log10(T)))
    if abs(xx)<0.7 or yy<0.0:
       fxy=1.0; 
    else:
       fxy=1.05+(0.39-1.25*xx-0.35*np.sin(4.5*xx)-0.3*np.exp(-np.power(4.5*xx+0.9,2.0)))*np.exp(-((min(0.0,yy-1.6+1.25*xx))/(0.57-0.25*xx))**2.0) 
    Qnu=3e21*np.power(Lambda(T),9.0)*np.power(Gamma,6.0)*np.exp(-Gamma)*(fTT+fL)*fxy
    return (Cv**2.0+n*Cvv**2.0)*Qnu


###################################################################################################################################################
#Bremsstrahlung neutrino process
###################################################################################################################################################
def Qgas(rho,T,n,A,Z):
    bb3=7.75e5*np.power(T/1e8,1.5)+247.0*np.power(T/1e8,3.85); 
    bb4=4.07+0.0024*np.power(T/1e8,1.40); bb5=4.59e-5*np.power(T/1e8,-0.110);
    aa0=23.5; aa1=6.83e4;aa2=7.81e8;aa3=230; aa4=6.70e5;aa5=7.66e9;bb1=4.47; bb2=0.0329;
    eta=rho/(7.05e6*np.power(T/1e8,1.5)+5.12e4*np.power(T/1e8,3))
    Fgas=1.0/(aa0+aa1/(T/1e8)**2.0+aa2*np.power(T/1e8,-5))+1.26*(1.0+1.0/eta)/(1.0+bb1/eta+bb2/eta**2);
    Ggas=1.0/((1.0+1e9*rho)*(aa3+aa4/(T/1e8)**2.0+aa5/np.power(T/1e8,5)))+1.0/(bb3/rho+bb4+bb5*np.power(rho,0.656));
    return 0.5738*(Z**2/A)*np.power(T/1e8,6.0)*rho*(0.5*((Cv**2.0+Ca**2.0)+n*(Cvv*2.0+Caa**2.0))*Fgas-0.5*((Cv**2.0-Ca**2.0)+n*(Cvv**2.0-Caa**2.0))*Ggas)

def Qliquid(rho,T,n,A,Z):
    u=2.0*np.pi*(np.log10(rho)-3.0)/10.0
    GammaL=2.275e-1*Z**2.0/(T/1e8)*np.power((rho/1e6)/A,1.0/3.0);
    v=-0.05483 - 0.01946*np.power(GammaL,-1.0/3.0) + 1.86310*np.power(GammaL,-2.0/3.0) - 0.78873/GammaL
    w=-0.06711 + 0.06859*np.power(GammaL,-1.0/3.0) + 1.74360*np.power(GammaL,-2.0/3.0) - 0.74498/GammaL
    fb=0.5*0.17946  + 0.00945*u+0.34529-0.05821*np.cos(u) - 0.04969*np.sin(u)-0.01089*np.cos(2.0*u)-0.01584*np.sin(2.0*u)-0.01147*np.cos(3.0*u) - 0.00504*np.sin(3.0*u)- 0.00656*cos(4.0*u) - 0.00281*np.sin(4.0*u)-0.00519*cos(5.0*u)
    ft=0.5*0.06781-0.02342*u+0.24819-0.00944*np.cos(u)-0.02213*np.sin(u)-0.01289*np.cos(2.0*u)-0.01136*np.sin(2.0*u)-0.00589*np.cos(3.0*u)- 0.00467*np.sin(3.0*u)-0.00404*np.cos(4.0*u)-0.00131*np.sin(4.0*u)-0.00330*np.cos(5.0*u) 
    gb=0.5*0.00766-0.01259*u+0.07917-0.00710*np.cos(u)+0.023*np.sin(u)-0.00028*np.cos(2.0*u)-0.01078*np.sin(2.0*u)+0.002232*np.cos(3.0*u)+ 0.00118*np.sin(3.0*u)+0.00044*np.cos(4.0*u)-0.00089*np.sin(4.0*u)+0.00158*np.cos(5.0*u)
    gt=-0.5*0.00769-0.00829*u+0.05211+0.00356*np.cos(u)+0.01052*np.sin(u)-0.00184*np.cos(2.0*u)-0.00354*np.sin(2.0*u)+0.00146*np.cos(3.0*u)- 0.00014*np.sin(3.0*u)+ 0.00031*np.cos(4.0*u)-0.00018*np.sin(4.0*u)+0.00069*np.cos(5.0*u) 
    Gliquid=w*gb+(1.0-w)*gt
    Fliquid=v*fb+(1.0-v)*ft
    return 0.5738*(Z**2/A)*np.power(T/1e8,6.0)*rho*(0.5*((Cv**2+Ca**2)+n*(Cvv**2+Caa**2))*Fliquid-0.5*((Cv**2-Ca**2)+n*(Cvv**2-Caa**2))*Gliquid)

def Qcrystal(rho,T,n,A,Z):
    fband=np.exp(-7.12e-2*Z*np.power(rho/(A*1e6),1.0/3.0)*np.power(T/1e8,-1.0))
    GammaL=2.275e-1*Z**2.0/(T/1e8)*np.power((rho/1e6)/A,1.0/3.0);
    u=2.0*np.pi*(np.log10(rho)-3.0)/9.0
    aa=[[0.03677,-0.01066,-0.00458,-0.00177,-0.00138],[0.04719,-0.01353,-0.00619,-0.00211,-0.00176],[0.00106,-0.00048,-0.00022,0.00019,-1e-5],[-0.00047,0.00063,-0.00064,0.0003,-0.00006],[0.02231,-0.00589,-0.00279,-0.00073,-0.00043],[0.00024,0.00018,-0.00028,0.00012,-0.00004]]
    bb=[[-0.00244,-0.00206,-0.00037],[0.00456,-0.00174,-0.00031],[0.00658,-0.00180,0.00036],[0.01013,-0.00247,0.00052],[-0.00095,-0.00059,0.00002],[0.00339,-0.00082,0.00015]]
    cc=[-0.01093,-0.02259,-0.00398,-0.00650,-0.00729,-0.00212]
    dd=[0.12431,0.20343,0.02499,0.04087,0.06630,0.01332]
    F1=aa[0][0]/2.0+cc[0]*u+dd[0]; F5=aa[1][0]/2.0+cc[1]*u+dd[1]
    G1=aa[2][0]/2.0+cc[2]*u+dd[2]; G5=aa[3][0]/2.0+cc[3]*u+dd[3]
    Fp=aa[4][0]/2.0+cc[4]*u+dd[4]
    Gp=aa[5][0]/2.0+cc[5]*u+dd[5]
    i=1; kk=1.0;
    while i<4:
       F1=F1+aa[0][i]*np.cos(kk*u)+bb[0][i-1]*np.sin(kk*u)
       F5=F5+aa[1][i]*np.cos(kk*u)+bb[1][i-1]*np.sin(kk*u)
       G1=G1+aa[2][i]*np.cos(kk*u)+bb[2][i-1]*np.sin(kk*u)
       G5=G5+aa[3][i]*np.cos(kk*u)+bb[3][i-1]*np.sin(kk*u)
       Fp=Fp+aa[4][i]*np.cos(kk*u)+bb[4][i-1]*np.sin(kk*u)
       Gp=Gp+aa[5][i]*np.cos(kk*u)+bb[5][i-1]*np.sin(kk*u)
       i=i+1; kk=kk+1.0
    v= 0.6252+10.6819*np.power(GammaL,-1.0/3.0)-70.6879*np.power(GammaL,-2.0/3.0)-44.3349/GammaL
    w=0.6307+10.4966*np.power(GammaL,-1.0/3.0)-68.7973*np.power(GammaL,-2.0/3.0)-50.0581/GammaL
    vv=0.5481-20.4731*np.power(GammaL,-1.0/3.0)+223.922*np.power(GammaL,-2.0/3.0)-534.94/GammaL
    ww=0.5413-20.2069*np.power(GammaL,-1.0/3.0)+220.7060*np.power(GammaL,-2.0/3.0) -524.1240/GammaL
    Flattice=(1.0-v)*F1+v*F5
    Glattice=(1.0-w)*G1+w*G5
    Fphonon=vv*Fp
    Gphonon=ww*Gp
    Qlattice=0.5738*(Z**2/A)*np.power(T/1e8,6.0)*rho*fband*(0.5*((Cv**2+Ca**2)+n*(Cvv**2+Caa**2))*Flattice-0.5*((Cv**2-Ca**2)+n*(Cvv**2-Caa**2))*Glattice)
    Qphonon=0.5738*(Z**2/A)*np.power(T/1e8,6.0)*rho*(0.5*((Cv**2+Ca**2)+n*(Cvv**2+Caa**2))*Fphonon-0.5*((Cv**2-Ca**2)+n*(Cvv**2-Caa**2))*Gphonon)
    return Qlattice

def Qbrem(rho,T,n,A,Z):
    if T>=0.3*(me*c**2/kappaB)*(np.sqrt(1.018*np.power(rho/(2.0*1e6),2.0/3.0)+1.0)-1.0):
       Brem=Qgas(rho,T,n,A,Z)
    else:
       if Gamma2(A,Z,rho)/T>=180:
          Brem=Qcrystal(rho,T,n,A,Z)
       else: 
          Brem=Qliquid(rho,T,n,A,Z)
    return Brem




