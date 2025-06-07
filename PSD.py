import numpy as np

C_SI = 3e8

def Sacc(f, SA):
    index = np.where(f != 0)
    sa = np.zeros_like(f)
    sa[index] = SA/(2*np.pi*f[index]*C_SI)**2 *(1+1e-4/f[index])
    return sa

def Spos(f, SP):
    return SP*(2*np.pi*f/C_SI)**2

def Sx(f, L=1.7e8/C_SI, SA = 1e-4, SP = 1e-4):
    omg = 2*np.pi*f*L
    return 16*np.sin(omg)**2 * (2*(1+np.cos(omg)**2)*Sacc(f, SA) + Spos(f, SP))

def Sxy(f, L=1.7e8/C_SI, SA = 1e-4, SP = 1e-4):
    omg = 2*np.pi*f*L
    return -8*np.sin(omg)**2 * np.cos(omg)*(4*Sacc(f, SA)+Spos(f, SP))