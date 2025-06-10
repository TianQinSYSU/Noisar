import numpy as np
from PSD import Sx,Sxy

DAY = 86400
Tobs = 10*DAY
df = 1/Tobs
dt = 0.5
fmax = 1/(2*dt)
freq = np.arange(0, fmax+0.1*df, df)

psdx0 = Sx(freq, SA=1e-30, SP=1e-24)
psdxy = Sxy(freq, SA=1e-30, SP=1e-24)

# the 0th element of psdx0&psdxy is 0.0
# in case the nan value caused by dividing by 0.0, artificially set:
psdx0[0]=psdx0[1]
psdxy[0]=psdxy[1]
# It's not influential, since we only consider frequency domain >10^-4Hz

# add some disturb to psdx in low frequency domain
freqlog = np.logspace(0,-6,len(freq))
disturb = 2e-46*(-3*np.sin(8*np.pi*freqlog)-0.5*np.cos(4*freqlog+90)**2+0.2*np.cos(freq)+0.5*np.sin(30*freqlog))
disturb[-300000:-1]=0
psdx = psdx0+disturb

sigmaxx = (0.25 * psdx / df)
sigmaxy = (0.25 * psdxy / df)

COV = np.array([[sigmaxx, sigmaxy, sigmaxy], [sigmaxy, sigmaxx,sigmaxy], [sigmaxy, sigmaxy, sigmaxx]])
NDIM = freq.shape[0]

nxr, nyr, nzr = np.array([np.random.multivariate_normal([0,0,0], COV[:,:,i]) for i in range(NDIM)]).T
nxi, nyi, nzi = np.array([np.random.multivariate_normal([0,0,0], COV[:,:,i]) for i in range(NDIM)]).T

nx = nxr+1j*nxi
ny = nyr+1j*nyi
nz = nzr+1j*nzi
nxt = np.fft.irfft(nx)
nyt = np.fft.irfft(ny)
nzt = np.fft.irfft(nz)

t = np.arange(0,Tobs,dt)
savearr = np.array([t,nxt,nyt,nzt])
np.save('figs/TQ_XYZ_t', savearr.T)
