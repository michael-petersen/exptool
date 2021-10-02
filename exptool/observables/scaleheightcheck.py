"""

THIS CODE IS A PLACEHOLDER; SHOULD BE UPDATED

import numpy as np

from scipy.optimize import curve_fit


from exptool.utils import kde_3d
from exptool.io import particle


def scaleheight(x,hscale,A):
    return A * ((np.cosh(x/hscale))**-2.)






PSP = particle.Input('/work/mpetersen/Disk001thick/OUT.run001t.01000','star')
PSPT = trapping.BarTransform(PSP)

xbins = np.linspace(-0.04,0.04,35)
ybins = np.linspace(-0.04,0.04,35)
zbins = np.linspace(0,0.006,25)

xyz = np.zeros([len(xbins),len(ybins),len(zbins)])


iX = np.floor((PSPT.xpos - np.min(xbins))/(xbins[1]-xbins[0]))
iY = np.floor((PSPT.ypos - np.min(ybins))/(ybins[1]-ybins[0]))
iZ = np.floor((abs(PSPT.zpos) - np.min(zbins))/(zbins[1]-zbins[0]))

for indx in range(0,len(PSPT.xpos)):
    if ((iX[indx] >= 0) & (iX[indx] < len(xbins)) & (iY[indx] >= 0) & (iY[indx] < len(ybins)) & (iZ[indx] >= 0) & (iZ[indx] < len(zbins))):
        xyz[iX[indx],iY[indx],iZ[indx]] += 1.


#
# try just as a function of radius
#
rbins = np.linspace(0.,0.05,51)
R = ( PSPT.xpos*PSPT.xpos + PSPT.ypos*PSPT.ypos)**0.5

iR = np.floor((R - np.min(rbins))/(rbins[1]-rbins[0]))
iZ = np.floor((abs(PSPT.zpos) - np.min(zbins))/(zbins[1]-zbins[0]))

rz = np.zeros([len(rbins),len(zbins)])


for indx in range(0,len(PSPT.xpos)):
    if ((iR[indx] >= 0) & (iR[indx] < len(rbins)) & (iZ[indx] >= 0) & (iZ[indx] < len(zbins))):
        rz[iR[indx],iZ[indx]] += 1.





sheight = np.zeros(len(rz[:,0]))
for indx in range(0,len(rz[:,0])):
    popt, pcov = curve_fit(scaleheight, zbins, rz[indx,:],p0=[0.001,0.1])
    sheight[indx] = abs(popt[0])


plt.plot(rbins,sheight)




sheight = np.zeros([len(xbins),len(ybins)])
for indx,valx in enumerate(xbins):
    for indy,valy in enumerate(ybins):
        try:
            popt, pcov = curve_fit(scaleheight, zbins, xyz[indx,indy,:],p0=[0.001,0.1])
            if abs(popt[0]) < 0.01:
                sheight[indx,indy] = abs(popt[0])
        except:
            pass


import matplotlib.cm as cm
xx,yy = np.meshgrid(xbins,ybins)
plt.contourf(xx,yy,sheight.T,16,cmap=cm.gnuplot)
cbh = plt.colorbar()
plt.xlabel('X',size=24)
plt.ylabel('Y',size=24)
cbh.set_label('Scaleheight',size=24)

plt.text(-0.02,0.02,'T=2.0',size=24)

plt.contour(xx,yy,np.log10(np.sum(xyz,axis=2).T),16,colors='black',lw=0.8)




popt, pcov = curve_fit(scaleheight, zbins, xyz[19,17,:],p0=[0.001,0.1])


plt.plot(zbins,rz[0,:])

plt.plot(zbins, scaleheight(zbins, popt[0],popt[1]), 'r-', label='fit')



plt.plot(zbins,xyz[35,35,:])

plt.plot(zbins,xyz[27,27,:]/np.max(xyz[27,27,:]))

plt.plot(zbins,rz[0,:]/np.max(rz[0,:]))

plt.plot(zbins,scaleheight(zbins,0.001)/np.max(scaleheight(zbins,0.001)))


        

        
"""
