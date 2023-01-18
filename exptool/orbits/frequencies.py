"""

 frequencies.py

    part of exptool: orbit input/output from simulations, pure data processing convenience.

    18 Jan 2023 separated out for better organisation

    TODO:

    WISHLIST:

"""


# exptool imports
import numpy as np

from ..io import particle
from ..analysis import trapping
from ..utils import utils
from ..utils import kde_3d


def find_fundamental_frequency_map(OrbitInstance,time='T',pos='X',vel='VX',hanning=True,window=[0,10000],order=4):
    '''


    outputs
    -----------------
    returns first three frequencies, labeled as O+pos+[1,2,3]

    '''
    #
    lo = window[0]
    hi = window[1]

    if hi > OrbitInstance[time].shape[-1]:
        hi = OrbitInstance[time].shape[-1]

    freq = np.fft.fftfreq(OrbitInstance[time][lo:hi].shape[-1],d=(OrbitInstance[time][1]-OrbitInstance[time][0]))

    #
    norb = OrbitInstance[pos].shape[1]

    OrbitInstance['O'+pos+'1'] = np.zeros(norb)
    OrbitInstance['O'+pos+'2'] = np.zeros(norb)
    OrbitInstance['O'+pos+'3'] = np.zeros(norb)

    for orbn in range(0,norb):
        ft = OrbitInstance[pos][lo:hi,orbn] + 1.j * OrbitInstance[vel][lo:hi,orbn]
        if hanning:
            spec = np.fft.fft( ft * np.hanning(len(ft)))
        else:
            spec = np.fft.fft( ft )

        omg,val = organize_frequencies(freq,spec,order=order)

        OrbitInstance['O'+pos+'1'][orbn] = omg[0]
        OrbitInstance['O'+pos+'2'][orbn] = omg[1]
        OrbitInstance['O'+pos+'3'][orbn] = omg[2]

    return OrbitInstance





def find_fundamental_frequency(OrbitInstance,time='T',pos='X',vel='VX',hanning=True,window=[0,10000],retall=False,order=3):
    lo = window[0]
    hi = window[1]
    if hi > OrbitInstance[time].shape[-1]:
        hi = OrbitInstance[time].shape[-1]
    freq = np.fft.fftfreq(OrbitInstance[time][lo:hi].shape[-1],d=(OrbitInstance[time][1]-OrbitInstance[time][0]))
    ft = OrbitInstance[pos][lo:hi] + 1.j * OrbitInstance[vel][lo:hi]
    if hanning:
        spec = np.fft.fft( ft * np.hanning(len(ft)))
    else:
        spec = np.fft.fft( ft )

    omg,val = organize_frequencies(freq,spec,order=order)

    OrbitInstance['O'+pos+'1'] = omg[0]
    OrbitInstance['O'+pos+'2'] = omg[1]
    OrbitInstance['O'+pos+'3'] = omg[2]

    if retall:
        return OrbitInstance,freq,spec
    else:
        return OrbitInstance



def organize_frequencies(freq,fftarr,order=4):
    '''
    organize_frequencies
    -----------------------------

    inputs
    --------------------




    outputs
    --------------------



    '''

    # find maxima in the frequency spectrum
    vals = utils.argrelextrema(np.abs(fftarr.real),np.greater,order=order)[0]

    # only select from positive side (not smart, should be absolute?)
    g = np.where(freq[vals] > 0.)[0]

    # get corresponding frequencies
    gomegas = freq[vals[g]]

    # get corresponding power
    gvals = np.abs(fftarr.real)[vals[g]]

    # sort by power
    freq_order = (-1.*gvals).argsort()


    return gomegas[freq_order],gvals[freq_order]






def find_orbit_frequencies(T,R,PHI,Z,window=[0,10000]):
    '''
    calculate the peak of the orbit frequency plot

    much testing/theoretical work to be done here (perhaps see the seminal papers?)

    what do we want the windowing to look like?

    '''

    if window[1] == 10000:
        window[1] = R.shape[0]

    # get frequency values
    freq = np.fft.fftfreq(T[window[0]:window[1]].shape[-1],d=(T[1]-T[0]))

    sp_r = np.fft.fft(  R[window[0]:window[1]])
    sp_t = np.fft.fft(PHI[window[0]:window[1]])
    sp_z = np.fft.fft(  Z[window[0]:window[1]])

    # why does sp_r have a zero frequency peak??
    sp_r[0] = 0.0

    OmegaR = abs(freq[np.argmax(((sp_r.real**2.+sp_r.imag**2.)**0.5))])
    OmegaT = abs(freq[np.argmax(((sp_t.real**2.+sp_t.imag**2.)**0.5))])
    OmegaZ = abs(freq[np.argmax(((sp_z.real**2.+sp_z.imag**2.)**0.5))])


    return OmegaR,OmegaT,OmegaZ







def find_orbit_map_frequencies(OrbitInstance,window=[0,10000]):
    '''
    calculate the peak of the orbit frequency plot

    much testing/theoretical work to be done here (perhaps see the seminal papers?)

    what do we want the windowing to look like?

    '''

    try:
        x = OrbitInstance['Rp']
    except:
        print('orbit.find_orbit_frequencies: must have polar_coordinates. calculating...')
        OrbitInstance.polar_coordinates()

    if window[1] == 10000:
        window[1] = OrbitInstance['Phi'].shape[0] - 1

    # get frequency values
    freq = np.fft.fftfreq(OrbitInstance['T'][window[0]:window[1]].shape[-1],d=(OrbitInstance['T'][1]-OrbitInstance['T'][0]))

    sp_r = np.fft.fft(OrbitInstance['Rp'][window[0]:window[1]],axis=0)
    sp_t = np.fft.fft(OrbitInstance['Phi'][window[0]:window[1]],axis=0)
    sp_z = np.fft.fft(OrbitInstance['Z'][window[0]:window[1]],axis=0)

    # why does sp_r have a zero frequency peak??
    try:
        sp_r[0,:] = np.zeros(OrbitInstance['Phi'].shape[1])
        sp_z[0,:] = np.zeros(OrbitInstance['Phi'].shape[1])

    except:
        sp_r[0] = 0.
        sp_z[0] = 0.

    OmegaR = abs(freq[np.argmax(((sp_r.real**2.+sp_r.imag**2.)**0.5),axis=0)])
    OmegaT = abs(freq[np.argmax(((sp_t.real**2.+sp_t.imag**2.)**0.5),axis=0)])
    OmegaZ = abs(freq[np.argmax(((sp_z.real**2.+sp_z.imag**2.)**0.5),axis=0)])

    # check the frequencies; restrict to obits with multiple rotation periods
    #minfreq = 4./(np.max(OrbitInstance['T'][window[0]:window[1]]) - np.min(OrbitInstance['T'][window[0]:window[1]]))
    #OmegaR[np.where(OmegaR <= minfreq)[0]] = np.nan*np.ones((np.where(OmegaR <= minfreq)[0]).size)
    #OmegaT[np.where(OmegaT <= minfreq)[0]] = np.nan*np.ones((np.where(OmegaT <= minfreq)[0]).size)
    #OmegaZ[np.where(OmegaZ <= minfreq)[0]] = np.nan*np.ones((np.where(OmegaZ <= minfreq)[0]).size)


    return OmegaR,OmegaT,OmegaZ
