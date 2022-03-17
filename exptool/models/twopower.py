'''


twopower (part of exptool.models)
    Implementation of Martin Weinberg's twopower.cc


20 Oct 2017  First construction
04 Feb 2021  revamp

Generalization is (BT 2.64)

                           rho0
   rho(r) = -------------------------------
            (r/a)^alpha (1+r/a)^(beta-alpha)

where alpha=1, beta=4 for the Hernquist model

(beta=4 is the dehnen family of models. dehnen models with alpha in [0.6,2] reasonably describe elliptical galaxy centers)
(alpha=2, beta=4 is Jaffe model)
(alpha=1, beta=3 is NFW model)

Enclosed mass is then

                                               s^{2-\alpha}
    M(r) = 4\pi \rho_0 a^3 \int_0^{r/a} ds --------------------
                                            (1+s)^{\beta-alpha}




if you'd like a MW-like model, try this:




'''

# general python imports
import numpy as np



def Hernquist(r,a,rc=0,rho0=1.,beta=0):
    """return unscaled Herquist density

    r      : the radius to evaluate
    a      : the scale radius
    rc     : the core radius
    rho0   : the central density


    """
    ra = r/a
    return rho0/((ra+rc)*((1+ra)**3.))

def NFW(r,a,rc,beta=0):
    """return unscaled NFW density"""
    ra = r/a
    return 1./((ra+rc)*((1+ra)**2.))


def powerhalo(r,rs=1.,rc=0.,alpha=1.,beta=1.e-7):
    """return generic twopower law distribution

    inputs
    ----------
    r      : (float) radius values
    rs     : (float, default=1.) scale radius
    rc     : (float, default=0. i.e. no core) core radius
    alpha  : (float, default=1.) inner halo slope
    beta   : (float, default=1.e-7) outer halo slope

    returns
    ----------
    densities evaluated at r

    notes
    ----------
    different combinations are known distributions.
    alpha=1,beta=2 is NFW
    alpha=1,beta=3 is Hernquist
    alpha=2.5,beta=0 is a typical single power law halo


    """
    ra = r/rs
    return 1./(((ra+rc)**alpha)*((1+ra)**beta))



def make_twopower_model(func,M,R,rs,alpha=1.,beta=1.e-7,rc=0.,\
                        pfile='',plabel='',\
                        verbose=False,truncate=True,\
                        perc=97.,numr=4000):
    """make a two power distribution

    inputs
    -------------
    func
    M
    R
    rs
    alpha
    beta
    rc
    pfile
    plabel
    verbose
    truncate
    perc
    numr


    """

    # set the radius values
    rvals = 10.**np.linspace(-6.,np.log10(2.),numr)

    # original flexible version
    rtrunc = np.nanpercentile(rvals,perc)

    # hardwired so edge is sharp
    rtrunc = np.nanpercentile(rvals,99.9)
    wtrunc = np.nanpercentile(rvals,99.9)-np.nanpercentile(rvals,perc)

    if verbose:
        print('Truncation settings: rtrunc={0:3.2f},wtrunc={1:3.2f}'.format(rtrunc,wtrunc))


    # query out the density values
    dvals = func(rvals,rs,rc,alpha=alpha,beta=beta)

    # apply the truncation
    if truncate:
        dvals *= 0.5*(1.0 - scipy.special.erf((rvals-rtrunc)/wtrunc))

    # make the mass and potential arrays
    mvals = np.zeros(dvals.size)
    pvals = np.zeros(dvals.size)
    pwvals = np.zeros(dvals.size)

    mvals[0] = 1.e-15
    pwvals[0] = 0.

    # evaluate mass enclosed
    for indx in range(1,dvals.size):
        mvals[indx] = mvals[indx-1] +\
          2.0*np.pi*(rvals[indx-1]*rvals[indx-1]*dvals[indx-1] +\
                 rvals[indx]*rvals[indx]*dvals[indx])*(rvals[indx] - rvals[indx-1]);
        pwvals[indx] = pwvals[indx-1] + \
          2.0*np.pi*(rvals[indx-1]*dvals[indx-1] + rvals[indx]*dvals[indx])*(rvals[indx] - rvals[indx-1]);

    # evaluate potential
    for indx in range(0,dvals.size):
        pvals[indx] = -mvals[indx]/(rvals[indx]+1.e-10) - (pwvals[dvals.size-1] - pwvals[indx])


    # prepare to rescale by potential energy
    W0 = 0.0;
    for indx in range(0,dvals.size):
        W0 += np.pi*(rvals[indx-1]*rvals[indx-1]*dvals[indx-1]*pvals[indx-1] + \
                rvals[indx]*rvals[indx]*dvals[indx]*pvals[indx-1])*(rvals[indx] - rvals[indx-1]);

    if verbose:
        print("orig PE = ",W0 )

    M0 = mvals[dvals.size-1]
    R0 = rvals[dvals.size-1]

    # compute scaling factors
    Beta = (M/M0) * (R0/R);
    Gamma = np.sqrt((M0*R0)/(M*R)) * (R0/R);
    if verbose:
        print("! Scaling:  R=",R,"  M=",M)

    rfac = np.power(Beta,-0.25) * np.power(Gamma,-0.5);
    dfac = np.power(Beta,1.5) * Gamma;
    mfac = np.power(Beta,0.75) * np.power(Gamma,-0.5);
    pfac = Beta;

    if verbose:
        print(rfac,dfac,mfac,pfac)

    # save file if desired
    if pfile != '':
        f = open(pfile,'w')
        print('! ',plabel,file=f)
        print('! R    D    M    P',file=f)

        print(rvals.size,file=f)

        for indx in range(0,rvals.size):
            print('{0:12.10f} {1:15.7f} {2:16.15f} {3:16.14f}'.format( rfac*rvals[indx],\
              dfac*dvals[indx],\
              mfac*mvals[indx],\
              pfac*pvals[indx]),file=f)

        f.close()

    return rvals*rfac,dfac*dvals,mfac*mvals,pfac*pvals


def check_concentration(R,D):
    """
    check the  concentration of a halo
    by finding where the power law is most similar to alpha^-2

    return 1./radius, which is the concentration.
    (so find the scale radius by taking 1./concentration)

    """
    func = np.log10(R**-2.)-np.log10(D)
    print('Concentration={}'.format(1./R[np.nanargmin(func)]))
