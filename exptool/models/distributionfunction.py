"""

class to build distribution functions from spherically symmetric systems

02-Mar-2021: introduction

This draws heavily from https://github.com/rerrani/nbopy ! Please
check it out, as well as the accompanying paper.

supporting:
-simplest way to paint stars on top of a dark matter distribution as
tracer particles



"""
import numpy as np

from scipy import interpolate


class DF:
    """simple and straightforward DF construction.


    """

    def __init__(self, filename,NE=1000,NR=1000,Rmax=2,Rmin=1e-4,verbose=0):
        """
        inputs
        ------------
        modelfile  : str
            the path to the modelfile to use

        the modelfile must follow a set system
        (fix to also take RDMP?
        
        """
        
        sphmod = np.genfromtxt(modfile,skip_header=4)

        self.R,self.D,self.M,self.P = sphmod[:,0],sphmod[:,1],sphmod[:,2],sphmod[:,3]

        # set the parameters
        self.NE = NE
        self.NR = NR


        self.R  = np.logspace(np.log10(Rmin),np.log10(Rmax),num=int(NR))
        

        self.EPSREL = 1e-6     # integration relative precision

        _construct_interpolations()
        _set_energy_range()




    def _construct_interpolations(self):
        # do spline representations of the other quantities to make a function
        self.nu   = interpolate.UnivariateSpline(R,D,s=0)
        self.Mcum = interpolate.UnivariateSpline(R,M,s=0)
        self.psi  = interpolate.UnivariateSpline(R,-1*P,s=0)

        # total mass
        self.Mtot = np.nanmax(M)

        # compute gradients (can always make sure this worked!)
        self.dndp   = np.gradient(nu,   psi,edge_order=2)
        self.d2nd2p = np.gradient(dndp, psi,edge_order=2)

        # make into interpolateable functions
        self.dndpfunc = interpolate.splrep(R,nu, s=0)
        self.psifunc = interpolate.splrep(R,psi, s=0)


    def _compute_distribution_function(self):

        if self.verbose:
            print("distributionfunction.DF: constructing distribution function...")

        # see equation (4.78a) in BT08
        f = np.vectorize( lambda e: 1./(np.sqrt(8)*np.pi*np.pi) * (quad( lambda p:  np.interp(p, self.psi[::-1], self.d2nd2p[::-1]) / np.sqrt(e-p) , 0., e,  epsrel=self.EPSREL)[0]  ) )


        self.DF = f(self.E)               # here we build the DF

        # check whether DF is physical - if not, check for rounding errors in integration, try increasing/decreasing NR, NE
        if np.any (self.DF < 0): 
            print("      *  Exit. DF < 0, see df.dat" )
        else: 
            print("      *  DF >= 0, all good" )

        if self.verbose:
            print("done!")
            
    def _set_energy_range(self):
        
        maxE = self.psi[0]           # most-bound energy
        minE = maxE/float(self.NE)   # least-bound energy
        self.E = np.linspace(minE,maxE,num=self.NE)


    def _dPdr(self,e,r): 
        """phase space volume element per energy accessible at given energy and fixed radius"""
        return np.sqrt( 2.*(np.interp(r, self.R, self.psi) - e )) * r*r     
    
    def _PLikelihood(self,e,r):
        """likelihood of a particle to have energy E at a fixed radius"""
        return np.interp(e,self.E,self.DF) * _dPdr(e,r)

    def _build_dP(self):
        """phase space volume accessible at a given energy, i.e. integrated over all admitted radii"""
        dP = np.vectorize( lambda e: quad( lambda r: dPdr(e,r), 0, np.interp(e,psi[::-1],R[::-1]),\
                                  epsrel=self.EPSREL )[0] ) # rmax = np.interp(e,psi,R)

        # differential energy distribution / (4pi)^2
        self.dNdE = self.DF * dP(self.E)


    def _find_maximum_likelihood(self):

        if self.verbose:
            print("distributionfunction.DF: finding maximum likelihood" )
            
        maxPLikelihood = []
        for RR in self.R:
            allowed = np.where (self.E <= np.interp(RR,self.R,psi) )[0]
            ThismaxPLikelihood = 1.1 * np.amax( PLikelihood(self.E[allowed],RR) )   # 10 per cent tolerance
            maxPLikelihood.append(ThismaxPLikelihood)

        self.maxPLikelihood = np.array(maxPLikelihood)


    def rhoS(r,aS=2.,bS=5.0,gS=0.1,rsS = 0.2):
        """alpha-beta-gamma density profile, massless tracers (think: stars)

           NB: gS needs to be >0, see Appendix A of EP19
           rsS is the ratio of stellar and DM scale radius):
        """

        return r**(-gS) * (rsS**aS + r**aS )**((gS-bS)/aS) 

    def simple_density(r,alphaS,rcore,rtrunc,wtrunc):   
        """
        one-power model with a core for formal non-divergence, PLUS a truncation.
    
        """
        rho = ((r+rcore)**-alphaS)# * ((r)**-alphaS))
    
        return rho *  (1. - erf( (r-rtrunc)/wtrunc ))


    def reweight(self,alpha,rcore=1.e-6,rtrunc=0.5,wtrunc=0.1):
        """
        reweight the distribution function to a different distribution


        """
        # non-normalized total mass
        MS  =  4 * pi * quad( lambda r: r*r * simple_density(r,alpha,rcore,rtrunc,wtrunc) , 0., np.inf)[0]
        MSr  = np.vectorize(lambda x: 4*pi *  quad( lambda r: r*r * stellar_density(r,alphaS=alphaS) /MS , 0., x  )[0]  )

        nuS  =   simple_density(self.R,alpha,rcore,rtrunc,wtrunc) / MS
        MScum = MSr(self.R)  # stellar cumulative mass

        dndpS   = np.gradient(nuS,   self.psi)
        d2nd2pS = np.gradient(dndpS, self.psi)
        fS = np.vectorize( lambda e: 1./(np.sqrt(8)*np.pi*np.pi) * (quad( lambda p:  np.interp(p, self.psi[::-1], d2nd2pS[::-1]) /np.sqrt(e-p) , 0., e,  epsrel=self.EPSREL)[0]  ) ) # + np.interp(0., psi, dndp) / np.sqrt(e)   == 0 due to B.C.

        # compute the distribution function
        self.DFS = fS(self.E)

        # now for each EE, compute the weights
        #probs = np.interp(EE, E, DFS)/ np.interp(EE, E, DF)
        #norm = np.sum(probs)
        #probs = probs/norm




    def realise_model(self):

        N=1e5
        Ndraw=10000

        print("      *  Allocating memory for N-body" )
        xx = np.zeros(int(N),dtype="float16")
        yy = np.zeros(int(N),dtype="float16")
        zz = np.zeros(int(N),dtype="float16")
        vx = np.zeros(int(N),dtype="float16")
        vy = np.zeros(int(N),dtype="float16")
        vz = np.zeros(int(N),dtype="float16")

        print("      *  Draw particles" )




        n=0             # current number of generated 'particles'
        Efails = 0      # rejection sampling failures in Energy


        # while we still need to generate 'particles'..
        while (n < N):
            # inverse transform sampling for R
            #randMcum = Nin/float(N)  + (1.-(Nin+Nout)/float(N)) * np.random.rand(int(Ndraw))
            #randMcum = Nin/float(N)  + (M-(Nin+Nout)/float(N)) * np.random.rand(int(Ndraw))
            randMcum = Nin/float(N)  + (1.-(Nin+Nout)/float(N)) * np.random.rand(int(Ndraw))
            # inverse gives radius distributed like rho(r) w/o particles inside/outside Rmin/Rmax 
            #randR = np.interp(randMcum ,Mcum,R)   

            #print(randMcum)
            randR = np.interp(randMcum ,Mcum,R)        # inverse gives radius distributed like rho(r)
            #print(randR)
            # rejection sampling for E
            psiR = np.interp(randR ,R,psi)             # potential at radius
            randE = np.random.rand(int(Ndraw)) * psiR  # random E with constraint that  E > 0 but E < Psi 
            rhoE  = PLikelihood(randE,randR)           # likelihood for E at given R
            randY = np.random.rand(int(Ndraw)) * np.interp(randR,R, maxPLikelihood) 
            Missidx = np.where(randY > rhoE)[0]        # sampled energies rejected
            Efails += len(Missidx)
  
            # repeat sampling at fixed R till we got all the energies we need
            while len(Missidx):
                randE[Missidx] = np.random.rand(len(Missidx)) * psiR[Missidx]  
                rhoE[Missidx]  = PLikelihood(randE[Missidx],randR[Missidx])
                randY[Missidx] = np.random.rand(len(Missidx)) * np.interp(randR[Missidx],R, maxPLikelihood)
                Missidx = np.where(randY > rhoE)[0]
                Efails += len(Missidx)
    
            okEidx = np.where(randY <= rhoE)[0]
  
            if len(okEidx) != int(Ndraw):             # this should never happen.
                print("      *  Particles went missing. Exit." )
  
            # Let's select as many R,E combinations as we're still missing to get N particles in total
            missing = int(N) - int(n)
            if len(okEidx) <= missing: 
                arraxIdx = n + np.arange(0,len(okEidx))
            else: 
                arraxIdx = n + np.arange(0,missing)
                okEidx = okEidx[:missing]

  
            # spherical symmetric model, draw random points on sphere
            Rtheta  = np.arccos (2. * np.random.rand(len(okEidx)) - 1.)
            Rphi    = np.random.rand(len(okEidx)) * 2*np.pi

            # isotropic velocity dispersion, draw random points on sphere
            Vtheta  = np.arccos (2. * np.random.rand(len(okEidx)) - 1.)
            Vphi    = np.random.rand(len(okEidx)) * 2*np.pi
            V =  np.sqrt( 2.*(psiR[okEidx] - randE[okEidx] ))  
  
            # spherical to cartesian coordinates 
            xx[arraxIdx] = randR[okEidx] * np.sin(Rtheta) * np.cos(Rphi)
            yy[arraxIdx] = randR[okEidx] * np.sin(Rtheta) * np.sin(Rphi)
            zz[arraxIdx] = randR[okEidx] * np.cos(Rtheta) 
  
            vx[arraxIdx] = V * np.sin(Vtheta) * np.cos(Vphi)
            vy[arraxIdx] = V * np.sin(Vtheta) * np.sin(Vphi)
            vz[arraxIdx] = V * np.cos(Vtheta)

            n += len(okEidx)
            print ("         %.2f per cent; E rejection ratio %.2f "%(100* n/float(N) ,  100* Efails/float(n)  ) )

            print("      *  Writing output file" )

            if modelname[-4:] == ".npy":
                np.save(modelname,np.column_stack((xx,yy,zz,vx,vy,vz)) )   # save npy array
            else:
                np.savetxt(modelname,np.column_stack((xx,yy,zz,vx,vy,vz)) )  # ASCII output


            print("      *  All done :o)" )

        
        
