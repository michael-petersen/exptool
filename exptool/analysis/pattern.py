'''

pattern.py (part of exptool)
     tools to find patterns in the global simulation outputs


MSP 29 Aug 2016 Added maximum radius capabilities to bar_fourier_compute
MSP 25 Oct 2016 Some redundancies noticed (bar_fourier_compute) and should be unified


BarTransform
BarDetermine


TODO:
-Filtering algorithms for bar determination (e.g. look at better time-series algorithms)
-Need a partial pattern calculator for bars that grow and
disappear. Perhaps also in eof.py?
-Filter bad values from the pattern speed of the bar somehow
-Combine multiple coefficient series for a better estimate

BASIC USAGE Examples:

# to transform a PSP output to have the bar on the X axis
PSPTransform = pattern.BarTransform(PSPInput)

# to read in an existing bar file
BarInstance = pattern.BarDetermine()
BarInstance.read_bar(bar_file)

'''


# general imports
import time
import numpy as np
import datetime
import os
from scipy import interpolate
from scipy.interpolate import UnivariateSpline


# exptool imports
from ..io import particle
from ..utils import kmeans
from ..utils import utils


class BarFromCoefficients:
    """
    Class to calculate and analyze the bar position from Fourier coefficients.
    
    Parameters
    ----------
    times : array-like
        Array of time values.
    coefs : array-like
        Array of complex Fourier coefficients.
    unwrap_threshold : float, optional
        Threshold for unwrapping the bar position phase (default is -1.).
    smooth : bool, optional
        Whether to smooth the unwrapped position (default is False).
    reverse : bool, optional
        Whether to reverse the unwrapping direction (default is False).
    adjust : float, optional
        Adjustment value used in unwrapping (default is np.pi).
    verbose : int, optional
        Verbosity level (default is 0).
    smth_derivative : int, optional
        Smoothing factor for the polynomial derivative calculation (default is 0).
    spline_derivative : int, optional
        Smoothing factor for the spline derivative calculation (default is 0).
    """
    def __init__(self, times, coefs, unwrap_threshold=-np.pi/2., smooth=False, reverse=False, adjust=2*np.pi, verbose=0, smth_derivative=0, spline_derivative=0):
        self.verbose = verbose
        self.time = times
        self.cos = np.real(coefs)
        self.sin = np.imag(coefs)
        self.barposition = np.arctan2(self.sin, self.cos)
        # compute the unwrapped position
        self._unwrap_position(unwrap_threshold, smooth, reverse, adjust)
        # compute the derivative
        self._frequency_and_derivative(smth_derivative,spline_derivative)
    
    def _unwrap_position(self, unwrap_threshold, smooth, reverse, adjust):
        """
        Unwrap the bar position to avoid discontinuities.

        Parameters
        ----------
        unwrap_threshold : float
            Threshold for phase unwrapping.
        smooth : bool
            Whether to smooth the unwrapped position.
        reverse : bool
            Whether to reverse the unwrapping direction.
        adjust : float
            Adjustment value used in unwrapping.
        """
        running_number_of_rotations = 0
        number_of_rotations = np.zeros_like(self.barposition)
        
        # which way are we rotating
        primarydirection = np.nanmedian(np.ediff1d(B.barposition))
        
        # start from the beginning and keep track of number of rotations
        for i in range(1, len(self.barposition)):
            if (primarydirection > 0):
                if (self.barposition[i] - self.barposition[i-1]) < unwrap_threshold:
                    running_number_of_rotations += 1
            else:
                if (self.barposition[i] - self.barposition[i-1]) > -1. * unwrap_threshold:
                    running_number_of_rotations += 1
            number_of_rotations[i] = running_number_of_rotations
            
        # now straighten out the zeros
        if (primarydirection > 0):
            unwrapped_barposition = self.barposition + number_of_rotations * adjust
        else:
            unwrapped_barposition = - self.barposition + number_of_rotations * adjust
        self.pos = unwrapped_barposition
    
    def _frequency_and_derivative(self, smth_derivative,spline_derivative):
        """
        Calculate the frequency and derivative of the unwrapped bar position.

        Parameters
        ----------
        spline_derivative : int
            Smoothing factor for the spline derivative calculation.
        """
        # make a numerical derivative estimate
        self.deriv = np.zeros_like(self.pos)
        for i in range(1, len(self.pos) - 1):
            self.deriv[i] = (self.pos[i+1] - self.pos[i-1]) / (2 * (self.time[i] - self.time[i-1]))
        
        if (smth_derivative):
            smth_params = np.polyfit(self.time, self.deriv, smth_derivative)
            pos_func = np.poly1d(smth_params)
            self.deriv = pos_func(self.time)

        # hard set as a cubic spline,
        # number is a smoothing factor between knots, see scipy.UnivariateSpline
        #
        # recommended: 7 for dt=0.002 spacing
        if spline_derivative:
            spl = UnivariateSpline(self.time, self.pos, k=3, s=spline_derivative)
            self.deriv = (spl.derivative())(self.time)
            self.dderiv = np.zeros_like(self.deriv)
            #
            # can also do a second deriv
            for indx, timeval in enumerate(self.time):
                self.dderiv[indx] = spl.derivatives(timeval)[2]
    
    def print_bar(self, outfile):
        """
        Print the bar position, its derivative, and time to a file.

        Parameters
        ----------
        outfile : str
            Path to the output file.
        """
        with open(outfile, 'w') as f:
            for i in range(len(self.time)):
                print(self.time[i], self.pos[i], self.deriv[i], file=f)
        return None




class BarTransform:
    """
    BarTransform: A class to calculate the bar position and transform particles into the bar frame.

    On its own, BarTransform will reset the particles to be in the bar frame (planar transformation).

    Parameters
    ----------
    ParticleInstanceIn : object
        The input particle instance.
    bar_angle : float, optional
        The known bar angle. If None, it will be computed (default is None).
    rel_bar_angle : float, optional
        The desired rotation angle relative to the bar major axis, counterclockwise (default is 0.).
    minr : float, optional
        The minimum radius of particles to use to compute the bar angle (default is 0.).
    maxr : float, optional
        The maximum radius of particles to use to compute the bar angle (default is 1.).

    Attributes
    ----------
    ParticleInstanceIn : object
        The input particle instance.
    bar_angle : float
        The computed or provided bar angle.
    data : dict
        Dictionary containing the transformed particle data.
    time : float
        The time of the particle instance.
    filename : str
        The filename of the particle instance.
    comp : str
        The component of the particle instance.

    Methods
    -------
    calculate_transform_and_return()
        Modify the input particle instance to be in the bar frame.
    bar_fourier_compute(posx, posy, minr=0., maxr=1.)
        Use m=2 Fourier analysis to compute the bar angle.
    """

    def __init__(self, ParticleInstanceIn, bar_angle=None, rel_bar_angle=0., minr=0., maxr=1.):
        self.ParticleInstanceIn = ParticleInstanceIn
        self.bar_angle = bar_angle
        self.data = dict()

        if self.bar_angle is None:
            self.bar_angle = -1. * self.bar_fourier_compute(self.ParticleInstanceIn.data['x'], self.ParticleInstanceIn.data['y'], minr=minr, maxr=maxr)

        # Apply relative bar angle rotation
        self.bar_angle += rel_bar_angle

        # Perform the transformation
        self.calculate_transform_and_return()

    def calculate_transform_and_return(self):
        """
        Modify the input particle instance to be in the bar frame.
        """
        # Transform positions
        transformed_x = self.ParticleInstanceIn.data['x'] * np.cos(self.bar_angle) - self.ParticleInstanceIn.data['y'] * np.sin(self.bar_angle)
        transformed_y = self.ParticleInstanceIn.data['x'] * np.sin(self.bar_angle) + self.ParticleInstanceIn.data['y'] * np.cos(self.bar_angle)

        # Transform velocities
        transformed_vx = self.ParticleInstanceIn.data['vx'] * np.cos(self.bar_angle) - self.ParticleInstanceIn.data['vy'] * np.sin(self.bar_angle)
        transformed_vy = self.ParticleInstanceIn.data['vx'] * np.sin(self.bar_angle) + self.ParticleInstanceIn.data['vy'] * np.cos(self.bar_angle)

        # Update the data dictionary
        self.data['x'] = transformed_x
        self.data['y'] = transformed_y
        self.data['z'] = np.copy(self.ParticleInstanceIn.data['z'])
        self.data['vx'] = transformed_vx
        self.data['vy'] = transformed_vy
        self.data['vz'] = np.copy(self.ParticleInstanceIn.data['vz'])
        self.data['m'] = self.ParticleInstanceIn.data['m']
        self.data['potE'] = self.ParticleInstanceIn.data['potE']

        # Update metadata
        self.time = self.ParticleInstanceIn.time
        self.filename = self.ParticleInstanceIn.filename
        self.comp = self.ParticleInstanceIn.comp

    def bar_fourier_compute(self, posx, posy, minr=0., maxr=1.):
        """
        Use x and y positions to compute the m=2 Fourier phase angle.

        Parameters
        ----------
        posx : array-like
            x positions of particles.
        posy : array-like
            y positions of particles.
        minr : float, optional
            Minimum radius to consider (default is 0.).
        maxr : float, optional
            Maximum radius to consider (default is 1.).

        Returns
        -------
        float
            The m=2 phase angle.
        """
        radius = np.sqrt(posx**2 + posy**2)
        w = np.where((radius > minr) & (radius < maxr))[0]

        aval = np.sum(np.cos(2. * np.arctan2(posy[w], posx[w])))
        bval = np.sum(np.sin(2. * np.arctan2(posy[w], posx[w])))

        return np.arctan2(bval, aval) / 2.




class BarDetermine():
    '''
    #
    # class to find the bar
    #

    '''

    def __init__(self,**kwargs):

        if 'file' in kwargs:
            try:
                # check to see if bar file has already been created
                self.read_bar(kwargs['file'])
                print('pattern.BarDetermine: BarInstance sucessfully read.')

            except:
                print('pattern.BarDetermine: no compatible bar file found.')


        return None

    def track_bar(self,filelist,verbose=0,maxr=1.,apse=False):

        self.slist = filelist
        self.verbose = verbose
        self.maxr = maxr

        if apse:
            BarDetermine.cycle_files_aps(self)

        else:
            BarDetermine.cycle_files(self)

        BarDetermine.unwrap_bar_position(self)

        BarDetermine.frequency_and_derivative(self)

    def parse_list(self):
        f = open(self.slist)
        s_list = []
        for line in f:
            d = [q for q in line.split()]
            s_list.append(d[0])

        self.SLIST = np.array(s_list)

        if self.verbose >= 1:
            print('BarDetermine.parse_list: Accepted {0:d} files.'.format(len(self.SLIST)))

    def cycle_files(self):

        if self.verbose >= 2:
                t1 = time.time()

        BarDetermine.parse_list(self)

        self.time = np.zeros(len(self.SLIST))
        self.pos = np.zeros(len(self.SLIST))

        for i in range(0,len(self.SLIST)):
                O = particle.Input(self.SLIST[i],comp='star',verbose=self.verbose)
                self.time[i] = O.time
                self.pos[i] = BarDetermine.bar_fourier_compute(self,O.data['x'],O.data['y'],maxr=self.maxr)
                #BarDetermine.bar_fourier_compute(self,O.xpos,O.ypos,maxr=self.maxr)


        if self.verbose >= 2:
                print('Computed {0:d} steps in {1:3.2f} minutes, for an average of {2:3.2f} seconds per step.'.format( len(self.SLIST),(time.time()-t1)/60.,(time.time()-t1)/float(len(self.SLIST)) ))

    def cycle_files_aps(self,threedee=False,nout=100000):
        '''
        go through files, but use only the aps positions to determine the bar position. Useful if strong secondary patterns exist in m=2 power.


        EXAMPLE
        A = pattern.BarDetermine()
        A.track_bar(filelist,apse=True,maxr=0.02)


        '''

        # eventually this could be flexible!
        comp='star'

        if self.verbose >= 2:
                t1 = time.time()

        BarDetermine.parse_list(self)

        self.time = np.zeros(len(self.SLIST))
        self.pos = np.zeros(len(self.SLIST))

        for i in range(1,len(self.SLIST)-1):

            # open three files to compare
            Oa = particle.Input(self.SLIST[i-1],comp=comp,verbose=0)
            #particle.Input(self.SLIST[i-1],legacy=True,comp=comp,nout=nout,verbose=0)
            Ob = particle.Input(self.SLIST[i],comp=comp,verbose=self.verbose)
            #particle.Input(self.SLIST[i],legacy=True,comp=comp,nout=nout,verbose=self.verbose)
            Oc = particle.Input(self.SLIST[i+1],comp=comp,verbose=0)
            #particle.Input(self.SLIST[i+1],legacy=True,comp=comp,nout=nout,verbose=0)

            # compute 2d radial positions
            if threedee:
                Oa.R = (Oa.data['x']*Oa.data['x'] + Oa.data['y']*Oa.data['y'] + Oa.data['z']*Oa.data['z'])**0.5
                #(Oa.xpos*Oa.xpos + Oa.ypos*Oa.ypos + Oa.zpos*Oa.zpos)**0.5
                Ob.R = (Ob.data['x']*Ob.data['x'] + Ob.data['y']*Ob.data['y'] + Ob.data['z']*Ob.data['z'])**0.5
                #(Ob.xpos*Ob.xpos + Ob.ypos*Ob.ypos + Ob.zpos*Ob.zpos)**0.5
                Oc.R = (Oc.data['x']*Oc.data['x'] + Oc.data['y']*Oc.data['y'] + Oc.data['z']*Oc.data['z'])**0.5
                #(Oc.xpos*Oc.xpos + Oc.ypos*Oc.ypos + Oc.zpos*Oc.zpos)**0.5

            else:
                Oa.R = (Oa.data['x']*Oa.data['x'] + Oa.data['y']*Oa.data['y'])**0.5
                #(Oa.xpos*Oa.xpos + Oa.ypos*Oa.ypos)**0.5
                Ob.R = (Ob.data['x']*Ob.data['x'] + Ob.data['y']*Ob.data['y'])**0.5
                #(Ob.xpos*Ob.xpos + Ob.ypos*Ob.ypos)**0.5
                Oc.R = (Oc.data['x']*Oc.data['x'] + Oc.data['y']*Oc.data['y'])**0.5
                #(Oc.xpos*Oc.xpos + Oc.ypos*Oc.ypos)**0.5

            # use logic to find aps
            aps = np.logical_and( Ob.R > Oa.R, Ob.R > Oc.R )


            xposlist = Ob.data['x'][aps] #Ob.xpos[aps]
            yposlist = Ob.data['y'][aps]#Ob.ypos[aps]

            self.time[i] = Ob.time
            self.pos[i] = BarDetermine.bar_fourier_compute(self,xposlist,yposlist,maxr=self.maxr)


        if self.verbose >= 2:
                print('Computed {0:d} steps in {1:3.2f} minutes, for an average of {2:3.2f} seconds per step.'.format( len(self.SLIST),(time.time()-t1)/60.,(time.time()-t1)/float(len(self.SLIST)) ))



    def bar_doctor_print(self):

        #
        # wrap the bar file
        #
        BarDetermine.unwrap_bar_position(self)

        BarDetermine.frequency_and_derivative(self)

        BarDetermine.print_bar(self,outfile)



    def unwrap_bar_position(self,jbuffer=-1.,smooth=False,reverse=False,adjust=np.pi):


        #
        # modify the bar position to smooth and unwrap
        #
        jnum = 0
        jset = np.zeros_like(self.pos)


        for i in range(1,len(self.pos)):

            if reverse:
                if (self.pos[i]-self.pos[i-1]) > -1.*jbuffer:   jnum -= 1

            else:
                if (self.pos[i]-self.pos[i-1]) < jbuffer:   jnum += 1

            jset[i] = jnum

        unwrapped_pos = self.pos + jset*adjust

        if (smooth):
            unwrapped_pos = helpers.savitzky_golay(unwrapped_pos,7,3)

        # to unwrap on twopi, simply do:
        #B.bar_upos%(2.*np.pi)

        self.pos = unwrapped_pos

        #
        # this implementation is not particularly robust, could revisit in future

    def frequency_and_derivative(self,smth_order=None,fft_order=None,spline_derivative=None,verbose=0):



        if (smth_order or fft_order):

            if (verbose):

                print('Cannot assure proper functionality of both order smoothing and low pass filtering.')

        self.deriv = np.zeros_like(self.pos)
        for i in range(1,len(self.pos)):
            self.deriv[i] = (self.pos[i]-self.pos[i-1])/(self.time[i]-self.time[i-1])


        if (smth_order):
            smth_params = np.polyfit(self.time, self.deriv, smth_order)
            pos_func = np.poly1d(smth_params)
            self.deriv = pos_func(self.time)

        if (fft_order):
            self.deriv = self.deriv

        if (spline_derivative):

            # hard set as a cubic spline,
            #    number is a smoothing factor between knots, see scipy.UnivariateSpline
            #
            #    recommended: 7 for dt=0.002 spacing

            spl = UnivariateSpline(self.time, self.pos, k=3, s=spline_derivative)
            self.deriv = (spl.derivative())(self.time)

            self.dderiv = np.zeros_like(self.deriv)
            #
            # can also do a second deriv
            for indx,timeval in enumerate(self.time):

                self.dderiv[indx] = spl.derivatives(timeval)[2]



    def bar_fourier_compute(self,posx,posy,maxr=1.):

        #
        # use x and y positions tom compute the m=2 power, and find phase angle
        #
        w = np.where( (posx*posx + posy*posy)**0.5 < maxr )[0]

        aval = np.sum( np.cos( 2.*np.arctan2(posy[w],posx[w]) ) )
        bval = np.sum( np.sin( 2.*np.arctan2(posy[w],posx[w]) ) )

        return np.arctan2(bval,aval)/2.



    def print_bar(self,outfile):

        #
        # print the barfile to file
        #

        # this will be broken in python 3 compatibility

        f = open(outfile,'w')

        for i in range(0,len(self.time)):
            print(self.time[i],self.pos[i],self.deriv[i],file=f)

        f.close()

        return None

    def place_ellipse(self):

        return None

    def read_bar(self,infile):

        #
        # read a printed bar file
        #

        f = open(infile)

        time = []
        pos = []
        deriv = []
        for line in f:
            q = [float(d) for d in line.split()]
            time.append(q[0])
            pos.append(q[1])
            try:
                deriv.append(q[2])
            except:
                pass

        self.time = np.array(time)
        self.pos = np.array(pos)
        self.deriv = np.array(deriv)

        if len(self.deriv) < 1:

            BarDetermine.frequency_and_derivative(self)





def find_barangle(time, BarInstance, interpolate=True):
    """
    Use a bar instance to match the output time to a bar position.

    Parameters
    ----------
    time : array-like
        Array of time values at which to find the bar position.
    BarInstance : object
        An instance of a class (such as `BarFromCoefficients`) that contains
        bar positions and corresponding times.
    interpolate : bool, optional
        Whether to interpolate the bar position using a spline. If False,
        the function finds the closest available bar position (default is True).

    Returns
    -------
    indx_barpos : array-like
        Array of bar positions corresponding to the input time values.

    Notes
    -----
    This function can take arrays as input. It currently handles only one
    direction of bar position matching and places a guard against NaN values
    in the bar positions.
    """
    # Place a guard against NaN values
    BarInstance.pos[np.isnan(BarInstance.pos)] = 0.

    sord = 0  # Should this be a variable?

    if interpolate:
        not_nan = np.where(np.isnan(BarInstance.pos) == False)
        bar_func = UnivariateSpline(BarInstance.time[not_nan], -BarInstance.pos[not_nan], s=sord)

    try:
        indx_barpos = np.zeros([len(time)])
        for indx, timeval in enumerate(time):
            if interpolate:
                indx_barpos[indx] = bar_func(timeval)
            else:
                indx_barpos[indx] = -BarInstance.pos[np.abs(timeval - BarInstance.time).argmin()]
    except TypeError: 
        if interpolate:
            indx_barpos = bar_func(time)
        else:
            indx_barpos = -BarInstance.pos[np.abs(time - BarInstance.time).argmin()]

    return indx_barpos

def find_barpattern(intime, BarInstance, smth_order=2):
    """
    Use a bar instance to match the output time to a bar pattern speed.

    Parameters
    ----------
    intime : array-like
        Array of time values at which to find the bar pattern speed.
    BarInstance : object
        An instance of a class (such as `BarFromCoefficients`) that contains
        bar positions, times, and their derivatives.
    smth_order : int, optional
        Smoothing factor for the derivative calculation (default is 2).

    Returns
    -------
    barpattern : array-like
        Array of bar pattern speeds corresponding to the input time values.
    """

    # Compute the derivative of the bar position at the specified smoothing order
    BarInstance._frequency_and_derivative(spline_derivative=smth_order)

    try:
        # Initialize an array to hold the bar pattern speeds
        barpattern = np.zeros([len(intime)])

        # Loop over each time value in the input array
        for indx, timeval in enumerate(intime):
            # Find the index of the closest time in BarInstance.time
            best_time = abs(timeval - BarInstance.time).argmin()

            # Get the derivative (bar pattern speed) at the closest time
            barpattern[indx] = BarInstance.deriv[best_time]

    except:
        # Handle the case where intime is a single value
        best_time = abs(intime - BarInstance.time).argmin()

        # Get the derivative (bar pattern speed) at the closest time
        barpattern = BarInstance.deriv[best_time]

    return barpattern


class BarFromFourier:
    """
    Class to compute the bar pattern speed and phase angle using Fourier analysis.
    
    This class replaces the EOF information if it appears to be incorrect. The output file
    formats are designed to be identical to those generated using EOF information.

    Parameters
    ----------
    inputfiles : str
        Path to a file containing a list of input files to be processed.

    Attributes
    ----------
    slist : str
        Path to the input files list.
    SLIST : array-like
        List of input files parsed from the file.
    pos : array-like
        Array of bar positions.
    deriv : array-like
        Array of bar pattern speeds (derivatives).
    time : array-like
        Array of time steps corresponding to the bar positions and speeds.
    """

    def __init__(self, inputfiles):
        self.slist = inputfiles

    def parse_list(self):
        """
        Parse the list of input files from the provided file path.

        This method reads the file specified in `self.slist` and stores the list
        of input files in the attribute `self.SLIST`.
        """
        with open(self.slist, 'r') as f:
            s_list = [line.split()[0] for line in f]
        self.SLIST = np.array(s_list)

    def bar_fourier_compute(self, posx, posy, maxr=0.5, minr=0.001):
        """
        Compute the m=2 Fourier phase angle from particle positions.

        Parameters
        ----------
        posx : array-like
            x positions of particles.
        posy : array-like
            y positions of particles.
        maxr : float, optional
            Maximum radius to consider (default is 0.5).
        minr : float, optional
            Minimum radius to consider (default is 0.001).

        Returns
        -------
        float
            The m=2 phase angle.
        """
        # Select particles within the specified radius range
        radius = np.sqrt(posx**2 + posy**2)
        w = np.where((radius < maxr) & (radius > minr))[0]

        # Compute m=2 Fourier components
        aval = np.sum(np.cos(2. * np.arctan2(posy[w], posx[w])))
        bval = np.sum(np.sin(2. * np.arctan2(posy[w], posx[w])))

        return np.arctan2(bval, aval) / 2.

    def bar_speed(self, filelist, comp='star'):
        """
        Compute the bar pattern speed from the list of input files.

        Parameters
        ----------
        filelist : array-like
            List of input files to process.
        comp : str, optional
            Component to analyze (default is 'star').

        Returns
        -------
        dict
            Dictionary containing arrays of time steps, bar positions, and bar pattern speeds.
        """
        self.slist = filelist
        self.parse_list()
        
        pos = particle.Input(self.SLIST[0], comp=comp, verbose=0)
        pos_p1 = particle.Input(self.SLIST[1], comp=comp, verbose=0)
        first_bar_angle = self.bar_fourier_compute(pos.data['x'], pos.data['y'])
        
        # Calculate the timestep
        timestep = pos_p1.time - pos.time
        tt = np.array([])
        pp = np.array([])
        rot = np.array([])
        
        for i in range(len(self.SLIST)):
            pos = particle.Input(self.SLIST[i], comp=comp, verbose=0)
            bar_angle = self.bar_fourier_compute(pos.data['x'], pos.data['y'])
            
            if i == 0:
                old_bar_angle = first_bar_angle
            
            pattern_speed = (old_bar_angle - bar_angle) / timestep
            
            if abs(old_bar_angle - bar_angle) >= (np.pi * 3 / 4):
                pattern_speed = (old_bar_angle - (bar_angle + np.pi)) / timestep
            
            pp = np.append(pp, bar_angle)
            rot = np.append(rot, pattern_speed)
            tt = np.append(tt, pos.time)
            
            old_bar_angle = bar_angle
        
        self.pos = pp
        self.deriv = rot
        self.time = tt
        
        return {'time': tt, 'pos': pp, 'deriv': rot}

    def print_bar(self, simulation_directory, simulation_name):
        """
        Print the bar positions and pattern speeds to a file.

        Parameters
        ----------
        simulation_directory : str
            Directory where the output file will be saved.
        simulation_name : str
            Base name for the output file.
        """
        output_file = simulation_directory + simulation_name + 'fourier_barpos.dat'
        
        with open(output_file, 'w') as f:
            for i in range(len(self.SLIST)):
                print(self.time[i], self.pos[i], self.deriv[i], file=f)
        
        return None
